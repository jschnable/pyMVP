"""
FWL+QR GLM implementation for fast per-SNP association scans.

Algorithm:
- Build covariate matrix X = [1 | CV] and compute thin QR: X = Q R.
- Residualize phenotype once: y_r = y - Q(Q^T y).
- Process SNPs in batches G:
  - Impute missing values (-9/NaN) to the per-SNP major allele.
  - Residualize genotypes: G_r = G - Q(Q^T G).
  - Vectorized stats per SNP j:
      gTy = G_r.T @ y_r
      gTg = sum(G_r^2, axis=0)
      beta = gTy / gTg
      SSE = y_r·y_r - (gTy^2)/gTg
      sigma2 = SSE / df,  df = n - p - 1
      se = sqrt(sigma2 / gTg)
      t = beta / se,   p = 2 * sf(|t|, df)

Effect/SE scaling:
- Match pyMVP/rMVP effect scale post-hoc (does not change p-values):
  divide by per-marker SD of imputed genotypes, then multiply by 0.656.

This file exposes MVP_GLM_ultrafast with the same signature as MVP_GLM.
Use tests/quick_validation_test.py to validate before integration.
"""

from typing import Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import special

from panicle.utils.data_types import GenotypeMatrix, AssociationResults


def _fast_t_pvalue(t_stats: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Fast vectorized two-tailed t-test p-value calculation.

    Uses a normal approximation (erfc) for speed on large arrays.

    Args:
        t_stats: Array of absolute t-statistics
        df: Array of degrees of freedom (same shape as t_stats)

    Returns:
        Two-tailed p-values
    """
    p = special.erfc(t_stats / np.sqrt(2.0))
    return np.clip(p, 0.0, 1.0)


def _load_genotype_batch(
    geno: Union[GenotypeMatrix, np.ndarray],
    start: int,
    end: int,
    use_gm: bool,
    is_imputed: bool,
    missing_fill_value: float
) -> np.ndarray:
    """Load and optionally impute a genotype batch. Thread-safe for prefetching."""
    if use_gm:
        if is_imputed:
            return geno.get_batch_imputed(start, end, fill_value=None, dtype=np.float32)
        else:
            return geno.get_batch_imputed(start, end, fill_value=missing_fill_value, dtype=np.float32)
    else:
        return _impute_numpy_batch_major_allele(
            geno[:, start:end], fill_value=missing_fill_value, dtype=np.float32
        )


def _impute_numpy_batch_major_allele(batch: np.ndarray,
                                     fill_value: Optional[float] = None,
                                     dtype: np.dtype = np.float64) -> np.ndarray:
    """Impute -9/NaN values for a numpy genotype batch.

    Args:
        batch: Raw genotype slice (n_individuals × n_markers)
        fill_value: Optional constant used to replace missing values. When
            provided, this overrides the major-allele strategy used by rMVP
            for backwards compatibility with legacy FarmCPU behaviour.
        dtype: Output dtype for the returned array.
    """
    out_dtype = np.dtype(dtype)
    G = np.array(batch, dtype=out_dtype, copy=True)
    missing = (G == -9) | np.isnan(G)
    if not missing.any():
        return G

    if fill_value is not None:
        G[missing] = out_dtype.type(fill_value)
        return G

    # Default behaviour: impute with per-SNP major allele (matches rMVP C++ helpers)
    with np.errstate(invalid="ignore"):
        c0 = np.sum(G == 0, axis=0, dtype=np.int32)
        c1 = np.sum(G == 1, axis=0, dtype=np.int32)
        c2 = np.sum(G == 2, axis=0, dtype=np.int32)
    counts = np.stack([c0, c1, c2], axis=0)
    maj_idx = np.argmax(counts, axis=0)
    maj_vals = maj_idx.astype(out_dtype)

    # Columns containing unexpected genotype codes fallback to unique counts
    valid_set_mask = (G == 0) | (G == 1) | (G == 2) | missing
    col_has_other = ~np.all(valid_set_mask, axis=0)
    if np.any(col_has_other):
        cols = np.where(col_has_other)[0]
        for j in cols:
            col = G[:, j]
            mm = (col == -9) | np.isnan(col)
            non_missing = col[~mm]
            if non_missing.size > 0:
                vals, cnts = np.unique(non_missing, return_counts=True)
                maj = vals[int(np.argmax(cnts))]
            else:
                maj = out_dtype.type(0.0)
            maj_vals[j] = out_dtype.type(maj)

    G[missing] = np.broadcast_to(maj_vals, G.shape)[missing]
    return G


def _compute_qr(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute thin QR; returns Q (n x p), R (p x p)."""
    # Use economy (reduced) QR for efficiency
    Q, R = np.linalg.qr(X, mode="reduced")
    return Q, R


def PANICLE_GLM_ultrafast(phe: np.ndarray,
                      geno: Union[GenotypeMatrix, np.ndarray],
                      CV: Optional[np.ndarray] = None,
                      maxLine: int = 5000,
                      cpu: int = 1,
                      verbose: bool = True,
                      missing_fill_value: float = 1.0,
                      return_cov_stats: bool = False,
                      cov_pvalue_agg: Optional[str] = None) -> AssociationResults:
    """FWL+QR GLM scan with vectorized residualization and statistics.

    Args:
        phe: n x 2 array [ID, trait]
        geno: GenotypeMatrix or numpy array (n x m)
        CV: n x k covariates (optional)
        maxLine: batch size (markers per block)
        cpu: unused (kept for signature compatibility)
        verbose: print brief progress
        missing_fill_value: value to impute for missing genotypes
        return_cov_stats: if True, return stats for all covariates (memory intensive!)
        cov_pvalue_agg: if set ("reward"/"penalty"/"mean"), compute aggregated
            covariate p-values instead of full 2D array. Much more memory efficient.
            Result will have .cov_pvalue_summary attribute with shape (n_covariates,).
    Returns:
        AssociationResults. If return_cov_stats is False and cov_pvalue_agg is None,
        arrays are 1D (markers). If return_cov_stats is True, arrays are 2D.
        If cov_pvalue_agg is set, arrays are 1D with .cov_pvalue_summary metadata.
    """
    # Extract phenotype vector y
    if not isinstance(phe, np.ndarray) or phe.shape[1] != 2:
        raise ValueError("Phenotype must be numpy array with 2 columns [ID, trait_value]")
    y = phe[:, 1].astype(np.float32)

    # Dimensions and genotype accessor
    if isinstance(geno, GenotypeMatrix):
        n, m = geno.n_individuals, geno.n_markers
        use_gm = True
    elif isinstance(geno, np.ndarray):
        n, m = geno.shape
        use_gm = False
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    # Build covariate matrix with intercept
    if CV is not None:
        if CV.shape[0] != n:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        CV_f32 = np.asarray(CV, dtype=np.float32)
        X = np.column_stack([np.ones(n, dtype=np.float32), CV_f32])
    else:
        X = np.ones((n, 1), dtype=np.float32)

    X = np.ascontiguousarray(X, dtype=np.float32)
    XT = X.T

    # Suppress warnings for expected numerical issues in initial covariate setup
    # These are properly handled by try-except and validity checks
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        XtX = XT @ X
        try:
            iXX = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            iXX = np.linalg.pinv(XtX, rcond=1e-10)
        xy = XT @ y
        beta_cov = iXX @ xy
        yy = float(y @ y)
        beta_cov = iXX @ xy
        yy = float(y @ y)
    p = X.shape[1]
    
    # Pre-extract diagonal of iXX for variance updates
    # iXX diagonal elements correspond to the variance of covariate estimates
    diag_iXX = np.diag(iXX).astype(np.float64)
    # Number of fixed effects (intercept + covariates)
    n_fixed = p 

    iXX = iXX.astype(np.float32, copy=False)
    xy = xy.astype(np.float32, copy=False)
    beta_cov = beta_cov.astype(np.float32, copy=False)
    xy_f64 = xy.astype(np.float64, copy=False)
    beta_cov_f64 = beta_cov.astype(np.float64, copy=False)
    yy_f64 = float(yy)

    df_full = int(n - p - 1)
    df_reduced = int(n - p)
    if df_full <= 0:
        raise ValueError("Degrees of freedom must be positive; check covariates")

    
    # Determine output mode:
    # - return_cov_stats=True: full 2D arrays (memory intensive)
    # - cov_pvalue_agg set: 1D marker arrays + aggregated covariate p-values (efficient)
    # - neither: 1D marker arrays only
    use_cov_agg = cov_pvalue_agg is not None and n_fixed > 1  # Need covariates to aggregate

    if return_cov_stats and not use_cov_agg:
        # Full 2D mode (legacy, memory intensive)
        n_cols = 1 + n_fixed
        effects = np.zeros((m, n_cols), dtype=np.float64)
        ses = np.zeros((m, n_cols), dtype=np.float64)
        pvals = np.ones((m, n_cols), dtype=np.float64)
        cov_pval_min = cov_pval_max = cov_pval_sum = cov_pval_count = None
    else:
        # 1D marker arrays only
        effects = np.zeros(m, dtype=np.float64)
        ses = np.zeros(m, dtype=np.float64)
        pvals = np.ones(m, dtype=np.float64)

        if use_cov_agg:
            # Initialize running aggregates for covariate p-values
            # Shape: (n_fixed,) for [intercept, cov1, cov2, ..., covN]
            cov_pval_min = np.full(n_fixed, np.inf, dtype=np.float64)
            cov_pval_max = np.full(n_fixed, -np.inf, dtype=np.float64)
            cov_pval_sum = np.zeros(n_fixed, dtype=np.float64)
            cov_pval_count = np.zeros(n_fixed, dtype=np.int64)
        else:
            cov_pval_min = cov_pval_max = cov_pval_sum = cov_pval_count = None

    batch_size = max(1, min(maxLine, m))
    is_imputed = use_gm and geno.is_imputed

    # Use prefetching for large datasets to overlap I/O with computation
    use_prefetch = m > batch_size * 2  # Only prefetch if more than 2 batches

    if use_prefetch:
        # Prefetch next batch while processing current batch
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Load first batch
            G = _load_genotype_batch(geno, 0, min(batch_size, m), use_gm, is_imputed, missing_fill_value)
            next_future = None

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)

                # Get current batch (from prefetch or first load)
                if next_future is not None:
                    G = next_future.result()

                # Submit prefetch for next batch
                next_start = start + batch_size
                if next_start < m:
                    next_end = min(next_start + batch_size, m)
                    next_future = executor.submit(
                        _load_genotype_batch, geno, next_start, next_end,
                        use_gm, is_imputed, missing_fill_value
                    )
                else:
                    next_future = None

                # Process current batch (code continues below)
                _process_glm_batch(
                    G, start, end, XT, iXX, beta_cov, beta_cov_f64, xy_f64, y, yy_f64,
                    df_full, df_reduced, diag_iXX, effects, ses, pvals,
                    return_cov_stats and not use_cov_agg, n_fixed,
                    cov_pval_min, cov_pval_max, cov_pval_sum, cov_pval_count
                )
    else:
        # Simple sequential processing for small datasets
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            G = _load_genotype_batch(geno, start, end, use_gm, is_imputed, missing_fill_value)
            _process_glm_batch(
                G, start, end, XT, iXX, beta_cov, beta_cov_f64, xy_f64, y, yy_f64,
                df_full, df_reduced, diag_iXX, effects, ses, pvals,
                return_cov_stats and not use_cov_agg, n_fixed,
                cov_pval_min, cov_pval_max, cov_pval_sum, cov_pval_count
            )

    if verbose:
        valid_tests = np.sum(np.isfinite(ses))
        print(f"FWL-QR GLM complete. {valid_tests}/{m} markers tested")
        if valid_tests > 0:
            print(f"Minimum p-value: {np.nanmin(pvals):.2e}")

    result = AssociationResults(effects, ses, pvals)

    # Attach aggregated covariate p-values if computed
    if use_cov_agg and cov_pval_min is not None:
        # Compute final aggregates based on method
        if cov_pvalue_agg == "reward":
            cov_summary = cov_pval_min
        elif cov_pvalue_agg == "penalty":
            cov_summary = cov_pval_max
        elif cov_pvalue_agg == "mean":
            with np.errstate(invalid='ignore'):
                cov_summary = cov_pval_sum / np.maximum(cov_pval_count, 1)
            cov_summary[cov_pval_count == 0] = 1.0
        else:
            # Default to reward (min)
            cov_summary = cov_pval_min

        # Replace inf with 1.0 for covariates with no valid tests
        cov_summary[~np.isfinite(cov_summary)] = 1.0
        result.cov_pvalue_summary = cov_summary

    return result


def _process_glm_batch(
    G: np.ndarray,
    start: int,
    end: int,
    XT: np.ndarray,
    iXX: np.ndarray,
    beta_cov: np.ndarray,
    beta_cov_f64: np.ndarray,
    xy_f64: np.ndarray,
    y: np.ndarray,
    yy_f64: float,
    df_full: int,
    df_reduced: int,
    diag_iXX: np.ndarray,
    effects: np.ndarray,
    ses: np.ndarray,
    pvals: np.ndarray,
    return_cov_stats: bool,
    n_fixed: int,
    cov_pval_min: Optional[np.ndarray] = None,
    cov_pval_max: Optional[np.ndarray] = None,
    cov_pval_sum: Optional[np.ndarray] = None,
    cov_pval_count: Optional[np.ndarray] = None
) -> None:
    """Process a single batch of genotypes for GLM statistics.

    If cov_pval_* arrays are provided, updates running aggregates for covariate
    p-values without storing the full 2D arrays.
    """
    # Suppress warnings for expected numerical issues in matrix operations
    # These are properly handled by validity checks below
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        xs = XT @ G                       # shape (p, b)
        xst = xs.T                        # shape (b, p)
        sy = G.T @ y                      # shape (b,)
        ss = np.sum(G * G, axis=0)        # shape (b,)

        B21 = xst @ iXX                   # shape (b, p)
        tmp = sy - (xst @ beta_cov)       # shape (b,)
        t2 = np.einsum('ij,ij->i', B21, xst)
        B22 = ss - t2

    B21 = B21.astype(np.float64)
    tmp = tmp.astype(np.float64)
    B22 = B22.astype(np.float64)
    sy = sy.astype(np.float64)

    valid = B22 > 1e-8
    invB22 = np.zeros_like(B22)
    invB22[valid] = 1.0 / B22[valid]

    beta_marker = invB22 * tmp

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        beta_cov_new = beta_cov_f64[np.newaxis, :] - (beta_marker[:, np.newaxis] * B21)
        rhs_cov = beta_cov_new @ xy_f64
    df_array = np.full_like(B22, df_full, dtype=float)
    df_array[~valid] = df_reduced

    ve = (yy_f64 - (rhs_cov + beta_marker * sy)) / df_array
    ve = np.maximum(ve, 0.0)
    se_marker = np.sqrt(ve * invB22)

    t_stats = np.zeros_like(beta_marker)
    finite_mask = (se_marker > 0) & np.isfinite(se_marker)
    t_stats[finite_mask] = np.abs(beta_marker[finite_mask] / se_marker[finite_mask])
    p_batch = np.ones_like(beta_marker)
    if np.any(finite_mask):
        p_batch[finite_mask] = _fast_t_pvalue(t_stats[finite_mask], df_array[finite_mask])

    # Handle singular cases as rMVP (set effect/SE to 0/NaN, p=1)
    beta_marker[~valid] = 0.0
    se_marker[~valid] = np.nan
    p_batch[~valid] = 1.0

    if return_cov_stats:
        # Full 2D mode - store all covariate stats
        b_cov_final = beta_cov_new
        if b_cov_final.ndim == 1:
            b_cov_final = b_cov_final[:, np.newaxis]

        var_inflation = (B21**2) * invB22[:, np.newaxis]
        diag_inv_new = diag_iXX[np.newaxis, :] + var_inflation
        se_cov_new = np.sqrt(ve[:, np.newaxis] * diag_inv_new)

        p_cov = np.ones_like(b_cov_final)
        valid_batch_idx = np.where(valid)[0]
        if len(valid_batch_idx) > 0:
            b_valid = b_cov_final[valid_batch_idx]
            se_valid = se_cov_new[valid_batch_idx]
            df_valid = df_array[valid_batch_idx]

            t_valid = np.abs(b_valid / se_valid)
            for col in range(t_valid.shape[1]):
                p_cov[valid_batch_idx, col] = _fast_t_pvalue(t_valid[:, col], df_valid)

            bad_mask = ~np.isfinite(se_valid) | (se_valid <= 0)
            p_cov[valid_batch_idx][bad_mask] = 1.0
            se_cov_new[valid_batch_idx][bad_mask] = np.nan

        effects[start:end, 0] = beta_marker
        effects[start:end, 1:] = b_cov_final
        ses[start:end, 0] = se_marker
        ses[start:end, 1:] = se_cov_new
        pvals[start:end, 0] = p_batch
        pvals[start:end, 1:] = p_cov

    elif cov_pval_min is not None:
        # Aggregation mode - compute covariate p-values but only keep running stats
        # This avoids storing the huge 2D array
        var_inflation = (B21**2) * invB22[:, np.newaxis]
        diag_inv_new = diag_iXX[np.newaxis, :] + var_inflation
        se_cov_new = np.sqrt(ve[:, np.newaxis] * diag_inv_new)

        valid_batch_idx = np.where(valid)[0]
        if len(valid_batch_idx) > 0:
            b_valid = beta_cov_new[valid_batch_idx]
            se_valid = se_cov_new[valid_batch_idx]
            df_valid = df_array[valid_batch_idx]

            t_valid = np.abs(b_valid / se_valid)
            # Compute p-values for each covariate and update running aggregates
            for col in range(t_valid.shape[1]):
                p_col = _fast_t_pvalue(t_valid[:, col], df_valid)
                # Handle bad values
                bad = ~np.isfinite(se_valid[:, col]) | (se_valid[:, col] <= 0)
                p_col[bad] = 1.0
                # Update running aggregates
                valid_p = p_col[~bad]
                if len(valid_p) > 0:
                    cov_pval_min[col] = min(cov_pval_min[col], np.min(valid_p))
                    cov_pval_max[col] = max(cov_pval_max[col], np.max(valid_p))
                    cov_pval_sum[col] += np.sum(valid_p)
                    cov_pval_count[col] += len(valid_p)

        # Store only marker results (1D)
        effects[start:end] = beta_marker
        ses[start:end] = se_marker
        pvals[start:end] = p_batch

    else:
        # Simple 1D mode - marker stats only
        effects[start:end] = beta_marker
        ses[start:end] = se_marker
        pvals[start:end] = p_batch
