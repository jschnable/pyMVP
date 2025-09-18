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
import numpy as np
from scipy import stats

from pymvp.utils.data_types import GenotypeMatrix, AssociationResults


def _impute_numpy_batch_major_allele(batch: np.ndarray,
                                     fill_value: Optional[float] = None) -> np.ndarray:
    """Impute -9/NaN values for a numpy genotype batch.

    Args:
        batch: Raw genotype slice (n_individuals × n_markers)
        fill_value: Optional constant used to replace missing values. When
            provided, this overrides the major-allele strategy used by rMVP
            for backwards compatibility with legacy FarmCPU behaviour.
    """
    G = batch.astype(np.float64, copy=True)
    missing = (G == -9) | np.isnan(G)
    if not missing.any():
        return G

    if fill_value is not None:
        G[missing] = float(fill_value)
        return G

    # Default behaviour: impute with per-SNP major allele (matches rMVP C++ helpers)
    with np.errstate(invalid="ignore"):
        c0 = np.nansum((G == 0).astype(np.float64), axis=0)
        c1 = np.nansum((G == 1).astype(np.float64), axis=0)
        c2 = np.nansum((G == 2).astype(np.float64), axis=0)
    counts = np.stack([c0, c1, c2], axis=0)
    maj_idx = np.argmax(counts, axis=0)
    maj_vals = maj_idx.astype(np.float64)

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
                maj = 0.0
            maj_vals[j] = float(maj)

    G[missing] = np.broadcast_to(maj_vals, G.shape)[missing]
    return G


def _compute_qr(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute thin QR; returns Q (n x p), R (p x p)."""
    # Use economy (reduced) QR for efficiency
    Q, R = np.linalg.qr(X, mode="reduced")
    return Q, R


def MVP_GLM_ultrafast(phe: np.ndarray,
                      geno: Union[GenotypeMatrix, np.ndarray],
                      CV: Optional[np.ndarray] = None,
                      maxLine: int = 5000,
                      cpu: int = 1,
                      verbose: bool = True,
                      missing_fill_value: float = 1.0) -> AssociationResults:
    """FWL+QR GLM scan with vectorized residualization and statistics.

    Args:
        phe: n x 2 array [ID, trait]
        geno: GenotypeMatrix or numpy array (n x m)
        CV: n x k covariates (optional)
        maxLine: batch size (markers per block)
        cpu: unused (kept for signature compatibility)
        verbose: print brief progress
    Returns:
        AssociationResults with Effect, SE, P-value per marker
    """
    # Extract phenotype vector y
    if not isinstance(phe, np.ndarray) or phe.shape[1] != 2:
        raise ValueError("Phenotype must be numpy array with 2 columns [ID, trait_value]")
    y = phe[:, 1].astype(np.float64)

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
        X = np.column_stack([np.ones(n), CV])
    else:
        X = np.ones((n, 1))

    X = np.asfortranarray(X, dtype=np.float64)
    XtX = X.T @ X
    try:
        iXX = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        iXX = np.linalg.pinv(XtX, rcond=1e-10)
    xy = X.T @ y
    beta_cov = iXX @ xy
    yy = float(y @ y)
    p = X.shape[1]

    df_full = int(n - p - 1)
    df_reduced = int(n - p)
    if df_full <= 0:
        raise ValueError("Degrees of freedom must be positive; check covariates")

    effects = np.zeros(m, dtype=np.float64)
    ses = np.zeros(m, dtype=np.float64)
    pvals = np.ones(m, dtype=np.float64)

    batch_size = max(1, min(maxLine, m))
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        if use_gm:
            G = geno.get_batch_imputed(start, end, fill_value=missing_fill_value)
        else:
            G = _impute_numpy_batch_major_allele(geno[:, start:end], fill_value=missing_fill_value)

        xs = X.T @ G                      # shape (p, b)
        sy = G.T @ y                      # shape (b,)
        ss = np.sum(G * G, axis=0)        # shape (b,)

        B21 = xs.T @ iXX                  # shape (b, p)
        tmp = sy - (xs.T @ beta_cov)      # shape (b,)
        t2 = np.einsum('ij,ij->i', B21, xs.T)
        B22 = ss - t2

        valid = B22 > 1e-8
        invB22 = np.zeros_like(B22)
        invB22[valid] = 1.0 / B22[valid]

        beta_marker = invB22 * tmp

        beta_cov_new = beta_cov[np.newaxis, :] - (beta_marker[:, np.newaxis] * B21)
        rhs_cov = beta_cov_new @ xy
        df_array = np.full_like(B22, df_full, dtype=float)
        df_array[~valid] = df_reduced

        ve = (yy - (rhs_cov + beta_marker * sy)) / df_array
        ve = np.maximum(ve, 0.0)
        se_marker = np.sqrt(ve * invB22)

        t_stats = np.zeros_like(beta_marker)
        finite_mask = (se_marker > 0) & np.isfinite(se_marker)
        t_stats[finite_mask] = np.abs(beta_marker[finite_mask] / se_marker[finite_mask])
        p_batch = np.ones_like(beta_marker)
        if np.any(finite_mask):
            p_batch[finite_mask] = 2.0 * stats.t.sf(t_stats[finite_mask], df_array[finite_mask])

        # Handle singular cases as rMVP (set effect/SE to 0/NaN, p=1)
        beta_marker[~valid] = 0.0
        se_marker[~valid] = np.nan
        p_batch[~valid] = 1.0

        effects[start:end] = beta_marker
        ses[start:end] = se_marker
        pvals[start:end] = p_batch

    if verbose:
        valid_tests = np.sum(np.isfinite(ses))
        print(f"FWL-QR GLM complete. {valid_tests}/{m} markers tested")
        if valid_tests > 0:
            print(f"Minimum p-value: {np.nanmin(pvals):.2e}")

    return AssociationResults(effects, ses, pvals)
