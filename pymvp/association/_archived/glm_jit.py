"""
JIT-optimized General Linear Model (GLM) for GWAS analysis
Uses single-marker processing (like original GLM) but with Numba JIT for performance
"""

import numpy as np
from typing import Optional, Union, Tuple
from scipy import stats
from ..utils.data_types import GenotypeMatrix, AssociationResults
import warnings

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def MVP_GLM_jit(phe: np.ndarray,
                geno: Union[GenotypeMatrix, np.ndarray],
                CV: Optional[np.ndarray] = None,
                maxLine: int = 5000,
                cpu: int = 1,
                verbose: bool = True) -> AssociationResults:
    """JIT-optimized General Linear Model for GWAS analysis
    
    Uses the same single-marker processing approach as the original GLM
    but applies Numba JIT compilation to the statistical computations for speed.
    
    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        CV: Covariate matrix (n_individuals × n_covariates), optional
        maxLine: Batch size for processing markers (matches original GLM interface)
        cpu: Number of CPU threads (currently ignored)
        verbose: Print progress information
    
    Returns:
        AssociationResults object containing Effect, SE, and P-value for each marker
    """
    
    # Handle input validation and data preparation (identical to original)
    if isinstance(phe, np.ndarray):
        if phe.shape[1] != 2:
            raise ValueError("Phenotype matrix must have 2 columns [ID, trait_value]")
        trait_values = phe[:, 1].astype(np.float64)
    else:
        raise ValueError("Phenotype must be numpy array")
    
    # Handle genotype input (identical to original)
    if isinstance(geno, GenotypeMatrix):
        genotype = geno
        n_individuals = geno.n_individuals
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        genotype = geno
        n_individuals, n_markers = geno.shape
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")
    
    if verbose:
        print(f"Running JIT-optimized GLM analysis on {n_individuals} individuals, {n_markers} markers")
        if HAS_NUMBA:
            print("⚡ Using Numba JIT compilation for performance boost")
        else:
            print("💡 Numba not available - using standard Python (install numba for speedup)")
    
    # Validate dimensions (identical to original)
    if len(trait_values) != n_individuals:
        raise ValueError("Number of phenotype observations must match number of individuals")
    
    # Set up design matrix X (identical to original)
    if CV is not None:
        if CV.shape[0] != n_individuals:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        X = np.column_stack([np.ones(n_individuals), CV])
        if verbose:
            print(f"Including {CV.shape[1]} covariates in the model")
    else:
        X = np.ones((n_individuals, 1))
    
    n_covariates = X.shape[1]
    
    if verbose:
        print(f"Design matrix: {n_individuals} × {n_covariates}")
    
    # Precompute (X'X)^-1 * X' for efficiency (identical to original)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        XtX_inv_Xt = XtX_inv @ X.T
    except np.linalg.LinAlgError:
        raise ValueError("Singular design matrix - covariates may be collinear")
    
    # Initialize results arrays
    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64) 
    p_values = np.ones(n_markers, dtype=np.float64)
    
    # Process markers in batches (same structure as original)
    n_batches = (n_markers + maxLine - 1) // maxLine
    
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)
        
        if verbose and n_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{n_batches} (markers {start_marker}-{end_marker-1})")
        
        # Get batch of markers
        if isinstance(genotype, GenotypeMatrix):
            G_batch = genotype.get_batch(start_marker, end_marker).astype(np.float64)
        else:
            G_batch = genotype[:, start_marker:end_marker].astype(np.float64)
        
        # Use JIT-compiled processing for linear algebra, scipy for p-values
        if HAS_NUMBA:
            batch_effects, batch_se, batch_t_stats, batch_dfs = _process_markers_jit_linalg(
                trait_values, X, G_batch, n_covariates)
            # Compute p-values using scipy for accuracy
            batch_pvals = np.zeros_like(batch_effects)
            for i in range(len(batch_t_stats)):
                if not np.isnan(batch_t_stats[i]) and batch_dfs[i] > 0:
                    batch_pvals[i] = 2 * (1 - stats.t.cdf(abs(batch_t_stats[i]), batch_dfs[i]))
                else:
                    batch_pvals[i] = 1.0
        else:
            batch_effects, batch_se, batch_pvals = _process_markers_python(
                trait_values, X, G_batch, n_covariates)
        
        # Store batch results
        effects[start_marker:end_marker] = batch_effects
        std_errors[start_marker:end_marker] = batch_se
        p_values[start_marker:end_marker] = batch_pvals
    
    if verbose:
        valid_tests = np.sum(~np.isnan(std_errors))
        print(f"JIT-optimized GLM analysis complete. {valid_tests}/{n_markers} markers successfully tested")
        if valid_tests > 0:
            min_p = np.nanmin(p_values)
            print(f"Minimum p-value: {min_p:.2e}")
    
    # Create results object
    return AssociationResults(effects, std_errors, p_values)


def _process_markers_python(y: np.ndarray, 
                           X: np.ndarray, 
                           G_batch: np.ndarray,
                           n_covariates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python implementation for fallback when numba is not available"""
    n_individuals, batch_size = G_batch.shape
    
    # Initialize output arrays
    effects = np.zeros(batch_size, dtype=np.float64)
    std_errors = np.zeros(batch_size, dtype=np.float64)
    p_values = np.ones(batch_size, dtype=np.float64)
    
    # Process each marker individually (same logic as original GLM)
    for marker_idx in range(batch_size):
        g = G_batch[:, marker_idx]
        
        # Handle missing data (-9 values) - impute with major allele like original
        missing_mask = (g == -9) | np.isnan(g)
        
        if np.sum(missing_mask) == len(g):
            continue
                
        if np.sum(missing_mask) > 0:
            non_missing_values = g[~missing_mask]
            if len(non_missing_values) == 0:
                continue
            
            # Find major allele (most common genotype)
            unique_vals, counts = np.unique(non_missing_values, return_counts=True)
            major_allele = unique_vals[np.argmax(counts)]
            
            # Impute missing values
            g_imputed = g.copy()
            g_imputed[missing_mask] = major_allele
            g_valid = g_imputed
        else:
            g_valid = g
        
        # Skip markers with no variation
        if np.var(g_valid) < 1e-10:
            continue
        
        # Augmented design matrix [X, g]
        X_aug = np.column_stack([X, g_valid])
        
        # Solve normal equations
        try:
            XtX_aug = X_aug.T @ X_aug
            Xty_aug = X_aug.T @ y
            beta_aug = np.linalg.solve(XtX_aug, Xty_aug)
            
            marker_effect = beta_aug[-1]
            
            # Calculate residuals and RSS
            y_pred = X_aug @ beta_aug
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)
            
            # Degrees of freedom
            df_residual = len(y) - (n_covariates + 1)
            
            if df_residual > 0:
                # Residual variance
                sigma2 = rss / df_residual
                
                # Variance-covariance matrix
                var_cov_matrix = sigma2 * np.linalg.inv(XtX_aug)
                
                # Standard error of marker effect
                marker_se = np.sqrt(var_cov_matrix[-1, -1])
                
                # T-statistic and p-value
                if marker_se > 0:
                    t_stat = marker_effect / marker_se
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_residual))
                else:
                    p_value = 1.0
            else:
                marker_se = np.nan
                p_value = 1.0
                
        except np.linalg.LinAlgError:
            marker_effect = 0.0
            marker_se = np.nan
            p_value = 1.0
        
        # Store results
        effects[marker_idx] = marker_effect
        std_errors[marker_idx] = marker_se
        p_values[marker_idx] = p_value
    
    return effects, std_errors, p_values


if HAS_NUMBA:
    @jit(nopython=True)
    def _process_markers_jit_linalg(y: np.ndarray, 
                                   X: np.ndarray, 
                                   G_batch: np.ndarray,
                                   n_covariates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """JIT-compiled linear algebra for marker processing, returns t-stats and df for scipy p-values"""
        n_individuals, batch_size = G_batch.shape
        
        # Initialize output arrays
        effects = np.zeros(batch_size, dtype=np.float64)
        std_errors = np.zeros(batch_size, dtype=np.float64)
        t_stats = np.full(batch_size, np.nan, dtype=np.float64)
        dfs = np.zeros(batch_size, dtype=np.float64)
        
        # Pre-allocate arrays to avoid repeated allocations
        X_aug = np.zeros((n_individuals, n_covariates + 1), dtype=np.float64)
        X_aug[:, :-1] = X  # Copy covariate columns
        
        # Process each marker
        for marker_idx in range(batch_size):
            g = G_batch[:, marker_idx]
            
            # Handle missing data - impute with major allele
            missing_count = 0
            for i in range(len(g)):
                if g[i] == -9 or np.isnan(g[i]):
                    missing_count += 1
            
            if missing_count == len(g):
                continue  # All missing
            
            g_valid = g.copy()
            if missing_count > 0:
                # Find major allele (most frequent non-missing value)
                # Count occurrences of each genotype
                max_count = 0
                major_allele = 0.0
                
                for val in [0.0, 1.0, 2.0]:  # Common genotype values
                    count = 0
                    for i in range(len(g)):
                        if g[i] == val:
                            count += 1
                    if count > max_count:
                        max_count = count
                        major_allele = val
                
                # Impute missing values with major allele
                for i in range(len(g_valid)):
                    if g_valid[i] == -9 or np.isnan(g_valid[i]):
                        g_valid[i] = major_allele
            
            # Check for variation
            g_mean = np.mean(g_valid)
            g_var = 0.0
            for i in range(len(g_valid)):
                diff = g_valid[i] - g_mean
                g_var += diff * diff
            g_var /= len(g_valid)
            
            if g_var < 1e-10:
                continue  # No variation
            
            # Set up augmented design matrix
            X_aug[:, -1] = g_valid
            
            # Solve normal equations
            XtX_aug = X_aug.T @ X_aug
            Xty_aug = X_aug.T @ y
            
            # Check for singularity
            det = np.linalg.det(XtX_aug)
            if abs(det) < 1e-12:
                continue
            
            # Solve using numpy (numba supports basic linalg)
            beta_aug = np.linalg.solve(XtX_aug, Xty_aug)
            marker_effect = beta_aug[-1]
            
            # Calculate residuals
            y_pred = X_aug @ beta_aug
            residuals = y - y_pred
            rss = np.sum(residuals * residuals)
            
            # Degrees of freedom
            df_residual = len(y) - (n_covariates + 1)
            
            if df_residual > 0:
                # Residual variance
                sigma2 = rss / df_residual
                
                # Variance of marker effect (diagonal element of inv(X'X))
                XtX_inv = np.linalg.inv(XtX_aug)
                marker_var = sigma2 * XtX_inv[-1, -1]
                marker_se = np.sqrt(marker_var)
                
                # T-statistic (let scipy compute p-value)
                if marker_se > 0:
                    t_stat = marker_effect / marker_se
                    t_stats[marker_idx] = t_stat
                    dfs[marker_idx] = df_residual
                else:
                    t_stats[marker_idx] = np.nan
                    dfs[marker_idx] = 0
            else:
                marker_se = np.nan
                t_stats[marker_idx] = np.nan
                dfs[marker_idx] = 0
            
            # Store results
            effects[marker_idx] = marker_effect
            std_errors[marker_idx] = marker_se
        
        return effects, std_errors, t_stats, dfs

else:
    # Define dummy functions when numba is not available
    def _process_markers_jit_linalg(y, X, G_batch, n_covariates):
        # Fallback to Python implementation, need to adapt return values
        effects, se, pvals = _process_markers_python(y, X, G_batch, n_covariates)
        # Extract t-stats and dfs from the python version for consistency
        t_stats = np.full_like(effects, np.nan)
        dfs = np.full_like(effects, 0.0)
        return effects, se, t_stats, dfs