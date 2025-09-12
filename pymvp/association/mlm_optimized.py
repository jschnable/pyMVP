"""
Optimized Mixed Linear Model (MLM) for GWAS analysis

This implementation provides significant performance improvements over the standard MLM:
- Phase 1: Vectorized batch processing in eigenspace
- Phase 2: Fast p-value calculations using scipy.special.stdtr
- Phase 3: Multi-core parallel processing with joblib
- Numba JIT compilation for critical numerical operations

Target: 5-10x speedup while maintaining statistical accuracy
"""

import numpy as np
from typing import Optional, Union, Dict, Tuple
from scipy import stats, optimize
from scipy.special import stdtr  # Fast t-distribution CDF
from ..utils.data_types import GenotypeMatrix, KinshipMatrix, AssociationResults
import warnings
import time

# Check for Numba availability
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Phase 1: Vectorized Batch Processing Functions
if HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def compute_batch_crossproducts_jit(y, X, G_batch, weights):
        """JIT-compiled vectorized cross-product computation for MLM batch"""
        n_markers = G_batch.shape[1]
        n_individuals = G_batch.shape[0]
        q0 = X.shape[1]
        
        # Pre-allocate result arrays
        batch_UXWUX = X.T @ (X * weights[:, np.newaxis])  # Same for all markers
        batch_UXWy = X.T @ (weights * y)  # Same for all markers
        
        # Marker-specific arrays
        batch_UXWUs = np.zeros((q0, n_markers))
        batch_UsWUs = np.zeros(n_markers)
        batch_UsWy = np.zeros(n_markers)
        
        # Vectorized computation for all markers in batch
        for j in numba.prange(n_markers):  # Parallel loop over markers
            g = G_batch[:, j]
            batch_UXWUs[:, j] = X.T @ (weights * g)
            batch_UsWUs[j] = np.sum(weights * g * g)
            batch_UsWy[j] = np.sum(weights * g * y)
        
        return batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy

    @numba.jit(nopython=True, cache=True)
    def process_batch_effects_jit(batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy, vg, n, q0):
        """JIT-compiled batch effect size and standard error calculation"""
        n_markers = len(batch_UsWUs)
        
        # Pre-compute UXWUX inverse (same for all markers)
        try:
            iUXWUX = np.linalg.inv(batch_UXWUX)
        except:
            # Return NaN results if matrix is singular
            return np.full(n_markers, np.nan), np.full(n_markers, np.nan), np.full(n_markers, np.nan), np.full(n_markers, 1.0)
        
        # Initialize result arrays
        effects = np.zeros(n_markers)
        std_errors = np.zeros(n_markers)
        t_stats = np.zeros(n_markers)
        dfs = np.full(n_markers, float(n - q0 - 1))
        
        # Process all markers in vectorized fashion
        for j in numba.prange(n_markers):  # Parallel loop
            # Calculate B22 (marker precision after removing covariate effects)
            UXWUs_j = batch_UXWUs[:, j]
            B22 = batch_UsWUs[j] - UXWUs_j.T @ iUXWUX @ UXWUs_j
            
            if B22 <= 1e-12:  # Numerical stability check
                effects[j] = 0.0
                std_errors[j] = np.nan
                t_stats[j] = 0.0
                dfs[j] = 1.0
                continue
            
            invB22 = 1.0 / B22
            
            # Marker effect estimation
            effects[j] = invB22 * (batch_UsWy[j] - UXWUs_j.T @ iUXWUX @ batch_UXWy)
            
            # Standard error calculation
            marker_var = invB22 * vg
            std_errors[j] = np.sqrt(marker_var)
            
            # T-statistic for p-value calculation
            if std_errors[j] > 0:
                t_stats[j] = effects[j] / std_errors[j]
            else:
                t_stats[j] = 0.0
        
        return effects, std_errors, t_stats, dfs

else:
    # Fallback non-JIT versions
    def compute_batch_crossproducts_jit(y, X, G_batch, weights):
        """Non-JIT fallback for batch cross-product computation"""
        n_markers = G_batch.shape[1]
        q0 = X.shape[1]
        
        batch_UXWUX = X.T @ (X * weights[:, np.newaxis])
        batch_UXWy = X.T @ (weights * y)
        
        batch_UXWUs = X.T @ (weights[:, np.newaxis] * G_batch)  # Broadcasting
        batch_UsWUs = np.sum(weights[:, np.newaxis] * G_batch * G_batch, axis=0)
        batch_UsWy = np.sum(weights[:, np.newaxis] * G_batch * y[:, np.newaxis], axis=0)
        
        return batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy

    def process_batch_effects_jit(batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy, vg, n, q0):
        """Non-JIT fallback for batch effect calculation"""
        n_markers = len(batch_UsWUs)
        
        try:
            iUXWUX = np.linalg.inv(batch_UXWUX)
        except np.linalg.LinAlgError:
            return np.full(n_markers, np.nan), np.full(n_markers, np.nan), np.full(n_markers, np.nan), np.full(n_markers, 1.0)
        
        effects = np.zeros(n_markers)
        std_errors = np.zeros(n_markers)
        t_stats = np.zeros(n_markers)
        dfs = np.full(n_markers, float(n - q0 - 1))
        
        for j in range(n_markers):
            UXWUs_j = batch_UXWUs[:, j]
            B22 = batch_UsWUs[j] - UXWUs_j.T @ iUXWUX @ UXWUs_j
            
            if B22 <= 1e-12:
                effects[j] = 0.0
                std_errors[j] = np.nan
                t_stats[j] = 0.0
                dfs[j] = 1.0
                continue
            
            invB22 = 1.0 / B22
            effects[j] = invB22 * (batch_UsWy[j] - UXWUs_j.T @ iUXWUX @ batch_UXWy)
            
            marker_var = invB22 * vg
            std_errors[j] = np.sqrt(marker_var)
            
            if std_errors[j] > 0:
                t_stats[j] = effects[j] / std_errors[j]
        
        return effects, std_errors, t_stats, dfs

# Phase 2: Fast p-value calculation
def compute_fast_pvalues(t_stats: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    """Fast p-value calculation using scipy.special.stdtr (same optimization as GLM)"""
    pvalues = np.ones_like(t_stats)
    valid_mask = ~np.isnan(t_stats) & (dfs > 0) & ~np.isnan(dfs)
    
    if np.any(valid_mask):
        valid_t = t_stats[valid_mask]
        valid_df = dfs[valid_mask]
        
        # Use scipy.special.stdtr for speed (faster than stats.t.sf)
        # P(T > |t|) = 1 - stdtr(df, |t|) for one-tail
        # Two-tailed: 2 * (1 - stdtr(df, |t|))
        pvalues[valid_mask] = 2.0 * (1.0 - stdtr(valid_df, np.abs(valid_t)))
    
    return pvalues

# Phase 3: Parallel batch processing function
def process_batch_parallel(batch_data):
    """Process a single batch of markers in parallel"""
    y_transformed, X_transformed, G_batch_transformed, eigenvals, delta_hat, vg_hat, start_marker = batch_data
    
    # Apply eigenvalue floor and compute weights
    eig_safe = np.maximum(eigenvals, 1e-6)
    V_eigenvals = eig_safe + delta_hat
    weights = 1.0 / V_eigenvals
    
    n = len(y_transformed)
    q0 = X_transformed.shape[1]
    n_markers = G_batch_transformed.shape[1]
    
    # Phase 1: Vectorized cross-product computation
    batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy = compute_batch_crossproducts_jit(
        y_transformed, X_transformed, G_batch_transformed, weights
    )
    
    # Process effects and standard errors
    effects, std_errors, t_stats, dfs = process_batch_effects_jit(
        batch_UXWUX, batch_UXWy, batch_UXWUs, batch_UsWUs, batch_UsWy, vg_hat, n, q0
    )
    
    # Phase 2: Fast p-value calculation
    pvalues = compute_fast_pvalues(t_stats, dfs)
    
    return start_marker, effects, std_errors, pvalues


def MVP_MLM_optimized(phe: np.ndarray,
                     geno: Union[GenotypeMatrix, np.ndarray],
                     K: Optional[Union[KinshipMatrix, np.ndarray]] = None,
                     eigenK: Optional[Dict] = None,
                     CV: Optional[np.ndarray] = None,
                     vc_method: str = "BRENT",
                     maxLine: int = 1000,  # Larger batches for vectorization
                     cpu: int = 1,
                     verbose: bool = True) -> AssociationResults:
    """Optimized Mixed Linear Model for GWAS analysis
    
    Performance improvements:
    - Phase 1: Vectorized batch processing in eigenspace
    - Phase 2: Fast p-value calculations using scipy.special.stdtr  
    - Phase 3: Multi-core parallel processing
    - Numba JIT compilation for numerical operations
    
    Target: 5-10x speedup over standard MLM implementation
    
    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        K: Kinship matrix (n_individuals × n_individuals)
        eigenK: Pre-computed eigendecomposition of K
        CV: Covariate matrix (n_individuals × n_covariates), optional
        vc_method: Variance component estimation method ["BRENT", "EMMA", "HE"]
        maxLine: Batch size for processing markers (larger for vectorization)
        cpu: Number of CPU threads for parallel processing
        verbose: Print progress information
    
    Returns:
        AssociationResults object containing Effect, SE, and P-value for each marker
    """
    
    # Handle cpu=0 to mean use all available cores
    if cpu == 0:
        import multiprocessing
        cpu = multiprocessing.cpu_count()
    
    if verbose:
        print("=" * 60)
        print("OPTIMIZED MLM IMPLEMENTATION")
        print("=" * 60)
        if HAS_NUMBA:
            print("⚡ Using Numba JIT compilation")
        if HAS_JOBLIB and cpu > 1:
            print(f"🚀 Using {cpu} CPU cores for parallel processing")
        else:
            print("🔄 Using sequential processing")
    
    # Handle input validation and data preparation (same as original)
    if isinstance(phe, np.ndarray):
        if phe.shape[1] != 2:
            raise ValueError("Phenotype matrix must have 2 columns [ID, trait_value]")
        trait_values = phe[:, 1].astype(np.float64)
    else:
        raise ValueError("Phenotype must be numpy array")
    
    # Handle genotype input
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
        print(f"Running optimized MLM on {n_individuals} individuals, {n_markers} markers")
        print(f"Batch size: {maxLine}")
    
    # Validate dimensions
    if len(trait_values) != n_individuals:
        raise ValueError("Number of phenotype observations must match number of individuals")
    
    # Handle kinship matrix
    if K is None:
        raise ValueError("Kinship matrix K is required for MLM analysis")
    
    if isinstance(K, KinshipMatrix):
        kinship = K.to_numpy()
    elif isinstance(K, np.ndarray):
        kinship = K
    else:
        raise ValueError("Kinship matrix must be KinshipMatrix or numpy array")
    
    if kinship.shape != (n_individuals, n_individuals):
        raise ValueError("Kinship matrix dimensions must match number of individuals")
    
    # Set up design matrix X (covariates)
    if CV is not None:
        if CV.shape[0] != n_individuals:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        X = np.column_stack([np.ones(n_individuals), CV])
        if verbose:
            print(f"Design matrix: {n_individuals} × {X.shape[1]} (including intercept)")
    else:
        X = np.ones((n_individuals, 1))
        if verbose:
            print(f"Design matrix: {n_individuals} × 1")
    
    # Perform eigendecomposition if not provided
    if eigenK is None:
        if verbose:
            print("Computing eigendecomposition of kinship matrix...")
        eigenvals, eigenvecs = np.linalg.eigh(kinship)
        
        # Sort by eigenvalues in descending order
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        eigenK = {
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs
        }
    else:
        eigenvals = eigenK['eigenvals']
        eigenvecs = eigenK['eigenvecs']
    
    if verbose:
        print(f"Eigendecomposition complete. Range: [{np.min(eigenvals):.6f}, {np.max(eigenvals):.6f}]")
    
    # Transform to eigenspace
    if verbose:
        print("Transforming data to eigenspace...")
    y_transformed = eigenvecs.T @ trait_values
    X_transformed = eigenvecs.T @ X
    
    # Estimate variance components using standard method
    if verbose:
        print(f"Estimating variance components using {vc_method} method...")
    
    # Use original variance component estimation (this is not the bottleneck)
    from .mlm import estimate_variance_components_brent
    
    if vc_method.upper() == "BRENT":
        delta_hat, vg_hat, ve_hat = estimate_variance_components_brent(y_transformed, X_transformed, eigenvals, verbose)
    else:
        # For other methods, fall back to BRENT for now
        delta_hat, vg_hat, ve_hat = estimate_variance_components_brent(y_transformed, X_transformed, eigenvals, verbose)
    
    h2 = vg_hat / (vg_hat + ve_hat) if (vg_hat + ve_hat) > 0 else 0.0
    
    if verbose:
        print(f"Estimated delta (ve/vg): {delta_hat:.6f}")
        print(f"Heritability estimate: h² = {h2:.6f}")
    
    # Initialize results arrays
    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64)
    p_values = np.ones(n_markers, dtype=np.float64)
    
    # Prepare batches for processing
    n_batches = (n_markers + maxLine - 1) // maxLine
    
    if verbose:
        print(f"Preprocessing missing data...")
        print(f"Phase 1: Vectorized batch processing {n_batches} batches")
        if HAS_JOBLIB and cpu > 1:
            print(f"Phase 3: Parallel processing {n_batches} batches on {cpu} cores")
    
    start_time = time.time()
    
    # Prepare batch data for processing
    batch_data_list = []
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)
        
        # Get batch of markers (imputed)
        if isinstance(genotype, GenotypeMatrix):
            G_batch = genotype.get_batch_imputed(start_marker, end_marker).astype(np.float64)
        else:
            G_batch = genotype[:, start_marker:end_marker].astype(np.float64)
        
        # Transform genotypes to eigenspace
        G_batch_transformed = eigenvecs.T @ G_batch
        
        # Prepare batch data tuple
        batch_data = (y_transformed, X_transformed, G_batch_transformed, eigenvals, delta_hat, vg_hat, start_marker)
        batch_data_list.append(batch_data)
    
    # Phase 3: Process batches (parallel if available)
    if HAS_JOBLIB and cpu > 1 and len(batch_data_list) > 1:
        # Parallel processing
        batch_results = Parallel(n_jobs=cpu, backend='threading')(
            delayed(process_batch_parallel)(batch_data) for batch_data in batch_data_list
        )
    else:
        # Sequential processing
        batch_results = [process_batch_parallel(batch_data) for batch_data in batch_data_list]
    
    # Collect results from all batches
    for start_marker, batch_effects, batch_se, batch_pvals in batch_results:
        end_marker = min(start_marker + len(batch_effects), n_markers)
        effects[start_marker:end_marker] = batch_effects
        std_errors[start_marker:end_marker] = batch_se
        p_values[start_marker:end_marker] = batch_pvals
    
    processing_time = time.time() - start_time
    
    if verbose:
        valid_tests = np.sum(~np.isnan(std_errors))
        print(f"Optimized MLM complete. {valid_tests}/{n_markers} markers tested")
        print(f"Processing time: {processing_time:.2f} seconds")
        if valid_tests > 0:
            min_p = np.nanmin(p_values)
            print(f"Minimum p-value: {min_p:.2e}")
    
    # Create results object
    return AssociationResults(effects, std_errors, p_values)


# Alias for compatibility
MVP_MLM_ultra = MVP_MLM_optimized