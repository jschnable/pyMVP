
import numpy as np
from scipy import stats, optimize
import warnings
from typing import Tuple

from .mlm import _calculate_neg_ml_likelihood

def _sanitize_array(arr: np.ndarray, clip: float = 1e6) -> np.ndarray:
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None and arr.size:
        max_abs = np.max(np.abs(arr))
        if max_abs > clip:
            arr = np.clip(arr, -clip, clip, out=arr)
    return arr


def fit_marker_lrt(y_transformed: np.ndarray, 
                   X_transformed: np.ndarray, 
                   g_transformed: np.ndarray, 
                   eigenvals: np.ndarray,
                   null_neg_loglik: float) -> Tuple[float, float, float, float]:
    """
    Perform Likelihood Ratio Test (LRT) for a single marker.
    
    Args:
        y_transformed: Phenotype vector in eigenspace (U'y)
        X_transformed: Covariate matrix in eigenspace (U'X) (Fixed effects)
        g_transformed: Genotype vector in eigenspace (U'g) (Marker effect)
        eigenvals: Eigenvalues of kinship matrix
        null_neg_loglik: Negative log-likelihood of the NULL model (pre-calculated)
        
    Returns:
        Tuple (LRT_statistic, p_value, beta_hat, se_hat)
    """
    
    # Construct Alternative Model Design Matrix: [X | g]
    # Note: g must be a column vector
    if g_transformed.ndim == 1:
        g_col = g_transformed[:, np.newaxis]
    else:
        g_col = g_transformed
        
    X_alt = np.hstack([X_transformed, g_col])
    X_alt = _sanitize_array(X_alt)
    y_transformed = _sanitize_array(y_transformed)
    
    # Define optimization function for Alternative Model (ML, not REML)
    def alt_neg_ml_likelihood(h2):
        return _calculate_neg_ml_likelihood(h2, y_transformed, X_alt, eigenvals)
    
    # Optimize h² for Alternative Model
    # We use bounded optimization similar to Null Model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize_scalar(
                alt_neg_ml_likelihood, 
                bounds=(0.001, 0.999), 
                method='bounded',
                options={'xatol': 1.22e-4, 'maxiter': 100} # Less iterations than null model usually okay
            )
            
        if result.success:
            alt_neg_loglik = result.fun
            h2_alt = float(result.x)
        else:
            return 0.0, 1.0, 0.0, float('inf') # Fail to converge -> Insignificant
            
    except Exception:
        return 0.0, 1.0, 0.0, float('inf')
    
    # Calculate LRT Statistic
    # LRT = 2 * (LL_alt - LL_null)
    # Since we have NEGATIVE LL:
    # LRT = 2 * ( (-alt_neg) - (-null_neg) )
    # LRT = 2 * (null_neg - alt_neg)
    
    lrt_stat = 2.0 * (null_neg_loglik - alt_neg_loglik)
    
    # Numerical stability check (stat should be >= 0)
    if lrt_stat < 0:
        lrt_stat = 0.0
        
    # Calculate P-value (Chi-square with 1 d.f.)
    p_value = stats.chi2.sf(lrt_stat, df=1)
    
    # Compute beta/se at alternative h2 using generalized least squares
    eig_safe = np.maximum(eigenvals, 1e-6)
    V0b = h2_alt * eig_safe + (1.0 - h2_alt) * np.ones_like(eig_safe)
    V0b = np.maximum(V0b, 1e-8)
    V0bi = np.nan_to_num(1.0 / V0b, nan=np.inf, posinf=np.inf, neginf=np.inf)
    if np.any(~np.isfinite(V0bi)):
        return float(lrt_stat), float(p_value), 0.0, float('inf')
    ViX = V0bi[:, np.newaxis] * X_alt
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        XViX = X_alt.T @ ViX
    if not np.all(np.isfinite(XViX)):
        return float(lrt_stat), float(p_value), 0.0, float('inf')
    try:
        XViX_inv = np.linalg.solve(XViX, np.eye(XViX.shape[0]))
    except np.linalg.LinAlgError:
        XViX_inv = np.linalg.pinv(XViX)
    beta_hat_vec = XViX_inv @ (ViX.T @ y_transformed)
    beta_marker = float(beta_hat_vec[-1])
    # SE from covariance matrix diag
    cov_beta = XViX_inv
    se_marker = float(np.sqrt(max(cov_beta[-1, -1], 0.0)))
    
    return float(lrt_stat), float(p_value), beta_marker, se_marker
