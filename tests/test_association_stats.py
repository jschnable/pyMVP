
import pytest
import numpy as np
from scipy import stats
from pymvp.association.glm import MVP_GLM
from pymvp.association.mlm import MVP_MLM
from pymvp.matrix.kinship import MVP_K_VanRaden

def test_glm_statistical_correctness():
    """Verify GLM p-values against standard scipy.stats regression"""
    np.random.seed(42)
    n = 100
    n_markers = 10
    
    # Random phenotype
    y = np.random.randn(n)
    phe = np.column_stack([np.arange(n), y])
    
    # 1. Test Simple Regression (No covariates)
    geno = np.random.randint(0, 3, size=(n, n_markers)).astype(float)
    
    # Run MVP GLM
    res = MVP_GLM(phe, geno, cpu=1, verbose=False)
    
    # Check each marker manually
    for j in range(n_markers):
        g = geno[:, j]
        slope, intercept, r_value, p_value, std_err = stats.linregress(g, y)
        
        # Tolerances
        # Note: MVP GLM p-values are t-tests. linregress is essentially the same.
        # Effect size scaling in MVP might differ (rMVP scaling), so we focus on p-values mostly.
        # Although rMVP/pyMVP scales effects/SEs, the t-stats and P-values should be identical.
        
        np.testing.assert_allclose(res.pvalues[j], p_value, rtol=1e-5, 
            err_msg=f"P-value mismatch at marker {j}")

def test_glm_covariates_correctness():
    """Verify GLM with covariates (Partial Regression)"""
    np.random.seed(1337)
    n = 200
    n_markers = 5
    n_cov = 2
    
    # Co-variates
    CV = np.random.randn(n, n_cov)
    
    # Phenotype with covariate effects
    y = CV @ np.array([0.5, -0.5]) + np.random.randn(n)
    phe = np.column_stack([np.arange(n), y])
    
    geno = np.random.randint(0, 3, size=(n, n_markers))
    
    res = MVP_GLM(phe, geno, CV=CV, verbose=False)
    
    # Validation using statsmodels (OLS)
    import statsmodels.api as sm
    
    for j in range(n_markers):
        g = geno[:, j]
        # Design matrix: Intercept + CV + g
        X = np.column_stack([np.ones(n), CV, g])
        model = sm.OLS(y, X).fit()
        
        # The last parameter is 'g'
        p_val_sm = model.pvalues[-1]
        
        np.testing.assert_allclose(res.pvalues[j], p_val_sm, rtol=1e-5,
             err_msg=f"Covariate P-value mismatch at marker {j}")

def test_mlm_returns_reasonable_values():
    """Smoke test for MLM (statistical correctness is harder to unit-test simply)"""
    np.random.seed(99)
    n = 50
    m = 100
    geno = np.random.randint(0, 3, size=(n, m))
    y = np.random.randn(n)
    phe = np.column_stack([np.arange(n), y])
    
    # Kinship
    K = MVP_K_VanRaden(geno)
    
    res = MVP_MLM(phe, geno, K=K, verbose=False)
    
    assert len(res.pvalues) == m
    assert np.all(res.pvalues >= 0) and np.all(res.pvalues <= 1)
    # Check nothing is NaN (unless variance is 0)
    assert not np.any(np.isnan(res.pvalues))
