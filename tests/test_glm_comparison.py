
import time
import numpy as np
import pytest
from pymvp.association.glm_fwl_qr import MVP_GLM_ultrafast
from pymvp.association.glm_optimized import MVP_GLM_ultra

def test_glm_comparison():
    # Generate synthetic data
    n_individuals = 1000
    n_markers = 10000
    n_covariates = 3
    
    np.random.seed(42)
    
    # Phenotype
    y = np.random.randn(n_individuals)
    phe = np.column_stack([np.arange(n_individuals), y])
    
    # Genotype (random 0, 1, 2)
    geno = np.random.randint(0, 3, size=(n_individuals, n_markers))
    
    # Covariates
    cv = np.random.randn(n_individuals, n_covariates)
    
    print(f"\nComparing GLM implementations with {n_individuals} samples, {n_markers} markers...")
    
    # Test 1: MVP_GLM_ultrafast (FWL+QR)
    start = time.time()
    res_fwl = MVP_GLM_ultrafast(phe, geno, CV=cv, verbose=False)
    time_fwl = time.time() - start
    print(f"FWL+QR Time: {time_fwl:.4f}s")
    
    # Test 2: MVP_GLM_ultra (Numba/Optimized)
    start = time.time()
    res_opt = MVP_GLM_ultra(phe, geno, CV=cv, verbose=False)
    time_opt = time.time() - start
    print(f"Optimized Time: {time_opt:.4f}s")
    
    # Compare results
    # effects
    diff_effects = np.abs(res_fwl.effects - res_opt.effects)
    max_diff_idx = np.nanargmax(diff_effects)
    # Ignore NaNs
    valid_mask = ~np.isnan(res_fwl.effects) & ~np.isnan(res_opt.effects)
    
    if np.sum(valid_mask) > 0:
        max_diff = np.max(np.abs(res_fwl.effects[valid_mask] - res_opt.effects[valid_mask]))
        print(f"Max effect difference: {max_diff:.2e}")
        
        # p-values
        max_p_diff = np.max(np.abs(res_fwl.pvalues[valid_mask] - res_opt.pvalues[valid_mask]))
        print(f"Max p-value difference: {max_p_diff:.2e}")
        
        # Check correlation
        corr = np.corrcoef(res_fwl.pvalues[valid_mask], res_opt.pvalues[valid_mask])[0, 1]
        print(f"P-value correlation: {corr:.6f}")
    else:
        print("No valid markers to compare!")

if __name__ == "__main__":
    test_glm_comparison()
