
import numpy as np
import pandas as pd
import time
from panicle.association.mlm_loco import MVP_MLM_LOCO
from panicle.association.hybrid_mlm import MVP_MLM_Hybrid
from panicle.matrix.kinship import MVP_K_VanRaden
from panicle.utils.data_types import GenotypeMatrix, KinshipMatrix

def test_hybrid_mlm():
    print("Generating synthetic data...")
    n_individuals = 200
    n_markers = 1000
    
    # Random genotypes (0, 1, 2)
    geno = np.random.randint(0, 3, size=(n_individuals, n_markers)).astype(np.int8)
    map_df = pd.DataFrame({
        'SNP': [f"SNP{i:04d}" for i in range(n_markers)],
        'CHROM': [f"Chr{(i % 5) + 1:02d}" for i in range(n_markers)],
        'POS': np.arange(n_markers) + 1
    })
    
    # Kinship
    kinship = MVP_K_VanRaden(geno, verbose=False)
    
    # Major Effect Locus (Marker 50)
    # Explain 10% of variance
    causal_idx = 50
    beta = 5.0
    g_causal = geno[:, causal_idx]
    
    # Polygenic background
    u = np.random.multivariate_normal(np.zeros(n_individuals), kinship.to_numpy() * 2.0)
    
    # Residual
    e = np.random.normal(0, 1.0, n_individuals)
    
    # Phenotype
    y = beta * g_causal + u + e
    
    phe = np.column_stack([np.arange(n_individuals), y])
    
    print("\n--- Running Standard MLM (Wald) ---")
    start = time.time()
    res_wald = MVP_MLM_LOCO(phe, geno, map_data=map_df, verbose=False)
    print(f"Wald Time: {time.time() - start:.2f}s")
    
    print("\n--- Running Hybrid MLM (Screen + LRT) ---")
    start = time.time()
    # Use loose threshold to guarantee refinement of our causal marker
    # The class name is still MVP_MLM_Hybrid (function name), but we refer to the method conceptually as HybridMLM
    res_hybrid = MVP_MLM_Hybrid(phe, geno, map_data=map_df, screen_threshold=0.01, verbose=True)
    print(f"Hybrid Time: {time.time() - start:.2f}s")
    
    # Compare
    p_wald = res_wald.pvalues[causal_idx]
    p_hybrid = res_hybrid.pvalues[causal_idx]
    
    print(f"\n--- Results for Causal Marker ({causal_idx}) ---")
    print(f"Wald P-value:   {p_wald:.4e}")
    print(f"Hybrid P-value: {p_hybrid:.4e}")
    
    if p_hybrid < p_wald:
        print("SUCCESS: Hybrid LRT improved significance for causal marker.")
    else:
        print("NOTE: Hybrid LRT provided similar or lower significance (expected if Wald was optimistic or sample small).")
        
    # Check consistency for non-refined markers
    non_refined = np.where(res_wald.pvalues >= 0.01)[0]
    if len(non_refined) > 0:
        diff = np.abs(res_wald.pvalues[non_refined] - res_hybrid.pvalues[non_refined])
        print(f"Mean difference for non-refined markers: {np.mean(diff):.4e}")
        assert np.allclose(diff, 0), "Non-refined markers should be identical!"

if __name__ == "__main__":
    test_hybrid_mlm()
