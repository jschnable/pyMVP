
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvp.pipelines.gwas import GWASPipeline
from pymvp.matrix.kinship import MVP_K_VanRaden
from pymvp.utils.data_types import GenotypeMatrix

# Config
VCF_PATH = "sorghum_data/SbDiv_811_maf0.05_het0.10.vcf.gz"
PHE_PATH = "sorghum_data/SpATS_genotype_BLUEs_all_traits_filtered_strict.csv"
TRAIT = "BLUE_PhiNPQ_corrected"

def check_array(name, arr):
    print(f"--- Checking {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Finite: {np.all(np.isfinite(arr))}")
    if not np.all(np.isfinite(arr)):
        print(f"Has NaNs: {np.any(np.isnan(arr))}")
        print(f"Has Infs: {np.any(np.isinf(arr))}")
    print(f"Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")
    print(f"Mean: {np.nanmean(arr)}, Std: {np.nanstd(arr)}")
    print("-----------------------")

def main():
    print("Starting Debugging Script...")
    
    pipeline = GWASPipeline(output_dir="debug_out")
    
    # 1. Load Data
    pipeline.load_data(
        phenotype_file=PHE_PATH,
        genotype_file=VCF_PATH,
        trait_columns=[TRAIT],
        loader_kwargs={'backend': 'cyvcf2'} 
    )
    pipeline.align_samples()
    
    # 2. Check Genotype Matrix (Sample)
    geno = pipeline.genotype_matrix
    print(f"Genotype Matrix loaded. {geno.n_individuals} x {geno.n_markers}")
    
    # Check first batch
    g_batch = geno.get_batch_imputed(0, 5000)
    check_array("G_batch (0-5000)", g_batch)
    
    # 3. Calculate Kinship
    print("\nCalculating Kinship...")
    K = MVP_K_VanRaden(geno, verbose=True)
    k_mat = K.to_numpy()
    check_array("Kinship Matrix", k_mat)
    
    # 4. Check Eigendecomposition
    print("\nCalculating Eigen Decomposition...")
    eigenvals, eigenvecs = np.linalg.eigh(k_mat)
    check_array("Eigenvalues", eigenvals)
    check_array("Eigenvectors", eigenvecs)
    
    # 5. Check Matmul (G_batch_transformed)
    print("\nChecking Transformation Matmul (All Batches)...")
    
    n_markers = geno.n_markers
    batch_size = 5000
    
    import warnings
    
    for start in range(0, n_markers, batch_size):
        end = min(start + batch_size, n_markers)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                g_batch = geno.get_batch_imputed(start, end)
                g_batch = np.nan_to_num(g_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Matmul
                res = eigenvecs.T @ g_batch
                
                if w:
                    print(f"Warning at batch {start}-{end}: {w[-1].message}")
                    print(f"  G_batch min/max: {g_batch.min()}, {g_batch.max()}")
                    print(f"  G_batch has NaNs: {np.any(np.isnan(g_batch))}")
                    print(f"  Eigenvecs min/max: {eigenvecs.min()}, {eigenvecs.max()}")

        except Exception as e:
             print(f"Error at batch {start}-{end}: {e}")

    # 6. Check MLM Inversion (XViX)
    print("\nChecking MLM Inversion...")
    # Mock X (Intercept + 5 PCs)
    n_ind = geno.n_individuals
    X = np.ones((n_ind, 1))
    
    # Compute PCs
    if pipeline.genotype_matrix.n_individuals == n_ind:
        pipeline.compute_population_structure(n_pcs=5)
        pcs = pipeline.pcs
        check_array("PCs", pcs)
        X = np.column_stack([X, pcs])
    
    check_array("Design Matrix X", X)
    
    # Mock Variance Component V
    # V = h2*Vg + (1-h2)*Ve ... essentially V scales with K and I.
    # Simplified check: Just invert X'X (if V=I)
    print("Checking inv(X'X)...")
    XtX = X.T @ X
    check_array("X'X", XtX)
    # 7. Run MLM fully
    print("\nRunning Full MLM to Reproduce Warnings...")
    y_sub, g_sub, cov_sub, k_sub = pipeline._prepare_trait_data(TRAIT)
    from pymvp.association.mlm import MVP_MLM
    
    # Calculate EigenK manually if needed or let MLM do it
    # MLM calculates it if not passed.
    try:
        res = MVP_MLM(y_sub, g_sub, K=k_sub, CV=cov_sub, verbose=True)
        print("MLM Completed Successfully.")
    except Exception as e:
        print(f"MLM Failed: {e}")

if __name__ == "__main__":
    main()
