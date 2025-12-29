
import numpy as np
import warnings
import time
from typing import Optional, Union, Dict

from ..utils.data_types import GenotypeMatrix, AssociationResults
from ..matrix.kinship_loco import PANICLE_K_VanRaden_LOCO, LocoKinship, _extract_chromosomes
from .mlm import estimate_variance_components_brent, _calculate_neg_ml_likelihood
from .mlm_loco import PANICLE_MLM_LOCO
from .lrt import fit_marker_lrt

def PANICLE_MLM_Hybrid(phe: np.ndarray,
                   geno: Union[GenotypeMatrix, np.ndarray],
                   map_data,
                   loco_kinship: Optional[LocoKinship] = None,
                   CV: Optional[np.ndarray] = None,
                   screen_threshold: float = 1e-4,
                   maxLine: int = 1000,
                   cpu: int = 1,
                   verbose: bool = True) -> AssociationResults:
    """
    Hybrid Mixed Linear Model (LOCO Wald Screen + LRT Refine)
    
    Strategy:
    1. Run LOCO MLM (Wald test with P3D approximation).
    2. Identify markers with p-value < screen_threshold.
    3. Re-test these markers using the exact Likelihood Ratio Test (LRT)
       by re-estimating variance components for each chromosome's LOCO model.
       
    Args:
        phe: Phenotype matrix (n x 2)
        geno: Genotype matrix
        map_data: Genetic map (must include CHROM labels)
        loco_kinship: Optional precomputed LOCO kinship container
        CV: Covariates
        screen_threshold: P-value threshold to trigger LRT refinement
        maxLine: Batch size
        cpu: Number of cores
        verbose: Verbosity
        
    Returns:
        AssociationResults with updated p-values for refined markers.
    """

    def _sanitize_array(arr: np.ndarray, name: str, clip: float = 1e6) -> np.ndarray:
        """Make array finite and optionally clip extreme magnitudes to avoid overflow."""
        arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            max_abs = np.max(np.abs(arr)) if arr.size else 0.0
            if max_abs > clip:
                if verbose:
                    warnings.warn(f"{name} had values > {clip:.0e}; clipping to +/-{clip:.0e}")
                arr = np.clip(arr, -clip, clip, out=arr)
        return arr
    
    if verbose:
        print("=" * 60)
        print("HYBRID MLM: WALD SCREEN + LRT REFINE")
        print(f"Screening threshold: {screen_threshold}")
        print("=" * 60)
        
    # -------------------------------------------------------------------------
    # Step 1: Data Preparation
    # -------------------------------------------------------------------------

    if map_data is None:
        raise ValueError("map_data is required for LOCO hybrid MLM")

    # Extract phenotype vector
    trait_values = phe[:, 1].astype(np.float64)
    trait_values = _sanitize_array(trait_values, "trait_values")
    n_individuals = len(trait_values)

    # Handle Covariates
    if CV is not None:
        CV = _sanitize_array(CV.astype(np.float64), "covariates")
        X = np.column_stack([np.ones(n_individuals), CV])
    else:
        X = np.ones((n_individuals, 1))

    if loco_kinship is None:
        loco_kinship = PANICLE_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)

    if isinstance(geno, GenotypeMatrix):
        n_markers = geno.n_markers
    else:
        n_markers = geno.shape[1]
    chrom_values = _extract_chromosomes(map_data, n_markers)

    # -------------------------------------------------------------------------
    # Step 2: Fast Wald Scan (Screening)
    # -------------------------------------------------------------------------
    
    if verbose:
        print("\n--- Phase 1: Fast Wald Screening ---")
        
    # Run LOCO MLM
    wald_results = PANICLE_MLM_LOCO(
        phe=phe,
        geno=geno,
        map_data=map_data,
        loco_kinship=loco_kinship,
        CV=CV,
        vc_method="BRENT",
        maxLine=maxLine,
        cpu=cpu,
        verbose=verbose
    )
    
    # -------------------------------------------------------------------------
    # Step 3: LRT Refinement
    # -------------------------------------------------------------------------
    
    # Identify candidates
    candidate_indices = np.where(wald_results.pvalues < screen_threshold)[0]
    n_candidates = len(candidate_indices)
    
    if verbose:
        print(f"\n--- Phase 2: LRT Refinement ---")
        print(f"Found {n_candidates} candidates (p < {screen_threshold})")
        
    if n_candidates == 0:
        if verbose:
            print("No candidates found for refinement.")
        return wald_results
        
    # Prepare for refinement
    # We modify the results in-place
    final_pvalues = wald_results.pvalues.copy()
    final_effects = wald_results.effects.copy()
    final_se = wald_results.se.copy()
    
    # Process candidates grouped by chromosome (shared null model per chromosome)
    null_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_null_model(chrom: str) -> Dict[str, np.ndarray]:
        if chrom in null_cache:
            return null_cache[chrom]

        eigen = loco_kinship.get_eigen(chrom)
        eigenvals = np.maximum(
            _sanitize_array(np.asarray(eigen['eigenvals'], dtype=np.float64), "eigenvals", clip=None),
            1e-6
        )
        eigenvecs = _sanitize_array(np.asarray(eigen['eigenvecs'], dtype=np.float64), "eigenvecs", clip=None)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            y_transformed = eigenvecs.T @ trait_values
            X_transformed = eigenvecs.T @ X
        y_transformed = _sanitize_array(y_transformed, "y_transformed")
        X_transformed = _sanitize_array(X_transformed, "X_transformed")

        delta_null, vg_null, ve_null = estimate_variance_components_brent(
            y_transformed, X_transformed, eigenvals, verbose=False, use_ml=True
        )
        h2_null = vg_null / (vg_null + ve_null) if (vg_null + ve_null) > 0 else 0.0
        null_neg_loglik = _calculate_neg_ml_likelihood(h2_null, y_transformed, X_transformed, eigenvals)

        payload = {
            "eigenvals": eigenvals,
            "eigenvecs": eigenvecs,
            "y_transformed": y_transformed,
            "X_transformed": X_transformed,
            "null_neg_loglik": null_neg_loglik,
        }
        null_cache[chrom] = payload
        return payload

    # Process candidates
    # Note: This loop is currently sequential. Could be parallelized if needed.
    
    start_time = time.time()
    for i, marker_idx in enumerate(candidate_indices):
        if verbose and i % 10 == 0:
            print(f"Refining marker {i+1}/{n_candidates}...", end='\r')

        chrom = str(chrom_values[marker_idx])
        null_model = _get_null_model(chrom)
            
        # Get Genotype
        # Handling GenotypeMatrix vs ndarray
        if isinstance(geno, GenotypeMatrix):
            # Efficiently fetch single column
            # Note: accessing memmap column by column can be slow if not contiguous.
            # But for sparse candidates it's usually acceptable.
            # Ideally we'd batch this too, but for simplicity:
            g_raw = geno.get_batch_imputed(marker_idx, marker_idx+1).flatten()
        else:
            g_raw = geno[:, marker_idx]
            
        g_raw = _sanitize_array(np.asarray(g_raw, dtype=np.float64), "genotype_marker")
        
        # Transform Genotype
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g_transformed = null_model["eigenvecs"].T @ g_raw
        g_transformed = _sanitize_array(g_transformed, "g_transformed")
        
        # Run LRT
        lrt_stat, lrt_p, lrt_beta, lrt_se = fit_marker_lrt(
            null_model["y_transformed"],
            null_model["X_transformed"],
            g_transformed, 
            null_model["eigenvals"],
            null_model["null_neg_loglik"]
        )
        
        # Update results if the LRT model is numerically stable
        if np.isfinite(lrt_p) and np.isfinite(lrt_beta) and np.isfinite(lrt_se):
            final_pvalues[marker_idx] = lrt_p
            final_effects[marker_idx] = lrt_beta
            final_se[marker_idx] = lrt_se
        
    duration = time.time() - start_time
    if verbose:
        print(f"\nRefinement complete in {duration:.2f}s ({duration/max(1,n_candidates):.3f}s/marker)")
        
    # Return updated results
    # Effects/SE are refined only for candidates with stable LRT fits.
    return AssociationResults(
        final_effects,
        final_se,
        final_pvalues
    )
