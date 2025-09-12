"""
FarmCPU with JIT-optimized GLM integration

This version integrates the JIT-optimized GLM into FarmCPU while maintaining
exact rMVP compliance. Uses the validated JIT GLM for performance improvement
while keeping all other FarmCPU logic identical to the original.
"""

import numpy as np
from typing import Optional, Union, Tuple
from ..utils.data_types import GenotypeMatrix, AssociationResults
from .farmcpu import MVP_FarmCPU as MVP_FarmCPU_original
from .glm_jit import MVP_GLM_jit
from .farmcpu import FarmCPUTimer


def MVP_FarmCPU_jit(phe: np.ndarray,
                    geno: Union[GenotypeMatrix, np.ndarray],
                    map_data,
                    CV: Optional[np.ndarray] = None,
                    maxLoop: int = 10,
                    p_threshold: float = 0.05,
                    QTN_threshold: float = 0.01,
                    bin_size = None,
                    method_bin: str = "static",
                    cpu: int = 1,
                    verbose: bool = True,
                    timing: bool = False) -> Tuple[AssociationResults, Optional[FarmCPUTimer]]:
    """FarmCPU with JIT-optimized GLM for improved performance
    
    This version replaces the original GLM calls with JIT-optimized GLM
    while maintaining exact rMVP compliance. All other FarmCPU logic
    remains identical to the original implementation.
    
    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix or GenotypeMatrix (with optimizations)
        map_data: GenotypeMap with SNP information
        CV: Covariate matrix (typically PCs)
        maxLoop: Maximum FarmCPU iterations
        p_threshold: P-value threshold for candidate QTN selection
        QTN_threshold: P-value threshold for final QTN selection
        bin_size: Bin sizes for hierarchical binning
        method_bin: Binning method ("static" for rMVP compatibility)
        cpu: Number of CPU cores (for future implementation)
        verbose: Print progress information
        timing: Enable detailed timing analysis
        
    Returns:
        Tuple of (AssociationResults, Optional[FarmCPUTimer])
    """
    
    # Initialize timer if requested
    timer = FarmCPUTimer() if timing else None
    
    if timing:
        timer.start("total")
        timer.start("preparation")
    
    if verbose:
        print("🧬 FarmCPU with JIT-optimized GLM")
        
        # Check numba availability
        try:
            import numba
            print("⚡ Using Numba JIT for ~2x GLM performance boost")
        except ImportError:
            print("💡 Numba not available - using standard GLM (install numba for speedup)")
        
        # Check for genotype optimizations
        if isinstance(geno, GenotypeMatrix):
            if hasattr(geno, '_major_alleles') and geno._major_alleles is not None:
                print("🚀 Using pre-computed major alleles for fast imputation")
            if hasattr(geno, '_memory_mapped') and geno._memory_mapped:
                print("💾 Using memory-mapped storage for large dataset support")
    
    # Ensure genotype matrix has optimizations if available
    if isinstance(geno, GenotypeMatrix):
        # Pre-compute major alleles if not already done
        if not hasattr(geno, '_major_alleles') or geno._major_alleles is None:
            if verbose:
                print("🔄 Pre-computing major alleles for fast imputation...")
            geno._precompute_major_alleles()
            if verbose:
                print("✅ Major alleles pre-computed")
    
    if timing:
        prep_time = timer.end("preparation")
        if verbose:
            print(f"✅ Preparation completed in {prep_time:.3f}s")
    
    # Call the original FarmCPU but with JIT GLM patch
    if timing:
        timer.start("farmcpu_algorithm")
    
    # Temporarily replace the original GLM with JIT GLM
    import pymvp.association.farmcpu as farmcpu_module
    original_glm = farmcpu_module.MVP_GLM
    
    try:
        # Patch the GLM function in the FarmCPU module
        farmcpu_module.MVP_GLM = MVP_GLM_jit
        
        # Call original FarmCPU with identical parameters (now using JIT GLM)
        results = MVP_FarmCPU_original(
            phe=phe,
            geno=geno,
            map_data=map_data,
            CV=CV,
            maxLoop=maxLoop,
            p_threshold=p_threshold,
            QTN_threshold=QTN_threshold,
            bin_size=bin_size,
            method_bin=method_bin,
            cpu=cpu,
            verbose=verbose
        )
        
    finally:
        # Restore original GLM function
        farmcpu_module.MVP_GLM = original_glm
    
    if timing:
        algorithm_time = timer.end("farmcpu_algorithm")
        total_time = timer.end("total")
        if verbose:
            print(f"✅ FarmCPU algorithm completed in {algorithm_time:.3f}s")
            print(f"🎉 Total analysis completed in {total_time:.3f}s")
    
    if verbose:
        print("✅ JIT-optimized FarmCPU complete - enhanced performance with exact rMVP compliance")
    
    return results, timer


# Convenience alias
MVP_FarmCPU_with_JIT = MVP_FarmCPU_jit