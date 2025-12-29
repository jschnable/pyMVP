"""
General Linear Model (GLM) for GWAS analysis.

This module provides the `MVP_GLM` function, which uses the Frisch-Waugh-Lovell (FWL)
theorem and QR decomposition for high-performance association testing.

Implementation details:
- pymvp.association.glm_fwl_qr: Fast vectorized implementation (default).
"""

from typing import Optional, Union
import numpy as np

from ..utils.data_types import GenotypeMatrix, AssociationResults
from .glm_fwl_qr import PANICLE_GLM_ultrafast

def PANICLE_GLM(phe: np.ndarray,
           geno: Union[GenotypeMatrix, np.ndarray],
           CV: Optional[np.ndarray] = None,
           maxLine: int = 5000,
           cpu: int = 1,
           verbose: bool = True,
           impute_missing: bool = True,
           major_alleles: Optional[np.ndarray] = None,
           missing_fill_value: float = 1.0) -> AssociationResults:
    """General Linear Model (GLM) for GWAS.
    
    Uses an optimized FWL+QR algorithm for speed.

    Args:
        phe: Phenotype array (n_individuals x 2) [ID, Value]
        geno: Genotype matrix (n_individuals x n_markers)
        CV: Covariates (n_individuals x n_covariates)
        maxLine: Batch size for processing
        cpu: Unused (kept for API compatibility)
        verbose: Print progress
        impute_missing: Unused (always handled internally by FWL+QR loader)
        major_alleles: Unused (always handled internally)
        missing_fill_value: Value to use for missing genotypes (default: 1.0)

    Returns:
        AssociationResults object with effects, SEs, and p-values.
    """
    return PANICLE_GLM_ultrafast(
        phe=phe,
        geno=geno,
        CV=CV,
        maxLine=maxLine,
        cpu=cpu,
        verbose=verbose,
        missing_fill_value=missing_fill_value,
    )
