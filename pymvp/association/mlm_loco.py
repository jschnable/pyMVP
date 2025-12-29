"""
LOCO (Leave-One-Chromosome-Out) MLM wrapper.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

from typing import Optional, Union
import numpy as np

from ..utils.data_types import GenotypeMatrix, AssociationResults
from ..matrix.kinship_loco import MVP_K_VanRaden_LOCO, LocoKinship, _extract_chromosomes, _group_markers_by_chrom
from .mlm import MVP_MLM


def _subset_genotypes(geno: Union[GenotypeMatrix, np.ndarray],
                      indices: np.ndarray) -> np.ndarray:
    """Return a genotype submatrix for a set of marker indices."""
    if isinstance(geno, GenotypeMatrix):
        return geno.get_columns_imputed(indices)
    return geno[:, indices]


def MVP_MLM_LOCO(phe: np.ndarray,
                 geno: Union[GenotypeMatrix, np.ndarray],
                 map_data,
                 loco_kinship: Optional[LocoKinship] = None,
                 CV: Optional[np.ndarray] = None,
                 vc_method: str = "BRENT",
                 maxLine: int = 1000,
                 cpu: int = 1,
                 verbose: bool = True) -> AssociationResults:
    """Run MLM with LOCO kinship matrices grouped by chromosome."""
    if isinstance(geno, GenotypeMatrix):
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        n_markers = geno.shape[1]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    chrom_values = _extract_chromosomes(map_data, n_markers)
    chrom_groups = _group_markers_by_chrom(chrom_values)

    if loco_kinship is None:
        loco_kinship = MVP_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)

    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64)
    p_values = np.ones(n_markers, dtype=np.float64)

    if verbose:
        print("=" * 60)
        print("LOCO MLM")
        print("=" * 60)
        print(f"Chromosomes: {len(chrom_groups)}")

    for chrom, indices in chrom_groups.items():
        if indices.size == 0:
            continue

        if verbose:
            print(f"Processing chromosome {chrom} ({indices.size} markers)")

        geno_subset = _subset_genotypes(geno, indices)
        eigenK = loco_kinship.get_eigen(chrom)
        K_loco = loco_kinship.get_loco(chrom)

        res = MVP_MLM(
            phe=phe,
            geno=geno_subset,
            K=K_loco,
            eigenK=eigenK,
            CV=CV,
            vc_method=vc_method,
            maxLine=maxLine,
            cpu=cpu,
            verbose=False,
        )

        effects[indices] = res.effects
        std_errors[indices] = res.se
        p_values[indices] = res.pvalues

    return AssociationResults(effects=effects, se=std_errors, pvalues=p_values)
