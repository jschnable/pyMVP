"""
LOCO (Leave-One-Chromosome-Out) MLM wrapper.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

from typing import Optional, Union, Tuple, Dict
import numpy as np

from ..utils.data_types import GenotypeMatrix, AssociationResults
from ..matrix.kinship_loco import PANICLE_K_VanRaden_LOCO, LocoKinship, _extract_chromosomes, _group_markers_by_chrom
from .mlm import PANICLE_MLM

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def _subset_genotypes(geno: Union[GenotypeMatrix, np.ndarray],
                      indices: np.ndarray) -> np.ndarray:
    """Return a genotype submatrix for a set of marker indices.

    For GenotypeMatrix, uses get_columns_imputed which handles -9 and NaN.
    For pre-imputed GenotypeMatrix, skips -9 checks for faster access.
    For numpy arrays, handles -9 sentinel and NaN values by imputing to 0.
    """
    if isinstance(geno, GenotypeMatrix):
        if geno.is_imputed:
            # Data is pre-imputed, skip -9 checks for faster access
            return geno._data[:, indices].astype(np.float32)
        return geno.get_columns_imputed(indices)
    # For numpy arrays, handle missing values
    subset = geno[:, indices].astype(np.float32)
    missing_mask = (subset == -9) | np.isnan(subset)
    if missing_mask.any():
        subset[missing_mask] = 0.0
    return subset


def _process_chromosome(chrom: str,
                        indices: np.ndarray,
                        geno_subset: np.ndarray,
                        phe: np.ndarray,
                        K_loco: np.ndarray,
                        eigenK: Dict[str, np.ndarray],
                        CV: Optional[np.ndarray],
                        vc_method: str,
                        maxLine: int) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single chromosome for LOCO MLM.

    This function is designed to be called in parallel for each chromosome.
    Takes pre-computed K_loco and eigenK to avoid serializing the full LocoKinship object.

    Returns:
        Tuple of (chrom, indices, effects, std_errors, pvalues)
    """
    res = PANICLE_MLM(
        phe=phe,
        geno=geno_subset,
        K=K_loco,
        eigenK=eigenK,
        CV=CV,
        vc_method=vc_method,
        maxLine=maxLine,
        cpu=1,  # Don't nest parallelism
        verbose=False,
    )

    return chrom, indices, res.effects, res.se, res.pvalues


def PANICLE_MLM_LOCO(phe: np.ndarray,
                 geno: Union[GenotypeMatrix, np.ndarray],
                 map_data,
                 loco_kinship: Optional[LocoKinship] = None,
                 CV: Optional[np.ndarray] = None,
                 vc_method: str = "BRENT",
                 maxLine: int = 1000,
                 cpu: int = 1,
                 verbose: bool = True) -> AssociationResults:
    """Run MLM with LOCO kinship matrices grouped by chromosome.

    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        map_data: Genetic map with chromosome information
        loco_kinship: Pre-computed LOCO kinship (computed if None)
        CV: Covariate matrix (n_individuals × n_covariates), optional
        vc_method: Variance component estimation method ["BRENT"]
        maxLine: Batch size for processing markers
        cpu: Number of CPU cores for parallel chromosome processing
        verbose: Print progress information

    Returns:
        AssociationResults object containing Effect, SE, and P-value for each marker
    """
    if isinstance(geno, GenotypeMatrix):
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        n_markers = geno.shape[1]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    chrom_values = _extract_chromosomes(map_data, n_markers)
    chrom_groups = _group_markers_by_chrom(chrom_values)

    if loco_kinship is None:
        loco_kinship = PANICLE_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)

    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64)
    p_values = np.ones(n_markers, dtype=np.float64)

    # Filter out empty chromosome groups
    chrom_items = [(chrom, indices) for chrom, indices in chrom_groups.items() if indices.size > 0]
    n_chroms = len(chrom_items)

    if verbose:
        print("=" * 60)
        print("LOCO MLM")
        print("=" * 60)
        print(f"Chromosomes: {n_chroms}")

    # Handle cpu=0 to mean use all available cores
    if cpu == 0:
        import multiprocessing
        cpu = multiprocessing.cpu_count()

    # Determine if we should use parallel processing
    use_parallel = HAS_JOBLIB and cpu > 1 and n_chroms > 1

    if use_parallel:
        if verbose:
            print(f"Using parallel processing with {min(cpu, n_chroms)} workers")

        # Pre-compute all data needed for parallel processing
        # This avoids serializing the full LocoKinship object to workers
        chrom_data = []
        for chrom, indices in chrom_items:
            geno_subset = _subset_genotypes(geno, indices)
            eigenK = loco_kinship.get_eigen(chrom)
            K_loco = loco_kinship.get_loco(chrom).to_numpy()
            chrom_data.append((chrom, indices, geno_subset, K_loco, eigenK))

        # Process chromosomes in parallel
        # Use 'loky' for CPU-bound work (releases GIL in numpy/numba)
        results = Parallel(n_jobs=min(cpu, n_chroms), backend='loky')(
            delayed(_process_chromosome)(
                chrom, indices, geno_subset, phe, K_loco, eigenK, CV, vc_method, maxLine
            )
            for chrom, indices, geno_subset, K_loco, eigenK in chrom_data
        )

        # Collect results
        for chrom, indices, eff, se, pvals in results:
            effects[indices] = eff
            std_errors[indices] = se
            p_values[indices] = pvals

    else:
        # Sequential processing (original behavior)
        if verbose and not HAS_JOBLIB and cpu > 1:
            print("Note: joblib not available, using sequential processing")

        for chrom, indices in chrom_items:
            if verbose:
                print(f"Processing chromosome {chrom} ({indices.size} markers)")

            geno_subset = _subset_genotypes(geno, indices)
            eigenK = loco_kinship.get_eigen(chrom)
            K_loco = loco_kinship.get_loco(chrom)

            res = PANICLE_MLM(
                phe=phe,
                geno=geno_subset,
                K=K_loco,
                eigenK=eigenK,
                CV=CV,
                vc_method=vc_method,
                maxLine=maxLine,
                cpu=1,
                verbose=False,
            )

            effects[indices] = res.effects
            std_errors[indices] = res.se
            p_values[indices] = res.pvalues

    return AssociationResults(effects=effects, se=std_errors, pvalues=p_values)
