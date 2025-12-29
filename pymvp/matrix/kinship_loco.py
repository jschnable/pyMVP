"""
LOCO (Leave-One-Chromosome-Out) kinship matrix utilities.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import warnings
import pandas as pd

from ..utils.data_types import GenotypeMatrix, GenotypeMap, KinshipMatrix


def _extract_chromosomes(map_data: Union[GenotypeMap, pd.DataFrame, np.ndarray, List],
                         n_markers: int) -> np.ndarray:
    """Extract chromosome labels aligned to genotype markers."""
    if isinstance(map_data, GenotypeMap):
        chroms = map_data.chromosomes
    elif isinstance(map_data, pd.DataFrame):
        if "CHROM" not in map_data.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        chroms = map_data["CHROM"]
    elif hasattr(map_data, "to_dataframe"):
        map_df = map_data.to_dataframe()
        if "CHROM" not in map_df.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        chroms = map_df["CHROM"]
    else:
        chroms = np.asarray(map_data)

    chroms = np.asarray(chroms).astype(str, copy=False)
    if chroms.ndim != 1 or len(chroms) != n_markers:
        raise ValueError("Chromosome labels must be a 1D array aligned to genotype markers")
    return chroms


def _group_markers_by_chrom(chrom_values: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ordered marker indices grouped by chromosome."""
    chrom_to_indices: Dict[str, List[int]] = {}
    for idx, chrom in enumerate(chrom_values):
        chrom_key = str(chrom)
        chrom_to_indices.setdefault(chrom_key, []).append(idx)
    return {chrom: np.asarray(indices, dtype=int) for chrom, indices in chrom_to_indices.items()}


class LocoKinship:
    """Container for LOCO kinship computations and cached eigendecompositions."""

    def __init__(self,
                 total_raw: np.ndarray,
                 total_diag: np.ndarray,
                 chrom_raw: Dict[str, np.ndarray],
                 chrom_diag: Dict[str, np.ndarray],
                 chrom_order: List[str]):
        self._total_raw = total_raw
        self._total_diag = total_diag
        self._chrom_raw = chrom_raw
        self._chrom_diag = chrom_diag
        self._chrom_order = list(chrom_order)

        self._loco_cache: Dict[str, KinshipMatrix] = {}
        self._eigen_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._full_cache: Optional[KinshipMatrix] = None

    @property
    def chromosomes(self) -> List[str]:
        """Chromosome labels in the order they appeared."""
        return list(self._chrom_order)

    def _normalize(self, raw: np.ndarray, diag: np.ndarray, label: str) -> KinshipMatrix:
        """Symmetrize and normalize a raw kinship matrix."""
        kin = (raw + raw.T) / 2.0
        mean_diag = float(np.mean(diag))
        if mean_diag > 0:
            kin = kin / mean_diag
        else:
            warnings.warn(f"Mean diagonal for {label} is non-positive; skipping normalization")
        return KinshipMatrix(kin)

    def get_full(self) -> KinshipMatrix:
        """Return the full (non-LOCO) kinship matrix."""
        if self._full_cache is None:
            self._full_cache = self._normalize(self._total_raw, self._total_diag, "full")
        return self._full_cache

    def get_loco(self, chrom: Union[str, int]) -> KinshipMatrix:
        """Return the LOCO kinship matrix for a chromosome."""
        chrom_key = str(chrom)
        if chrom_key in self._loco_cache:
            return self._loco_cache[chrom_key]
        if chrom_key not in self._chrom_raw:
            raise KeyError(f"Chromosome {chrom_key} not found in LOCO kinship")

        raw_loco = self._total_raw - self._chrom_raw[chrom_key]
        diag_loco = self._total_diag - self._chrom_diag[chrom_key]
        kin = self._normalize(raw_loco, diag_loco, f"loco:{chrom_key}")
        self._loco_cache[chrom_key] = kin
        return kin

    def get_eigen(self, chrom: Union[str, int]) -> Dict[str, np.ndarray]:
        """Return cached eigendecomposition for a LOCO kinship matrix."""
        chrom_key = str(chrom)
        if chrom_key in self._eigen_cache:
            return self._eigen_cache[chrom_key]

        kinship = self.get_loco(chrom_key).to_numpy()
        eigenvals, eigenvecs = np.linalg.eigh(kinship)
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        eigen = {"eigenvals": eigenvals, "eigenvecs": eigenvecs}
        self._eigen_cache[chrom_key] = eigen
        return eigen


def MVP_K_VanRaden_LOCO(M: Union[GenotypeMatrix, np.ndarray],
                        map_data: Union[GenotypeMap, pd.DataFrame, np.ndarray, List],
                        maxLine: int = 5000,
                        verbose: bool = True) -> LocoKinship:
    """Compute LOCO kinship using VanRaden-style raw cross-products."""
    if isinstance(M, GenotypeMatrix):
        genotype = M
        n_individuals = M.n_individuals
        n_markers = M.n_markers
    elif isinstance(M, np.ndarray):
        genotype = M
        n_individuals, n_markers = M.shape
    else:
        raise ValueError("M must be GenotypeMatrix or numpy array")

    chrom_values = _extract_chromosomes(map_data, n_markers)
    chrom_groups = _group_markers_by_chrom(chrom_values)
    chrom_order = list(chrom_groups.keys())

    if verbose:
        print(f"Calculating LOCO kinship for {n_individuals} individuals, {n_markers} markers")
        print(f"Chromosomes: {len(chrom_order)}")

    raw_total = np.zeros((n_individuals, n_individuals), dtype=np.float64)
    diag_total = np.zeros(n_individuals, dtype=np.float64)
    raw_by_chrom = {chrom: np.zeros((n_individuals, n_individuals), dtype=np.float64) for chrom in chrom_order}
    diag_by_chrom = {chrom: np.zeros(n_individuals, dtype=np.float64) for chrom in chrom_order}

    n_batches = (n_markers + maxLine - 1) // maxLine
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)

        if verbose and n_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{n_batches} (markers {start_marker}-{end_marker - 1})")

        if isinstance(genotype, GenotypeMatrix):
            Z_batch = genotype.get_batch_imputed(start_marker, end_marker).astype(np.float64)
        else:
            Z_batch = genotype[:, start_marker:end_marker].astype(np.float64)

        means_batch = np.nanmean(Z_batch, axis=0)
        means_batch[np.isnan(means_batch)] = 0.0
        Z_batch -= means_batch[np.newaxis, :]

        if not np.all(np.isfinite(Z_batch)):
            if verbose and batch_idx == 0:
                print("Warning: Z_batch contains NaNs or Infs. Replacing with 0.")
            Z_batch[~np.isfinite(Z_batch)] = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_total += Z_batch @ Z_batch.T

        diag_total += np.sum(Z_batch * Z_batch, axis=1)

        batch_chroms = chrom_values[start_marker:end_marker]
        for chrom in np.unique(batch_chroms):
            cols = batch_chroms == chrom
            if not np.any(cols):
                continue
            Z_sub = Z_batch[:, cols]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    raw_by_chrom[chrom] += Z_sub @ Z_sub.T
            diag_by_chrom[chrom] += np.sum(Z_sub * Z_sub, axis=1)

    raw_total = (raw_total + raw_total.T) / 2.0
    for chrom in chrom_order:
        raw_by_chrom[chrom] = (raw_by_chrom[chrom] + raw_by_chrom[chrom].T) / 2.0

    return LocoKinship(
        total_raw=raw_total,
        total_diag=diag_total,
        chrom_raw=raw_by_chrom,
        chrom_diag=diag_by_chrom,
        chrom_order=chrom_order,
    )
