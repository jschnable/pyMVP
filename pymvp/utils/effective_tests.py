"""
Effective number of independent marker estimation following the harmonised GEC
specification.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .data_types import GenotypeMap, GenotypeMatrix


class LDSparseMatrixProtocol:
    """Protocol-like base class describing the minimal LD interface we consume."""

    def get_all_unique_indexes(self) -> Sequence[int]:
        raise NotImplementedError

    def sub_dense_matrix(self, index_list: Sequence[int]) -> np.ndarray:
        raise NotImplementedError

    def release_ld_data(self) -> None:
        return None

    def convert_to_pvalue_coefficients(self, coeffs: Sequence[float]) -> None:
        raise NotImplementedError


@dataclass
class BlockStat:
    chrom: str
    start_pos: int
    end_pos: int
    n_snps: int
    Me: float
    largest_gap_bp: int


@dataclass
class ChromosomeResult:
    Me: float
    n_snps: int
    block_stats: List[BlockStat]
    missing_ld_snps: int = 0
    monomorphic_dropped: int = 0


DEFAULT_POLYNOMIAL_COEFFS: Tuple[float, ...] = (
    0.7723,
    -1.5659,
    1.2010,
    -0.2355,
    0.2184,
    0.6086,
    0.0,
)


def _apply_polynomial_transform(matrix: np.ndarray, coeffs: Sequence[float]) -> None:
    """In-place polynomial transform mirroring convert2PValueCoefficient."""
    if matrix.size == 0:
        return

    # Work on upper triangle including diagonal.
    triu_idx = np.triu_indices_from(matrix)
    values = matrix[triu_idx]

    x = values * values
    result = np.zeros_like(x)
    for coeff in coeffs[:-1]:
        result = (result + coeff) * x
    result += coeffs[-1]

    result = np.clip(result, 0.0, 1.0, out=result)

    matrix[triu_idx] = result
    matrix[(triu_idx[1], triu_idx[0])] = result  # mirror to lower triangle
    np.fill_diagonal(matrix, 1.0)


class _GenotypeLDSource(LDSparseMatrixProtocol):
    """Adapter exposing a genotype matrix chunk as an LD source."""

    def __init__(
        self,
        genotype: GenotypeMatrix,
        indices: Sequence[int],
        positions: Sequence[int],
        *,
        monomorphic_dropped: int = 0,
    ) -> None:
        if len(indices) != len(positions):
            raise ValueError("Indices and positions must be the same length")

        self._genotype = genotype
        self._indices = np.asarray(indices, dtype=int)
        self._positions = np.asarray(positions, dtype=int)
        order = np.argsort(self._positions, kind="mergesort")
        self._indices = self._indices[order]
        self._positions = self._positions[order]
        self._position_lookup: Dict[int, int] = {
            int(idx): int(pos) for idx, pos in zip(self._indices, self._positions)
        }
        self.monomorphic_dropped = int(monomorphic_dropped)

    def get_all_unique_indexes(self) -> Sequence[int]:
        return self._indices

    def get_position(self, index: int) -> int:
        return self._position_lookup[int(index)]

    def sub_dense_matrix(self, index_list: Sequence[int]) -> np.ndarray:
        if not index_list:
            raise ValueError("index_list must contain at least one marker")

        columns = self._genotype.get_columns_imputed(index_list)
        if columns.size == 0:
            raise ValueError("Genotype slice returned an empty matrix")

        columns = columns.astype(np.float64, copy=False)
        # Standardise columns to zero mean / unit variance; protect against zeros.
        means = np.mean(columns, axis=0, dtype=np.float64)
        columns -= means
        stds = np.std(columns, axis=0, ddof=1)
        zero_variance = stds == 0
        stds[zero_variance] = 1.0
        columns /= stds
        np.nan_to_num(columns, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            corr = (columns.T @ columns) / max(columns.shape[0] - 1, 1)
        np.nan_to_num(corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if zero_variance.any():
            zero_idx = np.where(zero_variance)[0]
            for idx in zero_idx:
                corr[idx, :] = 0.0
                corr[:, idx] = 0.0
            corr[zero_idx, zero_idx] = 1.0

        return corr

    def release_ld_data(self) -> None:
        return None

    def pair_correlation(self, idx_a: int, idx_b: int) -> Optional[float]:
        if idx_a == idx_b:
            return 1.0
        cols = self._genotype.get_columns_imputed([idx_a, idx_b])
        if cols.shape[1] != 2:
            return None
        a = cols[:, 0].astype(np.float64, copy=False)
        b = cols[:, 1].astype(np.float64, copy=False)
        a -= np.mean(a)
        b -= np.mean(b)
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
        if std_a == 0 or std_b == 0:
            return None
        corr = float(np.dot(a, b) / ((len(a) - 1) * std_a * std_b))
        if np.isnan(corr):
            return None
        return corr


def _normalise_ld_sources(
    ld_sources: Union[
        LDSparseMatrixProtocol,
        Mapping[str, LDSparseMatrixProtocol],
        Iterable[Tuple[str, LDSparseMatrixProtocol]],
    ],
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, LDSparseMatrixProtocol]:
    if isinstance(ld_sources, LDSparseMatrixProtocol):
        chrom = "unknown"
        if metadata is not None:
            chrom = str(metadata.get("chromosome", chrom))
        return {chrom: ld_sources}

    if isinstance(ld_sources, Mapping):
        return dict(ld_sources)

    normalised: Dict[str, LDSparseMatrixProtocol] = {}
    for chrom, source in ld_sources:
        normalised[str(chrom)] = source
    return normalised


def _construct_blocks(
    chrom: str,
    ld_source: LDSparseMatrixProtocol,
    *,
    max_window_bp: int,
    corr_cutoff: float,
    gap_snp_limit: int,
    span_bp_limit: int,
) -> Tuple[List[List[int]], List[BlockStat], int]:
    indices = list(ld_source.get_all_unique_indexes())
    if not indices:
        return [], [], 0

    if hasattr(ld_source, "get_position"):
        positions = [int(getattr(ld_source, "get_position")(idx)) for idx in indices]
    else:
        positions = list(range(len(indices)))

    entries = sorted(zip(indices, positions), key=lambda x: x[1])
    blocks: List[List[int]] = []
    stats: List[BlockStat] = []

    n = len(entries)
    start = 0
    while start < n:
        block_indices = [entries[start][0]]
        block_positions = [entries[start][1]]
        largest_gap = 0
        gap_run = 0
        end = start + 1

        while end < n:
            curr_idx, curr_pos = entries[end]
            prev_idx, prev_pos = entries[end - 1]

            if curr_pos - entries[start][1] > span_bp_limit:
                break
            if curr_pos - prev_pos > max_window_bp:
                break

            pair_corr = None
            if hasattr(ld_source, "pair_correlation"):
                pair_corr = getattr(ld_source, "pair_correlation")(prev_idx, curr_idx)

            if pair_corr is None:
                gap_run += 1
                if gap_run > gap_snp_limit:
                    break
            else:
                gap_run = 0
                if abs(pair_corr) < corr_cutoff:
                    break

            largest_gap = max(largest_gap, curr_pos - prev_pos)
            block_indices.append(curr_idx)
            block_positions.append(curr_pos)
            end += 1

        blocks.append(block_indices)
        stats.append(
            BlockStat(
                chrom=chrom,
                start_pos=block_positions[0],
                end_pos=block_positions[-1],
                n_snps=len(block_indices),
                Me=0.0,
                largest_gap_bp=largest_gap,
            )
        )
        start = max(end, start + 1)

    return blocks, stats, len(entries)


def _prune_redundant_snps(
    matrix: np.ndarray,
    indices: List[int],
    threshold: Optional[float],
) -> Tuple[np.ndarray, List[int]]:
    if threshold is None or matrix.shape[0] <= 1:
        return matrix, indices

    keep_mask = np.ones(matrix.shape[0], dtype=bool)
    for i in range(matrix.shape[0]):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, matrix.shape[0]):
            if not keep_mask[j]:
                continue
            if abs(matrix[i, j]) >= threshold:
                keep_mask[j] = False

    if keep_mask.all():
        return matrix, indices

    retained_indices = [idx for idx, keep in zip(indices, keep_mask) if keep]
    pruned_matrix = matrix[np.ix_(keep_mask, keep_mask)]
    return pruned_matrix, retained_indices


def _process_chromosome(
    chrom: str,
    ld_source: LDSparseMatrixProtocol,
    *,
    max_window_bp: int,
    corr_cutoff: float,
    gap_snp_limit: int,
    span_bp_limit: int,
    prune_redundant_threshold: Optional[float],
    polynomial_coeffs: Sequence[float],
) -> ChromosomeResult:
    converted = False
    convert_method = getattr(ld_source, "convert_to_pvalue_coefficients", None)
    if convert_method is not None:
        try:
            convert_method(polynomial_coeffs)
            converted = True
        except NotImplementedError:
            converted = False
        except Exception:
            converted = False

    blocks, stats, total_indices = _construct_blocks(
        chrom,
        ld_source,
        max_window_bp=max_window_bp,
        corr_cutoff=corr_cutoff,
        gap_snp_limit=gap_snp_limit,
        span_bp_limit=span_bp_limit,
    )

    Me_total = 0.0

    for block_indices, block_stat in zip(blocks, stats):
        ld_matrix = ld_source.sub_dense_matrix(block_indices)

        if ld_matrix.size == 0 or ld_matrix.shape[0] != ld_matrix.shape[1]:
            raise ValueError("LD matrix must be non-empty and square")

        if not converted:
            _apply_polynomial_transform(ld_matrix, polynomial_coeffs)

        ld_matrix, block_indices = _prune_redundant_snps(
            ld_matrix, list(block_indices), prune_redundant_threshold
        )

        ld_matrix = (ld_matrix + ld_matrix.T) / 2.0
        np.clip(ld_matrix, 0.0, 1.0, out=ld_matrix)
        np.fill_diagonal(ld_matrix, 1.0)

        try:
            eigenvalues = np.linalg.eigvalsh(ld_matrix)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Eigen decomposition failed for {chrom}") from exc

        capped = np.maximum(eigenvalues - 1.0, 0.0)
        Me_block = ld_matrix.shape[0] - float(np.sum(capped))
        if Me_block < 0:
            Me_block = 0.0

        Me_total += Me_block
        block_stat.n_snps = len(block_indices)
        block_stat.Me = Me_block

    covered = sum(len(block) for block in blocks)
    missing_ld_snps = max(total_indices - covered, 0)
    Me_total += float(missing_ld_snps)
    n_snps = total_indices

    monomorphic_dropped = getattr(ld_source, "monomorphic_dropped", 0)

    result = ChromosomeResult(
        Me=Me_total,
        n_snps=n_snps,
        block_stats=stats,
        missing_ld_snps=missing_ld_snps,
        monomorphic_dropped=monomorphic_dropped,
    )

    ld_source.release_ld_data()

    return result


def estimate_effective_tests(
    ld_sources: Union[
        LDSparseMatrixProtocol,
        Mapping[str, LDSparseMatrixProtocol],
        Iterable[Tuple[str, LDSparseMatrixProtocol]],
    ],
    *,
    max_window_bp: int = 3_000_000,
    corr_cutoff: float = 0.7,
    gap_snp_limit: int = 500,
    span_bp_limit: int = 3_000_000,
    prune_redundant_threshold: Optional[float] = 0.9988,
    polynomial_coeffs: Optional[Sequence[float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    polynomial_coeffs = (
        tuple(polynomial_coeffs) if polynomial_coeffs is not None else DEFAULT_POLYNOMIAL_COEFFS
    )

    sources = _normalise_ld_sources(ld_sources, metadata)
    if not sources:
        return {
            "Me": 0.0,
            "per_chromosome": {},
            "block_stats": [],
            "total_snps": 0,
            "parameters": {
                "max_window_bp": max_window_bp,
                "corr_cutoff": corr_cutoff,
                "gap_snp_limit": gap_snp_limit,
                "span_bp_limit": span_bp_limit,
                "prune_redundant_threshold": prune_redundant_threshold,
                "polynomial_coeffs": polynomial_coeffs,
            },
            "metadata": dict(metadata) if metadata else {},
        }

    per_chromosome: MutableMapping[str, ChromosomeResult] = {}

    if executor is None:
        for chrom, source in sources.items():
            per_chromosome[chrom] = _process_chromosome(
                chrom,
                source,
                max_window_bp=max_window_bp,
                corr_cutoff=corr_cutoff,
                gap_snp_limit=gap_snp_limit,
                span_bp_limit=span_bp_limit,
                prune_redundant_threshold=prune_redundant_threshold,
                polynomial_coeffs=polynomial_coeffs,
            )
    else:
        futures: Dict[Future, str] = {}
        for chrom, source in sources.items():
            future = executor.submit(
                _process_chromosome,
                chrom,
                source,
                max_window_bp=max_window_bp,
                corr_cutoff=corr_cutoff,
                gap_snp_limit=gap_snp_limit,
                span_bp_limit=span_bp_limit,
                prune_redundant_threshold=prune_redundant_threshold,
                polynomial_coeffs=polynomial_coeffs,
            )
            futures[future] = chrom

        for future, chrom in futures.items():
            per_chromosome[chrom] = future.result()

    total_me = sum(result.Me for result in per_chromosome.values())
    rounded_total_me = int(round(total_me))
    block_stats = [
        {
            "chrom": block.chrom,
            "start_pos": block.start_pos,
            "end_pos": block.end_pos,
            "n_snps": block.n_snps,
            "Me": block.Me,
            "largest_gap_bp": block.largest_gap_bp,
        }
        for result in per_chromosome.values()
        for block in result.block_stats
    ]
    total_snps = sum(result.n_snps + result.monomorphic_dropped for result in per_chromosome.values())
    total_monomorphic = sum(result.monomorphic_dropped for result in per_chromosome.values())

    return {
        "Me": rounded_total_me,
        "per_chromosome": {
            chrom: {
                "Me": int(round(result.Me)),
                "n_snps": result.n_snps,
                "block_stats": [
                    {
                        "chrom": block.chrom,
                        "start_pos": block.start_pos,
                        "end_pos": block.end_pos,
                        "n_snps": block.n_snps,
                        "Me": block.Me,
                        "largest_gap_bp": block.largest_gap_bp,
                    }
                    for block in result.block_stats
                ],
                "missing_ld_snps": result.missing_ld_snps,
                "monomorphic_dropped": result.monomorphic_dropped,
            }
            for chrom, result in per_chromosome.items()
        },
        "block_stats": block_stats,
        "total_snps": total_snps,
        "dropped_monomorphic_total": total_monomorphic,
        "parameters": {
            "max_window_bp": max_window_bp,
            "corr_cutoff": corr_cutoff,
            "gap_snp_limit": gap_snp_limit,
            "span_bp_limit": span_bp_limit,
            "prune_redundant_threshold": prune_redundant_threshold,
            "polynomial_coeffs": polynomial_coeffs,
        },
        "metadata": dict(metadata) if metadata else {},
    }


def make_ld_sources_from_genotype(
    genotype: GenotypeMatrix,
    genotype_map: GenotypeMap,
) -> Dict[str, LDSparseMatrixProtocol]:
    """Construct LD sources per chromosome from a genotype matrix + map."""
    map_df = genotype_map.to_dataframe()
    if "CHROM" not in map_df.columns or "POS" not in map_df.columns:
        raise ValueError("GenotypeMap must contain CHROM and POS columns")

    per_chrom_indices: Dict[str, List[int]] = {}
    per_chrom_positions: Dict[str, List[int]] = {}
    for idx, (chrom, pos) in enumerate(zip(map_df["CHROM"], map_df["POS"])):
        chrom_str = str(chrom)
        per_chrom_indices.setdefault(chrom_str, []).append(idx)
        per_chrom_positions.setdefault(chrom_str, []).append(int(pos))

    ld_sources: Dict[str, LDSparseMatrixProtocol] = {}
    for chrom, indices in per_chrom_indices.items():
        positions = per_chrom_positions[chrom]
        entries = list(zip(indices, positions))
        filtered_entries: List[Tuple[int, int]] = []
        dropped = 0

        if entries:
            chunk_size = 1024
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                chunk_indices = [idx for idx, _ in chunk]
                chunk_positions = [pos for _, pos in chunk]
                columns = genotype.get_columns_imputed(chunk_indices)
                variances = np.var(columns, axis=0)
                keep_mask = np.asarray(variances > 1e-12)
                if keep_mask.ndim == 0:
                    keep_mask = np.asarray([bool(keep_mask)])
                for (entry, keep) in zip(chunk, keep_mask.tolist()):
                    if keep:
                        filtered_entries.append(entry)
                    else:
                        dropped += 1
                if len(chunk) > len(keep_mask):
                    dropped += len(chunk) - len(keep_mask)

        if filtered_entries:
            filtered_indices = [int(idx) for idx, _ in filtered_entries]
            filtered_positions = [int(pos) for _, pos in filtered_entries]
        else:
            filtered_indices = []
            filtered_positions = []

        ld_sources[chrom] = _GenotypeLDSource(
            genotype,
            filtered_indices,
            filtered_positions,
            monomorphic_dropped=dropped,
        )

    return ld_sources


def estimate_effective_tests_from_genotype(
    genotype: GenotypeMatrix,
    genotype_map: GenotypeMap,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper combining LD construction and Me estimation."""
    ld_sources = make_ld_sources_from_genotype(genotype, genotype_map)
    metadata = kwargs.pop("metadata", None)
    return estimate_effective_tests(
        ld_sources,
        metadata=metadata,
        **kwargs,
    )
