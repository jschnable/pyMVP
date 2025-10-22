# Harmonized GEC-Style Effective Test Count Specification

## 1. Purpose & Scope
- Provide a Python module that reproduces the effective number of tests (`Me`) calculation used by the published Java GEC tool while fitting into an existing genomic analysis package.
- Assume the host package already offers genotype/LD loaders (PLINK, VCF, HapMap LD, linkage, haplotype formats) and standard QC utilities (MAF filtering, missing data imputation, call-rate checks).
- Focus on transforming LD information into block-wise p-value correlation matrices, applying the Li & Ji eigenvalue capping on those matrices, and returning summary statistics to the caller.
- No command-line interface or on-disk reporting is required; callers receive structured results via Python return values.

## 2. Module Layout & Integration
- Suggested location: `gec_effective/tests.py` (or analogous module within existing package namespace).
- External dependencies: `numpy`, `scipy` (for eigen decomposition), `typing` utilities. Host package supplies loaders, sparse LD objects, and logging infrastructure.
- The module imports a shared polynomial mapping utility that mirrors the Java `convert2PValueCoefficient` behavior (coefficients embedded, optional override via configuration).

## 3. Public API (Python)
```python
from concurrent.futures import Executor
from typing import Sequence, Mapping, Optional, List, Dict, Any, Union, Iterable, Tuple
import numpy as np

def estimate_effective_tests(
    ld_sources: Union[
        "LDSparseMatrixProtocol",
        Mapping[str, "LDSparseMatrixProtocol"],
        Iterable[Tuple[str, "LDSparseMatrixProtocol"]],
    ],
    *,
    max_window_bp: int = 3000000,
    corr_cutoff: float = 0.7,
    gap_snp_limit: int = 500,
    span_bp_limit: int = 3000000,
    prune_redundant_threshold: float = 0.9988,
    polynomial_coeffs: Optional[Sequence[float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    """
    Return {
        "Me": float,
        "per_chromosome": Dict[str, ChromosomeResult],
        "block_stats": List[BlockStat],
        "total_snps": int,
        "parameters": {...},
        "metadata": metadata or {}
    }.
    """
```
- `ld_sources` accepts either a single LD container or a mapping/iterable of `(chrom, ld_source)` pairs. Each value must expose the Java `LDSparseMatrix`-style interface: `get_all_unique_indexes()`, `sub_dense_matrix(index_list)`, `release_ld_data()`, and (optionally) `convert_to_pvalue_coefficients(coeffs)`. Adapters translate the host package’s LD representations into this protocol.
- `polynomial_coeffs`: defaults to bundled 7-term array derived from the Java implementation; callers may override to experiment with refits.
- `executor`: optional concurrent executor used to process chromosomes in parallel. Implementations submit one task per chromosome when provided; in its absence, chromosomes are processed sequentially.
- Return object contains:
  - `Me`: total effective test count.
  - `per_chromosome`: mapping from chromosome identifier to `{"Me": float, "n_snps": int, "block_stats": [...]}`.
  - `block_stats`: list of dictionaries with keys `chrom`, `start_pos`, `end_pos`, `n_snps`, `Me`, `largest_gap_bp`.
  - `total_snps`: count of unique SNPs processed.
  - `parameters`: echoes the effective window, cutoff, pruning, and coefficient identifiers.
  - `metadata`: optional user-supplied information propagated unchanged.

## 4. Processing Pipeline
1. **Input Acquisition**
   - Caller constructs one LD container per chromosome using the host package loaders (PLINK, VCF, HapMap LD, linkage, haplotype). Missing data handling and MAF/call-rate filtering occur upstream.
   - `estimate_effective_tests` assembles the per-chromosome inputs either from the mapping/iterable supplied or by treating a single container as one chromosome (metadata must then specify its identifier).
   - When an `executor` is supplied, each chromosome’s work item (steps 2–6) is scheduled concurrently; results are merged after all futures complete.

2. **Polynomial Mapping**
   - For each chromosome, if its `ld_source` offers an in-place transformation (e.g., `convert_to_pvalue_coefficients`), invoke it with the coefficient vector before block construction.
   - Otherwise, after materializing each dense block (see step 4), transform every LD value `r` (Pearson correlation) using the Java polynomial:
     ```
     x = r * r
     x = (((((0.7723 * x - 1.5659) * x + 1.2010) * x - 0.2355) * x + 0.2184) * x + 0.6086) * x
     rho_p = x
     ```
     which corresponds to a sixth-degree polynomial in `r²` with coefficients `[0.7723, -1.5659, 1.2010, -0.2355, 0.2184, 0.6086, 0.0]`.
   - Clamp resulting correlations to `[0, 1]`; set diagonals to `1.0`.

3. **Block Construction (Java Heuristic)**
   - Within each chromosome, sort unique genomic positions supplied by the LD container.
   - Iterate with two pointers (`start`, `end`) expanding the current block while:
     - `positions[end] - positions[start] <= max_window_bp`.
     - Absolute correlation between adjacent SNPs (queried from `ld_source`) remains ≥ `corr_cutoff`.
     - Number of consecutive missing correlations does not exceed `gap_snp_limit`.
     - Physical span of the block does not exceed `span_bp_limit`.
   - When any condition fails or the sequence ends, finalize the current block and start a new one from the next unprocessed position.
   - Record observed largest inter-SNP gap inside the block for diagnostics.
   - Track positions that never appear in any LD pair within the chromosome; after block processing, treat each as an independent SNP contributing `1.0` to `Me` (mirrors the Java `missingCountHolder` adjustment).

4. **Dense Block Assembly & Redundancy Pruning**
   - Build an index list for the current block and call `ld_source.sub_dense_matrix(index_list)` to obtain the LD matrix (already transformed if step 2 applied).
   - If `prune_redundant_threshold` is not `None`, run the Java-style redundancy removal: iteratively drop SNPs whose absolute correlation with any earlier SNP is ≥ threshold.
   - Convert to double precision (`np.float64`) and enforce symmetry by averaging with its transpose.

5. **Effective Test Count per Block**
   - Perform eigen decomposition (`numpy.linalg.eigvalsh`).
   - Apply Li & Ji capping in place: `Me_block = block_size - Σ(max(0, λ_i - 1))`.
   - Accumulate `Me += Me_block`; append block-level diagnostics to `block_stats`.

6. **Cleanup**
   - After processing each block, call `ld_source.release_ld_data()` to allow upstream caching strategies to free memory (mirrors Java behavior).
   - If `ld_source` supports caching to disk, this method delegates to the host package implementation; no temp paths are managed here.

## 5. Configuration Defaults & Rationale
- Polynomial coefficients: fixed six-degree mapping sourced verbatim from the Java implementation (see §4.2). Unless explicitly overridden, every LD source must apply this transform for parity.
- `prune_redundant_threshold=0.9988`: the only hard-coded constant inside Java’s estimator; dropping nearly identical SNPs improves numerical stability.
- Window, cutoff, gap, and span limits depend on the upstream LD data source (mirroring Java call sites):
  - **HapMap LD file (whole genome):** `max_window_bp=1_000`, `corr_cutoff=0.1`, `gap_snp_limit=10`, `span_bp_limit=5_000_000`.
  - **HapMap LD file (SNP list workflow):** `max_window_bp=1_000`, `corr_cutoff=0.2`, `gap_snp_limit=1`, `span_bp_limit=5_000_000`.
  - **PLINK binary / linkage / HapMap haplotype / VCF (bulk and SNP list):** `max_window_bp=100`, `corr_cutoff=0.1`, `gap_snp_limit=1`, `span_bp_limit=1_000_000`.
- Implementations should surface a small registry (e.g., dictionary keyed by data source type) so callers can request Java-identical defaults while retaining manual overrides.

## 6. Error Handling & Logging
- Raise `ValueError` if the LD matrix for a block is empty or not square.
- Raise `RuntimeError` when eigen decomposition fails to converge; propagate the block context for debugging.
- Use the host package logger to emit debug messages (block boundaries, redundancy pruning counts, eigen issues); avoid direct `print`.
- Emit an informational message whenever missing LD positions cause additional independent contributions (see §4.3); expose the count in the returned `per_chromosome` payload.

## 7. Testing Strategy
- Unit tests using synthetic LD matrices that mimic Java outputs:
  - Identity matrix → `Me = m`.
  - Perfect LD block (all ones) → `Me ≈ 1`.
  - Compare block partitioning against stored fixtures from the Java tool on representative chromosomes.
  - Validate polynomial conversion by asserting transformed coefficients match Java’s `convert2PValueCoefficient`.
- Property tests ensuring `Me` is additive across concatenated independent blocks.

## 8. Parallelization Considerations
- Chromosome-level work items are embarrassingly parallel once each LD container is instantiated. When `executor` is provided, submit one task per chromosome and merge the results deterministically.
- Within a chromosome, block construction remains sequential along genomic positions, but eigen decompositions can still use thread pools provided by `numpy`/`scipy` or by delegating to a per-block worker.
- Avoid overwhelming shared storage: caller should match executor parallelism to available I/O bandwidth when LD matrices are loaded on demand. Provide configuration hooks so the host package can throttle concurrency.
