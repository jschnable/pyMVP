# VCF loader: performance-preserving refactor

This documents the first refactor and its measurements. The subsequent
[streaming/FORMAT/cache optimization](vcf_streaming_changes_2026-09-06.md)
supersedes its fallback and matrix-finalization behavior.

## What changed

- `load_genotype_vcf.py` now separates backend selection, builtin-record
  decoding, cyvcf2-record decoding, and public cache/output orchestration.
- `vcf_records.py` owns shared **scalar** QC and column-oriented map accumulation.
  The duplicated cyvcf2/general `consider_variant` implementations are gone.
  General decoding no longer allocates a dictionary for every retained marker.
- `vcf_storage.py` owns marker-major temporary matrix storage. The old writer
  import remains available from `load_genotype_vcf` for compatibility.
- **The bulk fast path remains separate and vectorized.** It does not call the
  scalar filter or allocate a Python record object for each genotype/marker.
- Simple-GT decoding now writes contiguous marker-major batches. Appending a
  batch no longer performs a strided transpose copy. The returned genotype
  matrix is still C-contiguous, sample-major int8, including small files that
  never need a temporary writer. The cache format is unchanged.
- Reader and temporary-file cleanup now covers failed general decoding and
  failures while processing the final bulk batch.

The numerical dosage, filtering, imputation, and backend-selection policies are
unchanged. Builtin GT/DS handling and cyvcf2's existing GT/ploidy handling remain
distinct; this refactor does not assert that backends are equivalent for all VCFs.

## Validation

- Full suite: **387 passed, 1 existing skip**.
- 15 new cases cover scalar QC boundaries, bulk versus general results, both
  compiled and NumPy decoding, single/multiple batches, public memory layout,
  late fallback, and resource cleanup on success and failure.
- `scripts/validate_vcf_refactor.py`: **672 cases match baseline commit
  `ccb3e08` exactly**, including 28 pre-existing error outcomes. Cases combine
  plain/gzip input, both backends, both map return modes, seven filter settings,
  simple GT, GT:DP, GT:DS, DS-only, reordered FORMAT, multiallelic, missing,
  monomorphic, indel, haploid, polyploid, and empty inputs.
- Benchmark runs independently compare SHA-256 of genotypes, sample IDs and map
  tables. Every before/after comparison passed.

## Benchmark method and limitations

`scripts/benchmark_vcf.py` generates deterministic synthetic genotypes with
1,500 individuals and approximately 1% missing calls. Every timed load forces
cache rebuilding and includes parsing, QC, imputation, map construction,
temporary writes, final matrix conversion, and cache output. Filters are
`drop_monomorphic=True`, `max_missing=0.2`, `min_maf=0.01`.

Runs use the same input and Python environment, sequentially, with one warmup
and three timed repetitions per process. Imports/JIT warmup, input generation,
and checksum calculation are outside timing. Gzip fixtures use compression
level 1. This is **warm filesystem cache**, not sustained cold-disk throughput.
The cached-load timer measures opening lazy cached data, not reading every byte.

Environment: macOS 26.2 ARM64, Python 3.14.7, NumPy 2.5.1, pandas 3.0.5,
Numba 0.67.0 (10 default worker threads), cyvcf2 0.34.0.

The largest fixture is **1,000,000 markers × 1,500 individuals**, about 6.03 GB
of plain VCF and 1.5 GB of returned genotype data. No actual 10-million-marker
production file was benchmarked. At the requested 10M × 1,500 scale the genotype
array alone is 15 GB; temporary storage, maps, parsing buffers, and copies are
additional. Do not extrapolate these timings linearly across memory pressure or
different storage/compression/FORMAT layouts.

## Measured results

Median seconds, including cache creation:

| Input / backend | Markers | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Plain simple GT / builtin, repeat run | 1,000,000 | 13.104 | 8.357 | 1.57× |
| Gzipped simple GT / builtin | 100,000 | 1.289 | 0.829 | 1.55× |
| Plain GT:DP / general builtin | 10,000 | 1.876 | 1.863 | 1.01× |
| Plain simple GT / cyvcf2 | 100,000 | 8.099 | 7.841 | 1.03× |

The first one-million-marker baseline was variable (13.43–23.54 seconds), so
the headline uses a second, stable three-run comparison. All raw timings,
including initial 100k-marker experiments and the structural-refactor-only run,
are retained in `benchmarks/vcf_refactor_2026-09-06.json`; none are hidden.
The structural-only 100k run was slightly slower (0.945 → 1.000 seconds) before
the batch-layout optimization; this was not accepted as the final result.

For the stable one-million-marker comparison, process peak RSS was about
4.20 GB before and 4.25 GB after (macOS reports bytes). This is a throughput
improvement, **not a claim of lower peak memory**. The general-parser map-list
change reduces dictionary allocations but does not remove final-matrix memory.

Profiling the initial 100k case identified the strided `append_block` copy as a
hotspot. Its profiled cumulative time fell from about 0.35 to 0.057 seconds after
changing batch layout. Overall profiling runs themselves were noisy; the
repeated end-to-end measurements above are the performance evidence.

Reproduce with a baseline checkout and an environment containing the test dependencies:

```sh
PYTHONPATH=. python scripts/benchmark_vcf.py --generate --samples 1500 --markers 1000000 --input /tmp/panel.vcf
PYTHONPATH=/path/to/baseline python scripts/benchmark_vcf.py --input /tmp/panel.vcf --output /tmp/before.json
PYTHONPATH=. python scripts/benchmark_vcf.py --input /tmp/panel.vcf --output /tmp/after.json --compare /tmp/before.json
PYTHONPATH=/path/to/baseline python scripts/validate_vcf_refactor.py --output /tmp/cases-before.json
PYTHONPATH=. python scripts/validate_vcf_refactor.py --output /tmp/cases-after.json --compare /tmp/cases-before.json
PYTHONPATH=. MPLBACKEND=Agg python -m pytest tests/ -q
```

## Next performance opportunities (not implemented)

1. **Broaden batched decoding to GT with extra FORMAT fields.** The existing
   bulk path requires exactly `FORMAT=GT`. GT:DP/GT:AD:DP currently falls back to
   the general parser. This is a substantial candidate for real production VCFs,
   but must preserve GT/DS precedence, missingness, and variable-width fields.
2. **Avoid restarting the whole file after a late unsupported record.** The
   current conservative bulk attempt discards its partial output and reopens
   the file. A streaming hybrid needs explicit state for original missing-call
   counts and imputation, not just concatenation of already-imputed prefixes.
3. **Reduce final full-matrix copying and temporary/cache writes.** Consider
   tiled transposition directly into the final cache rather than first producing
   a full RAM matrix and then saving it. Preserve sample-major access efficiency
   for downstream PCA/kinship/GWAS; changing cache layout is not automatically a
   net win.
4. **Profile decompression on representative compressed production input.** Do
   not switch to cyvcf2 automatically merely because input is large: simple-GT
   bulk decoding was much faster on this fixture, and backend semantics differ.

No new dependencies were introduced. Changes are local until explicitly committed.
