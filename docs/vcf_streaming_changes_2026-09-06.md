# VCF streaming, FORMAT decoding, and cache finalization

## Changes

1. **Bulk extra-FORMAT decoding.** A parallel compiled byte scanner extracts
   diploid GT from GT:DP, GT:AD:DP, DP:GT, and mixed layouts. Depth/other fields
   may have variable widths. It feeds the existing bulk dosage/QC kernels
   without allocating Python strings for individual sample calls. DS-containing,
   multiallelic, non-diploid, and unsupported calls retain general decoding.
   Without Numba, extra-FORMAT records use the general parser rather than a
   slow Python byte-scanning loop.
2. **One-pass fallback.** Unsupported records are decoded in the open stream;
   bulk decoding resumes afterward. A rejected batch replays only its buffered
   records, not the completed file prefix. Batches retain the existing 32,768
   marker limit; extra-FORMAT batches also flush at 128 MiB of sample text
   (a single unusually long record can exceed that bound). Sanity-check state,
   GT/DS precedence, marker order, and the old bulk/general floating-point QC
   boundary differences are preserved, even after late fallback.
3. **Tiled finalization and direct caches.** The marker-major spool is transposed
   in 8,192-marker tiles. Outputs of at least 64 MiB are built directly in a
   temporary C-order `.npy`, then published to the existing v2 cache path.
   General/htslib columns are imputed tile-by-tile, avoiding full-matrix boolean
   masks. The normal cache-save step does not copy the genotype again.

Fresh large loads return a **writable copy-on-write NumPy memmap**, retaining
sample-major shape, int8 dtype, and C contiguity. It is ndarray-compatible;
changes made by callers do not modify the disk cache. Small outputs remain heap
arrays. Existing cache hits remain read-only memmaps. Cache creation failures
fall back to RAM. Before publishing a rebuilt genotype, the old filter
fingerprint is invalidated; failed sidecar writes cannot leave it falsely valid.
The cache format/version and backend selection are unchanged.

This does not eliminate the marker-major spool or all metadata allocations.
During finalization both spool and cache coexist. At 10M markers × 1,500 samples,
each genotype representation is approximately 15 GB, before spool capacity
headroom and metadata. Mapped pages still count toward RSS; avoiding a heap
copy does **not** imply a proportional reduction in measured peak RSS.

## Validation

- Full suite: **408 passed, 1 existing skip**.
- **1,024 baseline comparisons match exactly**, including 32 pre-existing error
  outcomes. These cover both backends, plain/gzip inputs, both map modes, eight
  filter configurations, variable/reordered FORMAT fields, GT/DS, haploidy,
  polyploid error behavior, multiallelic records, empty data, and missing calls.
  A two-marker batch limit explicitly exercises fallback after completed batches.
- New regression tests check one input open, resumed bulk decoding, imputation,
  QC rounding, sanity-check state, selected-column tiled output, cache write
  failures/cleanup, stale-fingerprint invalidation, and copy-on-write isolation.
- Every benchmark compares exact genotype, sample-ID, and map SHA-256 hashes.

## Measurements

Comparison is against the immediately preceding, uncommitted VCF refactor
documented in `vcf_loader_changes_2026-09-06.md`, preserved at
`/tmp/panicle-vcf-stream-before.Lg5Sm0` during testing—not against older git HEAD.
All inputs have 1,500 individuals and approximately 1% missing calls. Timings
include QC, imputation, map creation, and cache output, with a warm filesystem,
one warmup and three measured repetitions. JIT/imports, generation, and hashing
are excluded. Runs are sequential, not concurrent with tests or generation.
Environment: macOS ARM64, Python 3.14.7, NumPy 2.5.1, pandas 3.0.5,
Numba 0.67.0 (10 default threads), cyvcf2 0.34.0.

| Input / backend | Markers | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| GT:DP / builtin | 100,000 | 18.339 | 0.696 | 26.36× |
| GT:DP / builtin | 10,000 | 1.818 | 0.078 | 23.37× |
| Simple GT, final GT:DS record / builtin | 100,000 | 3.155 | 0.546 | 5.77× |
| Simple GT / builtin | 1,000,000 | 8.422 | 4.842 | 1.74× |
| Gzipped simple GT / builtin, initial pair | 100,000 | 0.829 | 0.883 | 0.94× |
| Simple GT / cyvcf2 | 100,000 | 7.706 | 7.802 | 0.99× |

**Limitations and tradeoffs:** No actual 10M-marker production file was tested.
GT:DP benchmark values use a one-digit depth; more complex FORMAT payloads may
have different throughput. Gzip uses compression level 1. The initial gzip pair
was about 6.5% slower; later repetitions were unstable (before median 1.254 s,
after 0.858 s, with individual after runs up to 1.132 s). No gzip speedup is
claimed. A trial disabling direct caching and a subsequent profiled trial also
ran around 1.26–1.29 s during that variation; these are retained in the raw data,
not used to claim a gain. Direct caching remains enabled at 64 MiB.

Peak process RSS (decimal GB, including warmup) for the 1M simple-GT case was
**4.179 → 4.215 GB**, essentially unchanged. The 100k GT:DP case used
**0.764 → 1.055 GB**: compiled bulk buffers trade memory for speed. For cyvcf2,
tiled imputation reduced peak RSS from **0.870 → 0.602 GB**, with approximately
unchanged runtime. RSS includes mapped file pages and compiler/runtime memory.
These are not isolated heap-allocation measurements.

All timings, hashes, and RSS values, including gzip repeats and experiments, are
in `benchmarks/vcf_streaming_2026-09-06.json`. A small final metadata-prefix slicing
optimization was applied after the initial pairs; gzip repeat and cyvcf2 runs
include it. It does not change decoding or matrix finalization.

## Reproduction

```sh
PYTHONPATH=. python scripts/benchmark_vcf.py --generate --samples 1500 --markers 100000 --extra-format --input /tmp/gtdp.vcf
PYTHONPATH=. python scripts/benchmark_vcf.py --generate --samples 1500 --markers 100000 --late-general --input /tmp/late.vcf
PYTHONPATH=/path/to/baseline python scripts/benchmark_vcf.py --input /tmp/gtdp.vcf --output /tmp/before.json
PYTHONPATH=. python scripts/benchmark_vcf.py --input /tmp/gtdp.vcf --output /tmp/after.json --compare /tmp/before.json
PYTHONPATH=/path/to/baseline python scripts/validate_vcf_refactor.py --batch-markers 2 --output /tmp/cases-before.json
PYTHONPATH=. python scripts/validate_vcf_refactor.py --batch-markers 2 --output /tmp/cases-after.json --compare /tmp/cases-before.json
PYTHONPATH=. MPLBACKEND=Agg python -m pytest tests/ -q
```

The next performance target supported by profiling is compressed-input
decompression: zlib accounted for about 0.41 s of a 1.40 s profiled gzip load.
Reducing spool/cache disk traffic further would require a different storage
layout or stronger size/index information; neither is silently introduced here.
