# Production maize VCF benchmark

## Input and location

Copied `Maize_Marcin1461_filtered_v2.vcf.gz` from
`james@beadledesktop:/home/james/data/Common_Data/Markers/Maize/` to:

```
/private/tmp/panicle-maize-v2.g7fPxq/Maize_Marcin1461_filtered_v2.vcf.gz
```

Source and destination SHA-256 both equal:

```
a55de26934e8d3e6eac996172e47829b2255dc7a05187023d113d7d805155e6e
```

The compressed file is 3,647,063,164 bytes and contains **12,435,165 markers ×
1,461 individuals**. All records have GT-only FORMAT. Remote VCFs and existing
PANICLE caches were not modified. Local input, caches, one-million-marker
subset, profiles, and raw results occupy approximately 23 GiB in scratch storage;
they were retained, not added to git. `/private/tmp` is temporary storage.

## Full-scale result

The current loader (commit `61be38c`) loaded the entire panel and built its cache
in **302.676 seconds (5 minutes 3 seconds)**. Output is sample-major int8 with
shape `(1461, 12435165)`. Peak process RSS was **15.45 GB / 14.39 GiB** on this
32 GiB Mac. System memory compression increased, but the system swap-out counter
was unchanged across the run. File-backed mappings can exceed resident memory;
RSS alone does not capture all memory pressure.

Filters match the source panel's documented cache settings: split multiallelic
sites, include indels, drop monomorphic markers, `max_missing=1`, `min_maf=0`.
All input markers were retained. Cached opening took 0.171 seconds; that measures
opening lazy mappings and metadata, **not reading the entire matrix**.

This was one full load, without a full-workload warmup. It includes first-load/JIT
effects and all loader/cache work, but excludes imports and output hashing.
The compressed input had just been read for checksum verification, so this is
not a controlled cold-disk measurement. No full baseline run was attempted on
the 32 GiB machine because the older loader requires a full-size heap matrix
in addition to the spool and metadata.

The source provenance's historical 37.5-minute cache build used a different
version/environment. It is **not** a controlled baseline or evidence of a
specific full-scale speedup.

## Controlled real-data comparison

The first 1,000,000 records, retaining all 1,461 samples, were extracted and
recompressed with gzip level 1. Both implementations consumed this identical
495,857,206-byte subset with identical QC settings. Each ran in a separate process,
sequentially, with one warmup and three timed force-recache loads. Generation,
hashing, and a subsequent profiling pass were outside the reported load times.

| Implementation | Timed loads (seconds) | Median | Peak RSS (GB) |
| --- | --- | ---: | ---: |
| Before streaming/cache optimization | 10.362, 10.201, 10.233 | 10.233 | 4.079 |
| Current | 6.971, 6.761, 6.775 | 6.775 | 4.142 |

The current loader is **1.51× faster (33.8% less time)** on this real subset.
Genotype, sample-ID, and map hashes match exactly across implementations and all
repetitions. Peak RSS includes the separate profiling pass. This prefix is not
a random sample of the full panel, and its recompression differs from the source.

The baseline is `/tmp/panicle-vcf-stream-before.Lg5Sm0`, the saved preceding
refactor used in the earlier streaming benchmark—not an older release or git
HEAD. Both processes use the same Python 3.14.7 environment, NumPy 2.5.1,
pandas 3.0.5, and Numba 0.67.0 on macOS ARM64.

## Correctness and test tooling

- Transfer verified by matching source/destination SHA-256.
- Full matrix has the expected sample/marker counts and int8 dtype.
- An independent streaming validator checks all record counts and FORMAT layouts,
  verifies sample order, and compares every sample's dosage at 126 markers spread
  through the file, including the first and last: **184,086 matching genotypes**.
  This is sampled dosage validation, not exhaustive full-baseline equivalence.
- Exact complete genotype/map/ID hashes match on the one-million-marker subset.
- Benchmark hashing now uses bounded chunks, preserving the existing digest
  definition without building a huge CSV string or large genotype byte string.
- Added configurable QC and `--skip-warmup` to the benchmark, plus the independent
  `scripts/validate_vcf_panel.py` utility for pre-imputed biallelic GT-only panels.
- Test suite: **412 passed, 1 existing skip**.

## Remaining hotspots

The separate one-million-marker cProfile runs show:

| Cumulative time | Before | Current |
| --- | ---: | ---: |
| Matrix finalization | 4.718 s | 1.408 s |
| Gzip reading/decompression | 2.648 s | 2.644 s |
| Map-cache serialization | 1.296 s | 1.273 s |

Zlib decompression itself accounts for about 2.08 seconds in both profiles.
Profiled totals are 13.21 and 10.18 seconds; profiler overhead makes these
different from the unprofiled benchmark times. These rows are cumulative and
should not be interpreted as exclusive CPU or physical-disk measurements.

The full run spent substantial observable time constructing the temporary final
cache under memory pressure. Exact full-scale stage timings were not instrumented,
so the subset profile must not be extrapolated linearly. The next useful work is
to instrument full-scale finalization and investigate its disk access pattern,
alongside faster decompression and more efficient marker-map serialization.

Raw results and profile summaries are recorded in
`benchmarks/maize_v2_production_2026-09-06.json`.

## Reproduce

```sh
PYTHONPATH=. python scripts/benchmark_vcf.py --input /path/to/Maize_Marcin1461_filtered_v2.vcf.gz --skip-warmup --repeats 1 --max-missing 1 --min-maf 0 --output /tmp/full-current.json
PYTHONPATH=. python scripts/validate_vcf_panel.py --input /path/to/Maize_Marcin1461_filtered_v2.vcf.gz --output /tmp/validation.json --prefix-output /tmp/maize-first-1m.vcf.gz
PYTHONPATH=/path/to/baseline python scripts/benchmark_vcf.py --input /tmp/maize-first-1m.vcf.gz --max-missing 1 --min-maf 0 --output /tmp/before.json
PYTHONPATH=. python scripts/benchmark_vcf.py --input /tmp/maize-first-1m.vcf.gz --max-missing 1 --min-maf 0 --output /tmp/after.json --compare /tmp/before.json
```

Run against a scratch copy: force-recache intentionally replaces caches beside
the supplied input. Do not point these commands at production cache paths.
