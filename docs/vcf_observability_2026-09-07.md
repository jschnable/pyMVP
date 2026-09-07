# Portable VCF stage measurements and cross-system baseline

## Scope

Added opt-in measurements, **not performance tuning**. Parsing, QC, imputation,
tile size (8,192 markers), bulk batch limits, native thread settings, cache format,
and output layout are unchanged. Normal library calls do not enable timing.

`panicle/data/vcf_metrics.py` provides context-local timing/counters. Storage has
small hooks at transpose, output assignment, imputation, and flush boundaries.
`scripts/vcf_observability.py` scopes/restores benchmark-only wrappers around
larger stages. It does not enable interpreter-wide profiling. The harness is
single-process/single-benchmark oriented; do not patch shared application calls
concurrently. General fallback wrappers may run per record, so overhead needs
checking separately on fallback-heavy workloads.

`scripts/benchmark_vcf.py --stages` records inclusive/exclusive wall times,
call counts, logical storage bytes, CPU/fault/I/O resource snapshots, normalized
RSS where supported, final cache file sizes/allocation, and runtime environment.
Measurements exclude output hashing. Hashing remains bounded-memory.

## Reading the measurements

| Stage | Meaning |
| --- | --- |
| `load_total` | Loader call, including QC and cache creation |
| `builtin_stream` exclusive | Parsing/decoding/QC and residual buffered I/O, excluding measured child stages |
| `gzip_read_decompress` | Gzip reader work including underlying reads; not exclusively zlib CPU |
| `spool_append` / `spool_growth` | Appending temporary columns / extending and remapping storage |
| `spool_flush` | Explicit temporary-map flushes, including those nested in growth/finalization |
| `finalization` | Complete matrix finalization, including its child stages |
| `transpose_tiles` | Marker-major to sample-major tile copies |
| `output_tile_writes` | Assigning tiles into the final matrix; may include memory/page-fault/writeback costs |
| `cache_flush` | Explicit final mapped-cache flush; earlier writeback can occur during assignment |
| `tile_imputation` | General/cyvcf2 imputation when needed |
| `map_serialization` | Encoding and saving marker-map cache |

Inclusive times **overlap**; do not sum them as independent costs. Exclusive
times partition `load_total`, including parent residuals. A short final flush
does not prove cache writes were cheap: much of that work can occur earlier.

RSS is normalized from macOS bytes and Linux KiB; unknown units/platforms yield
null rather than guessed values. RSS is a process-wide high-water mark and can
include warmup/hashing, not a resettable per-run peak. CPU and fault counters
are snapshots, allowing deltas over the loader interval. Fault definitions and
raw block counters should not be compared blindly across operating systems.

Linux `/proc/self/io` counters are process-attributed I/O, not complete device
traffic. The Mac reported zero block counters here; physical I/O is **unavailable**
from this interface, not zero. Logical byte counts describe algorithmic writes,
not physical writes. Spool-plus-output capacity is not peak allocated disk usage:
it includes sparse capacity and excludes input, metadata, and preexisting caches.
Final cache logical/allocated sizes are recorded separately.

## Hosts and method

| | Mac | beadledesktop |
| --- | --- | --- |
| Platform | macOS / ARM64 | Linux / x86-64, i9-12900K |
| RAM | 32 GiB | 125.6 GiB |
| Logical CPUs / native Numba threads | 10 / 10 | 24 / 24 |
| Python | 3.14.7 | 3.12.3 |
| NumPy / pandas / Numba | 2.5.1 / 3.0.5 / 0.67.0 | 1.26.4 / 2.1.4 / 0.62.1 |
| Scratch | local APFS volume | NVMe `/dev/nvme0n1p2` |

The same code archive and input SHA-256 values were verified on both hosts.
Linux uses an isolated copy of the source on NVMe, not the production data
volume. The Mac uses a symlink to the already verified scratch input, with new
sidecars beside the symlink. Spools also use each benchmark's scratch directory.
No global packages or runtime settings were changed.

The full maize v2 panel contains **12,435,165 markers × 1,461 individuals**.
QC matches its provenance: include indels, split multiallelic, drop monomorphic,
`max_missing=1`, `min_maf=0`. One full instrumented load was run per host, without
a full-workload warmup. First-use/JIT effects are included; imports and hashes
are excluded. Checksumming read the compressed input beforehand. These are not
controlled cold-storage tests or repeated full-scale medians.

The smaller case uses the identical first 1M records, recompressed as gzip level 1.
Each instrumentation-off/on condition has one warmup and three measured loads in
a separate process. Conditions ran sequentially per host, after setup, transfers
and tests; measurements on different hosts could run concurrently. Off precedes
on, so this is not a randomized experiment. An earlier Mac pilot overlapped input
transfer; it is preserved in raw data but excluded from the headline results.

## Full-panel results

Seconds, except RSS:

| Measurement | Mac | Linux |
| --- | ---: | ---: |
| Total load and cache creation | 312.14 | 230.71 |
| Gzip reading/decompression | 30.59 | 79.83 |
| Stream residual (exclusive) | 25.95 | 71.17 |
| Finalization, inclusive | 229.95 | 42.44 |
| — transpose tile copies | 23.31 | 26.24 |
| — output tile assignments | 205.73 | 12.03 |
| — explicit final cache flush | 0.19 | 1.58 |
| Map serialization | 9.92 | 17.69 |
| Peak process RSS, decimal GB | 16.02 | 39.24 |

**The expensive part on the Mac is output assignment, not transpose computation.**
Its much larger assignment time and kernel/fault activity are consistent with
memory-mapped page/writeback pressure. They do not isolate RAM as the sole cause:
OS, filesystem, CPU, dependency versions, native thread counts, and system state
also differ. More resident file pages on the larger Linux machine do not indicate
a memory leak or a larger logical matrix.

Both hosts append **18.168 GB** of logical genotype bytes to the spool and write
the same amount to the final matrix. The logical spool capacity high-water mark
is **24.512 GB**, with spool-plus-output capacity **42.679 GB**. Final genotype
and map files are approximately 18.168 GB and 0.744 GB. Linux reports approximately
37.11 GB of process-attributed writes over the load; its compressed input reads
were served from filesystem cache (`read_bytes` delta zero). These figures do
not justify assuming identical device traffic on the Mac.

The platforms exhibit different bottleneck rankings. This baseline does **not**
establish a hardware-only speed ratio, a universally best tile size, or that a
Mac-specific tuning change would help Linux.

## Smaller-case baseline and measurement overhead

Median seconds over three timed loads:

| Host | Timing off | Timing on | Difference |
| --- | ---: | ---: | ---: |
| Mac | 6.821 | 6.787 | −0.5% |
| Linux | 18.511 | 18.706 | +1.1% |

The slight Mac decrease is noise, not a speedup from timing. The Linux on-run
range is 18.58–19.41 seconds. These checks suggest low overhead for this GT-only
workload, not a universal zero-overhead guarantee. Full instrumented totals
remain diagnostic baselines rather than precise uninstrumented timing estimates.

## Validation and next decision

- Full-panel genotype, sample-ID, and map hashes match **across hosts** and the
  previously measured production output.
- All complete subset hashes match with instrumentation on/off across both hosts.
- Mac full suite: **419 passed, 1 existing skip**. Linux targeted instrumentation,
  benchmark, and VCF streaming tests: **32 passed**.
- Tests cover timing nesting, exception cleanup, restoration of scoped wrappers,
  RSS units, logical-byte accounting, and instrumented/uninstrumented output parity.

Next investigate a **memory-bounded, more sequential output-write strategy** and
stream reading/decompression as separate candidates. Accept changes only after
testing both hosts and both dataset sizes; retain conservative defaults and
bounded memory. Do not tune a fixed tile size solely to this Mac or infer that
transpose arithmetic is the main problem. More environments, including Windows
and network storage, remain untested.

## Usage and retained artifacts

```sh
PYTHONPATH=. TMPDIR=/path/to/scratch python scripts/benchmark_vcf.py --input /path/to/scratch/panel.vcf.gz --max-missing 1 --min-maf 0 --skip-warmup --repeats 1 --stages --output /path/to/scratch/stages.json
```

Use an isolated scratch copy or symlink: force-recache replaces sidecars beside
the supplied path. Drop `--stages` for an uninstrumented run. Omitting
`--skip-warmup` adds a full load before the measured repetitions.

Scratch artifacts are retained at `/private/tmp/panicle-cross-platform.qU2puv`
on the Mac and `james@beadledesktop:/tmp/panicle-cross-platform.qkSnyL` on Linux.
Production VCFs and caches were untouched. Temporary data/caches are not added
to git. Raw measurements, checksum identities and environment records are in
`benchmarks/vcf_cross_platform_2026-09-07.json`.
