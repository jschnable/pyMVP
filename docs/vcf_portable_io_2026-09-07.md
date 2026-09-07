# Portable VCF I/O experiments (2026-09-07)

## Decision

Use buffered writes for all direct VCF cache construction. The mapped-output
writer and `PANICLE_VCF_CACHE_WRITER` option have been removed. We accept the
small-file overhead in exchange for shorter multi-minute loads and one less
implementation to maintain. Small/uncached loads retain their existing RAM path.
No CPU-, host-, or RAM-dependent selection is introduced.

Standard-library gzip remains the default; the alternative decoder is opt-in:

```sh
# Optional gzip decoder; benchmark before enabling:
pip install 'panicle[vcf-fast]'
PANICLE_VCF_GZIP_BACKEND=isal python your_analysis.py
```

`PANICLE_VCF_GZIP_BACKEND` accepts `stdlib` (default), `isal`, or `auto`.
`auto` tries ISA-L and falls back on import/unavailable-library errors; it does
**not** benchmark the readers. Installing the extra alone changes no defaults.
An explicit unavailable `isal` request raises an installation error. Corrupt or
truncated input is never silently retried using another reader. These controls
affect fresh built-in VCF loads, not cache hits or cyvcf2's own decompression.

## Implementation

The buffered writer retains the 8,192-marker transpose tiles and accumulates
them in one reusable, approximately 64 MiB sample-major slab (at least one
column). It then seeks/writes each sample's slab into the existing C-order NPY
layout. This groups scattered writes without mapping the final output during
construction. It is not a fully sequential-file algorithm, and the slab budget
is a memory cap, not a claimed optimum. No worker threads or OS-specific I/O
primitives are introduced. The existing marker spool remains mapped.

Short writes are completed, the output is synced before publication, and
publication still uses a temporary file plus replacement. Failed writes fall
back to the existing RAM path. Returned large matrices remain writable
copy-on-write mappings; modifying them does not change the reusable cache.
Cache version, ordering, filtering, missing-value behavior, and imputation
are unchanged. As before, RAM fallback can itself exhaust memory on huge loads.

The optional reader uses the gzip-compatible interface from
[python-isal](https://github.com/pycompression/python-isal), tested at 1.8.0.
It uses no subprocesses or decompression worker threads. Unsupported package
platforms can continue to use the standard library without installing the extra.

## Measurement method

Inputs, hashes, host differences, and previous full-panel baselines are recorded
in [the instrumentation report](vcf_observability_2026-09-07.md). The full maize
panel has 12,435,165 markers and 1,461 samples; the subset contains its first
1,000,000 markers. Each load forces an isolated cache rebuild, with max_missing=1
and min_maf=0. Full-panel runs use one timed load without a full workload warmup;
subset comparisons use a warmup followed by three timed loads. Hashing is outside
the load timer. Benchmarks do not overlap on a host; they are not controlled
cold-storage tests. Original production data/caches are untouched.

The new gzip stage measures the public buffered-read boundary for both readers,
including copies; the earlier baseline measured a private standard-library read
method. Those stage times are not exactly interchangeable. Whole-load times
remain comparable, subject to run-to-run variation. Buffered `tile_buffer_copies`
and `cache_file_writes` replace mapped `output_tile_writes`; inclusive finalization
includes all these operations and the transpose.

## Initial full-panel measurements (before the default switch)

Seconds, one run per configuration; standard gzip in both cases:

| Host | Prior mapped baseline | Buffered | Prior finalization | Buffered finalization |
| --- | ---: | ---: | ---: | ---: |
| macOS / ARM64, 32 GiB | 312.14 | 180.53 | 229.95 | 94.96 |
| Linux / x86-64, 125 GiB | 230.71 | 182.20 | 42.44 | 30.77 |

Full genotype, sample-ID, and map hashes match the baseline on both hosts.
The buffered run is encouraging on both systems, but the earlier baseline was
not an interleaved/repeated control. In particular, the Linux improvement in
total time is larger than its finalization improvement: not all of it can be
attributed to writes. No universal speedup or architecture-only explanation is
claimed. Windows, network storage, and other RAM limits remain untested.

On Linux, keeping buffered writes enabled and selecting ISA-L reduced the full
load further, from **182.20 s to 134.62 s** (one run each), with identical full
output hashes. This supports offering the optional decoder, not changing the
default for all systems.

On the Mac subset, mapped baseline median was 6.957 s, buffered+stdlib 8.334 s,
and buffered+ISA-L 20.356 s. This small-workload regression and accelerator
regression originally motivated keeping both alternatives opt-in. The subsequent
decision above accepts the write overhead but keeps ISA-L optional. The buffered+stdlib
range was 8.202–12.564 s, showing material run-to-run variation too.

Subset medians (seconds, three timed loads after warmup):

| Configuration | Mac | Linux |
| --- | ---: | ---: |
| Fresh before snapshot, mapped+stdlib | 6.957 | 17.345 |
| Then-default, mapped+stdlib | 6.915 | 16.143 |
| Buffered+stdlib | 8.334 | 16.455 |
| Mapped+ISA-L | Not run | 12.233 |
| Buffered+ISA-L | 20.356 | Not run |

The then-default path is essentially unchanged on the Mac. Buffered writes alone
regress both subset comparisons against that default (substantially on
Mac, slightly on Linux). ISA-L helps Linux's subset, but it is not universally
faster: the Mac comparison holds buffered writes constant and gets worse.
Following that regression, a full-panel Mac ISA-L run was not pursued.

## Correctness before the default switch

Mac full suite: **437 passed, 1 existing skip**, both with default settings and
with `buffered`+`isal` selected. Each gzip reader matches **1,024 baseline
compatibility cases**, including 32 expected error outcomes. Separate direct
writer tests force tiny slabs to exercise selection, partial final slabs,
imputation, short writes, write/sync failures, and copy-on-write isolation.
Reader tests cover long lines, CRLF, concatenated members, BGZF including its
EOF block, truncation, CRC corruption, missing dependencies, and close behavior.
Linux's final full suite with `buffered`+`isal` also passed: **437 passed, 1 skip**.

## Artifacts

After switching to buffered writes unconditionally, the Mac full suite passed
**435 tests with 1 existing skip**, and all **1,024 baseline compatibility
cases** matched again. The two fewer tests reflect removal of writer selection
and the duplicate mapped-writer imputation case. Failure handling, buffered
imputation, and copy-on-write tests now run without any writer opt-in.

Scratch data and code snapshots are retained at
`/private/tmp/panicle-portable-io.T0yXqL` on the Mac and
`james@beadledesktop:/tmp/panicle-portable-io.Px3ynj` on Linux. The `before`
snapshot includes the prior uncommitted instrumentation changes. Initial
candidate benchmarks used unconditional buffered writes; the archived `final`
snapshot exposed that path via an explicit option. These snapshots and raw
measurements predate the subsequent removal of the mapped-output path and option.
Raw timings, environment records, output hashes, code-snapshot hashes, and test
summaries are retained in
[the benchmark artifact](benchmarks/vcf_portable_io_2026-09-07.json).
