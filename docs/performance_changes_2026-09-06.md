# Implemented performance improvements — 2026-09-06

Implemented items 2–5 from the performance review. The comparison uses an
independent checkout of baseline commit `da717f9` and the updated working tree,
with the same seeded inputs and Python environment. Measurements and numerical
checks are recorded by [benchmark_changes.py](../scripts/benchmark_changes.py);
individual timings are retained in [JSON](benchmarks/2026-09-06-implemented.json).

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| GLM | 0.386 s | 0.355 s | 1.09× |
| GLM with existing prefetch enabled | 0.256 s | 0.224 s | 1.14× |
| BLINK, all markers retained | 4.440 s | 2.219 s | 2.00× |
| BLINK, MAF threshold 0.49 | 1.162 s | 0.750 s | 1.55× |
| Top-level `PANICLE`, eight GLM traits together | 5.541 s | 0.855 s | 6.48× |
| Temporary matrix build/finalize, 100k markers | 0.149 s | 0.132 s | 1.13× |

Medians of three warm repetitions on macOS ARM64 with NumPy Accelerate. All
association measurements use 738 samples, one million complete int8 markers,
three covariates, and the same seed as the original review. The writer benchmark
uses 100,000 markers and 4,096-marker appends. Top-level API measurements include
validation and MAC filtering but stub out plotting for both revisions. Imports,
fixture generation, JIT warmup, result saving, and result comparisons are outside
timed regions. These warm-cache timings do not measure durable storage throughput.

The implementation changes are:

- [Genotype conversion](../panicle/utils/compact.py): small batches cast directly
  from their source layout to the final C-order float32 buffer. Supplied output
  buffers also avoid an intermediate int8 copy. Blocks of at least eight million
  elements retain the existing parallel conversion path. Every int8 value casts
  exactly, including missing-value sentinels.
- [BLINK](../panicle/association/blink.py): preserve the original genotype array
  and allele metadata when every marker survives filtering. Pass pre-imputed
  GenotypeMatrix metadata through MAF calculation, and materialize lazy row
  subsets once. Actual marker filtering and missing-genotype handling remain
  covered by regression tests.
- [Top-level API](../panicle/core/mvp.py): group GLM traits by their exact retained
  sample mask after phenotype/covariate missingness. A group shares preparation,
  MAC filtering, and one joint scan; each result is padded back to the original
  marker map. Distinct masks run separately. Preparation is evicted after the
  group's final trait and bounded to four cached groups; eviction can repeat
  preparation, but does not repeat the joint scan. Shared scan time is divided
  equally among the group's per-trait runtime entries. The pipeline already
  supported grouping, and its existing tests still pass. Existing prefetch
  controls are now documented and checked for numerical equality; defaults have
  not changed.
- [VCF writer](../panicle/data/load_genotype_vcf.py): flush, close, extend, and
  remap the temporary file without copying/re-writing its existing prefix.
  Finalization and discard explicitly close the mapping before removal. A failed
  remap leaves a cleanable temporary file. The public matrix layout and cache
  format are unchanged.

Validation: **292 passed, 1 skipped** with `python -m pytest tests/ -q --tb=short`.
The skip is an existing obsolete-loader test. Coverage includes read-only,
Fortran-order, sliced, reversed, broadcast, and memmapped genotype inputs;
conversion output buffers and the no-Numba fallback; BLINK identity/partial/all
filtering and lazy subsets; interleaved trait groups with phenotype, covariate,
and genotype missingness; MAC-filtered marker padding; single/joint prefetch
equivalence; mixed writer appends, mapping closure, and remap failure cleanup.

The million-marker baseline comparison verified exact equality for conversions,
single-trait GLM, both BLINK cases, and writer matrices. Grouped GLM agrees with
the original separate scans at `rtol=1e-5, atol=1e-6`; small floating-point
differences are expected from the joint matrix operations. Validation was run on
the local macOS environment, not the full CI Python/OS matrix.

Reproduce the paired comparison with a baseline checkout and installed dependencies:

```sh
PYTHONPATH=/path/to/baseline MPLBACKEND=Agg python scripts/benchmark_changes.py --output /tmp/panicle-before-new
PYTHONPATH=. MPLBACKEND=Agg python scripts/benchmark_changes.py --output /tmp/panicle-after-new --compare /tmp/panicle-before-new
```

Run sequentially and use new output directories. Original paired artifacts are
in `/tmp/panicle-changes-before` and `/tmp/panicle-changes-after`. The benchmark
loads code from `PYTHONPATH` and records its source path in `environment.json`.
CSV/binary result-format work and MLM algorithm changes were outside this change.
