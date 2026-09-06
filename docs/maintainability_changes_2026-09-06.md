# Maintainability refactor: items 1–4

## Scope and module ownership

| Responsibility | Implementation |
| --- | --- |
| Shared sample masks, grouping, marker selection, solver calls | `panicle/core/workflow.py` |
| Named trait, cache-key, preparation, selection, and method-result objects | `panicle/core/workflow.py` |
| Cache paths, freshness, filter fingerprints, fallback, read/write | `panicle/data/genotype_cache.py` |
| Packed lazy map columns and binary map serialization | `panicle/utils/map_cache.py` |
| Pure table assembly and JSON conversion | `panicle/reporting/tables.py` |
| Plot invocation and figure lifecycle | `panicle/reporting/plots.py` |
| Existing pipeline and one-call output formats | `panicle/reporting/pipeline.py`, `legacy.py` |

`PANICLE(...)` and `GWASPipeline` remain public adapters. They share workflow
primitives rather than forcing their historically different option defaults,
logging, error policies, population-structure reuse, and output conventions into
one new policy. Numerical kernels were not changed by this refactor.

The pipeline preparation cache is now one coherent `TraitPreparation` object,
keyed by ordered sample indices, PCs, kinship requirement, MAC, and dosage.
Internal consumers use named fields. Compatibility wrappers retain the old
private tuple-returning preparation and method helpers. The old map-serialization
imports from `data_types` and writer import from `core.mvp` remain available.

Format-specific parsers still own decoding and QC. Their shared cache preserves
the v2 filenames/format, strict source-mtime comparison, filter fingerprints,
PLINK's three source dependencies, legacy CSV-map migration, read-only memory
mapping, and nonfatal cache failures. No new dependency was added.

For future changes: add shared preparation or execution rules to `workflow`,
format-specific parsing to the relevant loader, and presentation-only changes to
`reporting`. Keep interface-specific defaults in the public adapters. Avoid adding
file output to workflow helpers or reintroducing positional cache state.

## Verification

- Full suite: **320 passed, 1 existing skip** (obsolete loader test).
- 28 added tests cover shared cache round-trips, invalidation by every PLINK
  source, force/filter changes, missing/corrupt data, legacy migration, nonfatal
  save failures, ordered grouping, typed-cache invalidation/clearing, lazy/eager
  filtering, method input identity, errors, joint scans, table purity, and figure
  lifecycle.
- `scripts/validate_refactor.py`: compared each public interface with its own
  pre-refactor working-tree snapshot, including the earlier performance changes.
  **62 array/table/metadata artifacts matched exactly**, including filenames,
  column ordering, standard errors, p-values, marker aliases, and scientific
  summaries. Only the runtime section of text summaries is excluded.
- Real GLM, global/LOCO MLM, FarmCPU, and BLINK runs exercised three traits,
  shared/distinct missing-data masks, covariates, two PCs, MAC filtering and
  full-map padding. Plotting was stubbed in this cross-revision check; existing
  visualization/integration tests and new lifecycle tests cover rendering paths.
- BayesLOCO and resampling retain existing unit/integration coverage and have
  shared-dispatch contract tests; they were not part of the cross-revision fixture.

### Paired performance smoke check

Same machine/environment, baseline then refactor, one warmup and three timed
repetitions, 738 samples × 100,000 markers. Median seconds:

| Stage | Before | After |
| --- | ---: | ---: |
| GLM | 0.03712 | 0.03561 |
| GLM with prefetch | 0.02318 | 0.02272 |
| BLINK | 0.18236 | 0.17894 |
| Filtered BLINK | 0.14967 | 0.15031 |
| Eight-trait one-call GLM | 0.08557 | 0.08392 |
| 100k-marker temporary writer | 0.12747 | 0.13204 |

No clear runtime regression in this small warm-cache check; this is not a
statistical performance guarantee or a cold-disk benchmark. Cast timings are
sub-millisecond to low-millisecond and noisy. Raw timing vectors are recorded in
`benchmarks/refactor_before.json` and `refactor_after.json`. Numerical comparisons
passed for all stages (exact except the benchmark's existing eight-trait tolerance
of rtol=1e-5/atol=1e-6; the separate compatibility fixture uses exact comparisons).

Reproduce with an environment containing the project's test dependencies:

```sh
PYTHONPATH=/path/to/before PYTHONHASHSEED=0 python scripts/validate_refactor.py --output /tmp/parity-before
PYTHONPATH=. PYTHONHASHSEED=0 python scripts/validate_refactor.py --output /tmp/parity-after --compare /tmp/parity-before
PYTHONPATH=/path/to/before python scripts/benchmark_changes.py --markers 100000 --repeats 3 --output /tmp/speed-before
PYTHONPATH=. python scripts/benchmark_changes.py --markers 100000 --repeats 3 --output /tmp/speed-after --compare /tmp/speed-before
PYTHONPATH=. MPLBACKEND=Agg python -m pytest tests/ -q
```

## Pre-existing limitation found during validation

The one-call global-MLM path attempts `np.asarray(KinshipMatrix)` before subsetting
missing samples, producing a zero-dimensional object array and an `IndexError`.
This also fails in the pre-refactor snapshot. Consequently,
the cross-revision fixture uses complete samples for one-call global MLM; LOCO
and both pipeline modes exercise missing samples.

Subsequent correctness fix: the one-call API now indexes `KinshipMatrix` directly,
preserving the full-panel kinship calculation and selecting both axes for each
trait. Three new regression cases (phenotype-only, covariate-only, and combined
missingness) reproduce the original crash and compare corrected results with
direct MLM on the corresponding subsets. The full suite now has 323 passing
tests and one existing skip. The cross-revision fixture retains complete samples
for one-call global MLM so it remains runnable against the unfixed baseline.
