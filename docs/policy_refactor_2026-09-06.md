# Configuration and reporting cleanup (items 1–3)

## Implementation

1. `panicle/core/thresholds.py` owns pure pipeline threshold resolution.
   `Threshold` carries the value, source label, and test count; `TraitThresholds`
   carries the base and per-method thresholds plus the effective solver count.
   The pipeline only logs and consumes these decisions.
2. `panicle/core/methods.py` owns canonical identities, CLI aliases, display
   names, and solver input contracts. Execution and report order remain separate
   constants. The CLI normalization function remains a compatibility entry point.
3. The internal worker now accepts `PreparedTrait` and `MethodOptions` instead
   of reconstructing a trait from separate arrays. `MethodReport` combines a
   method's result, diagnostics, and threshold; `TraitReport` and `ReportOptions`
   replace the long reporting argument list. Parallel result/diagnostic/threshold
   dictionaries are no longer maintained in the trait loop. Dictionaries needed
   by the existing plotting API are constructed only at that boundary.

The old `_execute_single_method`, `_run_single_method`, `_save_trait_results`,
and `write_trait_results` signatures remain as adapters. Internal calls use the
named interfaces. Numerical arrays are referenced without copying.

## Intentional behavior preservation

- General Bonferroni uses explicit `n_eff` before an enabled LD estimate.
- FarmCPU's solver effective count instead prefers an enabled LD estimate,
  truncated to an integer, before explicit `n_eff`.
- A fixed significance threshold wins over post-MAC reporting correction.
- Without a fixed threshold, active MAC filtering uses the tested marker count
  for the base reporting threshold, even when a full-panel estimate exists.
- FarmCPU's solver QTN input remains the original parameter, not the corrected
  reporting threshold. Corrected-QTN flags and resampling overrides retain their
  previous treatment.
- Resampling reports RMIP hits at 0.1; this is separate from its solver p-value
  threshold.
- CLI aliases are not newly accepted by the Python interfaces. Unknown names,
  deduplication, execution order, report order, and method-error handling retain
  their prior behavior. The one-call API's separate threshold policy is unchanged.

These historical differences are documented, not endorsed or altered here.
Changing them should involve an explicit statistical-policy decision.

## Validation

- After the selective-map correctness fix: **372 passed, 1 existing skip**.
- 39 new passing cases cover threshold precedence/overrides, method aliases and
  ordering, internal input identity, legacy worker/report compatibility, errors,
  LOCO inputs, plotting diagnostics, output selection, and resampling metadata.
- The original expected failure below is now a passing regression test. Nine
  additional cases cover empty/reordered/duplicate row selections across NumPy,
  Series, list and cache-backed columns, plus end-to-end significant-only output.
- `scripts/validate_refactor.py` compared against a clean snapshot of commit
  `1c8845f`: **62 array/table/metadata artifacts and scientific summaries matched
  exactly**. Real GLM, global/LOCO MLM, FarmCPU, and BLINK kernels ran; plotting
  was stubbed equally in that comparison. Existing visualization tests and new
  renderer-input tests cover reporting changes. Runtime text is excluded from
  scientific-summary comparison.
- No dependencies or numerical kernels changed.

## Separate bug discovered and subsequently fixed

`GenotypeMap.to_dataframe_at()` calls `data.take(indices).to_numpy()` whenever
the column exposes `take`. For NumPy-backed columns, `take` already returns an
array, which has no `to_numpy` method. This raises `AttributeError`, including
when significant-only reporting extracts hits without building the full table.

Reproduced with a DataFrame-created map in commit `1c8845f`. The subsequent fix
uses the existing column-materialization helper on the selected subset, handling
arrays and wrappers without decoding the full lazy column. The regression is now
`tests/test_named_pipeline_inputs.py::test_numpy_map_selective_rows`.
Real GLM significant-only output matches full-table-plus-significant output
exactly, including standard errors and MAF. The expected-failure annotation and
the Series-backed test workaround have been removed.
