# BLINK Integration Plan for pyMVP

## Goals
- Add BLINK (Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway) as a first-class GWAS method within `pyMVP`.
- Match methodological behavior of the R reference implementation (`BLINK/R/Blink.R` and helpers) while reusing existing pyMVP utilities for genotype handling, GLM fitting, and LD pruning.
- Provide a maintainable Python API (`MVP_BLINK`) that plugs into the existing `MVP` dispatcher, returns an `AssociationResults` object, and supports downstream visualization/reporting.

## Reference Notes
- Core algorithm flow: `BLINK/R/Blink.R` lines ~1-280 (iteration logic, convergence, substitution) plus helper scripts `Blink.BICselection.R`, `Blink.LDRemove.R`, and `Blink.SUB.R`.
- Prior handling relies on `FarmCPU.Prior` (see `rMVP/R/MVP.FarmCPU.r` lines ~1139-1156) to temper p-values using external priors.
- BLINK reuses FarmCPU's GLM engine (`FarmCPU.LM`) for marker rescans; pyMVP should reuse `MVP_GLM` with covariate controls to keep consistency with existing modules (`pymvp/association/farmcpu.py`, especially `_get_covariate_statistics`).

## Implementation Outline
### 1. New module: `pymvp/association/blink.py`
Create a dedicated module that mirrors the structure of other association methods. Expected public surface:
- `MVP_BLINK(...) -> AssociationResults`
  - Inputs: phenotype matrix (`phe`, n×2), genotype (`geno` as `GenotypeMatrix` or ndarray), map (`GenotypeMap`), optional covariates (`CV`), optional prior table, and keyword params such as `maxLoop`, `converge`, `ld_threshold`, `p_threshold`, `qtn_threshold`, `maf_threshold`, `bic_method`, `method_sub`, `verbose`.
  - Output: `AssociationResults` with effect/SE/p-value arrays aligned to markers.
  - Side data: optionally expose iteration diagnostics for testing similar to `MVP_FarmCPU` (e.g., `MVP_BLINK.last_iteration_details`).

Internal helpers (module-private unless future reuse is desired):
- `_prepare_inputs(phe, geno, map_data, CV, maf_threshold) -> Tuple[trait_array, genotype_obj, map_obj, cv_matrix, maf_mask]`
  - Validates shapes, coerces to numpy where needed, applies MAF filtering using `calculate_maf_from_genotypes` or `GenotypeMatrix.calculate_maf`, and returns indices of retained markers.
- `_apply_prior(pvalues, prior, map_obj) -> np.ndarray`
  - Replicates `FarmCPU.Prior`: matches prior SNP IDs to map, multiplies corresponding p-values by provided weights.
- `_select_candidate_qtns(pvalues, iteration, qtn_threshold, p_threshold, prev_selection, prior_selection) -> List[int]`
  - Encodes the iteration-specific thresholding logic from `Blink.R` (line blocks handling `theLoop==2` etc.). Accepts the currently filtered SNP indices (MAF mask applied) and returns candidate marker indices (relative to filtered set).
- `_ld_prune_candidates(geno, candidate_indices, orientation, ld_threshold, block_size, ld_limit, map_data) -> List[int]`
  - Implements the block-wise LD removal in `Blink.LDRemove`. Should reuse `remove_qtns_by_ld` where possible but maintain block sampling behavior (e.g., operate in windows of `block=1000`, respect `LD.num`, and only sample up to 200 individuals when `bound=TRUE` analog).
- `_bic_model_selection(geno, candidates, phenotype, cv_matrix, orientation, bic_method) -> Tuple[List[int], np.ndarray]`
  - Port of `Blink.BICselection`: incrementally fits models with growing subsets of candidate QTNs, computes BIC (or alternative heuristics), and returns the selected QTN indices along with their per-QTN p-values for substitution. Requires access to genotype slices and phenotype/covariate matrices; should deliver stable results by using numpy linear algebra and `np.linalg.pinv` (with safeguards against singularities).
- `_build_covariate_matrix(CV, geno, qtn_indices, fill_value) -> np.ndarray`
  - Returns covariate design matrix (existing CV plus imputed pseudo-QTN genotypes). Should call `GenotypeMatrix.get_columns_imputed` when available to ensure rMVP-compatible imputation (fill with major allele or heterozygote dosage `1.0`).
- `_run_glm_with_covariates(phe, geno, covariates, major_alleles, maxLine, verbose) -> AssociationResults`
  - Thin wrapper around `MVP_GLM` to enforce consistent configuration (imputation, batch size, CPU usage) and to capture raw numpy arrays for post-processing.
- `_substitute_qtn_statistics(results_array, qtn_indices, covariate_stats, method_sub) -> None`
  - Implements `Blink.SUB`: updates rows corresponding to selected QTNs using aggregated covariate p-values (`min`/`mean`/`median`/`penalty`/`onsite`). Operates in-place on `[effect, se, p]` columns plus optional recording of t-values if we expose them.
- `_compute_covariate_stats(phe, covariate_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Borrow logic from `_get_covariate_statistics` in FarmCPU to obtain per-covariate effect/SE/p-values for substitution.
- Optional: `_collect_iteration_state(...)` to structure diagnostics for testing (selected QTNs, thresholds, BIC scores).

### 2. Algorithm flow for `MVP_BLINK`
High-level pseudocode (indexes reflect filtered markers after MAF mask):
```python
trait = phe[:, 1].astype(np.float64)
G = ensure_genotype(geno)              # GenotypeMatrix or np.ndarray
map_df = map_data.to_dataframe()
maf_mask = apply_maf_filter(G, maf_threshold)
G_filtered = subset_genotype(G, maf_mask)
map_filtered = map_df.loc[maf_mask].reset_index(drop=True)
current_pvals = np.full(G_filtered.n_markers, np.nan)
seq_qtn_prev = []
seq_qtn_save = []
iteration_details = []

for iteration in range(1, maxLoop + 1):
    if iteration == 1:
        covariates = CV_filtered  # None or subsetted by non-missing trait
    else:
        pvals_with_prior = _apply_prior(current_pvals, Prior, map_filtered)
        candidate_indices = _select_candidate_qtns(
            pvals_with_prior, iteration, qtn_threshold, p_threshold,
            prev_selection=seq_qtn_save,
            prior_selection=seq_qtn_prev,
        )
        if not candidate_indices:
            seq_qtn = []
        else:
            ld_pruned = _ld_prune_candidates(
                G_filtered, candidate_indices,
                ld_threshold=ld_threshold,
                block_size=blink_block,
                ld_limit=ld_limit,
                map_data=map_filtered,
            )
            seq_qtn, bic_stats = _bic_model_selection(
                G_filtered, ld_pruned,
                phenotype=trait_subset,
                cv_matrix=CV_filtered,
                bic_method=bic_method,
            )
        covariates = _build_covariate_matrix(CV_filtered, G_filtered, seq_qtn, fill_value=1.0)

    glm_res = _run_glm_with_covariates(
        phe=subset_phe,
        geno=G_filtered,
        covariates=covariates,
        major_alleles=G.major_alleles if hasattr(G, "major_alleles") else None,
        maxLine=maxLine,
        verbose=verbose and iteration == 1,
    )
    result_array = glm_res.to_numpy()
    covar_p, covar_beta, covar_se = _compute_covariate_stats(trait_subset, covariates) if seq_qtn else (None, None, None)
    if seq_qtn:
        _substitute_qtn_statistics(result_array, seq_qtn, (covar_p, covar_beta, covar_se), method_sub)

    current_pvals = result_array[:, 2]
    record_iteration_details(...)

    if iteration > 1:
        overlap = len(set(seq_qtn) & set(seq_qtn_prev)) / max(len(set(seq_qtn) | set(seq_qtn_prev)), 1)
        if overlap >= converge:
            break
    seq_qtn_prev, seq_qtn_save = seq_qtn_save, seq_qtn

final_results = AssociationResults(
    effects=result_array[:, 0],
    se=result_array[:, 1],
    pvalues=result_array[:, 2],
    snp_map=GenotypeMap(map_filtered)
)
restore_full_length(final_results, maf_mask)
return final_results
```
`restore_full_length` will re-insert rows for filtered-out markers (fill with NaNs) so lengths match original genotype/map ordering, ensuring `AssociationResults` stays aligned for downstream plotting.

### 3. Integration Points
- `pymvp/association/__init__.py`: import `MVP_BLINK` and add to `__all__`.
- `pymvp/core/mvp.py`
  - Extend method dispatch so that `method` may include "BLINK"; run the new function with relevant kwargs (e.g., map_data, CV, thresholds).
  - Update summary bookkeeping (`analysis_results['results']['BLINK']`, significant marker count, runtime, etc.).
- `pymvp/__init__.py`: expose `MVP_BLINK` in top-level exports similar to other methods.
- Optionally, add lazy attributes (`MVP_BLINK.last_iteration_details`) for debugging/tests.

## Testing Strategy
1. **Unit tests (`tests/test_blink_basic.py`)**
   - Generate small synthetic phenotype/genotype/map triples and verify `MVP_BLINK` returns an `AssociationResults` with expected shape, finite statistics, and deterministic output under fixed seeds.
   - Validate MAF filtering behavior by crafting markers below/above threshold.
   - Exercise substitution logic by forcing `method_sub` variants.
2. **Integration test vs R reference (`tests/r_helpers/run_blink.R`)**
   - Reuse existing R harness pattern: run R BLINK on a shared dataset, serialize GWAS p-values, and assert close match (within tolerance) to Python output for the same inputs.
   - Focus on convergence cases (multiple iterations) and ensure selected QTNs align.
3. **End-to-end test through `MVP` dispatcher**
   - Update `tests/quick_validation_test.py` or add new scenario invoking `MVP(..., method=["BLINK"])` to confirm results are wired into summary/visualization.
4. **Performance/robustness smoke**
   - Optional test scaling to a few thousand markers to ensure iteration completes and no singular-matrix errors occur (with try/except to skip if runtime exceeds threshold).

## Documentation Updates
- Extend `README.md` and any user guides to mention BLINK support and example usage (adding method string `"BLINK"`).
- Document configurable parameters in `docs/` (create a short section or table summarizing BLINK-specific kwargs, mirroring the R manual's key options).
- Update API reference (if any) to include `MVP_BLINK` signature and return type.

## Open Questions & Assumptions
- **Prior input format**: assume a pandas DataFrame or ndarray with columns matching `Prior` in R (`[SNP, Chr, Pos, Weight]`). Clarify in docstring and normalize internally.
- **Prediction (`Blink.Pred`)**: initial scope omits genomic prediction output; revisit once core association is validated.
- **Acceleration flag**: R code references `acceleration`/`ac` but lacks full usage—plan to defer unless benchmarks show necessity.
- **Orientation handling**: pyMVP stores genotypes as individual×marker; plan assumes this orientation. If row-major inputs appear, rely on `GenotypeMatrix` construction or raise explicit errors.
- **LD pruning parity**: we will mimic block-based pruning, but further benchmarking against R may reveal constants (block length, LD.num) requiring adjustment.

