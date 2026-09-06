"""Pipeline table/metadata writer, with explicit inputs and pluggable plotting."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from ..utils.data_types import GenotypeMatrix, GenotypeMap, MARKER_ID_COLUMN, LEGACY_MARKER_ID_COLUMN, infer_marker_id_column
from ..utils.stats import calculate_maf_from_genotypes, calculate_maf_for_indices
from ..association.farmcpu_resampling import FarmCPUResamplingResults
from .tables import json_default
from .plots import render_and_close
from .models import MethodReport, TraitReport, ReportOptions
from ..core.thresholds import Threshold
from ..core.workflow import MethodRunResult
from ..core.methods import ordered_report_methods


@dataclass
class TraitOutputContext:
    output_dir: Path
    geno_map: Optional[GenotypeMap]
    genotype_matrix: GenotypeMatrix
    log: Callable
    renderer: Callable


def write_trait_results(
    context: TraitOutputContext,
    trait_name,
    results,
    threshold,
    alpha,
    n_tests,
    max_dosage,
    outputs,
    threshold_source,
    include_standard_errors: bool = False,
    method_thresholds=None,
    method_threshold_sources=None,
    method_lambda_gc=None,
    method_lambda_gc_is_approx=None,
    n_samples=None,
    n_markers=None,
    runtime_seconds=None,
    geno_for_maf: Optional[GenotypeMatrix] = None,
    maf_keep_indices: Optional[np.ndarray] = None,
):
    """Compatibility adapter for the former long-argument reporting API."""

    base = Threshold(threshold, threshold_source, n_tests)
    methods = {
        name: MethodReport(
            MethodRunResult(
                name, result,
                lambda_gc=(method_lambda_gc or {}).get(name),
                lambda_gc_is_approx=(method_lambda_gc_is_approx or {}).get(name, False),
            ),
            Threshold((method_thresholds or {}).get(name, threshold),
                      (method_threshold_sources or {}).get(name, threshold_source)),
        )
        for name, result in results.items()
    }
    trait = TraitReport(
        trait_name, methods, base, n_samples, n_markers, runtime_seconds,
        geno_for_maf, maf_keep_indices,
    )
    return write_trait_report(context, trait, ReportOptions(outputs, alpha, max_dosage, include_standard_errors))


def write_trait_report(context: TraitOutputContext, trait: TraitReport, options: ReportOptions):
    """Write one trait from coherent results, diagnostics, and threshold metadata."""
    summary_data = []
    want_full_table = 'all_marker_pvalues' in options.outputs
    if want_full_table and context.geno_map is not None:
        # One copy of the map frame. to_dataframe() already copies; do not
        # copy again.
        if hasattr(context.geno_map, "data") and getattr(context.geno_map, "_dataframe_cache", None) is not None:
            base_df = context.geno_map.data.copy()
        elif hasattr(context.geno_map, "to_dataframe"):
            base_df = context.geno_map.to_dataframe()
        else:
            base_df = pd.DataFrame()
    elif want_full_table:
        base_df = pd.DataFrame()
    else:
        base_df = pd.DataFrame()
    marker_id_col = infer_marker_id_column(base_df.columns) if not base_df.empty else None
    if marker_id_col is None and not base_df.empty:
        raise ValueError("Genotype map is missing a marker ID column")
    if marker_id_col is not None and marker_id_col != MARKER_ID_COLUMN:
        base_df[MARKER_ID_COLUMN] = base_df[marker_id_col].astype(str)
    if not base_df.empty and LEGACY_MARKER_ID_COLUMN not in base_df.columns:
        base_df[LEGACY_MARKER_ID_COLUMN] = base_df[MARKER_ID_COLUMN].astype(str)

    base_columns = base_df.columns.tolist()

    def _insert_maf_column(df: pd.DataFrame, maf_values: np.ndarray) -> None:
        if 'MAF' in df.columns:
            df['MAF'] = maf_values
            return
        insert_at = len(df.columns)
        if 'ALT' in df.columns:
            insert_at = df.columns.get_loc('ALT') + 1
        df.insert(insert_at, 'MAF', maf_values)

    def _ordered_base_columns(df: pd.DataFrame) -> List[str]:
        cols = list(base_columns)
        if 'MAF' in df.columns and 'MAF' not in cols:
            if 'ALT' in cols:
                cols.insert(cols.index('ALT') + 1, 'MAF')
            else:
                cols.append('MAF')
        return cols


    all_res_df = base_df
    sig_snps = []
    hits_by_method = {}
    resampling_hit_snps = set()
    rmip_hit_threshold = 0.1

    ordered_methods = ordered_report_methods(trait.methods)

    output_prefix_base = str(context.output_dir / f"GWAS_{trait.name}")

    for method in ordered_methods:
        method_report = trait.methods[method]
        Res = method_report.run.result
        method_threshold = method_report.threshold.value
        method_source = method_report.threshold.source
        method_alpha = options.alpha if method_threshold == trait.base_threshold.value else None
        method_n_tests = trait.base_threshold.n_tests if method_threshold == trait.base_threshold.value else float('nan')

        if isinstance(Res, FarmCPUResamplingResults):
            # Handle Resampling
            res_file = context.output_dir / f"GWAS_{trait.name}_{method}_RMIP.csv"
            df = Res.to_dataframe()
            if 'Chr' in df.columns or 'Pos' in df.columns:
                df = df.rename(columns={'Chr': 'CHROM', 'Pos': 'POS'})
            if 'RMIP' in df.columns:
                resampling_marker_col = infer_marker_id_column(df.columns)
                if resampling_marker_col is None:
                    raise ValueError("Resampling results are missing a marker ID column")
                resampling_hit_snps = set(
                    df.loc[df['RMIP'] >= rmip_hit_threshold, resampling_marker_col].astype(str)
                )
            df.to_csv(res_file, index=False)
            summary_data.append({
                'Trait': trait.name, 'Method': method,
                'Significant_Hits': len(resampling_hit_snps),
                'Threshold': rmip_hit_threshold,
                'Lambda_GC': float('nan'),  # Not applicable for resampling
                'N_Samples': trait.n_samples,
                'N_Markers': trait.n_markers,
                'Runtime_Seconds': round(trait.runtime_seconds, 2) if trait.runtime_seconds else None,
                'Info': f"{method_source}; RMIP>={rmip_hit_threshold}; Runs={Res.total_runs}; Clustered={Res.cluster_mode}"
            })

            # Generate RMIP Manhattan plot
            if 'manhattan' in options.outputs:
                try:
                    report = render_and_close(renderer=context.renderer,
                        results=Res,
                        map_data=context.geno_map,
                        output_prefix=output_prefix_base,
                        plot_types=['manhattan'],
                        verbose=False,
                        save_plots=True
                    )
                except Exception as e:
                    context.log(f"   RMIP plotting error {method}: {e}")

            continue

        # Standard Results
        if want_full_table:
            all_res_df[f'{method}_P'] = Res.pvalues
            all_res_df[f'{method}_Effect'] = Res.effects
            if options.include_standard_errors:
                all_res_df[f'{method}_SE'] = Res.se

        # Check significance
        hits = Res.pvalues <= method_threshold
        hits_by_method[method] = hits
        n_sig = int(hits.sum())

        # Get lambda GC for this method
        lambda_gc_value = method_report.run.lambda_gc
        if lambda_gc_value is None:
            lambda_gc_value = float('nan')

        summary_data.append({
            'Trait': trait.name, 'Method': method,
            'Significant_Hits': n_sig,
            'Threshold': method_threshold,
            'Lambda_GC': round(lambda_gc_value, 3) if not np.isnan(lambda_gc_value) else float('nan'),
            'N_Samples': trait.n_samples,
            'N_Markers': trait.n_markers,
            'Runtime_Seconds': round(trait.runtime_seconds, 2) if trait.runtime_seconds else None,
            'Info': method_source
        })

        method_metadata = getattr(Res, "metadata", None)
        if isinstance(method_metadata, dict) and method_metadata:
            meta_file = context.output_dir / f"GWAS_{trait.name}_{method}_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as handle:
                json.dump(method_metadata, handle, indent=2, sort_keys=True, default=json_default)
            summary_data[-1]["Metadata_File"] = meta_file.name

        # Plots
        if 'manhattan' in options.outputs or 'qq' in options.outputs:
            try:
                plot_types = []
                if 'manhattan' in options.outputs: plot_types.append('manhattan')
                if 'qq' in options.outputs: plot_types.append('qq')

                report = render_and_close(renderer=context.renderer,
                    results={method: Res}, map_data=context.geno_map,
                    output_prefix=output_prefix_base,
                    plot_types=plot_types,
                    threshold=method_threshold,
                    threshold_alpha=method_alpha,
                    threshold_n_tests=method_n_tests,
                    threshold_source=method_source,
                    method_lambda_gc={name: item.run.lambda_gc for name, item in trait.methods.items()
                                      if item.run.lambda_gc is not None},
                    method_lambda_gc_is_approx={name: item.run.lambda_gc_is_approx for name, item in trait.methods.items()
                                                if item.run.lambda_gc is not None},
                    verbose=False,
                    save_plots=True
                )

            except Exception as e:
                context.log(f"   Plotting error {method}: {e}")

    # Save merged tables
    method_columns = []
    for method in ordered_methods:
        if method == 'FarmCPUResampling':
            continue
        method_columns.extend([f'{method}_P', f'{method}_Effect'])
        if options.include_standard_errors:
            method_columns.append(f'{method}_SE')

    if 'all_marker_pvalues' in options.outputs:
        maf_source = trait.genotype_for_maf or context.genotype_matrix
        maf_all = calculate_maf_from_genotypes(maf_source, max_dosage=options.max_dosage)
        # Pad MAF vector back to full-map length when the genotype
        # passed to MAF is a per-trait MAC-filtered subset.
        if (
            trait.maf_keep_indices is not None
            and len(maf_all) == len(trait.maf_keep_indices)
            and len(maf_all) != len(all_res_df)
        ):
            padded = np.full(len(all_res_df), np.nan, dtype=float)
            padded[np.asarray(trait.maf_keep_indices, dtype=np.int64)] = np.asarray(maf_all, dtype=float)
            maf_all = padded
        _insert_maf_column(all_res_df, maf_all)
        ordered_base = _ordered_base_columns(all_res_df)
        if method_columns:
            all_res_df = all_res_df[ordered_base + method_columns]
        all_res_df.to_csv(context.output_dir / f"GWAS_{trait.name}_all_results.csv", index=False)

    if resampling_hit_snps:
        all_res_marker_col = infer_marker_id_column(all_res_df.columns)
        if all_res_marker_col is None:
            raise ValueError("Merged results are missing a marker ID column")
        resampling_mask = all_res_df[all_res_marker_col].astype(str).isin(resampling_hit_snps).to_numpy()
        hits_by_method['FarmCPUResampling'] = resampling_mask

    if hits_by_method and 'significant_marker_pvalues' in options.outputs:
        n_out = (
            all_res_df.shape[0]
            if want_full_table and not all_res_df.empty
            else int(next(iter(hits_by_method.values())).shape[0])
        )
        method_labels = [[] for _ in range(n_out)]
        for method in ordered_methods:
            hits = hits_by_method.get(method)
            if hits is None or not np.any(hits):
                continue
            for idx in np.where(hits)[0]:
                method_labels[idx].append(method)

        any_hits = np.array([bool(labels) for labels in method_labels], dtype=bool)
        if np.any(any_hits):
            if want_full_table and not all_res_df.empty:
                sig_df = all_res_df.loc[any_hits].copy()
            elif hasattr(context.geno_map, "to_dataframe_at"):
                sig_indices = np.where(any_hits)[0]
                sig_df = context.geno_map.to_dataframe_at(sig_indices)
                for method in ordered_methods:
                    if method == "FarmCPUResampling" or method not in trait.methods:
                        continue
                    res_obj = trait.methods[method].run.result
                    sig_df[f"{method}_P"] = res_obj.pvalues[sig_indices]
                    sig_df[f"{method}_Effect"] = res_obj.effects[sig_indices]
                    if options.include_standard_errors:
                        sig_df[f"{method}_SE"] = res_obj.se[sig_indices]
                if not base_columns:
                    base_columns = [
                        c for c in sig_df.columns
                        if not c.endswith("_P") and not c.endswith("_Effect") and not c.endswith("_SE")
                    ]
            else:
                sig_df = all_res_df.loc[any_hits].copy()
            if 'MAF' not in sig_df.columns:
                sig_indices = np.where(any_hits)[0]
                geno_source = trait.genotype_for_maf or context.genotype_matrix
                if trait.maf_keep_indices is not None and trait.genotype_for_maf is not None:
                    keep_indices_arr = np.asarray(trait.maf_keep_indices, dtype=np.int64)
                    full_to_local = {
                        int(full_idx): local_idx
                        for local_idx, full_idx in enumerate(keep_indices_arr)
                    }
                    maf_subset = np.full(sig_indices.size, np.nan, dtype=float)
                    local_indices = []
                    output_positions = []
                    for output_pos, full_idx in enumerate(sig_indices):
                        local_idx = full_to_local.get(int(full_idx))
                        if local_idx is not None:
                            output_positions.append(output_pos)
                            local_indices.append(local_idx)
                    if local_indices:
                        output_positions_arr = np.asarray(output_positions, dtype=np.int64)
                        maf_subset[output_positions_arr] = calculate_maf_for_indices(
                            geno_source,
                            np.asarray(local_indices, dtype=np.int64),
                            max_dosage=options.max_dosage,
                        )
                else:
                    maf_subset = calculate_maf_for_indices(
                        geno_source,
                        sig_indices,
                        max_dosage=options.max_dosage,
                    )
                _insert_maf_column(sig_df, maf_subset)
            sig_df['Method'] = [
                "|".join(method_labels[idx]) for idx in np.where(any_hits)[0]
            ]
            ordered_base = _ordered_base_columns(sig_df)
            if method_columns:
                sig_df = sig_df[ordered_base + method_columns + ['Method']]
            sig_df.to_csv(context.output_dir / f"GWAS_{trait.name}_significant.csv", index=False)

    return summary_data
