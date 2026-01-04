#!/usr/bin/env python3
"""
Comprehensive GWAS Analysis Script using pyMVP (Refactored Pipeline Version)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from panicle.cli.utils import parse_args
from panicle.pipelines.gwas import GWASPipeline

OUTPUT_CHOICES = (
    'all_marker_pvalues',
    'significant_marker_pvalues',
    'manhattan',
    'qq',
)


def normalize_outputs(outputs):
    """Normalize output selections with comma splitting and deduplication."""
    if not outputs:
        return list(OUTPUT_CHOICES)

    normalized = []
    seen = set()
    for item in outputs:
        for part in str(item).split(','):
            part = part.strip().lower()
            if not part:
                continue
            if part not in OUTPUT_CHOICES:
                raise ValueError(f"Invalid output choice: {part}")
            if part not in seen:
                normalized.append(part)
                seen.add(part)

    return normalized if normalized else list(OUTPUT_CHOICES)


def normalize_methods(methods):
    """Normalize method names to pipeline-supported identifiers."""
    if not methods:
        return []

    aliases = {
        "GLM": "GLM",
        "MLM": "MLM",
        "FARMCPU": "FARMCPU",
        "FARMCPU_RESAMPLING": "FarmCPUResampling",
        "FARMCPURESAMPLING": "FarmCPUResampling",
        "RESAMPLING": "FarmCPUResampling",
        "BLINK": "BLINK",
    }

    normalized = []
    seen = set()
    for m in methods:
        key = str(m).replace('-', '_').replace(' ', '_').strip().upper()
        if not key:
            continue
        method = aliases.get(key, key)
        if method not in seen:
            normalized.append(method)
            seen.add(method)
    return normalized


class FarmCPUResamplingProgressReporter:
    """Lightweight progress reporter for FarmCPU resampling runs."""

    def __init__(self, trait_name: str, total_runs: int):
        self.trait_name = trait_name
        self.total_runs = total_runs
        self._started = False

    def __call__(self, run_idx: int, total_runs: int, elapsed_seconds: float) -> None:
        if not self._started:
            print(f"[{self.trait_name}] started resampling ({total_runs} runs)")
            self._started = True

        if run_idx >= total_runs:
            print(f"[{self.trait_name}] finished resampling in {elapsed_seconds:.0f}s")
            return

        threshold = max(5, int(0.1 * total_runs))
        if run_idx >= threshold:
            remaining = max(total_runs - run_idx, 0)
            avg_per_run = elapsed_seconds / max(run_idx, 1)
            eta = remaining * avg_per_run
            print(
                f"[{self.trait_name}] progress {run_idx}/{total_runs} "
                f"(ETA {eta:.0f}s)"
            )

def main():
    args = parse_args()

    # Backward compatibility: older CLI definitions may not include method flags
    # Default them to False so downstream logic can fall back to --methods.
    for flag in ("glm", "mlm", "farmcpu", "resampling"):
        if not hasattr(args, flag):
            setattr(args, flag, False)
    
    # Initialize Pipeline
    pipeline = GWASPipeline(output_dir=args.outputdir)
    
    # 1. Load Data
    traits = [t.strip() for t in args.traits.split(',')] if args.traits else None
    cov_cols = [c.strip() for c in args.covariate_columns.split(',')] if args.covariate_columns else None
    
    loader_kwargs = {
        'drop_monomorphic': args.drop_monomorphic,
        'max_missing': args.max_missing,
        'min_maf': args.min_maf,
        'include_indels': not args.snps_only,
        'split_multiallelic': not args.no_split_multiallelic,
        'compute_effective_tests': args.compute_effective_tests or args.use_effective_tests
    }

    pipeline.load_data(
        phenotype_file=args.phenotype,
        phenotype_id_column=args.phenotype_id_column,
        genotype_file=args.genotype,
        map_file=args.map,
        genotype_format=args.format,
        trait_columns=traits,
        covariate_file=args.covariates,
        covariate_columns=cov_cols,
        covariate_id_column=args.covariate_id_column,
        loader_kwargs=loader_kwargs
    )
    
    # 2. Align
    pipeline.align_samples()
    
    # 3. Structure
    pipeline.compute_population_structure(n_pcs=args.n_pcs)

    # 4. Run Analysis
    methods = []
    if args.glm: methods.append('GLM')
    if args.mlm: methods.append('MLM')
    if args.farmcpu: methods.append('FARMCPU')
    if args.resampling: methods.append('FarmCPUResampling')
    
    # If args.methods is still used as a fallback or for other methods not covered by flags
    if args.methods:
        methods.extend([m for m in args.methods.split(',') if m.strip()])

    valid_methods = normalize_methods(methods)
    
    outputs = normalize_outputs(args.outputs)

    def _resolve_denom() -> float:
        if args.n_eff:
            return float(args.n_eff)
        if args.use_effective_tests or args.compute_effective_tests:
            if pipeline.effective_tests_info and pipeline.effective_tests_info.get("Me"):
                return float(pipeline.effective_tests_info["Me"])
        return float(pipeline.genotype_matrix.n_markers)

    farmcpu_params = {
        "resampling_runs": args.farmcpu_resampling_runs,
        "resampling_mask_proportion": args.farmcpu_resampling_mask_proportion,
        "resampling_cluster_markers": args.farmcpu_resampling_cluster,
        "resampling_ld_threshold": args.farmcpu_resampling_ld_threshold,
    }
    if args.farmcpu_resampling_seed is not None:
        farmcpu_params["resampling_random_seed"] = args.farmcpu_resampling_seed

    resampling_threshold = args.farmcpu_resampling_significance
    if resampling_threshold is None and args.farmcpu_resampling_alpha is not None:
        denom = _resolve_denom()
        resampling_threshold = args.farmcpu_resampling_alpha / max(denom, 1.0)

    if resampling_threshold is not None:
        farmcpu_params["resampling_significance_threshold"] = resampling_threshold

    qtn_threshold = args.farmcpu_qtn_threshold
    qtn_threshold_is_corrected = False
    if qtn_threshold is None and args.farmcpu_qtn_alpha is not None:
        denom = _resolve_denom()
        qtn_threshold = args.farmcpu_qtn_alpha / max(denom, 1.0)
        qtn_threshold_is_corrected = True
    if qtn_threshold is not None:
        farmcpu_params["QTN_threshold"] = qtn_threshold
        if args.farmcpu_qtn_threshold is not None:
            qtn_threshold_is_corrected = True
        if qtn_threshold_is_corrected:
            farmcpu_params["QTN_threshold_is_corrected"] = True
        if args.farmcpu_p_threshold is None:
            farmcpu_params["p_threshold"] = qtn_threshold

    if args.farmcpu_p_threshold is not None:
        farmcpu_params["p_threshold"] = args.farmcpu_p_threshold

    pipeline.run_analysis(
        traits=traits,
        methods=valid_methods,
        max_iterations=args.max_iterations,
        significance=args.significance,
        alpha=args.alpha,
        n_eff=args.n_eff,
        use_effective_tests=args.use_effective_tests or args.compute_effective_tests,
        max_genotype_dosage=args.max_genotype_dosage,
        farmcpu_params=farmcpu_params,
        outputs=outputs
    )

if __name__ == "__main__":
    main()
