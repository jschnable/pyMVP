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
    
    # Initialize Pipeline
    pipeline = GWASPipeline(output_dir=args.output)
    
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
    # The original code had a comma-separated string for methods.
    # The instruction implies a change to individual boolean flags for methods.
    # Assuming `parse_args` has been updated to include these flags.
    methods = []
    if args.glm: methods.append('GLM')
    if args.mlm: methods.append('MLM')
    if args.farmcpu: methods.append('FarmCPU') # Assuming this exists
    if args.resampling: methods.append('FarmCPUResampling') # New method flag
    
    # If args.methods is still used as a fallback or for other methods not covered by flags
    if args.methods:
        for m in args.methods.split(','):
            method_name = m.strip().upper().replace('-', '_')
            if method_name == 'FARMCPURESAMPLING':
                method_name = 'FarmCPUResampling' # Normalize to the new name
            elif method_name == 'FARMCPU':
                method_name = 'FarmCPU' # Normalize if needed
            # Add other normalizations if necessary
            if method_name not in methods: # Avoid duplicates if both flags and string are used
                methods.append(method_name)

    # Normalize method names (this block replaces the old normalization logic)
    valid_methods = []
    for m in methods:
        # The instruction was to rename 'FARMCPU_RESAMPLING' to 'FarmCPUResampling'.
        # If the input method was 'FARMCPURESAMPLING' (old string format), it should become 'FarmCPUResampling'.
        # If the input method was 'FarmCPUResampling' (from new flag), it's already correct.
        if m.upper() == 'FARMCPURESAMPLING': # Handle both old string format and potential uppercase from other sources
            valid_methods.append('FarmCPUResampling')
        elif m.upper() == 'FARMCPU':
            valid_methods.append('FarmCPU')
        else:
            valid_methods.append(m) # Keep other methods as is

    outputs = normalize_outputs(args.outputs)
    
    pipeline.run_analysis(
        traits=traits,
        methods=valid_methods,
        max_iterations=args.max_iterations,
        significance=args.significance,
        alpha=args.alpha,
        n_eff=args.n_eff,
        use_effective_tests=args.use_effective_tests or args.compute_effective_tests,
        max_genotype_dosage=args.max_genotype_dosage,
        outputs=outputs
    )

if __name__ == "__main__":
    main()
