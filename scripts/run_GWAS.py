#!/usr/bin/env python3
"""
Comprehensive GWAS Analysis Script using pyMVP (Refactored Pipeline Version)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymvp.cli.utils import parse_args, normalize_outputs
from pymvp.pipelines.gwas import GWASPipeline

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
    methods = [m.strip().upper().replace('-', '_') for m in args.methods.split(',')]
    # Normalize method names
    valid_methods = []
    for m in methods:
        if m == 'FARMCPURESAMPLING': m = 'FARMCPU_RESAMPLING'
        valid_methods.append(m)
        
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
