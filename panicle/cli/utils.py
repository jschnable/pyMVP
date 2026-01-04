import argparse
from typing import List

OUTPUT_CHOICES = (
    'all_marker_pvalues',
    'significant_marker_pvalues',
    'manhattan',
    'qq',
)

def normalize_outputs(outputs: List[str]) -> List[str]:
    """Helper to normalize output choices"""
    if not outputs:
        return list(OUTPUT_CHOICES)
    # Simplified normalization
    valid = []
    for o in outputs:
        o = o.strip().lower()
        if o in OUTPUT_CHOICES:
            valid.append(o)
    return valid if valid else list(OUTPUT_CHOICES)

def parse_args():
    """Parse command line arguments for GWAS pipeline"""
    parser = argparse.ArgumentParser(
        description="Comprehensive GWAS Analysis using pyMVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--phenotype", "-p", required=True,
                       help="Phenotype file (CSV/TSV with ID column and trait columns)")
    parser.add_argument("--phenotype-id-column", default='ID',
                       help="Column name for sample IDs in phenotype file")
    parser.add_argument("--genotype", "-g", required=True,
                       help="Genotype file")
    
    # Optional arguments
    parser.add_argument("--map", "-m", default=None,
                       help="Genetic map file (CSV/TSV)")
    parser.add_argument("--outputdir", "-o", default="./GWAS_results",
                       help="Output directory")
    parser.add_argument("--covariates", default=None,
                       help="Optional covariate file")
    parser.add_argument("--covariate-columns", default=None,
                       help="Comma-separated list of covariate column names")
    parser.add_argument("--covariate-id-column", default='ID',
                       help="Column name for sample IDs in covariate file")
    parser.add_argument("--format", "-f", default=None, 
                       choices=['csv', 'tsv', 'numeric', 'vcf', 'plink', 'hapmap'],
                       help="Genotype file format")
    parser.add_argument("--methods", default="GLM,MLM,FarmCPU,BLINK",
                       help="Methods to run (comma-separated)")
    parser.add_argument("--n-pcs", type=int, default=3,
                       help="Number of PCs")
    
    # Thresholds
    parser.add_argument("--significance", type=float, default=None,
                       help="Fixed p-value threshold")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Bonferroni alpha")
    parser.add_argument("--n-eff", type=int, default=None,
                       help="Effective number of markers")
    parser.add_argument("--compute-effective-tests", action='store_true',
                       help="Compute effective tests")
    parser.add_argument("--use-effective-tests", action='store_true', 
                       help="Use effective tests for Bonferroni")

    # Options
    parser.add_argument("--max-genotype-dosage", type=float, default=2.0,
                       help="Max dosage (e.g. 2 for diploid)")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Max iterations for FarmCPU/BLINK")
    parser.add_argument("--traits", default=None,
                       help="Comma-separated traits")
    
    # Output
    parser.add_argument("--outputs", nargs='+',
                       choices=list(OUTPUT_CHOICES),
                       default=list(OUTPUT_CHOICES),
                       help="Outputs to generate")

    # FarmCPU resampling options
    parser.add_argument("--resampling", action='store_true',
                       help="Enable FarmCPU resampling")
    parser.add_argument("--farmcpu-resampling-runs", type=int, default=100,
                       help="Number of FarmCPU resampling runs")
    parser.add_argument("--farmcpu-resampling-mask-proportion", type=float, default=0.1,
                       help="Proportion of phenotype values masked per resampling run")
    parser.add_argument("--farmcpu-resampling-significance", type=float, default=None,
                       help="Fixed p-value threshold for resampling (overrides alpha)")
    parser.add_argument("--farmcpu-resampling-alpha", type=float, default=None,
                       help="Alpha for resampling Bonferroni threshold (alpha/n)")
    parser.add_argument("--farmcpu-resampling-cluster", action='store_true',
                       help="Cluster resampling hits by LD")
    parser.add_argument("--farmcpu-resampling-ld-threshold", type=float, default=0.7,
                       help="LD threshold for resampling clustering")
    parser.add_argument("--farmcpu-resampling-seed", type=int, default=None,
                       help="Random seed for resampling")

    # FarmCPU thresholds
    parser.add_argument("--farmcpu-qtn-threshold", type=float, default=None,
                       help="Fixed p-value threshold for FarmCPU QTN selection")
    parser.add_argument("--farmcpu-qtn-alpha", type=float, default=None,
                       help="Alpha for FarmCPU QTN Bonferroni threshold (alpha/n)")
    parser.add_argument("--farmcpu-p-threshold", type=float, default=None,
                       help="Fixed p-value threshold for FarmCPU early stopping")

    # Loader options (simplified subset)
    parser.add_argument("--drop-monomorphic", action='store_true', dest='drop_monomorphic',
                       help="Drop monomorphic markers (now default behavior, kept for backward compatibility)")
    parser.add_argument("--keep-monomorphic", action='store_false', dest='drop_monomorphic',
                       help="Keep monomorphic markers (override default)")
    parser.add_argument("--max-missing", type=float, default=1.0)
    parser.add_argument("--min-maf", type=float, default=0.0)
    parser.add_argument("--snps-only", action='store_true')
    parser.add_argument("--no-split-multiallelic", action='store_true')

    # Set defaults
    parser.set_defaults(drop_monomorphic=True)

    return parser.parse_args()
