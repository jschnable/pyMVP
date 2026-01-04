#!/usr/bin/env python3
"""
Example 02: MLM with Population Structure Correction

This example demonstrates how to use Mixed Linear Models (MLM) to account for
population structure using kinship matrix and principal components.

MLM is recommended for:
- Diverse populations with population structure
- Related individuals (family-based studies)
- Avoiding spurious associations due to stratification

Prerequisites:
- phenotype.csv
- genotypes.vcf.gz
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panicle.pipelines.gwas import GWASPipeline

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Example 02: MLM with Population Structure")
    parser.add_argument("--phenotype", default=str(HERE / "example_phenotypes.csv"))
    parser.add_argument("--genotype", default=str(HERE / "example_genotypes.vcf.gz"))
    parser.add_argument("--map", default=None)
    parser.add_argument("--trait", default="PlantHeight")
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--outputdir", default="./example02_results")
    return parser.parse_args()

def main():
    print("=" * 70)
    print("EXAMPLE 02: MLM with Population Structure Correction")
    print("=" * 70)

    args = parse_args()

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir=args.outputdir)

    # Load data
    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file=args.phenotype,
        genotype_file=args.genotype,
        map_file=args.map
    )

    # Align samples
    print("\n2. Aligning samples...")
    pipeline.align_samples()

    # Compute population structure
    # This calculates:
    #   - Principal components (PCs) for covariates
    #   - Kinship matrix for random effects
    print("\n3. Computing population structure...")
    pipeline.compute_population_structure(
        n_pcs=args.n_pcs,         # Calculate principal components
        calculate_kinship=True   # Calculate kinship matrix (needed for MLM)
    )
    print("   PCs will be used as covariates")
    print("   Kinship matrix will account for relatedness")

    # Run MLM analysis
    print("\n4. Running MLM analysis...")
    pipeline.run_analysis(
        traits=[args.trait],     # Analyze requested trait
        methods=['MLM']          # Use Mixed Linear Model
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.outputdir}/")
    print("\nCompare with GLM:")
    print("- MLM typically has better control of false positives")
    print("- Check lambda_GC in QQ plots (should be closer to 1.0)")
    print("- MLM may have fewer but more reliable associations")


if __name__ == '__main__':
    main()
