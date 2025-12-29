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

from panicle.pipelines.gwas import GWASPipeline

def main():
    print("=" * 70)
    print("EXAMPLE 02: MLM with Population Structure Correction")
    print("=" * 70)

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir='./example02_results')

    # Load data
    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file='example_phenotypes.csv',
        genotype_file='example_genotypes.vcf.gz'
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
        n_pcs=5,                 # Calculate 5 principal components
        calculate_kinship=True   # Calculate kinship matrix (needed for MLM)
    )
    print("   PCs will be used as covariates")
    print("   Kinship matrix will account for relatedness")

    # Run MLM analysis
    print("\n4. Running MLM analysis...")
    pipeline.run_analysis(
        traits=['PlantHeight'],  # Analyze PlantHeight trait
        methods=['MLM']          # Use Mixed Linear Model
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to: ./example02_results/")
    print("\nCompare with GLM:")
    print("- MLM typically has better control of false positives")
    print("- Check lambda_GC in QQ plots (should be closer to 1.0)")
    print("- MLM may have fewer but more reliable associations")


if __name__ == '__main__':
    main()
