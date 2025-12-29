#!/usr/bin/env python3
"""
Example 01: Basic GWAS Analysis

This example demonstrates the simplest possible GWAS workflow using pyMVP.
We'll run a GLM analysis (no population structure correction) on a single trait.

Prerequisites:
- phenotype.csv: CSV file with ID column and trait columns
- genotypes.vcf.gz: VCF file with genetic variants
"""

from pymvp.pipelines.gwas import GWASPipeline

def main():
    print("=" * 70)
    print("EXAMPLE 01: Basic GWAS Analysis")
    print("=" * 70)

    # Initialize the pipeline with output directory
    pipeline = GWASPipeline(output_dir='./example01_results')

    # Load phenotype and genotype data
    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file='example_phenotypes.csv',
        genotype_file='example_genotypes.vcf.gz'
    )

    # Align samples (match individuals between phenotype and genotype)
    print("\n2. Aligning samples...")
    pipeline.align_samples()

    # Run GLM analysis
    # GLM is fast and doesn't require population structure correction
    print("\n3. Running GLM analysis...")
    pipeline.run_analysis(
        traits=['PlantHeight'],  # Analyze the 'PlantHeight' trait
        methods=['GLM']          # Use General Linear Model
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to: ./example01_results/")
    print("- GWAS_PlantHeight_all_results.csv       (all markers)")
    print("- GWAS_PlantHeight_significant.csv       (significant markers only)")
    print("- GWAS_PlantHeight_GLM_GWAS_manhattan.png (Manhattan plot)")
    print("- GWAS_PlantHeight_GLM_GWAS_qq.png       (QQ plot)")
    print("\nNext steps:")
    print("- Open the Manhattan plot to visualize results")
    print("- Check the QQ plot for genomic inflation")
    print("- Load GWAS_PlantHeight_all_results.csv in Python/R for further analysis")


if __name__ == '__main__':
    main()
