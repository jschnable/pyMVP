#!/usr/bin/env python3
"""
Example 04: GWAS with External Covariates

This example shows how to include external covariates in your GWAS analysis.

In this case, we're analyzing PlantHeight while controlling for DaysToFlower
(flowering time) as a covariate. This is a realistic scenario since:
- Flowering time can be correlated with plant height
- Controlling for flowering time helps isolate height-specific genetic effects
- This demonstrates how to use any measured trait as a covariate

External covariates can help control for:
- Other traits (as shown here)
- Environmental effects (location, year)
- Experimental design factors (block, treatment)
- Other measured confounders
"""

from pymvp.pipelines.gwas import GWASPipeline

def main():
    print("=" * 70)
    print("EXAMPLE 04: GWAS with External Covariates")
    print("=" * 70)

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir='./example04_results')

    # Load data including external covariates
    print("\n1. Loading data with covariates...")
    pipeline.load_data(
        phenotype_file='example_phenotypes.csv',
        genotype_file='example_genotypes.vcf.gz',
        covariate_file='example_covariates.csv',  # Contains DaysToFlower
        covariate_columns=['DaysToFlower']        # Use flowering time as covariate
    )
    print("   Loaded covariate: DaysToFlower")

    # Align samples
    print("\n2. Aligning samples...")
    pipeline.align_samples()

    # Compute population structure
    # PCs will be automatically combined with external covariates
    print("\n3. Computing population structure...")
    pipeline.compute_population_structure(
        n_pcs=3,
        calculate_kinship=True
    )
    print("   Final covariates: DaysToFlower, PC1, PC2, PC3")

    # Run MLM with combined covariates
    print("\n4. Running MLM with all covariates...")
    pipeline.run_analysis(
        traits=['PlantHeight'],
        methods=['MLM']
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to: ./example04_results/")
    print("\nCovariates used in analysis:")
    print("- External: DaysToFlower (from example_covariates.csv)")
    print("- PCs: PC1, PC2, PC3 (for population structure)")
    print("\nThis helps control for:")
    print("- Correlation between flowering time and plant height")
    print("- Population stratification")
    print("- Increases power to detect height-specific genetic associations")
    print("\nNext steps:")
    print("- Compare results with example02 (MLM without flowering time covariate)")
    print("- Check if controlling for flowering time reveals new height-specific QTLs")


if __name__ == '__main__':
    main()
