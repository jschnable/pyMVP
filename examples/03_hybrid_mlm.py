#!/usr/bin/env python3
"""
Example 03: Hybrid MLM for Increased Power

This example demonstrates the Hybrid MLM approach which combines:
- Fast Wald test screening of all markers
- Exact LRT (Likelihood Ratio Test) refinement of top hits

Benefits:
- LRT-quality p-values for significant associations
- Minimal runtime overhead (~2-5% vs standard MLM)
- Increased statistical power for detecting large-effect loci

Use when:
- You need accurate p-values for top associations
- You're studying traits with moderate-to-large effect loci
- Avoiding false positives is critical
"""

from pymvp.pipelines.gwas import GWASPipeline
import pandas as pd
import numpy as np
import time

def main():
    print("=" * 70)
    print("EXAMPLE 03: Hybrid MLM for Increased Power")
    print("=" * 70)

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir='./example03_results')

    # Load data
    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file='example_phenotypes.csv',
        genotype_file='example_genotypes.vcf.gz'
    )

    pipeline.align_samples()

    # Compute population structure
    print("\n2. Computing population structure...")
    pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)

    # Run both standard MLM and Hybrid MLM for comparison
    print("\n3. Running standard MLM...")
    start_mlm = time.time()
    pipeline.run_analysis(
        traits=['PlantHeight'],
        methods=['MLM']
    )
    mlm_time = time.time() - start_mlm

    print("\n4. Running Hybrid MLM...")
    start_hybrid = time.time()
    pipeline.run_analysis(
        traits=['PlantHeight'],
        methods=['MLM_Hybrid'],
        hybrid_params={
            'screen_threshold': 1e-4  # Refine markers with p < 0.0001
        }
    )
    hybrid_time = time.time() - start_hybrid

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Standard MLM runtime: {mlm_time:.2f} seconds")
    print(f"Hybrid MLM runtime:   {hybrid_time:.2f} seconds")
    print(f"Overhead:             {hybrid_time - mlm_time:.2f} seconds " +
          f"({100*(hybrid_time/mlm_time - 1):.1f}%)")

    # Load and compare p-values
    results = pd.read_csv('example03_results/GWAS_PlantHeight_all_results.csv')

    # Find markers where LRT improved significance
    refined = results[results['MLM_P'] < 1e-4].copy()
    if len(refined) > 0:
        refined['P_ratio'] = refined['MLM_P'] / refined['MLM_Hybrid_P']
        refined['Log10_improvement'] = (-refined['MLM_Hybrid_P'].apply(lambda x: -999 if x <= 0 else np.log10(x)) -
                                       -refined['MLM_P'].apply(lambda x: -999 if x <= 0 else np.log10(x)))

        print(f"\nMarkers refined by LRT: {len(refined)}")
        print(f"Average p-value improvement: {refined['P_ratio'].mean():.2f}×")
        print(f"Maximum p-value improvement: {refined['P_ratio'].max():.2f}×")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey insight:")
    print("Hybrid MLM provides LRT-quality results with minimal time cost")


if __name__ == '__main__':
    main()
