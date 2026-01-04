#!/usr/bin/env python3
"""
Example 06: FarmCPU Resampling (RMIP)

This example runs FarmCPU resampling to estimate RMIP (resampling model
inclusion probability) for a single trait. The output includes an RMIP
table and a Manhattan plot showing RMIP by marker.
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
    parser = argparse.ArgumentParser(description="Example 06: FarmCPU Resampling")
    parser.add_argument("--phenotype", default=str(HERE / "example_phenotypes.csv"))
    parser.add_argument("--genotype", default=str(HERE / "example_genotypes.vcf.gz"))
    parser.add_argument("--map", default=None)
    parser.add_argument("--trait", default="PlantHeight")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--mask-proportion", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="./example06_results")
    return parser.parse_args()


def main():
    print("=" * 70)
    print("EXAMPLE 06: FarmCPU Resampling (RMIP)")
    print("=" * 70)

    args = parse_args()
    pipeline = GWASPipeline(output_dir=args.output)

    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file=args.phenotype,
        genotype_file=args.genotype,
        map_file=args.map,
    )

    print("\n2. Aligning samples...")
    pipeline.align_samples()

    print("\n3. Computing population structure...")
    pipeline.compute_population_structure(n_pcs=args.n_pcs)

    n_markers = pipeline.genotype_matrix.n_markers
    resampling_threshold = args.alpha / max(n_markers, 1)

    farmcpu_params = {
        "resampling_runs": args.runs,
        "resampling_mask_proportion": args.mask_proportion,
        "resampling_significance_threshold": resampling_threshold,
        "resampling_cluster_markers": False,
        "resampling_ld_threshold": 0.7,
    }

    print("\n4. Running FarmCPU resampling...")
    pipeline.run_analysis(
        traits=[args.trait],
        methods=["FarmCPUResampling"],
        farmcpu_params=farmcpu_params,
        outputs=["manhattan"],
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output}/")
    print(f"- GWAS_{args.trait}_FarmCPUResampling_RMIP.csv")
    print(f"- GWAS_{args.trait}_FarmCPUResampling_rmip_manhattan.png")


if __name__ == "__main__":
    main()
