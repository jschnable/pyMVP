#!/usr/bin/env python3
"""
Run FarmCPU v2 on PlantHeight (sorghum) with PCs and generate a Manhattan plot.
"""

from pathlib import Path
import sys

import pandas as pd

# Allow running from repo root without installation
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panicle.pipelines.gwas import GWASPipeline
from panicle.association.farmcpu_v2 import PANICLE_FarmCPU
from panicle.visualization.manhattan import PANICLE_Report
from panicle.utils.data_types import GenotypeMap


def main():
    phenotype_path = "sorghum_data/SbDiv_NE2021_Phenos_spats.csv"
    genotype_path = "sorghum_data/SbDiv_RNAseq_GeneticMarkers_Mangal2025.vcf.gz"
    map_path = "sorghum_data/SbDiv_RNAseq_GeneticMarkers_Mangal2025.vcf.gz.panicle.map.csv"

    output_dir = Path("preview_gwas_plots/farmcpu_v2_script")
    output_dir.mkdir(parents=True, exist_ok=True)
    trait = "PlantHeight"

    # Load data via pipeline
    pipeline = GWASPipeline(output_dir=str(output_dir))
    pipeline.load_data(
        phenotype_file=phenotype_path,
        genotype_file=genotype_path,
        map_file=map_path,
    )
    pipeline.align_samples()
    pipeline.compute_population_structure(n_pcs=3)  # keep PCs

    # Build numeric chromosome map for farmcpu_v2
    map_df = pipeline.geno_map.to_dataframe()
    chroms = map_df["CHROM"].astype(str).tolist()
    unique_chroms = sorted(set(chroms))
    chrom_to_num = {c: i + 1 for i, c in enumerate(unique_chroms)}
    map_df_num = map_df.copy()
    map_df_num["CHROM"] = map_df_num["CHROM"].astype(str).map(chrom_to_num)
    map_numeric = GenotypeMap(map_df_num)

    # Prepare trait data (uses aligned samples and PCs from pipeline)
    tdata = pipeline._prepare_trait_data(trait)
    if not tdata:
        raise SystemExit("No valid data for trait")
    y_sub, g_sub, cov_sub, k_sub = tdata

    # FarmCPU v2 run
    qtn_threshold = 0.01
    p_threshold = 0.05
    res = PANICLE_FarmCPU(
        phe=y_sub,
        geno=g_sub,
        map_data=map_numeric,
        CV=cov_sub,          # includes PCs
        maxLoop=10,
        p_threshold=p_threshold,
        QTN_threshold=qtn_threshold,
        verbose=True,
    )

    # Plot using the original map for labeling (Bonferroni line)
    bonf_threshold = 0.05 / map_numeric.n_markers
    report = PANICLE_Report(
        results={"FarmCPU_v2": res},
        map_data=pipeline.geno_map,
        threshold=bonf_threshold,
        threshold_alpha=0.05,
        threshold_n_tests=map_numeric.n_markers,
        threshold_source="Bonferroni",
        plot_types=["manhattan"],
        output_prefix=str(output_dir / f"GWAS_{trait}_FarmCPU_v2"),
        verbose=False,
        save_plots=True,
    )

    print("Created files:", report["files_created"])


if __name__ == "__main__":
    main()
