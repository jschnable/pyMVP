#!/usr/bin/env python3
"""Compare FarmCPU vs BLINK power on huge2 traits."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_ld_r_squared(geno1, geno2):
    """Calculate r² (linkage disequilibrium) between two markers"""
    missing_mask1 = np.isnan(geno1) | (geno1 == -9)
    missing_mask2 = np.isnan(geno2) | (geno2 == -9)
    valid_mask = (~missing_mask1) & (~missing_mask2)

    if np.sum(valid_mask) < 10:
        return 0.0

    geno1_clean = geno1[valid_mask]
    geno2_clean = geno2[valid_mask]

    try:
        correlation = np.corrcoef(geno1_clean, geno2_clean)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return correlation ** 2
    except:
        return 0.0

def calculate_power(pvalues, known_qtns, genotype_matrix, snp_positions, threshold=0.05/250000):
    """Calculate power metrics"""
    significant_indices = np.where(pvalues < threshold)[0]
    significant_markers = set(significant_indices)
    n_significant = len(significant_markers)

    # Calculate LD-adjusted metrics
    ld_detected_qtns = set()
    all_associated_markers = set()

    for qtn_idx in known_qtns:
        if qtn_idx >= len(snp_positions) or qtn_idx >= genotype_matrix.shape[1]:
            continue

        # Get QTN position
        qtn_chrom = snp_positions.iloc[qtn_idx]['Chr']
        qtn_pos = snp_positions.iloc[qtn_idx]['Pos']

        # Find candidates within 10Mb
        same_chrom_mask = snp_positions['Chr'] == qtn_chrom
        distance_mask = np.abs(snp_positions['Pos'] - qtn_pos) <= 10000000
        candidates = snp_positions[same_chrom_mask & distance_mask].index.tolist()

        # Calculate LD associations
        associated_markers = [qtn_idx]
        for candidate_idx in candidates:
            if candidate_idx != qtn_idx and candidate_idx < genotype_matrix.shape[1]:
                r_squared = calculate_ld_r_squared(
                    genotype_matrix[:, qtn_idx],
                    genotype_matrix[:, candidate_idx]
                )
                if r_squared >= 0.5:
                    associated_markers.append(candidate_idx)

        all_associated_markers.update(associated_markers)
        if set(associated_markers) & significant_markers:
            ld_detected_qtns.add(qtn_idx)

    # Calculate metrics
    strict_tp = len(set(known_qtns) & significant_markers)
    strict_tpr = strict_tp / len(known_qtns) if len(known_qtns) > 0 else 0
    ld_tpr = len(ld_detected_qtns) / len(known_qtns) if len(known_qtns) > 0 else 0

    false_positives = significant_markers - all_associated_markers
    ld_fdr = len(false_positives) / n_significant if n_significant > 0 else 0

    return {
        'strict_tpr': strict_tpr,
        'ld_tpr': ld_tpr,
        'ld_fdr': ld_fdr,
        'n_significant': n_significant,
        'strict_detected': strict_tp,
        'ld_detected': len(ld_detected_qtns)
    }

def main():
    # Load QTN data
    data_dir = Path("pyMVP/DevTests/validation_data/performance_test_huge2")
    with open(data_dir / "dataset_summary.json") as f:
        dataset_info = json.load(f)
    known_qtns = dataset_info['qtns']['indices']

    # Load genotype and map data
    npz_data = np.load(data_dir / "genotypes.npz")
    genotype_matrix = npz_data['genotypes']
    snp_positions = pd.read_csv(data_dir / "map.csv")

    print("🔬 FarmCPU vs BLINK Power Comparison on huge2 Dataset")
    print("=" * 60)

    traits = [f'trait{i}' for i in range(1, 11)]

    rows = []

    for trait in traits:
        print(f"\n📊 {trait.upper()} Results:")

        # Load FarmCPU results
        farmcpu_file = Path(f"pyMVP/DevTests/gwas_analysis_results/{trait}_farmcpu_full_results.csv")
        farmcpu_df = pd.read_csv(farmcpu_file)
        farmcpu_pvals = farmcpu_df['P-value'].values

        # Load BLINK results (use improved version)
        blink_file = Path(f"test_improved_blink_results/results_{trait}/GWAS_{trait}_all_results.csv")
        blink_df = pd.read_csv(blink_file)
        blink_pvals = blink_df['BLINK_Pvalue'].values

        # Calculate power for both methods
        farmcpu_power = calculate_power(farmcpu_pvals, known_qtns, genotype_matrix, snp_positions)
        blink_power = calculate_power(blink_pvals, known_qtns, genotype_matrix, snp_positions)

        print(f"FarmCPU - TPR: {farmcpu_power['ld_tpr']:.3f} ({farmcpu_power['ld_detected']}/25), FDR: {farmcpu_power['ld_fdr']:.3f}, Sig: {farmcpu_power['n_significant']}")
        print(f"BLINK   - TPR: {blink_power['ld_tpr']:.3f} ({blink_power['ld_detected']}/25), FDR: {blink_power['ld_fdr']:.3f}, Sig: {blink_power['n_significant']}")

        tpr_ratio = blink_power['ld_tpr'] / farmcpu_power['ld_tpr'] if farmcpu_power['ld_tpr'] > 0 else 0
        print(f"BLINK TPR is {tpr_ratio:.2f}x FarmCPU TPR")

        rows.append({
            'trait': trait,
            'farmcpu_ld_tpr': farmcpu_power['ld_tpr'],
            'farmcpu_ld_fdr': farmcpu_power['ld_fdr'],
            'blink_ld_tpr': blink_power['ld_tpr'],
            'blink_ld_fdr': blink_power['ld_fdr'],
            'blink_vs_farmcpu_tpr_ratio': tpr_ratio,
        })

    summary_df = pd.DataFrame(rows)
    print("\n📋 LD-adjusted summary across traits:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

if __name__ == "__main__":
    main()
