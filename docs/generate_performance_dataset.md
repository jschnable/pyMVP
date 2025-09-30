# GWAS Dataset Generation Guide

## Overview

The `generate_performance_dataset.py` script creates realistic simulated GWAS datasets for performance testing, method validation, and benchmarking. It generates genotype matrices, phenotypes with known QTN effects, and comprehensive metadata to facilitate controlled testing of GWAS methods.

## Key Features

- **Realistic Population Structure**: Wright-Fisher model with configurable FST-based population differentiation
- **Linkage Disequilibrium**: Distance-based LD decay within chromosomes
- **Flexible MAF Spectra**: U-shaped (default), centered Beta, or empirical site frequency spectrum
- **MAF-Effect Coupling**: Configurable relationship between allele frequency and effect size
- **Heritability Control**: Target trait heritability with environmental noise
- **Study Designs**: Outbred (human-like) or inbred (crop-like) populations
- **Multiple Output Formats**: NumPy compressed arrays, CSV, Parquet for compatibility

## Installation

The script is included with pyMVP. For optimal performance, install optional dependencies:

```bash
pip install numba tqdm  # ~5-10x speedup + progress bars
```

## Quick Start

### Basic Usage

Generate a small test dataset (200 samples, 10K SNPs):

```bash
python scripts/generate_performance_dataset.py \
  --n-samples 200 \
  --n-snps 10000 \
  --n-qtns 50 \
  --output-dir test_dataset
```

### Large-Scale Dataset

Generate a realistic GWAS dataset (2000 samples, 250K SNPs):

```bash
python scripts/generate_performance_dataset.py \
  --n-samples 2000 \
  --n-snps 250000 \
  --n-qtns 100 \
  --n-traits 10 \
  --output-dir large_gwas_dataset
```

### "Hard Mode" Benchmark Scenario

Generate a challenging dataset with rare variants and low heritability:

```bash
python scripts/generate_performance_dataset.py \
  --scenario hard_mode \
  --output-dir hard_benchmark
```

This preset configures:
- U-shaped MAF spectrum (enriched for rare variants)
- MAF-effect coupling (κ = 0.5, rare variants have larger effects)
- Low heritability (h² = 0.2)
- Highly inbred design (F = 0.98)
- 200 causal variants

## Configuration Parameters

### Core Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-samples` | 2000 | Number of individuals/samples |
| `--n-snps` | 200000 | Number of genetic markers (SNPs) |
| `--n-qtns` | 100 | Number of causal variants (QTNs) |
| `--n-chromosomes` | 10 | Number of chromosomes |
| `--n-traits` | 10 | Number of phenotypic traits |
| `--seed` | 42 | Random seed for reproducibility |

### Population Structure

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population-structure` | True | Enable population structure simulation |
| `--population-sizes` | "600,800,600" | Comma-separated population sizes |
| `--fst` | 0.05 | Population differentiation (Wright's FST) |

**What this controls**: Simulates genetic ancestry groups with different allele frequencies. FST values:
- 0.01-0.05: Little differentiation (European populations)
- 0.05-0.15: Moderate differentiation (continental groups)
- 0.15-0.25: Large differentiation (distinct ancestries)

### MAF Spectrum and Genetic Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--maf-spectrum` | `beta_u` | MAF distribution: `beta_u` (U-shaped), `beta_centered`, `empirical_sfs` |
| `--maf-beta-a` | 0.5 | Beta distribution α parameter |
| `--maf-beta-b` | 0.5 | Beta distribution β parameter |
| `--empirical-sfs-file` | None | Path to empirical SFS CSV (frequency, count columns) |
| `--effect-maf-coupling-kappa` | 0.25 | MAF-effect coupling: \|β\| ∝ [MAF(1-MAF)]^(-κ/2) |

**MAF Spectrum Options**:
- **`beta_u`** (default): U-shaped distribution Beta(0.5, 0.5) - enriched for rare and common variants
- **`beta_centered`**: Centered distribution Beta(2, 2) - enriched for intermediate frequencies
- **`empirical_sfs`**: Load real-world site frequency spectrum from CSV

**MAF-Effect Coupling** (κ parameter):
- **κ = 0**: No coupling, effects independent of MAF
- **κ = 0.25** (default): Mild coupling, rare variants ~1.5x larger effects
- **κ = 0.5**: Strong coupling, rare variants ~3x larger effects
- **κ = 1.0**: Very strong coupling, mimics strong negative selection

### Heritability and Effect Sizes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--heritability` | 0.7 | Target broad-sense heritability (h²) |
| `--effect-size-dist` | `normal` | Effect distribution: `normal` or `t` |
| `--effect-size-scale` | 0.3 | Scale parameter for effect size distribution |

**Heritability**: Proportion of phenotypic variance due to genetics (0-1 scale)
- High (0.7-0.9): Strongly genetic traits (e.g., height, flowering time)
- Moderate (0.4-0.6): Mixed genetic/environmental (e.g., yield, disease resistance)
- Low (0.1-0.3): Mostly environmental (e.g., stress response, complex diseases)

### Study Design

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--study-design` | `inbred` | Design: `outbred` (human-like) or `inbred` (crop-like) |
| `--inbreeding-f` | 0.95 | Inbreeding coefficient for inbred design |

**Study Design Options**:
- **`outbred`**: Human-like populations, high heterozygosity (H_obs ≈ H_exp)
- **`inbred`**: Crop-like populations (maize, wheat, rice), low heterozygosity, F ≈ 0.95

### Data Quality

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--missing-data-rate` | 0.02 | Proportion of missing genotype data (0-1) |

## Output Files

The script generates a complete dataset directory with the following files:

### Primary Data Files

| File | Description |
|------|-------------|
| `genotypes.npz` | Compressed NumPy array (fast loading, recommended) |
| `genotype_numeric.parquet` | Parquet format (efficient, medium datasets) |
| `genotype_numeric.csv` | CSV format (compatibility, small/medium datasets) |
| `phenotype.csv` | Observed phenotypes (Taxa + trait columns) |
| `phenotype_null.csv` | Null phenotypes (all zeros, negative control) |
| `map.csv` | SNP map (SNP, Chr, Pos columns) |

### Metadata and Truth Files

| File | Description |
|------|-------------|
| `dataset_summary.json` | Comprehensive metadata (parameters, dimensions, QTNs, design metrics) |
| `design_summary.tsv` | Study design metrics (heterozygosity, inbreeding coefficient) |
| `truth.tsv` | Ground truth QTN positions and effects |
| `true_qtns.csv` | Detailed QTN information with map coordinates |
| `sample_names.csv` | Sample identifiers |
| `snp_names.csv` | SNP identifiers |

### Loading Generated Data

#### With pyMVP

```python
from pymvp.data.loaders import load_genotype_file, load_phenotype_file
from pathlib import Path

# Load genotype data (automatically detects .npz format)
geno_matrix, individual_ids, snp_map = load_genotype_file(
    "test_dataset/genotypes.npz"
)

# Load phenotype data
phenotype_df = load_phenotype_file(
    "test_dataset/phenotype.csv",
    trait_columns=["trait1", "trait2"]
)

# Load truth data for validation
import pandas as pd
truth = pd.read_csv("test_dataset/truth.tsv", sep="\t")
```

#### With NumPy

```python
import numpy as np

# Load compressed genotype data
data = np.load("test_dataset/genotypes.npz")
genotypes = data['genotypes']      # Shape: (n_samples, n_snps)
sample_ids = data['sample_ids']    # Sample names
snp_ids = data['snp_ids']          # SNP names

print(f"Genotypes: {genotypes.shape}")
print(f"Encoding: 0=ref/ref, 1=ref/alt, 2=alt/alt, -9=missing")
```

## Built-in Assumptions and Limitations

### Genetic Model Assumptions

1. **Diploid Organisms**: All genotypes are coded as 0, 1, 2 (dosage of alternate allele)
2. **Bi-allelic SNPs**: Each marker has exactly two alleles (ref and alt)
3. **Additive Effects**: QTN effects are purely additive (no dominance or epistasis)
4. **Linear Genetic Model**: Phenotype = Σ(genotype × effect) + environmental noise
5. **Trait Independence**: Each trait has independent environmental variance (though QTNs may be shared)

### Population Structure Assumptions

1. **Wright-Fisher Model**: Populations evolve under standard Wright-Fisher assumptions (constant size, random mating within populations, no selection)
2. **FST-Based Differentiation**: Allele frequency differences follow Balding-Nichols FST model
3. **Fixed Population Sizes**: No demographic changes over time
4. **Migration Pattern**: No ongoing migration after initial divergence

### Linkage Disequilibrium Model

1. **Distance-Dependent Decay**: LD strength = exp(-recombination_rate × 100)
2. **Recombination Rate**: 1 cM per Mb (standard for many species)
3. **Within-Chromosome Only**: No LD between chromosomes (perfect linkage equilibrium)
4. **LD Blocks**: Clustering pattern creates ~30% markers in tight LD blocks (<10kb spacing)

### Limitations

- **No Dominance**: All QTN effects are additive only
- **No Epistasis**: No gene-gene interactions
- **No Selection**: Allele frequencies follow neutral evolution
- **No Rare Structural Variants**: Only SNPs, no CNVs, inversions, etc.
- **Simplified LD**: Real genomes have more complex recombination hotspots
- **No Genotyping Error**: Missing data is truly missing, not miscalled

## Example Workflows

### Benchmarking Power and FDR

Generate datasets with known QTNs to evaluate method performance:

```bash
# Generate 10 replicate datasets
for i in {1..10}; do
  python scripts/generate_performance_dataset.py \
    --n-samples 1000 \
    --n-snps 100000 \
    --n-qtns 50 \
    --heritability 0.5 \
    --seed $((1000 + i)) \
    --output-dir power_test_rep${i}
done

# Run GWAS on each dataset
for i in {1..10}; do
  python scripts/run_GWAS.py \
    -p power_test_rep${i}/phenotype.csv \
    -g power_test_rep${i}/genotypes.npz \
    --methods GLM,MLM,FarmCPU,BLINK \
    -o power_results_rep${i}
done

# Compare results to truth.tsv to calculate power and FDR
```

### Testing Population Structure Correction

Generate datasets with strong stratification:

```bash
python scripts/generate_performance_dataset.py \
  --n-samples 2000 \
  --population-sizes "500,500,500,500" \
  --fst 0.15 \
  --n-snps 50000 \
  --n-qtns 30 \
  --output-dir stratification_test

# Test without PCs (expect inflation)
python scripts/run_GWAS.py \
  -p stratification_test/phenotype.csv \
  -g stratification_test/genotypes.npz \
  --methods GLM \
  --n-pcs 0 \
  -o results_no_correction

# Test with PCs (expect proper control)
python scripts/run_GWAS.py \
  -p stratification_test/phenotype.csv \
  -g stratification_test/genotypes.npz \
  --methods GLM \
  --n-pcs 10 \
  -o results_with_pcs
```

### Heritability Sweep

Test method performance across heritability range:

```bash
for h2 in 0.1 0.3 0.5 0.7 0.9; do
  python scripts/generate_performance_dataset.py \
    --n-samples 1000 \
    --n-snps 50000 \
    --heritability ${h2} \
    --seed 42 \
    --output-dir h2_${h2}
done
```

### Rare Variant Architecture

Test methods on datasets enriched for rare causal variants:

```bash
python scripts/generate_performance_dataset.py \
  --maf-spectrum beta_u \
  --effect-maf-coupling-kappa 0.5 \
  --n-qtns 200 \
  --heritability 0.3 \
  --output-dir rare_variant_test
```

## Performance Considerations

### Speed Optimization

- **Numba JIT**: Install `numba` for ~5-10x speedup in genotype generation
- **Batch Size**: Default settings are optimized for most hardware
- **File Format**:
  - Use `.npz` for fastest loading (seconds)
  - Skip CSV generation for very large datasets (>200M genotypes)

### Memory Requirements

Approximate memory usage (genotype matrix storage):

| Samples | SNPs | Memory (int8) | Memory (CSV) |
|---------|------|---------------|--------------|
| 1,000 | 100K | ~100 MB | ~500 MB |
| 2,000 | 250K | ~500 MB | ~2.5 GB |
| 5,000 | 500K | ~2.5 GB | ~12 GB |
| 10,000 | 1M | ~10 GB | ~50 GB |

**Recommendations**:
- For datasets >200M genotypes, use only `.npz` and `.parquet` formats
- Enable Numba for datasets >100K SNPs
- Consider multiple smaller chromosomes for very large datasets

### Typical Generation Times

On modern hardware (Apple M3 Pro, OpenBLAS):

| Configuration | Genotypes | Phenotypes | I/O | Total |
|---------------|-----------|------------|-----|-------|
| 1K × 100K | 8s | 0.5s | 2s | ~11s |
| 2K × 250K | 45s | 1s | 8s | ~54s |
| 5K × 500K | 180s | 3s | 30s | ~213s |

## Validation and Quality Control

The script includes built-in validation metrics in `dataset_summary.json`:

```json
{
  "design_summary": {
    "observed_heterozygosity": 0.021,
    "expected_heterozygosity": 0.245,
    "inbreeding_coefficient": 0.914,
    "study_design": "inbred"
  },
  "qtns": {
    "effect_range": [-1.234, 1.456],
    "indices": [45, 892, 1203, ...]
  },
  "dimensions": {
    "n_samples": 2000,
    "n_snps": 250000,
    "n_qtns": 100
  }
}
```

**Quality Checks**:
1. **Inbreeding Coefficient**: Should match `--inbreeding-f` for inbred designs
2. **Heterozygosity**: Should be ~0 for inbred, ~2p(1-p) for outbred
3. **Missing Data Rate**: Should match `--missing-data-rate`
4. **QTN Count**: Should match `--n-qtns`

## Troubleshooting

### Issue: Slow Generation

**Solution**: Install Numba: `pip install numba`

### Issue: Out of Memory

**Solutions**:
- Reduce `--n-samples` or `--n-snps`
- Generate multiple smaller datasets
- Skip CSV generation (edit script line 944)

### Issue: Low MAF-Effect Correlation

**Check**: Verify `effect_maf_coupling_kappa > 0` in config
**Note**: Correlation is stochastic; expect r > 0.4 for κ = 0.5

### Issue: Heritability Mismatch

**Explanation**: Realized heritability varies due to:
- Random QTN selection
- Population structure effects
- Sample size

Typical variation: ±0.05 from target

## Citation and References

If using this simulation tool for publications, please cite:

- **pyMVP**: This Python implementation of MVP for GWAS analysis
- **rMVP**: Yin et al. (2021), original R implementation
- **Population Genetics Model**: Balding & Nichols (1995) FST model

### Related Documentation

- Main pyMVP README: `../README.md`
- Running GWAS: `run_GWAS.py --help`
- Input formats: See README Input Formats section
