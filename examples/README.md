# PANICLE Examples

This directory contains example scripts demonstrating different PANICLE workflows.

## Overview

| Example | Description | Difficulty | Time |
|---------|-------------|------------|------|
| [01_basic_gwas.py](01_basic_gwas.py) | Simplest GWAS with GLM | Beginner | 2 min |
| [02_mlm_with_structure.py](02_mlm_with_structure.py) | MLM with population structure correction | Beginner | 5 min |
| [03_hybrid_mlm.py](03_hybrid_mlm.py) | Hybrid MLM for increased power | Intermediate | 10 min |
| [04_with_covariates.py](04_with_covariates.py) | Including external covariates | Intermediate | 5 min |
| [05_reading_results.py](05_reading_results.py) | Analyzing and visualizing results | Intermediate | 5 min |

## Running the Examples

### Prerequisites

**Example data files are provided!** This directory includes:
- `example_phenotypes.csv`: Plant height data for 738 sorghum lines
- `example_genotypes.vcf.gz`: 1,000 SNP markers (includes known significant QTLs)
- `example_covariates.csv`: Flowering time data for use as a covariate

See [EXAMPLE_DATA.md](EXAMPLE_DATA.md) for details about the dataset.

All examples use these file names by default. You can also use your own data or the full sorghum dataset in `../sorghum_data/`.

### Running an Example

```bash
# Make sure PANICLE is installed
cd /path/to/PANICLE
pip install -e .

# Run an example
cd examples
python 01_basic_gwas.py
```

Or with custom data:

```bash
python 01_basic_gwas.py \
    --phenotype my_phenos.csv \
    --genotype my_genos.vcf.gz
```

## Example Details

### 01: Basic GWAS
**What it does:** Runs a simple GLM analysis on one trait
**When to use:** Quick screening, no population structure
**Output:** Manhattan plot, QQ plot, results CSV

```python
pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data('phenos.csv', 'genos.vcf.gz')
pipeline.align_samples()
pipeline.run_analysis(traits=['Height'], methods=['GLM'])
```

### 02: MLM with Population Structure
**What it does:** Uses kinship matrix and PCs to control for population structure
**When to use:** Diverse populations, related individuals
**Output:** Corrected association results

```python
pipeline.compute_population_structure(n_pcs=5, calculate_kinship=True)
pipeline.run_analysis(traits=['Height'], methods=['MLM'])
```

### 03: Hybrid MLM
**What it does:** Combines fast Wald screening with exact LRT for top hits
**When to use:** Need accurate p-values with minimal runtime cost
**Output:** LRT-refined results for significant markers

```python
pipeline.run_analysis(
    traits=['Height'],
    methods=['MLM_Hybrid'],
    hybrid_params={'screen_threshold': 1e-4}
)
```

### 04: With Covariates
**What it does:** Analyzes plant height while controlling for flowering time
**When to use:** When other measured traits may confound your analysis
**Output:** Results adjusted for covariates

```python
pipeline.load_data(
    phenotype_file='example_phenotypes.csv',
    genotype_file='example_genotypes.vcf.gz',
    covariate_file='example_covariates.csv',
    covariate_columns=['DaysToFlower']
)
```

### 05: Reading Results
**What it does:** Shows how to load, filter, and visualize results
**When to use:** Post-analysis exploration
**Output:** Custom plots, filtered results, exports

```python
import pandas as pd
results = pd.read_csv('results/GWAS_Height_all_results.csv')
sig_snps = results[results['MLM_P'] < 0.05/len(results)]
```

## Using Your Own Data

To adapt these examples for your data:

1. **Phenotype file format:**
```csv
ID,Trait1,Trait2,Trait3
Sample1,1.5,20,3.2
Sample2,1.8,22,3.5
```

2. **Genotype file:** VCF, HapMap, or Plink format

3. **Covariate file format:**
```csv
ID,Field,Year,Treatment
Sample1,A,2023,Control
Sample2,B,2023,Treatment1
```

4. **Modify the script:**
```python
pipeline.load_data(
    phenotype_file='YOUR_PHENOS.csv',
    genotype_file='YOUR_GENOS.vcf.gz'
)

pipeline.run_analysis(
    traits=['YOUR_TRAIT'],
    methods=['MLM']
)
```

## Troubleshooting

### "No common individuals"
- Check that sample IDs match exactly between files (case-sensitive)
- Verify first column in phenotype file is 'ID'

### "Kinship matrix missing"
- Run `pipeline.compute_population_structure(calculate_kinship=True)` before MLM

### Very slow analysis
- Start with GLM for quick results
- Use smaller sample of markers for testing
- Consider Hybrid MLM instead of full LRT

## Next Steps

- Read the [Quick Start Guide](../docs/quickstart.md)
- See [API Reference](../docs/api_reference.md) for detailed documentation
- Check [Output Files](../docs/output_files.md) for file format specifications
- Try the interactive [Jupyter notebook](../docs/hybrid_mlm_demo.ipynb)

## Getting Help

- Check documentation in `docs/`
- Open an issue on GitHub
- Email: [your contact]
