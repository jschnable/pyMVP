# pyMVP: Python Based GWAS

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

pyMVP is a high-performance Python package for Genome Wide Association Studies (GWAS). It implements **GLM** (General Linear Model), **MLM** (Mixed Linear Model), **FarmCPU**, and **BLINK**.

It offers a comprehensive **Python API** for interactive analysis (e.g., in Jupyter Notebooks) and a **CLI** for batch processing.

**Performance:** pyMVP's GLM implementation is optimized using FWL-QR decomposition, making it significantly faster than traditional approaches.

### Key Features
*   **Algorithms**: GLM, MLM, FarmCPU, BLINK, FarmCPU-Resampling (RMIP).
*   **Data Support**: VCF/BCF, PLINK, HapMap, CSV/TSV.
*   **Automatic Population Structure**: PCA and Kinship matrix calculation.
*   **Visualization**: Manhattan and QQ plots.
*   **Modular Architecture**: Reusable `GWASPipeline` class.

## Index

- [Installation](#installation)
- [Python API Usage](#python-api-usage)
- [CLI Usage](#cli-usage)
- [Input Formats](#input-formats)
- [Benchmarks](#benchmarks)
- [License](#license)

## Installation

Requirements: Python 3.7+

```bash
git clone https://github.com/jschnable/pyMVP.git
cd pyMVP
pip install -e .
```

*Optional dependencies for VCF/PLINK:*
```bash
pip install -e .[all]
```

## Python API Usage

The recommended way to use pyMVP is via the `GWASPipeline` class. This allows full control over the analysis steps within Python scripts or Notebooks.

```python
from pymvp.pipelines.gwas import GWASPipeline

# 1. Initialize Pipeline
pipeline = GWASPipeline(output_dir="./results")

# 2. Load Data
pipeline.load_data(
    phenotype_file="data/phenotype.csv",
    genotype_file="data/genotype.vcf.gz",
    trait_columns=["Height", "Yield"]  # Optional: specific traits
)

# 3. Quality Control & Alignment
pipeline.align_samples()

# 4. Population Structure (PCA + Kinship)
pipeline.compute_population_structure(n_pcs=3)

# 5. Run Analysis
pipeline.run_analysis(
    methods=['GLM', 'MLM', 'FarmCPU'],
    alpha=0.05
)
```

## CLI Usage

The backward-compatible CLI wrapper `run_GWAS.py` is available for terminal use.

**Basic Run:**
```bash
python scripts/run_GWAS.py \
  --phenotype data/phenotype.csv \
  --genotype data/geno.vcf.gz \
  --methods GLM,MLM,FarmCPU \
  --output ./results
```

**Common Options:**
*   `--methods`: Comma-separated list (default: GLM,MLM,FarmCPU,BLINK).
*   `--n-pcs`: Number of Principal Components (default: 3).
*   `--covariates`: External covariate file (CSV/TSV).
*   `--alpha`: p-value threshold for Bonferroni correction (default: 0.05).
*   `--max-iterations`: Max iterations for FarmCPU/BLINK (default: 10).

## Input Formats

### Phenotype & Covariates
CSV or TSV files with an ID column (e.g., `ID`, `Taxa`, `Sample`) and numeric columns for traits/covariates.

### Genotype
*   **VCF/BCF**: `.vcf`, `.vcf.gz`, `.bcf`.
*   **CSV/TSV**: Numeric matrix (rows=samples, cols=markers).
*   **PLINK**: `.bed` + `.bim` + `.fam`.
*   **HapMap**: `.hmp.txt`.

## Benchmarks

Average wall-clock time (1,000 samples, 250,000 markers, M3 Pro CPU):

| Method  | pyMVP | rMVP (R) |
|---------|-------|----------|
| GLM     | 0.6s  | 4.5s     |
| MLM     | 7.8s  | 21.9s    |
| FarmCPU | 8.9s  | 20.5s    |

*Note: pyMVP GLM outputs match rMVP exactly. FarmCPU reports similarly but may have minor discrepancies.*

## License

Distributed under the MIT license. See [LICENSE](LICENSE).

---
**Disclaimer:** This is an independent Python implementation inspired by [rMVP](https://github.com/xiaolei-lab/rMVP) and [BLINK](https://doi.org/10.1093/gigascience/giy154).
