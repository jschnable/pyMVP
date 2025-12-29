# pyMVP: Python Based GWAS Pipeline

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

pyMVP is a high-performance **Python package for Genome Wide Association Studies (GWAS)**. It implements **GLM** (General Linear Model), **MLM** (Mixed Linear Model, with optional LOCO), **Hybrid MLM** (Wald screen + LRT refinement), **FarmCPU**, **BLINK**, and **FarmCPU Resampling (RMIP)**, optimized for speed and huge datasets.

It features a **vectorized VCF loader**, **automatic binary caching**, **parallel execution**, and **decimated plotting**, making it significantly faster than unoptimized implementations while maintaining the flexibility of Python.

## Key Features

*   **Algorithms**: GLM, MLM (LOCO when a map is available), Hybrid MLM (`MLM_Hybrid`), FarmCPU, BLINK, FarmCPUResampling (RMIP).
*   **High Performance**:
    *   **Vectorized Loading**: `cyvcf2` + NumPy optimization (~26x faster VCF loading).
    *   **Binary Caching**: First run caches genotypes; subsequent runs load instantly (~1.5s).
    *   **Parallel Processing**: Concurrent execution of GWAS methods across traits.
    *   **Decimated Plotting**: Fast Manhattan plots via smart point downsampling.
*   **Data Support**: VCF/BCF, PLINK, HapMap, CSV/TSV.
*   **Automatic Population Structure**: Step-wise PCA and Kinship matrix calculation (VanRaden).
*   **Robustness**: Graceful handling of missing data and numerical instability.
*   **Hybrid MLM**: Two-stage Wald screen + LRT refinement for top hits.

## Installation

Requires Python 3.7+.

```bash
git clone https://github.com/jschnable/pyMVP.git
cd pyMVP
pip install -e .
```

*Optional dependencies for VCF/PLINK support:*
```bash
pip install -e .[all]
```

## CLI Usage (Quick Start)

The `run_GWAS.py` script provides a powerful command-line interface for batch processing.

```bash
python scripts/run_GWAS.py \
  --phenotype data/phenotype.csv \
  --genotype data/genotypes.vcf.gz \
  --traits Trait1,Trait2 \
  --methods GLM,MLM,MLM_Hybrid \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs manhattan qq significant_marker_pvalues \
  --output ./results
```

### Important Parameters

| Argument | Description | Default |
| :--- | :--- | :--- |
| **`--phenotype`** | Path to phenotype CSV/TSV (must contain ID column). | **Required** |
| **`--genotype`** | Path to genotype VCF/BCF/CSV. | **Required** |
| **`--map`** | Optional map file (SNP, CHROM, POS). Recommended for numeric CSV/TSV and LOCO/Hybrid methods. | None |
| **`--format`** | Genotype format override: `vcf`, `plink`, `hapmap`, `csv`, `tsv`, `numeric`. | Auto |
| **`--traits`** | Comma-separated list of columns to analyze. | All numeric |
| **`--methods`** | GWAS methods: `GLM`, `MLM`, `MLM_Hybrid`, `FarmCPU`, `BLINK`, `FarmCPUResampling`. | All |
| **`--n-pcs`** | Number of Principal Components for population structure. | 3 |
| **`--compute-effective-tests`** | Calculate Effective SNP Number (Me) for Bonferroni correction. | False |
| **`--use-effective-tests`** | Use Me (if available) for Bonferroni correction. | False |
| **`--alpha`** | Significance level (e.g., 0.05). Threshold = `alpha / Me` (or `M`). | 0.05 |
| **`--significance`** | Fixed p-value threshold (overrides Bonferroni). | None |
| **`--n-eff`** | Effective number of markers (overrides Me). | None |
| **`--covariates`** | External covariate file. | None |
| **`--covariate-columns`** | Comma-separated covariate column names. | All except ID |
| **`--covariate-id-column`** | ID column name in covariate file. | ID |
| **`--max-iterations`** | Max iterations for FarmCPU/BLINK. | 10 |
| **`--max-genotype-dosage`** | Max dosage (e.g., 2 for diploid). | 2.0 |
| **`--outputs`** | Outputs to generate: `all_marker_pvalues`, `significant_marker_pvalues`, `manhattan`, `qq`. | All |

Other useful filters:
- `--max-missing` (default 1.0), `--min-maf` (default 0.0)
- `--drop-monomorphic` / `--keep-monomorphic`
- `--snps-only`, `--no-split-multiallelic`

## Python API Usage

The recommended way to integrate pyMVP into your scripts or Jupyter Notebooks is via the `GWASPipeline` class.

```python
from pymvp.pipelines.gwas import GWASPipeline

# 1. Initialize
pipeline = GWASPipeline(output_dir="./results")

# 2. Load Data (Auto-caches for speed)
pipeline.load_data(
    phenotype_file="data/phenotype.csv",
    genotype_file="data/genotype.vcf.gz",
    map_file="data/genotype.map",  # Optional unless format lacks positions
    trait_columns=["Height", "Yield"],
    loader_kwargs={'compute_effective_tests': True}  # Enable Me calculation
)

# 3. Pre-process
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=5)

# 4. Run Analysis (runs in parallel by default)
pipeline.run_analysis(
    methods=['GLM', 'MLM', 'MLM_Hybrid'],
    hybrid_params={'screen_threshold': 1e-4},
    alpha=0.05
)
```

## Input Formats

### Phenotype & Covariates
CSV or TSV files with an **ID column** (e.g., `ID`, `Taxa`, `Sample`) and numeric columns for traits/covariates.

### Genotype
*   **VCF/BCF**: `.vcf`, `.vcf.gz`, `.bcf` (Preferred for performance).
*   **CSV/TSV**: Numeric matrix (rows=samples, cols=markers).
*   **PLINK**: `.bed` + `.bim` + `.fam`.
*   **HapMap**: `.hmp.txt`.

### Genetic Map (Optional but recommended)
CSV/TSV with `SNP`, `CHROM`, and `POS` columns (case-insensitive aliases like `Chr`, `Pos` are accepted).
Recommended for numeric genotype matrices and for LOCO-based methods like `MLM_Hybrid`.

## Performance Tips

1.  **First Run**: The first run on a large VCF takes longer (to parse and cache). **Subsequent runs** will use the `.npy` sidecar cache automatically and be extremely fast.
2.  **Effective Tests**: Use `--compute-effective-tests` to calculate a less stringent, more accurate Bonferroni threshold based on marker linkage (`Me`).
3.  **Parallelism**: By default, methods run in parallel. Ensure you have enough RAM for the number of concurrent processes (Dataset size × Threads).

## Benchmarks

Average wall-clock time (1,000 samples, 250,000 markers, M3 Pro CPU):

| Method  | pyMVP | rMVP (R) |
|---------|-------|----------|
| GLM     | 0.6s  | 4.5s     |
| MLM     | 7.8s  | 21.9s    |
| FarmCPU | 8.9s  | 20.5s    |

*Note: pyMVP optimized implementation (v2.0) is typically 2-4x faster than rMVP.*

## License

Distributed under the MIT license. See [LICENSE](LICENSE).

---
**Disclaimer:** This is an independent Python implementation inspired by [rMVP](https://github.com/xiaolei-lab/rMVP) and [BLINK](https://doi.org/10.1093/gigascience/giy154).
