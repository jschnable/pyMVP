# PANICLE: Python Algorithms for Nucleotide-phenotype Inference and Chromosome-wide Locus Evaluation

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PANICLE is a **Python package for Genome Wide Association Studies (GWAS)**. It implements GLM, MLM, FarmCPU, and BLINK. PANICLE seeks to achieve speeds comparable or better to other implementations while supporting multiple input data formats, providing multiple quality of life features (native effect marker number testing, leave one chromosome out MLM, calculation of resampling model inclusion probabilities, etc), and allowing modern GWAS algorithms to be natively integrated into python-based analysis and data analysis pipelines and ecosystems.

## Key Features

*   **Multiple Algorithms**: GLM, MLM, FarmCPU, BLINK
*   **Optmizied Performance**:
    *   **Vectorized Loading**: Optional optimized loading via `cyvcf2` (~26x faster VCF loading).
    *   **Binary Caching**: Option to cache genotype data in binary during an initial run, speeds future runs dramatically.
    *   **Decimated Plotting**: Fast Manhattan plots via smart point downsampling.
*   **Supported Genotype Formats**: VCF/BCF, PLINK, HapMap, CSV/TSV.
*   **Robustness**: Graceful handling of missing data.

## Installation

Requires Python 3.7+.

```bash
git clone https://github.com/jschnable/PANICLE.git
cd PANICLE
pip install -e .
```

*Optional dependencies for VCF/PLINK support:*
```bash
pip install -e .[all]
```

### Dependencies

**Core dependencies** (installed automatically):
- `numpy` ≥1.19.0
- `scipy` ≥1.6.0
- `pandas` ≥1.2.0
- `numba` ≥0.50.0 (JIT compilation for performance)
- `scikit-learn` ≥0.24.0 (includes joblib for parallel processing)
- `matplotlib` ≥3.3.0, `seaborn` ≥0.11.0 (plotting)
- `statsmodels` ≥0.12.0
- `h5py` ≥3.0.0, `tables` ≥3.6.0 (HDF5 support)
- `tqdm` ≥4.60.0 (progress bars)

**Optional dependencies**:
- `cyvcf2` ≥0.30.0 — Fast VCF/BCF parsing (~26x faster than pure Python)
- `bed-reader` ≥1.0.0 — PLINK .bed/.bim/.fam format support

## CLI Usage (Quick Start)

The `run_GWAS.py` script provides a command-line interface for batch processing.

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

### Parameters

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

Integrate PANICLE into scripts or Jupyter Notebooks is via the `GWASPipeline` class.

```python
from panicle.pipelines.gwas import GWASPipeline

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
*   **CSV/TSV**: Numeric matrix (rows=samples, cols=markers) + genetic map file with `SNP`, `CHROM`, and `POS` columns (case-insensitive aliases like `Chr`, `Pos` are accepted).
*   **PLINK**: `.bed` + `.bim` + `.fam`.
*   **HapMap**: `.hmp.txt`.

## Tips

1.  **Effective Tests**: Use `--compute-effective-tests` to calculate a less stringent, more accurate Bonferroni threshold based on marker linkage (`Me`).
2.  **Parallelism**: By default, methods run in parallel. Ensure you have enough RAM for the number of concurrent processes (Dataset size × Threads).
3.  **Genotype Subsetting**: If you align or filter samples manually, use `GenotypeMatrix.subset_individuals(...)` to preserve pre-imputed fast paths.

## Documentation & Examples

### Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes
- **[API Reference](docs/api_reference.md)** - Complete API documentation for all functions and classes
- **[Output Files](docs/output_files.md)** - Understanding result file formats and columns

### Interactive Tutorial

- **[Sorghum GWAS Tutorial](docs/sorghum_gwas_tutorial.ipynb)** - Jupyter notebook with complete GWAS workflow using real sorghum data (725 samples, 170K markers)

### Example Scripts

The [`examples/`](examples/) directory contains runnable example scripts with included test data:

| Example | Description |
|---------|-------------|
| [01_basic_gwas.py](examples/01_basic_gwas.py) | Simplest GWAS with GLM |
| [02_mlm_with_structure.py](examples/02_mlm_with_structure.py) | MLM with population structure correction |
| [03_hybrid_mlm.py](examples/03_hybrid_mlm.py) | Hybrid MLM (Wald + LRT) for accurate p-values |
| [04_with_covariates.py](examples/04_with_covariates.py) | Including external covariates |
| [05_reading_results.py](examples/05_reading_results.py) | Analyzing and visualizing results |

Run any example:
```bash
cd examples
python 01_basic_gwas.py
```

## Algorithms

### GLM

General Linear Model for fast single-marker association testing. Uses the Frisch-Waugh-Lovell (FWL) theorem combined with QR decomposition for computational efficiency. The algorithm residualizes the phenotype and genotypes against the covariate matrix (PCs + intercept), then computes per-marker regression statistics in vectorized batches. Missing genotypes are imputed to the per-SNP major allele, matching rMVP behavior. GLM is the fastest method and serves as the workhorse for initial genome scans in both FarmCPU and BLINK.

### MLM

Mixed Linear Model accounting for population structure and cryptic relatedness via a kinship matrix. The model is: y = Xβ + gα + u + e, where u ~ N(0, σ²_g K) captures polygenic effects.

**Key design decisions:**
- **LOCO by default**: Leave-One-Chromosome-Out kinship avoids proximal contamination (testing a marker against a kinship matrix that includes that marker), increasing power to detect true associations.
- **Eigenspace transformation**: Data is transformed via eigendecomposition of the kinship matrix, converting the correlated mixed model into an equivalent weighted least squares problem.
- **REML variance components**: Heritability (h²) is estimated using Brent's method optimization of the REML likelihood, matching rMVP's approach.
- **Wald test**: Uses fast Wald statistics by default. For markers near significance thresholds, consider MLM_Hybrid for more accurate p-values.
- **Performance optimizations**: Numba JIT compilation for critical numerical operations; optional multi-core parallel batch processing via joblib.

**MLM_Hybrid** extends MLM with a two-phase approach: (1) fast Wald screening of all markers, (2) Likelihood Ratio Test (LRT) refinement for markers passing a screening threshold (default p < 1e-4). LRT re-estimates variance components per marker and provides more accurate p-values near genome-wide significance thresholds.

### FarmCPU

Fixed and random model Circulating Probability Unification. FarmCPU iteratively alternates between a fixed-effect model (GLM) and random-effect model to identify associated markers while controlling for polygenic background.

**Algorithm:**
1. **Initial GLM scan**: Test all markers; identify candidates below threshold
2. **Multi-scale binning**: Select representative QTNs using static binning at three scales (500KB, 5MB, 50MB) to capture associations at different LD extents
3. **LD pruning**: Remove redundant QTNs with high correlation (r > 0.7) within chromosomes
4. **Covariate updating**: Add selected pseudo-QTNs as fixed-effect covariates
5. **Iterate**: Re-scan all markers with updated model; repeat until QTN set converges

**Key design decisions:**
- **Bound on QTN count**: Maximum pseudo-QTNs per iteration is √n / √log₁₀(n), preventing model overfitting
- **Reward substitution**: Final p-values for pseudo-QTNs use the minimum p-value observed across all iterations (more conservative than single-iteration estimates)
- **rMVP compatibility**: Binning, thresholding, and early-stop logic match rMVP for reproducibility

FarmCPU Citation: Liu, X., Huang, M., Fan, B., Buckler, E. S., & Zhang, Z. (2016). Iterative usage of fixed and random effect models for powerful and efficient genome-wide association studies. _PLoS genetics_, _12_(2), e1005767.

### BLINK

Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway. BLINK builds on FarmCPU's iterative framework but uses BIC-based model selection to optimize the pseudo-QTN set.

**Algorithm:**
1. **GLM scan**: Identify candidate markers below a significance threshold
2. **LD pruning**: Block-based LD removal (mirrors GAPIT's Blink.LDRemoveBlock)
3. **BIC model selection**: Evaluate nested models with increasing numbers of QTNs; select the model minimizing BIC
4. **Covariate updating**: Include BIC-selected QTNs as covariates
5. **Iterate**: Repeat until Jaccard similarity between consecutive QTN sets exceeds convergence threshold (default 1.0 = identical sets)

**Key design decisions:**
- **BIC strategies**: Multiple evaluation schemes (naive, even, lg, ln, fixed) control the positions tested during forward selection, trading thoroughness for speed
- **FDR option**: Iteration-2 threshold can use FDR-based cutoff instead of Bonferroni
- **Union preservation**: QTNs from previous iterations are preserved through the union step, ensuring stability across iterations
- **Substitution methods**: Final pseudo-QTN statistics can use reward (min p), penalty (max p), mean, median, or onsite strategies

Blink Citation: Huang, M., Liu, X., Zhou, Y., Summers, R. M., & Zhang, Z. (2019). BLINK: a package for the next level of genome-wide association studies with both individuals and markers in the millions. _Gigascience_, _8_(2), giy154.

### Effective Marker Number Estimates

A python based implementation of the effective marker number estimation method implemented in GEC. Accounts for linkage disequilibrium between markers to provide a less conservative multiple testing correction than standard Bonferroni.

GEC citation: Li MX, Yeung JM, Cherny SS, Sham PC. Evaluating the effective numbers of independent tests and significant p-value thresholds in commercial genotyping arrays and public imputation reference datasets. Hum Genet. 2012 May;131(5):747-56.

## Benchmarks

Benchmarks run on sorghum diversity panel data (862 samples, 5,751,024 markers) on Apple M3 Pro.

### Data Loading

Data loading is shared across all algorithms (cached VCF, large dataset):

| Step                | Time    |
|---------------------|---------|
| Genotype loading    | 1.42s   |
| Phenotype loading   | 0.005s  |
| Sample alignment    | 8.42s   |
| PCA (3 components)  | 1.47s   |
| **Total**           | **11.31s**|

*Note: First run includes VCF parsing; subsequent runs use binary cache.*

### Analysis Times

Time to run each algorithm (excludes data loading and result writing):

| Method      | Time    | Notes                              |
|-------------|---------|------------------------------------|
| GLM         | 5.77s   | ~997K markers/second (5M markers)  |
| MLM (LOCO)  | 20.63s  | + 13.34s kinship computation       |
| MLM_Hybrid  | 22.82s  | + 13.34s kinship computation       |
| FarmCPU     | 83.63s  | 10 max iterations                  |
| BLINK       | 54.64s  | 10 max iterations                  |

### Scaling by Marker Count

Performance scaling with 862 samples at varying marker densities (SAP large dataset).
Timings include cached data loading, sample alignment, PCA, and (for MLM/MLM_Hybrid) LOCO kinship.

| Markers    | GLM     | MLM     | MLM_Hybrid | FarmCPU  | BLINK   |
|------------|---------|---------|------------|----------|---------|
| 50,000     | 11.50s  | 12.13s  | 11.75s     | 12.15s   | 11.83s  |
| 500,000    | 11.98s  | 14.36s  | 14.08s     | 18.44s   | 15.16s  |
| 5,000,000  | 18.31s  | 41.68s  | 43.42s     | 86.34s   | 54.50s  |

## License

Distributed under the MIT license. See [LICENSE](LICENSE).

---
**Disclaimer:** This is an independent Python implementation of algorithms drawn from [rMVP](https://github.com/xiaolei-lab/rMVP) and [BLINK](https://doi.org/10.1093/gigascience/giy154). Any errors are mine alone.
