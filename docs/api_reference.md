# pyMVP API Reference

Complete reference for the pyMVP package.

## Table of Contents
- [GWASPipeline](#gwaspipeline)
- [Association Methods](#association-methods)
- [Data Loaders](#data-loaders)
- [Utility Functions](#utility-functions)

---

## GWASPipeline

The main high-level interface for running GWAS analyses.

### Class: `pymvp.pipelines.gwas.GWASPipeline`

```python
GWASPipeline(output_dir='./GWAS_results')
```

**Parameters:**
- `output_dir` (str): Directory where results and plots will be saved. Default: `'./GWAS_results'`

**Attributes:**
- `genotype_matrix` (GenotypeMatrix): Aligned genotype data (n_individuals × n_markers)
- `geno_map` (GenotypeMap): Genetic map with SNP information (ID, chromosome, position)
- `phenotype_df` (DataFrame): Aligned phenotype data with 'ID' column + trait columns
- `covariate_df` (DataFrame): External covariates (if loaded)
- `pcs` (ndarray): Principal components (n_individuals × n_pcs)
- `kinship` (ndarray): Kinship matrix (n_individuals × n_individuals)
- `output_dir` (Path): Output directory path
- `effective_tests_info` (dict): Effective number of independent tests (if computed)

**Example:**
```python
from pymvp.pipelines.gwas import GWASPipeline

pipeline = GWASPipeline(output_dir='./my_results')
```

---

### Method: `load_data()`

Load and validate phenotype, genotype, and optional covariate data.

```python
pipeline.load_data(
    phenotype_file,
    genotype_file,
    map_file=None,
    genotype_format=None,
    trait_columns=None,
    covariate_file=None,
    covariate_columns=None,
    covariate_id_column='ID',
    loader_kwargs=None
)
```

**Parameters:**
- `phenotype_file` (str): Path to phenotype CSV file (first column must be 'ID' or individual IDs)
- `genotype_file` (str): Path to genotype file (VCF, HapMap, or Plink format)
- `map_file` (str, optional): Override genetic map file. If None, map is extracted from genotype file
- `genotype_format` (str, optional): Format of genotype file ('vcf', 'hapmap', 'plink'). Auto-detected if None
- `trait_columns` (list, optional): Which phenotype columns to load. If None, loads all numeric columns
- `covariate_file` (str, optional): Path to external covariate CSV file
- `covariate_columns` (list, optional): Which covariate columns to use
- `covariate_id_column` (str): Column name for individual IDs in covariate file. Default: `'ID'`
- `loader_kwargs` (dict, optional): Additional arguments for genotype loader:
  - `compute_effective_tests` (bool): Calculate effective number of independent tests
  - `effective_test_kwargs` (dict): Parameters for effective test calculation

**Side Effects:**
- Sets `self.phenotype_df`, `self.genotype_matrix`, `self.geno_map`
- Sets `self.covariate_df` if covariate file provided
- Sets `self.effective_tests_info` if `compute_effective_tests=True`

**Example:**
```python
pipeline.load_data(
    phenotype_file='phenotypes.csv',
    genotype_file='genotypes.vcf.gz',
    covariate_file='field_locations.csv',
    covariate_columns=['Field', 'Year']
)
```

---

### Method: `align_samples()`

Match and align individuals between phenotype, genotype, and covariate datasets.

```python
pipeline.align_samples()
```

**Parameters:** None

**Side Effects:**
- Updates `self.phenotype_df` with only matched individuals
- Updates `self.genotype_matrix` with only matched individuals
- Updates `self.covariate_df` with only matched individuals (if applicable)
- Prints summary of sample matching

**Raises:**
- `ValueError`: If no common individuals found between datasets

**Example:**
```python
pipeline.align_samples()
# Output:
#    Original phenotypes: 1000
#    Original genotypes: 800
#    Matched Intersection: 750
```

---

### Method: `compute_population_structure()`

Calculate principal components and/or kinship matrix for population structure correction.

```python
pipeline.compute_population_structure(
    n_pcs=3,
    calculate_kinship=True
)
```

**Parameters:**
- `n_pcs` (int): Number of principal components to compute. Default: 3. Set to 0 to skip PCA
- `calculate_kinship` (bool): Whether to calculate kinship matrix. Default: True

**Side Effects:**
- Sets `self.pcs` (n_individuals × n_pcs array)
- Sets `self.pc_names` (list of PC names: ['PC1', 'PC2', ...])
- Sets `self.kinship` (n_individuals × n_individuals matrix) if `calculate_kinship=True`

**Notes:**
- PCs are automatically used as covariates in subsequent analyses
- Kinship matrix is required for MLM, FarmCPU, and BLINK methods
- Uses VanRaden (2008) method for kinship calculation

**Example:**
```python
# Compute 5 PCs and kinship matrix
pipeline.compute_population_structure(n_pcs=5, calculate_kinship=True)

# Skip PCA, only compute kinship
pipeline.compute_population_structure(n_pcs=0, calculate_kinship=True)
```

---

### Method: `run_analysis()`

Run GWAS analysis using specified methods and traits.

```python
pipeline.run_analysis(
    traits=None,
    methods=['GLM', 'MLM'],
    max_iterations=10,
    significance=None,
    alpha=0.05,
    n_eff=None,
    use_effective_tests=True,
    max_genotype_dosage=2.0,
    farmcpu_params=None,
    blink_params=None,
    hybrid_params=None,
    outputs=['all_marker_pvalues', 'significant_marker_pvalues', 'manhattan', 'qq']
)
```

**Parameters:**
- `traits` (list, optional): Which traits to analyze. If None, analyzes all numeric columns in phenotype data
- `methods` (list): GWAS methods to run. Options:
  - `'GLM'`: General Linear Model (fast, no population structure correction)
  - `'MLM'`: Mixed Linear Model (accounts for population structure via kinship)
  - `'MLM_Hybrid'`: Hybrid MLM (Wald screen + LRT refinement for top hits)
  - `'FarmCPU'`: Fixed and random model Circulating Probability Unification
  - `'BLINK'`: Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway
- `max_iterations` (int): Maximum iterations for iterative methods (FarmCPU, BLINK). Default: 10
- `significance` (float, optional): Fixed p-value threshold. If None, uses Bonferroni correction
- `alpha` (float): Significance level for Bonferroni correction. Default: 0.05
- `n_eff` (int, optional): Effective number of tests for Bonferroni. If None, uses total markers or M_eff
- `use_effective_tests` (bool): Use effective tests (if computed) instead of total markers. Default: True
- `max_genotype_dosage` (float): Maximum genotype dosage for MAF calculation. Default: 2.0
- `farmcpu_params` (dict, optional): Parameters for FarmCPU
- `blink_params` (dict, optional): Parameters for BLINK
- `hybrid_params` (dict, optional): Parameters for Hybrid MLM:
  - `screen_threshold` (float): P-value cutoff for LRT refinement. Default: 1e-4
  - `max_line` (int): Batch size for processing. Default: 1000
- `outputs` (list): Which outputs to generate. Options:
  - `'all_marker_pvalues'`: Full results CSV
  - `'significant_marker_pvalues'`: Significant SNPs only CSV
  - `'manhattan'`: Manhattan plot PNG
  - `'qq'`: QQ plot PNG

**Side Effects:**
- Writes result files to `self.output_dir/`
- Prints analysis progress and summary statistics

**Output Files** (per trait):
- `GWAS_{trait}_all_results.csv`: Full association results for all markers
- `GWAS_{trait}_significant.csv`: Only significant markers (p < threshold)
- `GWAS_{trait}_{method}_GWAS_manhattan.png`: Manhattan plot
- `GWAS_{trait}_{method}_GWAS_qq.png`: QQ plot
- `GWAS_summary_by_traits_methods.csv`: Summary table (shared across traits)

**Example:**
```python
# Run MLM and Hybrid MLM
pipeline.run_analysis(
    traits=['Height', 'FloweringTime'],
    methods=['MLM', 'MLM_Hybrid'],
    hybrid_params={'screen_threshold': 1e-4}
)

# Run all methods with custom threshold
pipeline.run_analysis(
    traits=['Yield'],
    methods=['GLM', 'MLM', 'FarmCPU', 'BLINK'],
    significance=1e-6,
    outputs=['all_marker_pvalues', 'manhattan']
)
```

---

## Association Methods

Low-level association testing functions. These are called internally by `GWASPipeline` but can also be used directly.

### `MVP_GLM()`

General Linear Model association test.

```python
from pymvp.association.glm import MVP_GLM

results = MVP_GLM(
    phe,           # Phenotype array (n × 2): [ID, value]
    geno,          # Genotype matrix (n × m)
    CV=None,       # Covariates (n × p)
    verbose=True
)
```

**Returns:** `AssociationResults` object with attributes:
- `effects`: Effect sizes (m,)
- `se`: Standard errors (m,)
- `pvalues`: P-values (m,)

---

### `MVP_MLM()`

Mixed Linear Model association test.

```python
from pymvp.association.mlm import MVP_MLM

results = MVP_MLM(
    phe,              # Phenotype array (n × 2)
    geno,             # Genotype matrix (n × m)
    K,                # Kinship matrix (n × n)
    eigenK=None,      # Pre-computed eigendecomposition (optional)
    CV=None,          # Covariates (n × p)
    vc_method='BRENT', # Variance component estimation: 'BRENT' only
    maxLine=1000,     # Batch size
    cpu=1,            # Number of CPU cores
    verbose=True
)
```

**Returns:** `AssociationResults`

---

### `MVP_MLM_Hybrid()`

Hybrid MLM: Wald test screening + LRT refinement for top hits.

```python
from pymvp.association.hybrid_mlm import MVP_MLM_Hybrid

results = MVP_MLM_Hybrid(
    phe,
    geno,
    K,
    eigenK=None,
    CV=None,
    screen_threshold=1e-4,  # P-value cutoff for LRT refinement
    maxLine=1000,
    cpu=1,
    verbose=True
)
```

**Returns:** `AssociationResults` with LRT-refined p-values for markers passing screen

---

### `MVP_FarmCPU()`

Fixed and random model Circulating Probability Unification.

```python
from pymvp.association.farmcpu import MVP_FarmCPU

results = MVP_FarmCPU(
    phe,
    geno,
    map_data,      # Genetic map (SNP, Chr, Pos)
    CV=None,
    maxLoop=10,    # Maximum iterations
    method='FaST-LMM',
    verbose=True
)
```

**Returns:** `AssociationResults`

---

### `MVP_BLINK()`

Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway.

```python
from pymvp.association.blink import MVP_BLINK

results = MVP_BLINK(
    phe,
    geno,
    map_data,
    CV=None,
    maxLoop=10,
    verbose=True
)
```

**Returns:** `AssociationResults`

---

## Data Loaders

Functions for loading different data file formats.

### `load_phenotype_file()`

```python
from pymvp.data.loaders import load_phenotype_file

pheno_df = load_phenotype_file(
    filename,
    trait_columns=None  # Load all if None
)
```

**Returns:** DataFrame with 'ID' column + trait columns

---

### `load_genotype_file()`

```python
from pymvp.data.loaders import load_genotype_file

geno_matrix, individual_ids, geno_map = load_genotype_file(
    filename,
    file_format='vcf',  # or 'hapmap', 'plink'
    compute_effective_tests=False,
    effective_test_kwargs=None
)
```

**Returns:** Tuple of (GenotypeMatrix, list of IDs, GenotypeMap)

---

### `load_covariate_file()`

```python
from pymvp.data.loaders import load_covariate_file

cov_df = load_covariate_file(
    filename,
    covariate_columns=None,
    id_column='ID'
)
```

**Returns:** DataFrame with covariates

---

## Utility Functions

### Population Structure

```python
from pymvp.matrix.pca import MVP_PCA
from pymvp.matrix.kinship import MVP_K_VanRaden

# Compute PCs
pcs = MVP_PCA(M, pcs_keep=5, verbose=True)

# Compute kinship matrix
K = MVP_K_VanRaden(M, verbose=True)
```

---

### Statistical Functions

```python
from pymvp.utils.stats import genomic_inflation_factor, calculate_maf_from_genotypes

# Calculate genomic inflation factor (lambda GC)
lambda_gc = genomic_inflation_factor(pvalues)

# Calculate minor allele frequencies
maf = calculate_maf_from_genotypes(geno_matrix, max_dosage=2.0)
```

---

### Effective Tests

```python
from pymvp.utils.effective_tests import estimate_effective_tests_from_genotype

eff_info = estimate_effective_tests_from_genotype(
    geno_matrix,
    geno_map,
    composite_ld_threshold=0.2,
    window_size_bp=1000000
)

# Returns dict with:
# - 'Me': Effective number of independent tests
# - 'total_snps': Total number of SNPs
# - 'ld_blocks': Number of LD blocks detected
```

---

### Visualization

```python
from pymvp.visualization.manhattan import MVP_Report

# Generate Manhattan and QQ plots
MVP_Report(
    results=assoc_results,
    map_data=geno_map,
    output_prefix='my_gwas',
    threshold=0.05/n_markers,
    file_type='png',
    dpi=300
)
```

---

## Data Types

### AssociationResults

Named tuple containing GWAS results:

```python
from pymvp.utils.data_types import AssociationResults

results = AssociationResults(
    effects=effect_array,   # Effect sizes
    se=se_array,           # Standard errors
    pvalues=pvalue_array   # P-values
)

# Access attributes
print(results.pvalues)
print(results.effects)
```

### GenotypeMatrix

Wrapper around genotype data (usually memory-mapped):

```python
from pymvp.utils.data_types import GenotypeMatrix

geno = GenotypeMatrix(data_array)

# Attributes
geno.n_individuals  # Number of samples
geno.n_markers      # Number of markers

# Indexing
subset = geno[0:10, :]  # First 10 individuals, all markers
```

---

## Common Workflows

### Workflow 1: Standard GWAS

```python
from pymvp.pipelines.gwas import GWASPipeline

pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data('phenos.csv', 'genos.vcf.gz')
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['Trait1'], methods=['GLM', 'MLM'])
```

### Workflow 2: Direct Association Testing

```python
from pymvp.association.mlm import MVP_MLM
from pymvp.matrix.kinship import MVP_K_VanRaden
import numpy as np

# Prepare data
phe = np.column_stack([np.arange(n), phenotype_values])
K = MVP_K_VanRaden(genotype_matrix)

# Run MLM
results = MVP_MLM(phe, genotype_matrix, K)

# Get significant SNPs
sig_indices = results.pvalues < 0.05/len(results.pvalues)
```

---

## See Also

- [Quick Start Guide](quickstart.md)
- [Output File Formats](output_files.md)
- [Examples](../examples/)
- [Hybrid MLM Tutorial](hybrid_mlm_demo.ipynb)
