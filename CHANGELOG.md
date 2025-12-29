# Changelog

All notable changes to pyMVP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX (Unreleased)

### Added
- Initial public release of pyMVP
- Core GWAS methods: GLM, MLM, FarmCPU, BLINK
- **Hybrid MLM method** combining Wald test screening with LRT refinement
  - 2-3% runtime overhead vs standard MLM
  - Orders of magnitude p-value improvement for significant associations
- High-level `GWASPipeline` API for streamlined workflows
- Multiple genotype format support:
  - VCF/BCF with automatic binary caching (~26x faster loading)
  - PLINK binary format (.bed/.bim/.fam)
  - HapMap format
  - CSV/TSV matrices
- Automatic population structure correction:
  - Step-wise PCA calculation
  - VanRaden kinship matrix computation
- Effective tests calculation for accurate Bonferroni correction
- Parallel execution of multiple GWAS methods
- Comprehensive visualization:
  - Manhattan plots with decimated rendering
  - QQ plots with genomic inflation factor
  - Results comparison plots
- Command-line interface via `scripts/run_GWAS.py`
- Binary genotype caching tool: `pymvp-cache-genotype`

### Documentation
- Complete API reference for all classes and functions
- Quick start guide with 6 common scenarios
- Output file format specifications
- 5 runnable example scripts demonstrating different workflows
- Interactive Jupyter notebook for Hybrid MLM demonstration
- PDF report generator for publication-ready results
- Detailed algorithm documentation for Hybrid MLM method

### Performance
- Vectorized VCF loading with cyvcf2 optimization
- Binary caching for instant subsequent loads (~1.5s)
- Numba JIT acceleration for computationally intensive operations
- 2-4x faster than R-based rMVP implementation

### Dependencies
- Core: numpy, scipy, pandas, h5py, tables, statsmodels, scikit-learn, matplotlib, seaborn, tqdm, numba
- Optional: cyvcf2 (VCF support), bed-reader (PLINK support)

[0.1.0]: https://github.com/jschnable/pyMVP/releases/tag/v0.1.0
