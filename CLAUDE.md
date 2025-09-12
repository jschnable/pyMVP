# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Development Setup
Due to modern Python packaging requirements and externally-managed environments, use one of these approaches:

**Option 1: Virtual Environment (Recommended)**
```bash
# Create and activate virtual environment
python3 -m venv pymvp-env
source pymvp-env/bin/activate  # On Windows: pymvp-env\Scripts\activate

# Install package in development mode
pip install -e .
```

**Option 2: System-wide with override (Use with caution)**
```bash
# Install package in development mode (overrides system restrictions)
pip install -e . --break-system-packages

# Install all dependencies if needed
pip install -r requirements.txt --break-system-packages
```

**Option 3: User-local installation**
```bash
# Install to user directory
pip install -e . --user
```

### Performance Optimization
The package uses Numba JIT compilation for performance-critical operations, providing near C++ performance with pure Python code. No compilation is required as Numba compiles functions at runtime.

### Running Analysis Scripts

**Production GWAS Script (scripts/run_GWAS.py)**
The main command-line interface for GWAS analysis:
```bash
# Basic usage
cd scripts
python run_GWAS.py --phenotype data.csv --genotype geno.csv --output results/

# With all options
python run_GWAS.py \
    --phenotype phenotype.csv \
    --genotype genotype.vcf.gz \
    --map map.csv \
    --output ./results \
    --methods GLM,MLM,FarmCPU \
    --n-pcs 3 \
    --significance 5e-8 \
    --max-iterations 10
```

**Example Analysis Scripts**
```bash
# Run comprehensive GWAS analysis
python comprehensive_gwas_analysis.py

# Run FarmCPU validation (quick check)
python farmcpu_quickcheck.py

# Run with timeout for long analyses
timeout 300 python3 comprehensive_gwas_analysis.py
```

### Testing
The repository uses pytest for testing. Tests are organized by functionality:
```bash
# Run all tests
python -m pytest

# Run specific test modules (examples based on typical GWAS testing patterns)
python -m pytest tests/test_phase1_data_structures.py -v
python -m pytest tests/test_phase1_data_structures.py::TestDataTypes::test_phenotype_loading -v
python -m pytest tests/test_phase1_data_structures.py::TestDataTypes::test_kinship_matrix_loading -v
```

### Implementation Validation (CRITICAL)
**⚠️  MANDATORY VALIDATION PROTOCOL FOR ALL ALGORITHMIC CHANGES ⚠️**

Before making ANY changes to the core GWAS algorithms (GLM, MLM, FarmCPU), you MUST follow this validation protocol:

#### 1. Separate Implementation Files
- All performance improvements must be implemented in separate files first
- Never modify `glm.py`, `mlm.py`, or `farmcpu.py` directly
- Use naming convention: `glm_optimized.py`, `mlm_fast.py`, `farmcpu_improved.py`, etc.

#### 2. Validation Dataset Setup
Create the validation dataset (one-time setup):
```bash
python tests/create_validation_dataset.py
```

This creates a validation dataset with:
- 500 individuals, 5,000 markers, 25 true QTNs
- Known genetic architecture for validation
- Population structure and realistic LD patterns
- Fast loading for rapid testing (<2 seconds)

#### 3. Validate Implementation Changes
Before integration, test your implementation:
```bash
# Quick validation test
python tests/quick_validation_test.py GLM association/glm_optimized.py
python tests/quick_validation_test.py MLM association/mlm_fast.py
python tests/quick_validation_test.py FarmCPU association/farmcpu_improved.py

# Comprehensive validation
python tests/validate_implementation_changes.py
```

#### 4. Validation Criteria (STRICT)
Your implementation MUST meet these criteria to pass validation:

**Critical Requirements:**
- **Log p-value correlation ≥ 0.999** (Pearson correlation of -log10(p-values))
- **Effect size correlation ≥ 0.9999** (Pearson correlation of effect estimates)  
- **True positive rate change ≤ ±5%** (detection of known QTNs)
- **No runtime errors** on validation dataset
- **Numerical stability** (no NaN/Inf values in results)

**Performance Goals:**
- Speedup > 1.0x (faster than original)
- Memory usage ≤ original implementation
- Maintain statistical accuracy

#### 5. Integration Protocol
Only after passing validation:
1. Document validation results in implementation file header
2. Create pull request with validation report attached
3. Replace original implementation only after code review
4. Update performance benchmarks

#### 6. Validation Framework Components
- `tests/validate_implementation_changes.py` - Main validation framework
- `tests/quick_validation_test.py` - Fast validation for development  
- `tests/create_validation_dataset.py` - Dataset creation utility
- `tests/validation_datasets/` - Standard validation dataset storage

**Example Validation Usage:**
```python
# In your implementation file (e.g., glm_optimized.py)
from tests.validate_implementation_changes import ValidationConfig, ValidationDataset, ImplementationValidator

# Run validation
config = ValidationConfig()
dataset = ValidationDataset(config)
dataset.load_dataset()
validator = ImplementationValidator(config, dataset)

# Test your implementation
results = validator.validate_glm_implementation(your_function, "Your Implementation Name")
```

**Validation Report Interpretation:**
- ✅ **PASSED**: Implementation ready for integration
- ❌ **FAILED**: Implementation needs more work
- 🔴 **CRITICAL**: Major algorithmic differences detected - do not integrate

This validation framework ensures that all performance optimizations maintain the statistical accuracy and reliability that pyMVP users depend on.

## Architecture

### Core Package Structure
- `pymvp/` - Main Python package
  - `core/mvp.py` - Primary GWAS analysis interface (MVP function)
  - `data/` - Data loading and conversion utilities
  - `matrix/` - Matrix operations (kinship, PCA)
  - `association/` - GWAS methods (GLM, MLM, FarmCPU)
  - `utils/data_types.py` - Core data structures (Phenotype, GenotypeMatrix, etc.)
  - `visualization/` - Result plotting and reporting

### Performance Features
- **Numba JIT compilation** for computationally intensive functions
- **Memory-mapped arrays** for handling large genotype datasets
- **Vectorized NumPy operations** using optimized BLAS/LAPACK libraries
- **Efficient data structures** designed for genomic-scale analyses

### Data Types
The package uses custom data structures that mirror R rMVP format:
- `Phenotype` - n×2 matrix (ID, Trait values)
- `GenotypeMatrix` - Genotype data with missing value handling
- `GenotypeMap` - SNP map information (CHR, SNP, CM, POS)
- `KinshipMatrix` - Relationship matrix for mixed models
- `AssociationResults` - GWAS output with p-values and effects

### GWAS Methods
1. **GLM** - General Linear Model (basic association)
2. **MLM** - Mixed Linear Model (with kinship correction)
3. **FarmCPU** - Fixed and random model Circulating Probability Unification
   - Uses iterative approach with pseudo-QTN selection
   - Implements LD-based binning (50Mb → 5Mb → 0.5Mb)
   - Includes genomic inflation correction

### Validation Data
- `validation_data/` - Test datasets for algorithm validation
- `performance_test_large/` - Large-scale performance benchmarks
- Synthetic datasets with known QTN effects for validation

### Analysis Scripts
- `scripts/run_GWAS.py` - Production command-line GWAS interface with comprehensive data format support
- `comprehensive_gwas_analysis.py` - Complete workflow demonstration
- `farmcpu_quickcheck.py` - rMVP-compatible FarmCPU validation with detailed logging

### Data Format Support
The package supports multiple genotype input formats:
- **CSV/TSV**: Numeric matrices (individuals × markers, 0/1/2 coding)
- **VCF/BCF**: Standard genomics format (.vcf, .vcf.gz, .vcf.bgz, .bcf)
- **PLINK**: Binary format (.bed + .bim + .fam files)  
- **HapMap**: Text format (.hmp, .hmp.txt, supports gzipped)

Key features:
- Automatic format detection
- Quality control filtering (MAF, missingness, monomorphic)
- Sample deduplication and ID matching
- Missing value imputation (major allele)

## Development Notes

### Implementation Status
- Core GWAS algorithms implemented and validated
- Pure Python implementation with Numba JIT acceleration
- Data structure compatibility with R rMVP format
- FarmCPU implementation matches rMVP behavior exactly

### File Handling
- Uses pathlib.Path for cross-platform compatibility
- Supports CSV, HDF5, and various genomics formats
- Large dataset handling through memory-mapped arrays

### Performance Considerations
- Numba JIT compilation provides near C++ performance for numerical code
- NumPy/SciPy for matrix operations using optimized libraries
- Memory-mapped arrays for efficient handling of large genomic datasets
- Vectorized operations minimize Python overhead

### Manhattan Plot Formatting
The visualization module (`pymvp/visualization/manhattan.py`) has been optimized for publication-ready plots:
- **Large point size** (default: 10.0-12.0) for high visibility
- **No titles** on Manhattan plots (clean appearance)
- **Single significance threshold** (no suggestive line)
- **No background grid** for clean appearance
- **Chromosome-labeled x-axis** with no gaps between chromosomes
- **Alternating chromosome colors** for clear distinction

Key parameters in `create_manhattan_plot()` and `MVP_Report()`:
- `point_size`: Controls SNP point visibility (default: 10.0+)
- `threshold`: Significance line (default: 5e-8)  
- `suggestive_threshold`: Set to 0 to disable
- `figsize`: Plot dimensions (default: (12,6) or (8,4))