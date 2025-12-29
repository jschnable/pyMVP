# Example Data for PANICLE

This directory contains real sorghum data curated for testing and demonstration purposes.

## Files

### `example_phenotypes.csv`
- **Description**: Plant height measurements from field trials
- **Source**: Subset of SbDiv diversity panel grown in Nebraska 2021
- **Samples**: 738 individuals
- **Traits**: 1 (PlantHeight, in meters)
- **Format**: CSV with columns `ID,PlantHeight`

### `example_genotypes.vcf.gz`
- **Description**: SNP genotype data
- **Source**: RNA-seq derived markers from SbDiv diversity panel
- **Samples**: 738 individuals (matching phenotypes)
- **Markers**: 1,000 SNPs across all 10 chromosomes
- **Format**: Compressed VCF (standard variant call format)

### `example_covariates.csv`
- **Description**: Flowering time data for use as a covariate
- **Source**: Same SbDiv diversity panel, Nebraska 2021
- **Samples**: 738 individuals (matching phenotypes and genotypes)
- **Covariates**: 1 (DaysToFlower, days from planting to flowering)
- **Format**: CSV with columns `ID,DaysToFlower`
- **Use case**: Demonstrates controlling for flowering time when analyzing plant height

## Data Curation

The example data was created by:
1. Extracting all 738 samples that have both genotype and phenotype data
2. Randomly sampling 997 markers from the full 170,072 marker dataset
3. **Including 3 key significant markers** from published GWAS results:
   - `Chr06_44163616` (p = 5.46e-11) - Most significant on chromosome 6
   - `Chr07_63464311` (p = 1.55e-07) - Most significant on chromosome 7
   - `Chr09_60149640` (p = 5.30e-08) - Most significant on chromosome 9

These three markers are known to be associated with plant height variation and ensure that the example data can demonstrate detection of real genetic associations.

## Usage

All example scripts in this directory are designed to work with these files:

```python
from panicle.pipelines.gwas import GWASPipeline

# Basic GWAS
pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data(
    phenotype_file='example_phenotypes.csv',
    genotype_file='example_genotypes.vcf.gz'
)
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['PlantHeight'], methods=['MLM'])

# GWAS with covariates
pipeline = GWASPipeline(output_dir='./results_with_covariates')
pipeline.load_data(
    phenotype_file='example_phenotypes.csv',
    genotype_file='example_genotypes.vcf.gz',
    covariate_file='example_covariates.csv',
    covariate_columns=['DaysToFlower']
)
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['PlantHeight'], methods=['MLM'])
```

## Expected Results

When analyzing this data with appropriate population structure correction (MLM or MLM_Hybrid):
- You should detect significant associations on chromosomes 6, 7, and 9
- The chromosome 6 QTL typically shows the strongest signal
- Lambda GC inflation factors should be reasonable (~1.0-1.3) with proper correction

## Data Size

- Phenotype file: ~11 KB (text)
- Genotype file: ~132 KB (compressed VCF)
- Covariate file: ~11 KB (text)
- Total: ~154 KB

This small dataset allows for:
- Fast testing (~10-30 seconds per analysis)
- Demonstration of all PANICLE features
- Verification of installation and setup
- Learning the PANICLE workflow

## Original Data Source

Data derives from:
- **SbDiv diversity panel**: 378 sorghum lines representing global diversity
- **Field trials**: Nebraska 2021 growing season
- **Genotyping**: RNA-seq based SNP calling (Mangal et al. 2025)

For the full dataset, see the `sorghum_data/` directory in the parent repository.

## Citation

If you use this example data in publications, please cite:
- The PANICLE package (citation TBD)
- Original SbDiv panel and genotyping work (Mangal et al. 2025, in prep)
