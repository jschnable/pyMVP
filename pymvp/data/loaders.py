"""
Data loading utilities for various file formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Any
import warnings
import time
import sys

from ..utils.data_types import GenotypeMatrix, GenotypeMap
from ..utils.memmap_utils import load_full_from_metadata
from ..utils.effective_tests import estimate_effective_tests_from_genotype
try:
    # Local VCF loader (builtin parser by default, cyvcf2 optional)
    from .load_genotype_vcf import load_genotype_vcf as _load_genotype_vcf
except Exception:
    _load_genotype_vcf = None
try:
    # HapMap loader
    from .load_genotype_hapmap import load_genotype_hapmap as _load_genotype_hapmap
except Exception:
    _load_genotype_hapmap = None
try:
    # PLINK bed loader via bed-reader
    from .load_genotype_plink import load_genotype_plink as _load_genotype_plink
except Exception:
    _load_genotype_plink = None

NON_BINARY_FORMATS = {
    'vcf', 'csv', 'tsv', 'numeric', 'hapmap'
}

LOAD_TIME_WARNING_THRESHOLD = 300.0  # seconds (5 minutes)


def detect_file_format(filepath: Union[str, Path]) -> str:
    """Detect file format based on extension and content
    
    Args:
        filepath: Path to file
        
    Returns:
        Detected format: 'csv', 'tsv', 'vcf', 'plink', 'hmp', 'numeric'
    """
    filepath = Path(filepath)
    
    # Check extension first (handle multi-suffix like .vcf.gz)
    name_lower = filepath.name.lower()
    if (
        name_lower.endswith('.vcf')
        or name_lower.endswith('.vcf.gz')
        or name_lower.endswith('.vcf.bgz')
        or name_lower.endswith('.bcf')
    ):
        return 'vcf'
    elif filepath.suffix.lower() in ['.bed', '.bim', '.fam']:
        return 'plink'
    elif name_lower.endswith('.hmp') or name_lower.endswith('.hmp.txt') or name_lower.endswith('.hmp.gz') or name_lower.endswith('.hmp.txt.gz'):
        return 'hapmap'
    elif filepath.suffix.lower() in ['.tsv', '.txt']:
        return 'tsv'
    elif filepath.suffix.lower() == '.csv':
        return 'csv'
    elif filepath.suffix.lower() == '.npz':
        try:
            with np.load(filepath, allow_pickle=True) as meta:
                if 'memmap_file' in meta:
                    return 'memmap'
        except Exception:
            return 'unknown'

    # Try to detect by content / neighboring files
    try:
        # Check for PLINK triad if a bare prefix is given
        if filepath.suffix == '':
            pref = filepath
            if pref.with_suffix('.bed').exists() and pref.with_suffix('.bim').exists() and pref.with_suffix('.fam').exists():
                return 'plink'
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        if first_line.startswith('##fileformat=VCF'):
            return 'vcf'
        # Detect HapMap header
        if first_line.split('\t')[:11] == ['rs#','alleles','chrom','pos','strand','assembly#','center','protLSID','assayLSID','panelLSID','QCcode']:
            return 'hapmap'
        elif '\t' in first_line and ',' not in first_line:
            return 'tsv'
        elif ',' in first_line:
            return 'csv'
        else:
            return 'numeric'
    except:
        return 'unknown'


def _detect_numeric_separator(filepath: Union[str, Path]) -> str:
    """Heuristically determine the delimiter for numeric genotype matrices."""
    filepath = Path(filepath)
    try:
        with filepath.open('r') as handle:
            # Inspect up to the first 10 non-empty header lines to decide.
            for _ in range(10):
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                comma_count = line.count(',')
                tab_count = line.count('\t')
                if tab_count or comma_count:
                    # Prefer the character that appears more frequently.
                    return '\t' if tab_count >= comma_count else ','
    except OSError:
        pass

    # Default to comma if the structure was ambiguous; this avoids collapsing
    # every marker into a single column when a comma-separated file is treated
    # as tab-delimited.
    return ','

def load_phenotype_file(filepath: Union[str, Path], 
                       trait_columns: Optional[List[str]] = None,
                       id_column: str = 'ID') -> pd.DataFrame:
    """Load phenotype file in various formats
    
    Args:
        filepath: Path to phenotype file
        trait_columns: List of trait column names (if None, auto-detect)
        id_column: Name of ID column
        
    Returns:
        DataFrame with ID and trait columns
    """
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)
    
    # Load based on format
    # Robust NA handling: recognize common missing tokens
    na_values = [
        '', 'NA', 'NaN', 'nan', 'NAN', 'na', 'N/A', 'n/a', 'Null', 'NULL',
        '.', '-', '--'
    ]
    read_kwargs = dict(na_values=na_values, keep_default_na=True)
    if file_format == 'csv':
        df = pd.read_csv(filepath, **read_kwargs)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t', **read_kwargs)
    else:
        # Try both comma and tab separation
        try:
            df = pd.read_csv(filepath, **read_kwargs)
        except:
            df = pd.read_csv(filepath, sep='\t', **read_kwargs)
    
    # Standardize column names
    if id_column not in df.columns:
        # Try common ID column names (select leftmost if multiple are present)
        possible_id_cols = [
            'ID', 'id', 'IID',
            'sample', 'Sample',
            'Taxa', 'taxa',
            'Genotype', 'genotype',
            'Accession', 'accession',
        ]
        present_candidates = [c for c in df.columns if c in possible_id_cols]
        if present_candidates:
            if len(present_candidates) > 1:
                warnings.warn(
                    "Multiple potential ID columns found: {}. Selecting leftmost '{}' as ID.".format(
                        present_candidates, present_candidates[0]
                    )
                )
            df = df.rename(columns={present_candidates[0]: 'ID'})
        else:
            # Use first column as ID (emit a gentle warning)
            first_col = df.columns[0]
            warnings.warn(
                "No recognized ID column found; using first column '{}' as ID.".format(first_col)
            )
            df = df.rename(columns={first_col: 'ID'})
    
    # Auto-detect trait columns if not specified
    if trait_columns is None:
        # Attempt to coerce non-ID columns to numeric to improve detection
        numeric_probe = {}
        for col in df.columns:
            if col == 'ID':
                continue
            s = pd.to_numeric(df[col], errors='coerce')
            numeric_probe[col] = s
        # Choose columns that are numeric after coercion
        trait_columns = [col for col, s in numeric_probe.items() if pd.api.types.is_numeric_dtype(s)]
        # Ensure stored df has numeric dtype for selected traits
        if trait_columns:
            df[trait_columns] = df[trait_columns].apply(pd.to_numeric, errors='coerce')
    else:
        # Coerce declared trait columns to numeric
        present = [c for c in trait_columns if c in df.columns]
        if present:
            df[present] = df[present].apply(pd.to_numeric, errors='coerce')

    # Keep only ID and specified trait columns
    columns_to_keep = ['ID'] + [col for col in trait_columns if col in df.columns]
    df = df[columns_to_keep]

    # Deduplicate phenotype IDs by aggregating trait means (skip NaNs)
    if df['ID'].duplicated().any():
        n_dups = int(df['ID'].duplicated().sum())
        if len(columns_to_keep) > 1:
            # Aggregate by ID using mean for each trait column, NaN excluded by default
            agg_cols = {col: 'mean' for col in columns_to_keep if col != 'ID'}
            df = df.groupby('ID', as_index=False).agg(agg_cols)
            warnings.warn(
                f"Detected {n_dups} duplicated phenotype records by ID; deduplicated by computing per-ID mean of trait columns (missing values ignored)."
            )
        else:
            # No trait columns found; just keep first occurrence per ID
            df = df.drop_duplicates(subset=['ID'], keep='first')
            warnings.warn(
                f"Detected {n_dups} duplicated phenotype records by ID; no numeric trait columns detected, so retained only the first record per ID."
            )

    return df

def load_covariate_file(filepath: Union[str, Path],
                        covariate_columns: Optional[List[str]] = None,
                        id_column: str = 'ID') -> pd.DataFrame:
    """Load covariate file and ensure numeric covariate columns."""
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)

    na_values = [
        '', 'NA', 'NaN', 'nan', 'NAN', 'na', 'N/A', 'n/a', 'Null', 'NULL',
        '.', '-', '--'
    ]
    read_kwargs = dict(na_values=na_values, keep_default_na=True)

    if file_format == 'csv':
        cov_df = pd.read_csv(filepath, **read_kwargs)
    elif file_format == 'tsv':
        cov_df = pd.read_csv(filepath, sep='	', **read_kwargs)
    else:
        try:
            cov_df = pd.read_csv(filepath, **read_kwargs)
        except Exception:
            cov_df = pd.read_csv(filepath, sep='	', **read_kwargs)

    if id_column not in cov_df.columns:
        possible_id_cols = [
            'ID', 'id', 'IID',
            'sample', 'Sample',
            'Taxa', 'taxa',
            'Genotype', 'genotype',
            'Accession', 'accession',
        ]
        present_candidates = [c for c in cov_df.columns if c in possible_id_cols]
        if present_candidates:
            if len(present_candidates) > 1:
                warnings.warn(
                    "Multiple potential ID columns found: {}. Selecting leftmost '{}' as ID.".format(
                        present_candidates, present_candidates[0]
                    )
                )
            cov_df = cov_df.rename(columns={present_candidates[0]: 'ID'})
        else:
            first_col = cov_df.columns[0]
            warnings.warn(
                "No recognized ID column found; using first column '{}' as ID.".format(first_col)
            )
            cov_df = cov_df.rename(columns={first_col: 'ID'})
    else:
        if id_column != 'ID':
            cov_df = cov_df.rename(columns={id_column: 'ID'})

    cov_df['ID'] = cov_df['ID'].astype(str)

    available_covariates = [c for c in cov_df.columns if c != 'ID']
    if covariate_columns is not None:
        requested = [c for c in covariate_columns if c not in ('ID', id_column)]
        missing = [c for c in requested if c not in cov_df.columns]
        if missing:
            raise ValueError(
                "Requested covariate columns missing from file '{}': {}".format(
                    filepath, missing
                )
            )
        selected_cols = [c for c in covariate_columns if c in cov_df.columns and c not in ('ID', id_column)]
    else:
        selected_cols = available_covariates

    if not selected_cols:
        raise ValueError(
            "No covariate columns found in file '{}'. Provide at least one numeric covariate column.".format(
                filepath
            )
        )

    covariate_data = cov_df[['ID'] + selected_cols].copy()
    dropped_for_nan: List[str] = []

    for col in list(selected_cols):
        series = covariate_data[col]
        converted = pd.to_numeric(series, errors='coerce')
        invalid_mask = series.notna() & converted.isna()
        if invalid_mask.any():
            invalid_examples = sorted(series[invalid_mask].astype(str).unique()[:5])
            example_str = ', '.join(invalid_examples)
            raise ValueError(
                "Covariate column '{}' contains non-numeric values (e.g. {}).".format(
                    col, example_str
                )
            )
        covariate_data[col] = converted
        if converted.notna().sum() == 0:
            dropped_for_nan.append(col)

    if dropped_for_nan:
        warnings.warn(
            "Dropping covariate columns with no numeric values after conversion: {}".format(
                dropped_for_nan
            )
        )
        covariate_data = covariate_data.drop(columns=dropped_for_nan)

    covariate_columns_present = [c for c in covariate_data.columns if c != 'ID']
    if not covariate_columns_present:
        raise ValueError(
            "No usable covariate columns remained after processing file '{}'.".format(
                filepath
            )
        )

    if covariate_data['ID'].duplicated().any():
        n_dups = int(covariate_data['ID'].duplicated().sum())
        agg_cols = {col: 'mean' for col in covariate_columns_present}
        covariate_data = covariate_data.groupby('ID', as_index=False).agg(agg_cols)
        warnings.warn(
            f"Detected {n_dups} duplicated covariate records by ID; deduplicated by computing per-ID mean of covariate columns (missing values ignored)."
        )
    else:
        covariate_data = covariate_data.reset_index(drop=True)

    covariate_columns_present = [c for c in covariate_data.columns if c != 'ID']
    covariate_data = covariate_data[['ID'] + covariate_columns_present]

    return covariate_data


def _deduplicate_genotype_samples(individual_ids: List[str], geno_np: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Deduplicate genotype sample IDs by retaining the first occurrence per ID.

    Emits a warning when duplicates are found. Returns (new_ids, new_geno).
    """
    seen = set()
    keep_indices: List[int] = []
    for i, sid in enumerate(individual_ids):
        if sid not in seen:
            seen.add(sid)
            keep_indices.append(i)
    if len(keep_indices) != len(individual_ids):
        dropped = len(individual_ids) - len(keep_indices)
        warnings.warn(
            f"Detected {dropped} duplicated genotype records by sample ID; retaining only the first occurrence for each duplicated ID."
        )
        new_ids = [individual_ids[i] for i in keep_indices]
        new_geno = geno_np[keep_indices, :]
        return new_ids, new_geno
    return individual_ids, geno_np

def load_genotype_file(filepath: Union[str, Path],
                      file_format: Optional[str] = None,
                      compute_effective_tests: bool = False,
                      effective_test_kwargs: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Load genotype file in various formats
    
    Args:
        filepath: Path to genotype file
        file_format: File format ('csv', 'tsv', 'numeric' or None for auto-detect)
        
    Returns:
        Tuple of (GenotypeMatrix, individual_ids, GenotypeMap)
    """
    filepath = Path(filepath)
    precompute_alleles = kwargs.pop('precompute_alleles', True)
    load_start = time.time()

    if file_format is None:
        file_format = detect_file_format(filepath)

    def _maybe_warn(elapsed_seconds: float) -> None:
        if file_format in NON_BINARY_FORMATS and elapsed_seconds >= LOAD_TIME_WARNING_THRESHOLD:
            print(
                f"Warning: Loading genotype file '{filepath}' took {elapsed_seconds / 60:.1f} minutes. "
                "Consider caching it with `pymvp-cache-genotype` for faster future runs.",
                file=sys.stderr,
            )

    if file_format == 'memmap':
        genotype, individual_ids, geno_map = load_full_from_metadata(
            filepath,
            precompute_alleles=precompute_alleles,
        )
        if geno_map is None:
            raise ValueError("Metadata file does not contain genotype map information")
        individual_ids = list(individual_ids)
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                genotype,
                geno_map,
                **effective_tests_kwargs,
            )
        return genotype, individual_ids, geno_map

    if file_format == 'vcf':
        if _load_genotype_vcf is None:
            raise ImportError("VCF loading requires 'load_genotype_vcf' module."
                              " Ensure pymvp/data/load_genotype_vcf.py is present.")
        geno_np, individual_ids, geno_map_df = _load_genotype_vcf(str(filepath), **kwargs)
        # Deduplicate genotype sample IDs centrally
        individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
        geno_matrix = GenotypeMatrix(geno_np, precompute_alleles=precompute_alleles)
        geno_map = GenotypeMap(
            geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df)
        )
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                geno_matrix,
                geno_map,
                **effective_tests_kwargs,
            )
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return geno_matrix, individual_ids, geno_map
    
    if file_format == 'plink':
        if _load_genotype_plink is None:
            raise ImportError("PLINK loading requires 'bed-reader' and 'load_genotype_plink'."
                              " pip install bed-reader.")
        geno_np, individual_ids, geno_map_df = _load_genotype_plink(str(filepath), **kwargs)
        # Deduplicate genotype sample IDs centrally
        individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
        geno_matrix = GenotypeMatrix(geno_np, precompute_alleles=precompute_alleles)
        geno_map = GenotypeMap(
            geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df)
        )
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                geno_matrix,
                geno_map,
                **effective_tests_kwargs,
            )
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return geno_matrix, individual_ids, geno_map

    if file_format == 'hapmap':
        if _load_genotype_hapmap is None:
            raise ImportError("HapMap loading requires 'load_genotype_hapmap' module.")
        geno_np, individual_ids, geno_map_df = _load_genotype_hapmap(str(filepath), **kwargs)
        # Deduplicate genotype sample IDs centrally
        individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
        geno_matrix = GenotypeMatrix(geno_np, precompute_alleles=precompute_alleles)
        geno_map = GenotypeMap(
            geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df)
        )
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                geno_matrix,
                geno_map,
                **effective_tests_kwargs,
            )
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return geno_matrix, individual_ids, geno_map

    if file_format in ['csv', 'tsv', 'numeric']:
        # Load as CSV/TSV with numeric genotypes
        if file_format == 'csv':
            separator = ','
        elif file_format == 'tsv':
            separator = '\t'
        else:
            separator = _detect_numeric_separator(filepath)

        # Use fast, consistent parsing and avoid mixed-type inference
        df = pd.read_csv(filepath, sep=separator, low_memory=False)

        # First column should be individual IDs
        individual_ids = df.iloc[:, 0].astype(str).tolist()

        # Remaining columns are markers
        marker_names = df.columns[1:].tolist()
        data_df = df.iloc[:, 1:]

        # Vectorized conversion to numeric with robust NA handling
        # Coerce any non-numeric (e.g., 'NA', 'N', '.') to NaN, then fill with sentinel -9
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        data_df = data_df.fillna(-9)

        # Cast to compact integer type; -9 and 0/1/2 fit in int8
        try:
            genotype_matrix = data_df.to_numpy(dtype=np.int8, copy=False)
        except Exception:
            # Fallback path in case of unexpected wide range; use int16
            genotype_matrix = data_df.to_numpy(dtype=np.int16, copy=False)

        # Deduplicate genotype sample IDs centrally
        individual_ids, genotype_matrix = _deduplicate_genotype_samples(individual_ids, genotype_matrix)

        # Create GenotypeMatrix
        geno_matrix = GenotypeMatrix(genotype_matrix, precompute_alleles=precompute_alleles)

        # Create basic GenotypeMap (assumes no map file provided)
        map_data = pd.DataFrame({
            'SNP': marker_names,
            'CHROM': [1] * len(marker_names),  # Default to chromosome 1
            'POS': list(range(1, len(marker_names) + 1))  # Sequential positions
        })
        geno_map = GenotypeMap(map_data)
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                geno_matrix,
                geno_map,
                **effective_tests_kwargs,
            )

        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return geno_matrix, individual_ids, geno_map
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_genotype_vcf(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load VCF files using the optimized loader.

    Accepts the same keyword args as `load_genotype_vcf` in `pymvp.data.load_genotype_vcf`,
    including `backend` ('auto'|'cyvcf2'|'builtin'), `split_multiallelic`, and filters.
    """
    if _load_genotype_vcf is None:
        raise ImportError("VCF loading requires 'load_genotype_vcf' module.")
    geno_np, individual_ids, geno_map_df = _load_genotype_vcf(str(filepath), **kwargs)
    # Deduplicate genotype sample IDs centrally
    individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
    geno_matrix = GenotypeMatrix(geno_np)
    geno_map = GenotypeMap(geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df))
    return geno_matrix, individual_ids, geno_map

def load_genotype_plink(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load PLINK .bed files.

    Accepts `prefix_or_bed` and optional `bim`/`fam` in kwargs, plus filters
    matching the VCF loader (drop_monomorphic, max_missing, min_maf).
    """
    if _load_genotype_plink is None:
        raise ImportError("PLINK loading requires 'bed-reader' and 'load_genotype_plink'."
                          " pip install bed-reader.")
    geno_np, individual_ids, geno_map_df = _load_genotype_plink(str(filepath), **kwargs)
    # Deduplicate genotype sample IDs centrally
    individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
    geno_matrix = GenotypeMatrix(geno_np)
    geno_map = GenotypeMap(geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df))
    return geno_matrix, individual_ids, geno_map

def load_genotype_hapmap(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load HapMap .hmp(.txt) files.

    Supports plain and gzipped files. QC filters align with other loaders.
    """
    if _load_genotype_hapmap is None:
        raise ImportError("HapMap loading requires 'load_genotype_hapmap' module.")
    geno_np, individual_ids, geno_map_df = _load_genotype_hapmap(str(filepath), **kwargs)
    # Deduplicate genotype sample IDs centrally
    individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
    geno_matrix = GenotypeMatrix(geno_np)
    geno_map = GenotypeMap(geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df))
    return geno_matrix, individual_ids, geno_map

def load_map_file(filepath: Union[str, Path]) -> GenotypeMap:
    """Load genetic map file
    
    Args:
        filepath: Path to map file
        
    Returns:
        GenotypeMap object
    """
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)
    
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t')
    else:
        try:
            df = pd.read_csv(filepath)
        except:
            df = pd.read_csv(filepath, sep='\t')
    
    # Standardize column names
    col_mapping = {
        'Chr': 'CHROM', 'chr': 'CHROM', 'chromosome': 'CHROM',
        'Pos': 'POS', 'pos': 'POS', 'position': 'POS', 'bp': 'POS',
        'snp': 'SNP', 'marker': 'SNP', 'rs': 'SNP'
    }
    
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return GenotypeMap(df)


def match_individuals(phenotype_df: pd.DataFrame,
                     individual_ids: List[str],
                     covariate_df: Optional[pd.DataFrame] = None
                     ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[int], Dict]:
    """Match individuals across phenotype, genotype, and optional covariates."""
    phenotype_df = phenotype_df.copy()
    if 'ID' not in phenotype_df.columns:
        raise ValueError("Phenotype dataframe must contain an 'ID' column.")
    phenotype_df['ID'] = phenotype_df['ID'].astype(str)

    phe_ids = set(phenotype_df['ID'])
    genotype_ids = [str(ind_id) for ind_id in individual_ids]
    geno_ids = set(genotype_ids)

    common_ids = phe_ids & geno_ids

    summary: Dict[str, int] = {
        'n_phenotype_original': len(phe_ids),
        'n_genotype_original': len(geno_ids),
        'n_common': len(common_ids),
        'n_phenotype_dropped': len(phe_ids - common_ids),
        'n_genotype_dropped': len(geno_ids - common_ids),
    }

    if len(common_ids) == 0:
        raise ValueError("No common individuals found between phenotype and genotype data")

    matched_phenotype = phenotype_df[phenotype_df['ID'].isin(common_ids)].copy()
    matched_phenotype = matched_phenotype.sort_values('ID').reset_index(drop=True)
    sorted_ids = matched_phenotype['ID'].tolist()

    id_to_index: Dict[str, int] = {}
    for idx, raw_id in enumerate(genotype_ids):
        if raw_id not in id_to_index:
            id_to_index[raw_id] = idx

    try:
        matched_indices = [id_to_index[sid] for sid in sorted_ids]
    except KeyError as exc:
        missing_id = exc.args[0]
        raise RuntimeError(
            f"Internal inconsistency while aligning genotype indices; ID '{missing_id}' was expected but not found."
        ) from None

    matched_covariate: Optional[pd.DataFrame] = None
    if covariate_df is not None:
        if 'ID' not in covariate_df.columns:
            raise ValueError("Covariate dataframe must contain an 'ID' column.")

        covariate_df = covariate_df.copy()
        covariate_df['ID'] = covariate_df['ID'].astype(str)
        cov_ids = set(covariate_df['ID'])

        summary['n_covariate_provided'] = len(cov_ids)

        missing_covariate_ids = sorted(common_ids - cov_ids)
        if missing_covariate_ids:
            preview = ', '.join(missing_covariate_ids[:5])
            if len(missing_covariate_ids) > 5:
                preview += ', ...'
            raise ValueError(
                "Covariate file is missing {} individuals required after phenotype/genotype matching: {}".format(
                    len(missing_covariate_ids), preview
                )
            )

        unused_covariate_ids = cov_ids - common_ids
        summary['n_covariate_unused'] = len(unused_covariate_ids)

        if covariate_df['ID'].duplicated().any():
            n_dups = int(covariate_df['ID'].duplicated().sum())
            agg_cols = {col: 'mean' for col in covariate_df.columns if col != 'ID'}
            covariate_df = covariate_df.groupby('ID', as_index=False).agg(agg_cols)
            warnings.warn(
                f"Detected {n_dups} duplicated covariate records by ID; deduplicated by computing per-ID mean of covariate columns (missing values ignored)."
            )

        matched_covariate = covariate_df.set_index('ID').loc[sorted_ids].reset_index()
        cov_cols = [c for c in matched_covariate.columns if c != 'ID']
        matched_covariate = matched_covariate[['ID'] + cov_cols]
        summary['n_covariate_matched'] = len(sorted_ids)
    else:
        summary['n_covariate_provided'] = 0
        summary['n_covariate_unused'] = 0
        summary['n_covariate_matched'] = 0

    return matched_phenotype, matched_covariate, matched_indices, summary
