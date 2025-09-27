"""
Core data structures for pyMVP package
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path

class Phenotype:
    """Phenotype data structure compatible with R rMVP format
    
    Expected format: n × 2 matrix where:
    - Column 1: Individual IDs 
    - Column 2: Trait values
    """
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, str, Path]):
        if isinstance(data, (str, Path)):
            # Load from file - try with header first
            try:
                self.data = pd.read_csv(data, header=0)
            except:
                self.data = pd.read_csv(data, header=None)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, np.ndarray):
            # Convert to DataFrame
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be array, DataFrame, or file path")
            
        # Validate structure
        if self.data.shape[1] != 2:
            raise ValueError(f"Phenotype must have 2 columns, got {self.data.shape[1]}")
            
        # Set standard column names
        self.data.columns = ['ID', 'Trait']
        
    @property
    def ids(self) -> pd.Series:
        """Individual IDs"""
        return self.data['ID']
    
    @property 
    def values(self) -> pd.Series:
        """Trait values"""
        return self.data['Trait']
    
    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return len(self.data)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.data.values


class GenotypeMap:
    """SNP map information compatible with R rMVP format
    
    Expected columns: [SNP_ID, Chr, Pos, REF, ALT]
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, Path]):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be DataFrame or file path")
            
        # Validate required columns
        required_cols = ['SNP', 'CHROM', 'POS']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    @property
    def snp_ids(self) -> pd.Series:
        """SNP identifiers"""
        return self.data['SNP']
    
    @property
    def chromosomes(self) -> pd.Series:
        """Chromosome numbers"""
        return self.data['CHROM']
    
    @property
    def positions(self) -> pd.Series:
        """Physical positions"""
        return self.data['POS']
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self.data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return self.data.copy()


class GenotypeMatrix:
    """Memory-efficient genotype matrix with lazy loading support
    
    Handles large genotype matrices that may not fit in memory.
    Compatible with R rMVP memory-mapped format.
    Includes pre-computed major alleles for efficient missing data imputation.
    """
    
    def __init__(self, data: Union[np.ndarray, str, Path], 
                 shape: Optional[Tuple[int, int]] = None,
                 dtype: np.dtype = np.int8,
                 precompute_alleles: bool = True):
        
        if isinstance(data, np.ndarray):
            self._data = data
            self._is_memmap = False
        elif isinstance(data, (str, Path)):
            # Memory-mapped file
            if shape is None:
                raise ValueError("Shape required for memory-mapped files")
            self._data = np.memmap(data, dtype=dtype, mode='r', shape=shape)
            self._is_memmap = True
        else:
            raise ValueError("Data must be array or file path")
        
        # Pre-compute major alleles for efficient imputation
        self._major_alleles = None
        self._missing_masks = None
        if precompute_alleles:
            self._precompute_major_alleles()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (n_individuals, n_markers)"""
        return self._data.shape
    
    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return self.shape[0]
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return self.shape[1]
    
    def __getitem__(self, key):
        """Support array indexing"""
        return self._data[key]
    
    def get_marker(self, marker_idx: int) -> np.ndarray:
        """Get genotypes for a specific marker"""
        return self._data[:, marker_idx]
    
    def get_individual(self, ind_idx: int) -> np.ndarray:
        """Get genotypes for a specific individual"""
        return self._data[ind_idx, :]
    
    def get_batch(self, marker_start: int, marker_end: int) -> np.ndarray:
        """Get batch of markers for efficient processing"""
        return self._data[:, marker_start:marker_end]
    
    def calculate_allele_frequencies(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate allele frequencies for all markers.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        n_markers = self.n_markers
        frequencies = np.zeros(n_markers)

        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)
            # Frequency of alt allele = mean(genotype) / max_dosage
            frequencies[start:end] = np.mean(batch, axis=0) / max(max_dosage, 1e-12)

        return frequencies

    def calculate_maf(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate minor allele frequencies.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        frequencies = self.calculate_allele_frequencies(
            batch_size=batch_size,
            max_dosage=max_dosage,
        )
        return np.minimum(frequencies, 1 - frequencies)
    
    def _precompute_major_alleles(self, batch_size: int = 1000):
        """Pre-compute major alleles for all markers to optimize missing data imputation
        
        This matches rMVP's missing data imputation strategy exactly.
        """
        n_markers = self.n_markers
        self._major_alleles = np.zeros(n_markers, dtype=self._data.dtype)
        
        # Process in batches to handle large datasets
        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)
            
            # For each marker in batch, find major allele
            for i, marker_idx in enumerate(range(start, end)):
                marker = batch[:, i]
                
                # Handle missing data (-9 values) - exactly like rMVP
                missing_mask = (marker == -9) | np.isnan(marker)
                
                if np.sum(missing_mask) == len(marker):
                    # All samples are missing - use 0 as default (major allele)
                    self._major_alleles[marker_idx] = 0
                    continue
                
                if np.sum(missing_mask) > 0:
                    # Some samples have missing data
                    non_missing_values = marker[~missing_mask]
                    if len(non_missing_values) == 0:
                        self._major_alleles[marker_idx] = 0
                        continue
                    
                    # Find major allele (most common genotype) - exactly like rMVP
                    unique_vals, counts = np.unique(non_missing_values, return_counts=True)
                    major_allele_idx = np.argmax(counts)
                    self._major_alleles[marker_idx] = unique_vals[major_allele_idx]
                else:
                    # No missing data - still compute for consistency
                    unique_vals, counts = np.unique(marker, return_counts=True)
                    major_allele_idx = np.argmax(counts)
                    self._major_alleles[marker_idx] = unique_vals[major_allele_idx]
    
    def get_marker_imputed(self, marker_idx: int, *, fill_value: Optional[float] = None) -> np.ndarray:
        """Get genotypes for a specific marker with optional missing-data imputation."""
        marker = self._data[:, marker_idx].copy().astype(np.float64)

        missing_mask = (marker == -9) | np.isnan(marker)
        if not missing_mask.any():
            return marker

        if fill_value is not None:
            marker[missing_mask] = float(fill_value)
        elif self._major_alleles is not None:
            marker[missing_mask] = self._major_alleles[marker_idx]
        else:
            marker[missing_mask] = 0.0

        return marker

    def get_batch_imputed(self,
                          marker_start: int,
                          marker_end: int,
                          *,
                          fill_value: Optional[float] = None) -> np.ndarray:
        """Get batch of markers with missing data imputed.

        Args:
            marker_start: Inclusive start index
            marker_end: Exclusive end index
            fill_value: Optional constant to impute missing genotypes.
                If None, the pre-computed major allele is used (rMVP default).
        """
        batch = self._data[:, marker_start:marker_end].copy().astype(np.float64)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = float(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[marker_start:marker_end].astype(np.float64)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = 0.0

        return batch

    def get_columns_imputed(self,
                             indices: Union[np.ndarray, list],
                             *,
                             fill_value: Optional[float] = None) -> np.ndarray:
        """Get arbitrary marker columns with missing data imputed.

        Optimized for fetching non-contiguous SNPs (e.g., pseudo-QTNs) in one call.
        Returns a float64 array of shape (n_individuals, len(indices)).
        """
        if isinstance(indices, list):
            indices = np.array(indices, dtype=int)
        # Slice and copy to ensure we don't mutate underlying storage
        batch = self._data[:, indices].copy().astype(np.float64)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = float(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[indices].astype(np.float64)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = 0.0

        return batch
    
    @property
    def major_alleles(self) -> Optional[np.ndarray]:
        """Pre-computed major alleles for all markers"""
        return self._major_alleles


class AssociationResults:
    """GWAS association results structure
    
    Standard format: [Effect, SE, P-value] for each marker
    """
    
    def __init__(self, effects: np.ndarray, se: np.ndarray, pvalues: np.ndarray,
                 snp_map: Optional[GenotypeMap] = None):
        
        if not (len(effects) == len(se) == len(pvalues)):
            raise ValueError("All result arrays must have same length")
            
        self.effects = effects
        self.se = se  
        self.pvalues = pvalues
        self.snp_map = snp_map
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self.effects)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'Effect': self.effects,
            'SE': self.se,
            'P-value': self.pvalues
        })
        
        if self.snp_map is not None:
            df['SNP'] = self.snp_map.snp_ids.values
            df['Chr'] = self.snp_map.chromosomes.values  
            df['Pos'] = self.snp_map.positions.values
            df = df[['SNP', 'Chr', 'Pos', 'Effect', 'SE', 'P-value']]
            
        return df
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [Effect, SE, P-value]"""
        return np.column_stack([self.effects, self.se, self.pvalues])


class KinshipMatrix:
    """Kinship matrix with validation and properties
    
    Must be symmetric positive semi-definite matrix
    """
    
    def __init__(self, data: Union[np.ndarray, str, Path]):
        if isinstance(data, (str, Path)):
            # Load from file - handle CSV with headers
            try:
                df = pd.read_csv(data, header=0)
                self._data = df.values.astype(float)
            except:
                # Try without headers  
                self._data = np.loadtxt(data, delimiter=',', skiprows=1)
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            raise ValueError("Data must be array or file path")
            
        # Validate properties
        if self._data.ndim != 2:
            raise ValueError("Kinship matrix must be 2D")
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Kinship matrix must be square")
        if not np.allclose(self._data, self._data.T, atol=1e-10):
            raise ValueError("Kinship matrix must be symmetric")
            
        self.n = self._data.shape[0]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape"""
        return self._data.shape
    
    def __getitem__(self, key):
        """Support array indexing"""
        return self._data[key]
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data.copy()
    
    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigendecomposition for MLM"""
        eigenvals, eigenvecs = np.linalg.eigh(self._data)
        return eigenvals, eigenvecs


def load_validation_data(data_path: Union[str, Path]) -> Dict[str, Any]:
    """Load validation data for testing
    
    Returns dictionary with test data matching R rMVP outputs
    """
    data_path = Path(data_path)
    
    result = {}
    
    # Load test data
    if (data_path / "test_phenotype.csv").exists():
        result['phenotype'] = Phenotype(data_path / "test_phenotype.csv")
    
    if (data_path / "test_genotype_full.csv").exists():
        geno_data = pd.read_csv(data_path / "test_genotype_full.csv").values
        result['genotype'] = GenotypeMatrix(geno_data)
    
    if (data_path / "test_map.csv").exists():
        result['map'] = GenotypeMap(data_path / "test_map.csv")
    
    # Load expected results
    if (data_path / "test_glm_results.csv").exists():
        glm_data = pd.read_csv(data_path / "test_glm_results.csv").values
        result['expected_glm'] = AssociationResults(
            glm_data[:, 0], glm_data[:, 1], glm_data[:, 2]
        )
    
    if (data_path / "test_mlm_results.csv").exists():
        mlm_data = pd.read_csv(data_path / "test_mlm_results.csv").values  
        result['expected_mlm'] = AssociationResults(
            mlm_data[:, 0], mlm_data[:, 1], mlm_data[:, 2]
        )
    
    if (data_path / "test_kinship.csv").exists():
        result['expected_kinship'] = KinshipMatrix(data_path / "test_kinship.csv")
        
    if (data_path / "test_pca_results.csv").exists():
        result['expected_pca'] = pd.read_csv(data_path / "test_pca_results.csv").values
    
    return result
