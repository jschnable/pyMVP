"""
Kinship matrix computation using VanRaden method
"""

import numpy as np
from typing import Optional, Union, Tuple
from ..utils.data_types import GenotypeMatrix, KinshipMatrix
import warnings

def MVP_K_VanRaden(M: Union[GenotypeMatrix, np.ndarray], 
                   marker_freq: Optional[np.ndarray] = None,
                   maxLine: int = 5000, 
                   cpu: int = 1,
                   verbose: bool = True,
                   return_eigen: bool = False) -> Union[KinshipMatrix, Tuple[KinshipMatrix, dict]]:
    """VanRaden kinship matrix calculation
    
    Implements the VanRaden method for kinship matrix calculation:
    K = ZZ' / mean(diag(ZZ'))
    where Z = M - 2p (centered genotype matrix)
    
    Args:
        M: Genotype matrix (n_individuals × n_markers)
        marker_freq: Optional marker allele frequencies 
        maxLine: Batch size for processing markers
        cpu: Number of CPU threads (currently ignored)
        verbose: Print progress information
        return_eigen: If True, also return eigendecomposition for MLM speedup
    
    Returns:
        KinshipMatrix object or tuple (KinshipMatrix, eigendecomposition_dict)
    """
    # Handle different input types
    if isinstance(M, GenotypeMatrix):
        genotype = M
        n_individuals = M.n_individuals
        n_markers = M.n_markers
    elif isinstance(M, np.ndarray):
        genotype = M
        n_individuals, n_markers = M.shape
    else:
        raise ValueError("M must be GenotypeMatrix or numpy array")
    
    if verbose:
        print(f"Calculating kinship matrix for {n_individuals} individuals, {n_markers} markers")
    
    # Initialize kinship matrix
    kin = np.zeros((n_individuals, n_individuals), dtype=np.float64)
    
    if verbose:
        print("Computing kinship matrix in batches...")
    
    # Process markers in batches to manage memory
    n_batches = (n_markers + maxLine - 1) // maxLine
    
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)
        
        if verbose and n_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{n_batches} (markers {start_marker}-{end_marker-1})")
        
        # Get batch of markers
        if isinstance(genotype, GenotypeMatrix):
            # Use major-genotype imputed data to match rMVP
            Z_batch = genotype.get_batch_imputed(start_marker, end_marker).astype(np.float64)
        else:
            Z_batch = genotype[:, start_marker:end_marker].astype(np.float64)
        
        # Center the genotype matrix by subtracting per-marker means (2p)
        means_batch = np.mean(Z_batch, axis=0)
        Z_batch -= means_batch[np.newaxis, :]
        
        # Compute cross products: kin += Z_batch @ Z_batch.T
        # Using numpy's optimized matrix multiplication
        kin += Z_batch @ Z_batch.T
    
    if verbose:
        print("Symmetrizing and normalizing kinship matrix...")
    
    # Ensure symmetry (should already be symmetric, but numerical precision)
    kin = (kin + kin.T) / 2.0
    
    # Normalize by mean diagonal element (VanRaden scaling)
    mean_diag = np.mean(np.diag(kin))
    if mean_diag > 0:
        kin /= mean_diag
    else:
        warnings.warn("Mean diagonal element is zero or negative, skipping normalization")
    
    if verbose:
        print(f"Kinship matrix matrix computation complete. Mean diagonal: {mean_diag:.6f}")
    
    kinship_matrix = KinshipMatrix(kin)
    
    # Optionally compute eigendecomposition for MLM speedup
    if return_eigen:
        if verbose:
            print("Computing eigendecomposition for MLM optimization...")
        eigenvals, eigenvecs = np.linalg.eigh(kin)
        
        # Sort eigenvalues and eigenvectors in descending order (for numerical stability)
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        eigenK = {
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs
        }
        
        if verbose:
            print(f"Eigendecomposition complete. Range of eigenvalues: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
        
        return kinship_matrix, eigenK
    
    return kinship_matrix


def MVP_K_IBS(M: Union[GenotypeMatrix, np.ndarray], 
              maxLine: int = 5000,
              verbose: bool = True) -> KinshipMatrix:
    """Identity by State (IBS) kinship matrix calculation
    
    Alternative kinship calculation based on identity by state.
    K_ij = (number of shared alleles) / (total markers)
    
    Args:
        M: Genotype matrix (n_individuals × n_markers) 
        maxLine: Batch size for processing
        verbose: Print progress information
    
    Returns:
        KinshipMatrix object
    """
    # Handle input types
    if isinstance(M, GenotypeMatrix):
        genotype = M
        n_individuals = M.n_individuals
        n_markers = M.n_markers
    elif isinstance(M, np.ndarray):
        genotype = M
        n_individuals, n_markers = M.shape
    else:
        raise ValueError("M must be GenotypeMatrix or numpy array")
    
    if verbose:
        print(f"Calculating IBS kinship matrix for {n_individuals} individuals, {n_markers} markers")
    
    # Initialize kinship matrix
    kin = np.zeros((n_individuals, n_individuals), dtype=np.float64)
    
    # Process in batches
    n_batches = (n_markers + maxLine - 1) // maxLine
    
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)
        
        if verbose and n_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{n_batches}")
        
        # Get batch of markers
        if isinstance(genotype, GenotypeMatrix):
            batch = genotype.get_batch(start_marker, end_marker)
        else:
            batch = genotype[:, start_marker:end_marker]
        
        # Calculate IBS for this batch
        for i in range(n_individuals):
            for j in range(i, n_individuals):
                # Count shared alleles
                geno_i = batch[i, :]
                geno_j = batch[j, :]
                
                # IBS calculation: 1 - |geno_i - geno_j| / 2
                # This gives 1 for identical genotypes, 0.5 for one allele shared, 0 for no shared alleles
                ibs = 1.0 - np.abs(geno_i - geno_j) / 2.0
                kin[i, j] += np.sum(ibs)
                if i != j:
                    kin[j, i] += np.sum(ibs)
    
    # Normalize by total number of markers
    kin /= n_markers
    
    if verbose:
        print("IBS kinship matrix computation complete")
    
    return KinshipMatrix(kin)


def validate_kinship_matrix(K: Union[KinshipMatrix, np.ndarray], 
                           tolerance: float = 1e-10) -> Tuple[bool, list]:
    """Validate kinship matrix properties
    
    Args:
        K: Kinship matrix to validate
        tolerance: Numerical tolerance for checks
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Convert to numpy if needed
    if isinstance(K, KinshipMatrix):
        matrix = K.to_numpy()
    else:
        matrix = K
    
    # Check if square
    if matrix.shape[0] != matrix.shape[1]:
        errors.append("Matrix is not square")
        return False, errors
    
    # Check if symmetric
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        errors.append("Matrix is not symmetric")
    
    # Check if positive semi-definite (all eigenvalues >= 0)
    try:
        eigenvals = np.linalg.eigvals(matrix)
        if np.any(eigenvals < -tolerance):
            errors.append("Matrix is not positive semi-definite")
    except np.linalg.LinAlgError:
        errors.append("Failed to compute eigenvalues")
    
    # Check diagonal elements (should generally be positive)
    diag_elements = np.diag(matrix)
    if np.any(diag_elements < 0):
        errors.append("Some diagonal elements are negative")
    
    return len(errors) == 0, errors
