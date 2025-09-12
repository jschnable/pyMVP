"""
Principal Component Analysis for GWAS
"""

import numpy as np
from typing import Optional, Union, Tuple
from ..utils.data_types import GenotypeMatrix, KinshipMatrix
import warnings

def MVP_PCA(M: Optional[Union[GenotypeMatrix, np.ndarray]] = None,
           K: Optional[Union[KinshipMatrix, np.ndarray]] = None, 
           pcs_keep: int = 5,
           maxLine: int = 5000,
           verbose: bool = True) -> np.ndarray:
    """Principal Component Analysis for genotype or kinship data
    
    Performs PCA either on:
    1. Genotype matrix M (markers × individuals covariance)
    2. Kinship matrix K (individuals × individuals relationship)
    
    Args:
        M: Genotype matrix (n_individuals × n_markers), optional
        K: Kinship matrix (n_individuals × n_individuals), optional  
        pcs_keep: Number of principal components to return
        maxLine: Batch size for genotype processing (if using M)
        verbose: Print progress information
    
    Returns:
        Principal components matrix (n_individuals × pcs_keep)
    """
    
    if M is None and K is None:
        raise ValueError("Either genotype matrix M or kinship matrix K must be provided")
    
    if M is not None and K is not None:
        warnings.warn("Both M and K provided, using kinship matrix K")
    
    if K is not None:
        # PCA on kinship matrix (relationship-based PCA)
        return MVP_PCA_kinship(K, pcs_keep, verbose)
    else:
        # PCA on genotype matrix (marker-based PCA)
        return MVP_PCA_genotype(M, pcs_keep, maxLine, verbose)


def MVP_PCA_kinship(K: Union[KinshipMatrix, np.ndarray],
                   pcs_keep: int = 5,
                   verbose: bool = True) -> np.ndarray:
    """PCA on kinship matrix using eigendecomposition
    
    Args:
        K: Kinship matrix (n_individuals × n_individuals)
        pcs_keep: Number of principal components to return
        verbose: Print progress information
    
    Returns:
        Principal components matrix (n_individuals × pcs_keep)
    """
    
    # Convert to numpy if needed
    if isinstance(K, KinshipMatrix):
        kinship_matrix = K.to_numpy()
        n = K.n
    else:
        kinship_matrix = K
        n = K.shape[0]
    
    if verbose:
        print(f"Performing PCA on kinship matrix ({n}×{n})")
    
    # Eigendecomposition of kinship matrix
    if verbose:
        print("Computing eigendecomposition...")
    
    try:
        eigenvals, eigenvecs = np.linalg.eigh(kinship_matrix)
        
        # Sort by eigenvalues in descending order
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        # Keep only the requested number of components
        pcs_keep = min(pcs_keep, n)  # Can't have more PCs than individuals
        
        if verbose:
            print(f"Keeping top {pcs_keep} principal components")
            print(f"Eigenvalues: {eigenvals[:pcs_keep]}")
            explained_var = eigenvals[:pcs_keep] / np.sum(eigenvals) * 100
            print(f"Explained variance: {explained_var}")
        
        return eigenvecs[:, :pcs_keep]
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute eigendecomposition: {e}")


def MVP_PCA_genotype(M: Union[GenotypeMatrix, np.ndarray],
                    pcs_keep: int = 5, 
                    maxLine: int = 5000,
                    verbose: bool = True) -> np.ndarray:
    """PCA on genotype matrix using covariance decomposition
    
    Computes PCA on the genotype covariance matrix G×G'/m where
    G is the centered genotype matrix and m is the number of markers.
    
    Args:
        M: Genotype matrix (n_individuals × n_markers)
        pcs_keep: Number of principal components to return
        maxLine: Batch size for processing markers
        verbose: Print progress information
    
    Returns:
        Principal components matrix (n_individuals × pcs_keep)
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
        print(f"Performing PCA on genotype matrix ({n_individuals}×{n_markers})")
        print("Computing genotype covariance matrix...")
    
    # Compute genotype covariance matrix G×G'/m efficiently
    # This is similar to kinship calculation but without normalization
    covariance = np.zeros((n_individuals, n_individuals), dtype=np.float64)
    
    # Process in batches
    n_batches = (n_markers + maxLine - 1) // maxLine
    
    for batch_idx in range(n_batches):
        start_marker = batch_idx * maxLine
        end_marker = min(start_marker + maxLine, n_markers)
        
        if verbose and n_batches > 1:
            print(f"Processing batch {batch_idx + 1}/{n_batches}")
        
        # Get batch of markers
        if isinstance(genotype, GenotypeMatrix):
            # Use imputed genotypes to match rMVP PCA inputs
            G_batch = genotype.get_batch_imputed(start_marker, end_marker).astype(np.float64)
        else:
            G_batch = genotype[:, start_marker:end_marker].astype(np.float64)
        
        # Center the genotype matrix by per-marker means
        means_batch = np.mean(G_batch, axis=0)
        G_batch -= means_batch[np.newaxis, :]
        
        # Add to covariance matrix
        covariance += G_batch @ G_batch.T
    
    # Normalize by number of markers
    covariance /= n_markers
    
    if verbose:
        print("Computing eigendecomposition of covariance matrix...")
    
    # Eigendecomposition
    try:
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        
        # Sort by eigenvalues in descending order
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        # Keep only positive eigenvalues and corresponding eigenvectors
        positive_mask = eigenvals > 1e-10
        eigenvals = eigenvals[positive_mask]
        eigenvecs = eigenvecs[:, positive_mask]
        
        # Keep only the requested number of components
        pcs_keep = min(pcs_keep, len(eigenvals), n_individuals)
        
        if verbose:
            print(f"Keeping top {pcs_keep} principal components")
            if len(eigenvals) > 0:
                print(f"Eigenvalues: {eigenvals[:pcs_keep]}")
                explained_var = eigenvals[:pcs_keep] / np.sum(eigenvals) * 100
                print(f"Explained variance: {explained_var}")
        
        return eigenvecs[:, :pcs_keep]
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute eigendecomposition: {e}")


def MVP_PCA_SVD(M: Union[GenotypeMatrix, np.ndarray],
                pcs_keep: int = 5,
                center: bool = True,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """PCA using Singular Value Decomposition (alternative method)
    
    Uses SVD for more numerically stable PCA computation.
    Suitable for cases where covariance matrix approach may be unstable.
    
    Args:
        M: Genotype matrix (n_individuals × n_markers)
        pcs_keep: Number of principal components to return
        center: Whether to center the genotype matrix
        verbose: Print progress information
    
    Returns:
        Tuple of (principal_components, explained_variance_ratio)
    """
    
    # Handle input types
    if isinstance(M, GenotypeMatrix):
        # For large matrices, this method may require too much memory
        if M.n_markers > 10000:
            warnings.warn("SVD method may require too much memory for large genotype matrices")
        
        genotype = M[:].astype(np.float64)
    elif isinstance(M, np.ndarray):
        genotype = M.astype(np.float64)
    else:
        raise ValueError("M must be GenotypeMatrix or numpy array")
    
    n_individuals, n_markers = genotype.shape
    
    if verbose:
        print(f"Performing SVD-based PCA on genotype matrix ({n_individuals}×{n_markers})")
    
    # Center the genotype matrix if requested
    if center:
        if verbose:
            print("Centering genotype matrix...")
        marker_means = np.mean(genotype, axis=0)
        genotype -= marker_means[np.newaxis, :]
    
    # Perform SVD
    if verbose:
        print("Computing SVD...")
    
    try:
        U, s, Vt = np.linalg.svd(genotype, full_matrices=False)
        
        # Principal components are the columns of U
        # Eigenvalues are s² / (n_markers - 1)
        eigenvals = (s ** 2) / (n_markers - 1)
        
        # Keep only the requested number of components
        pcs_keep = min(pcs_keep, len(eigenvals))
        
        pcs = U[:, :pcs_keep]
        explained_variance_ratio = eigenvals[:pcs_keep] / np.sum(eigenvals)
        
        if verbose:
            print(f"Keeping top {pcs_keep} principal components")
            print(f"Explained variance ratio: {explained_variance_ratio}")
        
        return pcs, explained_variance_ratio
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute SVD: {e}")


def validate_pca_results(pcs: np.ndarray, 
                        tolerance: float = 1e-10) -> Tuple[bool, list]:
    """Validate PCA results
    
    Args:
        pcs: Principal components matrix (n_individuals × n_components)
        tolerance: Numerical tolerance for orthogonality check
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if pcs.ndim != 2:
        errors.append("PCA results must be a 2D matrix")
        return False, errors
    
    n_individuals, n_components = pcs.shape
    
    # Check orthogonality of principal components
    if n_components > 1:
        dot_product = pcs.T @ pcs
        should_be_identity = np.eye(n_components)
        
        if not np.allclose(dot_product, should_be_identity, atol=tolerance):
            errors.append("Principal components are not orthogonal")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(pcs)) or np.any(np.isinf(pcs)):
        errors.append("PCA results contain NaN or infinite values")
    
    return len(errors) == 0, errors
