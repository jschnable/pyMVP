"""
Statistical utilities for GWAS analysis
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats

def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Apply Bonferroni correction for multiple testing
    
    Args:
        pvalues: Array of p-values
        alpha: Family-wise error rate (default: 0.05)
        
    Returns:
        Tuple of (corrected_pvalues, corrected_threshold)
    """
    n_tests = len(pvalues)
    corrected_threshold = alpha / n_tests
    corrected_pvalues = np.minimum(pvalues * n_tests, 1.0)
    
    return corrected_pvalues, corrected_threshold

def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05, method: str = 'bh') -> Tuple[np.ndarray, np.ndarray]:
    """Apply False Discovery Rate correction (Benjamini-Hochberg)
    
    Args:
        pvalues: Array of p-values
        alpha: False discovery rate (default: 0.05)
        method: Method ('bh' for Benjamini-Hochberg)
        
    Returns:
        Tuple of (rejected_hypotheses, corrected_pvalues)
    """
    pvalues = np.asarray(pvalues)
    pvalues_sortind = np.argsort(pvalues)
    pvalues_sorted = pvalues[pvalues_sortind]
    sortrevind = pvalues_sortind.argsort()
    
    if method == 'bh':
        # Benjamini-Hochberg procedure
        n = len(pvalues)
        i = np.arange(1, n + 1)
        corrected = pvalues_sorted * n / i
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected_pvalues = corrected[sortrevind]
        rejected = corrected_pvalues <= alpha
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return rejected, corrected_pvalues

def calculate_maf_from_genotypes(genotypes: np.ndarray, missing_value: int = -9) -> np.ndarray:
    """Calculate minor allele frequencies from genotype matrix
    
    Args:
        genotypes: Genotype matrix (individuals × markers)
        missing_value: Value representing missing data
        
    Returns:
        Array of minor allele frequencies for each marker
    """
    n_individuals, n_markers = genotypes.shape
    maf = np.zeros(n_markers)
    
    for i in range(n_markers):
        marker = genotypes[:, i]
        # Remove missing values
        valid_mask = (marker != missing_value) & (~np.isnan(marker))
        valid_genotypes = marker[valid_mask]
        
        if len(valid_genotypes) == 0:
            maf[i] = 0.0
            continue
            
        # Calculate allele frequency (assuming 0,1,2 coding)
        allele_freq = np.mean(valid_genotypes) / 2.0
        # MAF is minimum of freq and 1-freq
        maf[i] = min(allele_freq, 1.0 - allele_freq)
    
    return maf

def genomic_inflation_factor(pvalues: np.ndarray) -> float:
    """Calculate genomic inflation factor (lambda)
    
    Args:
        pvalues: Array of p-values
        
    Returns:
        Genomic inflation factor (lambda)
    """
    valid_pvals = pvalues[np.isfinite(pvalues) & (pvalues > 0)]
    if len(valid_pvals) == 0:
        return 1.0
        
    chi2_values = stats.chi2.ppf(1 - valid_pvals, df=1)
    median_chi2 = np.median(chi2_values)
    expected_median = stats.chi2.ppf(0.5, df=1)
    
    lambda_gc = median_chi2 / expected_median
    return lambda_gc

def qq_plot_data(pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for Q-Q plot
    
    Args:
        pvalues: Array of observed p-values
        
    Returns:
        Tuple of (expected_pvalues, observed_pvalues) for plotting
    """
    valid_pvals = pvalues[np.isfinite(pvalues) & (pvalues > 0)]
    valid_pvals = np.sort(valid_pvals)
    n = len(valid_pvals)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Expected p-values under null hypothesis
    expected_pvals = np.arange(1, n + 1) / (n + 1)
    
    return expected_pvals, valid_pvals