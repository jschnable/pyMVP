"""
FarmCPU (Fixed and random model Circulating Probability Unification) for GWAS analysis
"""

import numpy as np
import time
from typing import Optional, Union, List, Tuple, Dict, Sequence
from scipy import stats
from ..utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults
from .glm import MVP_GLM
from .mlm import MVP_MLM
import warnings

# rMVP preprocessing replaces missing genotypes (-9/NA) with the heterozygote dosage (1).
# Use the same fill value so that GLM/FarmCPU statistics remain comparable.
MISSING_FILL_VALUE = 1.0

# Check for JIT availability
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class FarmCPUTimer:
    """Timer class to track FarmCPU performance"""
    
    def __init__(self):
        self.timings = {}
        self.step_timings = []
        self.iteration_timings = []
        
    def start(self, step_name: str):
        """Start timing a step"""
        if step_name not in self.timings:
            self.timings[step_name] = []
        self.start_time = time.time()
        self.current_step = step_name
        
    def end(self, step_name: str = None):
        """End timing a step"""
        if step_name is None:
            step_name = self.current_step
        elapsed = time.time() - self.start_time
        self.timings[step_name].append(elapsed)
        return elapsed
    
    def get_summary(self) -> Dict:
        """Get timing summary statistics"""
        summary = {}
        for step, times in self.timings.items():
            if times:
                summary[step] = {
                    'total_time': sum(times),
                    'mean_time': np.mean(times),
                    'count': len(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
        return summary
    
    def print_summary(self):
        """Print detailed timing summary"""
        print("\n" + "="*60)
        print("🕐 FarmCPU PERFORMANCE TIMING REPORT")
        print("="*60)
        
        summary = self.get_summary()
        total_runtime = sum(sum(times) for times in self.timings.values())
        
        print(f"📊 Total Runtime: {total_runtime:.3f} seconds")
        print(f"🔄 Iterations: {len(self.iteration_timings)}")
        if self.iteration_timings:
            print(f"⏱️  Average per iteration: {np.mean(self.iteration_timings):.3f} seconds")
        
        print("\n📋 Step-by-step breakdown:")
        print("-" * 60)
        
        for step, stats in sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True):
            pct = (stats['total_time'] / total_runtime) * 100
            print(f"{step:25s} | {stats['total_time']:8.3f}s ({pct:5.1f}%) | "
                  f"{stats['count']:3d} calls | avg: {stats['mean_time']:6.3f}s")
        
        print("-" * 60)

def MVP_FarmCPU(phe: np.ndarray,
               geno: Union[GenotypeMatrix, np.ndarray],
               map_data: GenotypeMap,
               CV: Optional[np.ndarray] = None,
               maxLoop: int = 10,
               p_threshold: Optional[float] = 0.05,
               QTN_threshold: float = 0.01,
               bin_size: Optional[List[int]] = None,
               method_bin: str = "static",
               maxLine: int = 5000,
               cpu: int = 1,
               reward_method: str = "min",
               verbose: bool = True) -> AssociationResults:
    """FarmCPU method for GWAS analysis
    
    Fixed and random model Circulating Probability Unification iteratively:
    1. Uses GLM to identify candidate QTNs
    2. Bins markers and selects representative QTNs
    3. Uses MLM with selected QTNs as covariates
    4. Repeats until convergence
    
    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        map_data: Genetic map with SNP positions
        CV: Covariate matrix (n_individuals × n_covariates), optional
        maxLoop: Maximum number of iterations
        p_threshold: P-value threshold for first iteration
        QTN_threshold: P-value threshold for selecting pseudo-QTNs
        bin_size: List of bin sizes for iterations
        method_bin: Binning method ["static", "EMMA", "FaST-LMM"]
        maxLine: Batch size for processing markers
        cpu: Number of CPU threads (currently ignored)
        reward_method: How to substitute pseudo-QTN p-values in final results.
            'min' (default, matches rMVP "reward"): minimum covariate p-value across iterations.
            'last': use covariate p-value from the final iteration only.
        verbose: Print progress information
    
    Returns:
        AssociationResults object containing final Effect, SE, and P-value for each marker
    """
    
    # Handle input validation
    if isinstance(phe, np.ndarray):
        if phe.shape[1] != 2:
            raise ValueError("Phenotype matrix must have 2 columns [ID, trait_value]")
        trait_values = phe[:, 1].astype(np.float64)
    else:
        raise ValueError("Phenotype must be numpy array")
    
    # Handle genotype input
    if isinstance(geno, GenotypeMatrix):
        genotype = geno
        n_individuals = geno.n_individuals
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        genotype = geno
        n_individuals, n_markers = geno.shape
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    # Compute (or obtain) major alleles once to accelerate repeated GLM scans
    # This keeps results identical while avoiding per-chunk bincounts
    precomputed_major_alleles: Optional[np.ndarray] = None
    if isinstance(genotype, GenotypeMatrix):
        precomputed_major_alleles = genotype.major_alleles
    else:
        # Vectorized major allele computation for numpy arrays
        # Treat -9/NaN as missing; counts for 0/1/2 per column
        G = genotype
        # Ensure float64 for NaN checks without copying entire matrix unnecessarily
        # Use logical ops on original dtype to avoid allocation when possible
        if np.issubdtype(G.dtype, np.floating):
            is_miss = np.isnan(G) | (G == -9)
        else:
            is_miss = (G == -9)
        # Build counts for 0,1,2 excluding missing
        counts0 = np.sum((G == 0) & (~is_miss), axis=0)
        counts1 = np.sum((G == 1) & (~is_miss), axis=0)
        counts2 = np.sum((G == 2) & (~is_miss), axis=0)
        counts = np.stack([counts0, counts1, counts2], axis=0)
        major_codes = np.argmax(counts, axis=0).astype(np.int64)  # 0,1,2
        precomputed_major_alleles = major_codes
    
    # Apply rMVP parameter logic: QTN.threshold = max(p.threshold, QTN.threshold)
    # This matches rMVP line: if(!is.na(p.threshold)) QTN.threshold = max(p.threshold, QTN.threshold)
    if p_threshold is not None:
        QTN_threshold = max(p_threshold, QTN_threshold)
        if verbose and QTN_threshold != 0.01:  # Only print if changed from default
            print(f"Applied rMVP logic: QTN_threshold adjusted to {QTN_threshold} (max of p_threshold={p_threshold}, original QTN_threshold)")
    
    if verbose:
        print(f"FarmCPU analysis: {n_individuals} individuals, {n_markers} markers")
        print(f"Parameters: maxLoop={maxLoop}, p_threshold={p_threshold}, QTN_threshold={QTN_threshold}")
        if HAS_NUMBA:
            print("⚡ Using Numba JIT compilation for GLM performance boost")
        else:
            print("💡 Install numba for GLM performance improvements")
    
    # Validate dimensions
    if len(trait_values) != n_individuals:
        raise ValueError("Number of phenotype observations must match number of individuals")
    
    # Set default bin sizes if not provided (matches R default)
    if bin_size is None:
        # R default: bin.size=c(5e5,5e6,5e7)  
        bin_size = [500000, 5000000, 50000000]  # 500KB, 5MB, 50MB
    
    # Initialize variables for iteration (matching R FarmCPU)
    current_covariates = CV.copy() if CV is not None else None
    selected_qtns = []  # seqQTN - current QTNs
    selected_qtns_save = []  # seqQTN.save - previous QTNs  
    selected_qtns_pre = []  # seqQTN.pre - QTNs before previous
    iteration_results = []
    
    # Track latest covariate statistics for each pseudo-QTN
    qtn_latest_stats: Dict[int, Dict[str, float]] = {}
    
    # Expect 3 PCs (rMVP standard). If not provided, warn and proceed.
    if CV is None:
        if verbose:
            print("Warning: No PCs provided — rMVP uses exactly 3 PCs.")
    elif CV.shape[1] != 3 and verbose:
        print(f"Warning: rMVP uses exactly 3 PCs, got {CV.shape[1]}")
    
    if verbose:
        print(f"Starting FarmCPU with multi-scale binning approach")
        print(f"Number of PCs included in FarmCPU: {CV.shape[1]}")
    
    # Loop 1 (initial GLM pass)
    if verbose:
        print("Current loop: 1 out of maximum of {0}".format(maxLoop))
        print("Step 1: Running initial GLM (candidate scan)...")
    glm_results_initial = MVP_GLM(
        phe=phe,
        geno=genotype,
        CV=current_covariates,
        maxLine=maxLine,
        verbose=False,
        impute_missing=True,
        major_alleles=precomputed_major_alleles,
        missing_fill_value=MISSING_FILL_VALUE
    )
    glm_array = glm_results_initial.to_numpy()
    # P-values used to drive binning; updated each iteration after rescans
    current_pvalues = glm_array[:, 2]
    # Preserve initial on-site p-values (P0) for optional 'onsite' substitution
    pvalues_initial = current_pvalues.copy()
    min_p = np.min(current_pvalues) if current_pvalues.size > 0 else 1.0
    # Early stop rule (rMVP): if min p > p_threshold (or 0.01/n if NA), return GLM
    if p_threshold is None:
        p_cut = 0.01 / n_markers
    else:
        p_cut = p_threshold
    if min_p > p_cut:
        if verbose:
            print(f"Early stop: min p ({min_p:.2e}) > threshold ({p_cut:.2e}). Returning GLM results.")
        return glm_results_initial

    # Main FarmCPU iteration loop (loops 2..maxLoop)
    for iteration in range(2, maxLoop + 1):
        if verbose:
            print(f"Current loop: {iteration} out of maximum of {maxLoop}")
            print("Step 2: Binning and QTN selection (static method)...")

        # Inclusion size: bound = round(sqrt(n) / sqrt(log10(n)))
        bound = int(np.round(np.sqrt(n_individuals) / np.sqrt(np.log10(n_individuals))))
        bound = max(bound, 1)

        # Prepare p-values for binning: substitute pseudo-QTNs using previous iteration's selection
        # rMVP applies FarmCPU.SUB before using P for subsequent BIN. Here we emulate 'onsite'
        # substitution to drive binning.
        pvals_for_binning = current_pvalues.copy()
        if selected_qtns_save:
            idx = np.array(selected_qtns_save, dtype=int)
            # On-site substitution for selected QTNs
            pvals_for_binning[idx] = pvalues_initial[idx]

        final_selected_qtns, selection_meta = select_pseudo_qtns_rmvp_iteration(
            pvalues=pvals_for_binning,
            map_data=map_data,
            genotype_matrix=genotype,
            iteration=iteration,
            n_individuals=n_individuals,
            qtn_threshold=QTN_threshold,
            p_threshold=p_threshold,
            previous_qtns=selected_qtns_save,
            bin_sizes=bin_size,
            ld_threshold=0.7,
            bound=bound,
            verbose=verbose,
        )

        bin_size_history = selection_meta.get('bin_size')

        if verbose:
            if selection_meta.get('stopped'):
                print("Early stop triggered by rMVP p-value threshold check.")
            print(f"Selected {len(final_selected_qtns)} QTNs after rMVP-style optimization")
            if final_selected_qtns:
                print(f"seqQTN: {' '.join(map(str, final_selected_qtns))}")
                print(f"number of covariates in current loop is: {3 + len(final_selected_qtns)}")

        if selection_meta.get('stopped'):
            selected_qtns = []
            break

        if len(final_selected_qtns) == 0:
            if verbose:
                print("No QTNs passed QTN optimization threshold. Stopping FarmCPU.")
            selected_qtns = []
            break

        if iteration > 0:
            current_qtns_sorted = sorted(final_selected_qtns)
            previous_qtns_sorted = sorted(selected_qtns_save)
            exact_match = current_qtns_sorted == previous_qtns_sorted

            if verbose and exact_match:
                print("Converged!")
            if exact_match:
                break

        # Update QTN history (shift for next iteration)
        selected_qtns_pre = selected_qtns_save.copy()
        selected_qtns_save = selected_qtns.copy()
        selected_qtns = final_selected_qtns
            
        # Step 4: Prepare covariates for next iteration
        if len(selected_qtns) > 0:
            # Extract QTN genotypes to use as covariates (vectorized)
            if isinstance(genotype, GenotypeMatrix):
                # Fetch non-contiguous columns with imputation in one call
                qtn_genotypes = (
                    genotype.get_columns_imputed(selected_qtns, fill_value=MISSING_FILL_VALUE)
                    if len(selected_qtns) > 0 else None
                )
            else:
                # Numpy array: no imputation needed here because GLM handles it consistently
                qtn_genotypes = genotype[:, selected_qtns].astype(np.float64)
                missing_mask = (qtn_genotypes == -9) | np.isnan(qtn_genotypes)
                if missing_mask.any():
                    qtn_genotypes = qtn_genotypes.copy()
                    qtn_genotypes[missing_mask] = MISSING_FILL_VALUE
            
            # Combine with existing covariates
            if CV is not None and qtn_genotypes is not None:
                current_covariates = np.column_stack([CV, qtn_genotypes])
            elif CV is not None:
                current_covariates = CV
            else:
                current_covariates = qtn_genotypes
                
            if verbose:
                print(f"Created covariate matrix: {current_covariates.shape} (including {len(selected_qtns)} QTNs)")
        else:
            current_covariates = CV

        # Step 5: Re-scan all SNPs with updated covariates to refresh P for next loop
        # This mirrors rMVP's FarmCPU.LM step
        rescan = MVP_GLM(
            phe=phe,
            geno=genotype,
            CV=current_covariates,
            maxLine=maxLine,
            verbose=False,
            impute_missing=True,
            major_alleles=precomputed_major_alleles,
            missing_fill_value=MISSING_FILL_VALUE
        )
        current_pvalues = rescan.to_numpy()[:, 2]
        
        # Track iteration history for reward method
        if len(selected_qtns) > 0:
            covariate_pvals, covariate_effects, covariate_se = _get_covariate_statistics(
                phe, current_covariates, verbose=False)

            pc_covariates = current_covariates.shape[1] - len(selected_qtns)
            for i, qtn_idx in enumerate(selected_qtns):
                covariate_col_idx = pc_covariates + i
                if covariate_col_idx >= len(covariate_pvals):
                    if verbose:
                        print(f"  Warning: covariate index {covariate_col_idx} out of bounds for QTN {qtn_idx}")
                    continue

                reward_pval = float(covariate_pvals[covariate_col_idx])
                reward_effect = float(covariate_effects[covariate_col_idx])
                reward_se = float(covariate_se[covariate_col_idx])

                qtn_latest_stats[qtn_idx] = {
                    'iteration': iteration,
                    'p_value': reward_pval,
                    'effect': reward_effect,
                    'se': reward_se,
                    'bin_size': bin_size_history,
                }

                if 0 <= qtn_idx < len(current_pvalues):
                    current_pvalues[qtn_idx] = reward_pval

        iteration_results.append({
            'iteration': iteration,
            'bin_size': selection_meta.get('bin_size'),
            'n_selected_qtns': len(selected_qtns),
            'selected_indices': selected_qtns.copy(),
            'min_pvalue_glm': float(np.min(current_pvalues)) if len(current_pvalues) > 0 else 1.0,
            'qtn_covariate_pvals': {qtn_idx: stats['p_value'] for qtn_idx, stats in qtn_latest_stats.items()
                                   if qtn_idx in selected_qtns}
        })
    
    # Final GLM analysis with selected QTNs as covariates (standard FarmCPU output)
    if verbose:
        print(f"\n=== Final GLM Analysis ===")
        print(f"Using {len(selected_qtns)} selected QTNs as covariates")
    
    # Run final GLM with selected QTNs as covariates to get GWAS results
    final_results = MVP_GLM(
        phe=phe,
        geno=genotype,
        CV=current_covariates,
        maxLine=maxLine,
        verbose=verbose,
        impute_missing=True,
        major_alleles=precomputed_major_alleles,
        missing_fill_value=MISSING_FILL_VALUE
    )
    
    # FarmCPU substitution step (rMVP-style): Replace pseudo-QTN rows with covariate stats
    if len(selected_qtns) > 0:
        results_array = final_results.to_numpy()
        covariate_pvals, covariate_effects, covariate_se = _get_covariate_statistics(
            phe, current_covariates, verbose=False)

        pc_covariates = current_covariates.shape[1] - len(selected_qtns)
        for i, qtn_idx in enumerate(selected_qtns):
            if not (0 <= qtn_idx < len(results_array)):
                continue

            covariate_col_idx = pc_covariates + i
            if covariate_col_idx >= len(covariate_pvals):
                if verbose:
                    print(f"Warning: Unable to substitute QTN {qtn_idx}; covariate index {covariate_col_idx} out of range")
                continue

            reward_effect = float(covariate_effects[covariate_col_idx])
            reward_se = float(covariate_se[covariate_col_idx])
            reward_pval = float(covariate_pvals[covariate_col_idx])

            qtn_latest_stats[qtn_idx] = {
                'iteration': 'final',
                'p_value': reward_pval,
                'effect': reward_effect,
                'se': reward_se,
                'bin_size': None,
            }

            results_array[qtn_idx, 0] = reward_effect
            results_array[qtn_idx, 1] = reward_se
            results_array[qtn_idx, 2] = reward_pval

        final_results = AssociationResults(
            effects=results_array[:, 0],
            se=results_array[:, 1],
            pvalues=results_array[:, 2],
            snp_map=map_data
        )

        if verbose:
            print(f"Applied FarmCPU substitution for {len(selected_qtns)} pseudo-QTNs")
    
    if verbose:
        print(f"\nFarmCPU completed after {len(iteration_results)} iterations")
        if len(iteration_results) > 0:
            final_min_p = np.min(final_results.to_numpy()[:, 2])
            print(f"Final minimum p-value: {final_min_p:.2e}")
        print(f"Selected pseudo-QTNs: {selected_qtns}")

    # Expose latest selection for debugging/testing utilities
    MVP_FarmCPU.last_selected_qtns = selected_qtns.copy()
    MVP_FarmCPU.last_iteration_details = iteration_results

    return final_results


def _get_covariate_statistics(phe: np.ndarray, covariates: np.ndarray, verbose: bool = False):
    """
    Get covariate p-values, effects, and standard errors by fitting a linear model
    
    This replicates rMVP's approach to extract fixed effect statistics for pseudo-QTNs
    """
    from scipy import stats
    
    # Extract trait values
    trait_values = phe[:, 1].astype(np.float64)
    
    # Set up design matrix with intercept and covariates
    X = np.column_stack([np.ones(len(trait_values)), covariates])
    
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)
        Xt_y = X.T @ trait_values
        beta = XtX_inv @ Xt_y

        y_pred = X @ beta
        residuals = trait_values - y_pred
        rss = float(np.sum(residuals ** 2))

        df_residual = len(trait_values) - X.shape[1]
        if df_residual <= 0:
            return (np.ones(covariates.shape[1]), np.zeros(covariates.shape[1]), np.ones(covariates.shape[1]))

        sigma2 = rss / df_residual
        var_beta = sigma2 * np.diag(XtX_inv)
        se_beta = np.sqrt(np.maximum(var_beta, 0.0))

        covariate_effects = beta[1:]
        covariate_se = se_beta[1:]

        covariate_pvals = np.ones_like(covariate_effects)
        valid = covariate_se > 0
        if np.any(valid):
            t_stats = np.zeros_like(covariate_effects)
            t_stats[valid] = covariate_effects[valid] / covariate_se[valid]
            covariate_pvals[valid] = 2 * stats.t.sf(np.abs(t_stats[valid]), df_residual)

        return covariate_pvals, covariate_effects, covariate_se

    except np.linalg.LinAlgError:
        if verbose:
            print("Warning: Failed to compute covariate statistics due to singular matrix")
    
    # Fallback: return neutral values
    n_covariates = covariates.shape[1]
    return (np.ones(n_covariates), 
            np.zeros(n_covariates), 
            np.ones(n_covariates))


def select_qtns_multiscale_binning_rmvp(candidate_indices: np.ndarray,
                                        pvalues: np.ndarray,
                                        map_data: 'GenotypeMap',
                                        bin_sizes: List[int],
                                        method: str = "static",
                                        percentiles: List[int] = None,
                                        verbose: bool = False) -> List[int]:
    """rMVP multi-scale binning algorithm - exact implementation
    
    Implements rMVP's multi-scale binning approach:
    Each percentile is applied across ALL bin sizes simultaneously:
    - 500K bp bins (fine-scale)
    - 5M bp bins (medium-scale)  
    - 50M bp bins (coarse-scale)
    
    This allows multiple QTNs per genomic region and explains why
    rMVP selects 18+ QTNs instead of just 10 percentiles.
    
    Args:
        candidate_indices: Indices of candidate QTNs
        pvalues: P-values for all markers
        map_data: Genetic map information
        bin_sizes: List of bin sizes [500K, 5M, 50M] for multi-scale binning
        method: Binning method ("static" for rMVP compliance)
        percentiles: List of percentiles [10,20,...,100]
        verbose: Print progress
    
    Returns:
        List of selected QTN indices from multi-scale binning
    """
    
    if len(candidate_indices) == 0:
        return []
    
    if percentiles is None:
        percentiles = list(range(10, 101, 10))  # [10,20,30...100]
    
    if verbose:
        print(f"rMVP multi-scale binning: {len(candidate_indices)} candidates, {len(percentiles)} percentiles, {len(bin_sizes)} scales")
    
    # Get map information
    try:
        map_df = map_data.to_dataframe()
        chromosomes = map_df['CHROM'].values.astype(np.float64)
        positions = map_df['POS'].values.astype(np.float64)
    except Exception as e:
        if verbose:
            warnings.warn(f"Could not access map data for binning: {e}")
        # Fallback: select by percentiles directly from p-values
        sorted_indices = candidate_indices[np.argsort(pvalues[candidate_indices])]
        n_total = len(sorted_indices)
        selected_indices = []
        for pct in percentiles:
            idx = min(int(n_total * pct / 100) - 1, n_total - 1)
            if idx >= 0:
                selected_indices.append(sorted_indices[idx])
        return list(set(selected_indices))  # Remove duplicates
    
    if method.lower() != "static":
        if verbose:
            print(f"Warning: rMVP uses static binning, got {method}")
    
    # rMVP multi-scale static binning implementation
    MaxBP = 1e10  # Maximum base pair value for chromosome encoding
    
    # Step 1: Create SNP ID = position + chromosome * MaxBP (rMVP standard)
    snp_ids = positions[candidate_indices] + chromosomes[candidate_indices] * MaxBP
    candidate_pvalues = pvalues[candidate_indices]
    
    all_selected_qtns = set()  # Use set to avoid duplicates across scales
    
    # Step 2: Apply each percentile across ALL bin sizes (multi-scale approach)
    for bin_size in bin_sizes:
        if verbose:
            print(f"  Processing bin size: {bin_size:,.0f} bp")
        
        # Create bin IDs for this scale
        bin_ids = np.floor(snp_ids / bin_size).astype(np.int64)
        
        # Create binning data with position-based tie breaking
        bin_data = []
        for i in range(len(candidate_indices)):
            bin_data.append({
                'bin_id': bin_ids[i],
                'snp_idx': candidate_indices[i],
                'snp_id': snp_ids[i],
                'pvalue': candidate_pvalues[i],
                'position': positions[candidate_indices[i]],
                'original_order': i
            })
        
        # Sort by p-value first, then by position for tie breaking
        bin_data.sort(key=lambda x: (x['pvalue'], x['position'], x['original_order']))
        
        # Group by bin and select best QTN from each bin
        bin_representatives = {}
        for item in bin_data:
            bin_id = item['bin_id']
            if bin_id not in bin_representatives:
                bin_representatives[bin_id] = item  # Take best (first) QTN per bin
        
        # Get all representative QTNs from this scale
        scale_qtns = list(bin_representatives.values())
        scale_qtns.sort(key=lambda x: (x['pvalue'], x['position'], x['original_order']))
        
        # Apply percentile selection to this scale's QTNs
        n_scale_qtns = len(scale_qtns)
        if n_scale_qtns > 0:
            for pct in percentiles:
                pct_idx = min(int(n_scale_qtns * pct / 100) - 1, n_scale_qtns - 1)
                if pct_idx >= 0:
                    qtn_idx = scale_qtns[pct_idx]['snp_idx']
                    all_selected_qtns.add(qtn_idx)
    
    # Convert set back to list and sort by p-value
    selected_qtns = list(all_selected_qtns)
    if len(selected_qtns) > 1:
        qtn_pvalues = [(idx, pvalues[idx]) for idx in selected_qtns]
        qtn_pvalues.sort(key=lambda x: x[1])  # Sort by p-value
        selected_qtns = [idx for idx, _ in qtn_pvalues]
    
    if verbose:
        print(f"rMVP multi-scale binning result: {len(selected_qtns)} total QTNs from {len(bin_sizes)} scales")
    
    return selected_qtns


def _union_preserve_order(primary: Sequence[int], secondary: Sequence[int]) -> List[int]:
    """Return ordered union matching rMVP's base::union behaviour."""
    seen: set[int] = set()
    merged: List[int] = []
    for idx in list(primary) + list(secondary):
        idx_int = int(idx)
        if idx_int not in seen:
            merged.append(idx_int)
            seen.add(idx_int)
    return merged


def select_pseudo_qtns_rmvp_iteration(
    pvalues: np.ndarray,
    map_data: 'GenotypeMap',
    genotype_matrix: Union[GenotypeMatrix, np.ndarray],
    iteration: int,
    n_individuals: int,
    qtn_threshold: float,
    p_threshold: Optional[float],
    previous_qtns: Sequence[int],
    *,
    bin_sizes: Optional[Sequence[int]] = None,
    ld_threshold: float = 0.7,
    bound: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[int], Dict[str, Union[int, float, List[int]]]]:
    """Exact rMVP pseudo-QTN selection for a single FarmCPU iteration."""

    pvalues = np.asarray(pvalues, dtype=float)
    n_markers = pvalues.shape[0]

    if bin_sizes is None:
        bin_sizes = (500_000, 5_000_000, 50_000_000)
    if len(bin_sizes) < 3:
        raise ValueError("bin_sizes must contain at least three elements")

    if bound is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            bound_est = np.sqrt(float(n_individuals)) / np.sqrt(np.log10(max(n_individuals, 2)))
        bound = max(int(round(bound_est)), 1)

    if iteration == 2:
        bin_size_use = bin_sizes[-1]
    elif iteration == 3:
        bin_size_use = bin_sizes[-2]
    else:
        bin_size_use = bin_sizes[0]

    initial_candidates = select_qtns_static_binning_rmvp(
        pvalues=pvalues,
        map_data=map_data,
        bin_size=bin_size_use,
        top_k=bound,
        qtn_p_threshold=qtn_threshold,
        verbose=verbose,
    )

    metadata: Dict[str, Union[int, float, List[int]]] = {
        'iteration': iteration,
        'bin_size': bin_size_use,
        'initial_candidates': list(initial_candidates),
    }

    finite_mask = np.isfinite(pvalues)
    min_p = float(np.min(pvalues[finite_mask])) if np.any(finite_mask) else float('inf')
    metadata['min_p'] = min_p

    if iteration == 2:
        if p_threshold is not None and not np.isnan(p_threshold):
            threshold_cut = float(p_threshold)
        else:
            threshold_cut = 0.01 / max(n_markers, 1)
        metadata['p_threshold'] = threshold_cut
        if min_p > threshold_cut:
            metadata['stopped'] = True
            return [], metadata

    prev_list = list(previous_qtns) if previous_qtns is not None else []
    candidates = _union_preserve_order(initial_candidates, prev_list)
    metadata['after_union'] = candidates.copy()

    if not candidates:
        metadata['after_threshold'] = []
        metadata['after_ld'] = []
        return [], metadata

    candidate_pvals = pvalues[candidates]
    keep_mask = np.isfinite(candidate_pvals)

    if iteration == 2:
        keep_mask &= (candidate_pvals < qtn_threshold)
    else:
        below = candidate_pvals < qtn_threshold
        prev_set = set(prev_list)
        keep_prev = np.array([idx in prev_set for idx in candidates]) if prev_list else np.zeros(len(candidates), dtype=bool)
        keep_mask &= (below | keep_prev)

    filtered_candidates = [idx for idx, keep in zip(candidates, keep_mask) if keep]
    filtered_pvals = [float(val) for val, keep in zip(candidate_pvals, keep_mask) if keep]
    metadata['after_threshold'] = filtered_candidates.copy()

    if not filtered_candidates:
        metadata['after_ld'] = []
        return [], metadata

    order = np.argsort(filtered_pvals)
    sorted_candidates = [filtered_candidates[i] for i in order]

    pruned = remove_qtns_by_ld(
        selected_qtns=sorted_candidates,
        genotype_matrix=genotype_matrix,
        correlation_threshold=ld_threshold,
        max_individuals=100000,
        map_data=map_data,
        within_chrom_only=True,
        verbose=verbose,
        debug=verbose,
    )

    metadata['after_ld'] = pruned.copy()
    return pruned, metadata


def select_qtns_static_binning_rmvp(pvalues: np.ndarray,
                                    map_data: 'GenotypeMap',
                                    bin_size: int,
                                    top_k: int,
                                    qtn_p_threshold: float = 0.01,
                                    verbose: bool = False) -> List[int]:
    """Static binning (rMVP default) for FarmCPU QTN selection

    Steps per rMVP:
    - Compute SNP ID = CHR * MaxBP + POS (MaxBP = 1e9)
    - binID = floor(ID / bin_size)
    - Within each bin, select SNP with smallest GLM p-value
    - Collect bin representatives, sort by p, filter by p < qtn_p_threshold, take top_k
    """
    map_df = map_data.to_dataframe()
    chrom = map_df['CHROM'].to_numpy()
    pos = map_df['POS'].to_numpy()
    # rMVP uses fixed MaxBP = 1e10 and ID = POS + CHR*MaxBP
    MaxBP = 10_000_000_000
    ids = pos.astype(np.int64) + chrom.astype(np.int64) * MaxBP
    bin_ids = (ids // bin_size).astype(np.int64)

    n = len(pvalues)
    idx_all = np.arange(n)
    # Keep finite p-values only
    finite_mask = np.isfinite(pvalues)
    idx_all = idx_all[finite_mask]
    if idx_all.size == 0:
        return []
    # rMVP sorts by p, then by bin ID. Use stable sorts to keep min-p per bin.
    idx_p = idx_all[np.argsort(pvalues[idx_all], kind='mergesort')]
    idx_pb = idx_p[np.argsort(bin_ids[idx_p], kind='mergesort')]
    # Take first SNP per bin as representative
    seen_bins = set()
    reps = []
    for i in idx_pb:
        b = bin_ids[i]
        if b not in seen_bins:
            seen_bins.add(b)
            reps.append(i)
    reps_before = len(reps)
    # Sort reps by p ascending and take top_k (thresholding applied by caller for rMVP parity)
    reps.sort(key=lambda i: pvalues[i])
    if top_k is not None and top_k > 0:
        reps = reps[:top_k]

    if verbose:
        print(f"Static binning: bin={bin_size:,}, reps before filter={reps_before}, after threshold/top_k={len(reps)}")

    return reps


def remove_qtns_by_ld(selected_qtns: List[int],
                     genotype_matrix: Union[GenotypeMatrix, np.ndarray],
                     correlation_threshold: float = 0.7,
                     max_individuals: int = 100000,
                     map_data: Optional[GenotypeMap] = None,
                     within_chrom_only: bool = False,
                     verbose: bool = False,
                     debug: bool = False) -> List[int]:
    """Remove correlated QTNs using LD-based filtering (matches R FarmCPU.Remove)
    
    Based on R implementation lines 1021-1121 in FarmCPU.Remove:
    1. Calculate correlation matrix on subset of individuals (max 100,000)
    2. Use first N individuals, not random sampling
    3. Remove correlated QTNs using lower triangular masking
    4. Preserve QTNs in order of significance
    
    Args:
        selected_qtns: List of QTN indices to filter
        genotype_matrix: Genotype data (individuals × markers)
        correlation_threshold: Correlation threshold for removal (default 0.7)
        max_individuals: Maximum individuals to use for correlation calculation
        verbose: Print progress
    
    Returns:
        Filtered list of QTN indices
    """
    
    if len(selected_qtns) <= 1:
        return selected_qtns  # Nothing to filter
    
    if verbose:
        scope = "within-chromosome" if within_chrom_only else "genome-wide"
        print(f"LD filtering: {len(selected_qtns)} QTNs, threshold={correlation_threshold} ({scope})")
    
    # Step 1: Extract genotype data for selected QTNs
    if isinstance(genotype_matrix, GenotypeMatrix):
        # Extract QTN genotypes (imputed) in a single call
        qtn_genotypes = genotype_matrix.get_columns_imputed(selected_qtns)
        n_individuals = genotype_matrix.n_individuals
    else:
        qtn_genotypes = genotype_matrix[:, selected_qtns]
        n_individuals = genotype_matrix.shape[0]
    
    # Step 2: Subset individuals for correlation calculation (R: use first N, not random)
    sampled_individuals = min(n_individuals, max_individuals)
    qtn_subset = qtn_genotypes[:sampled_individuals, :]  # Use first N individuals
    
    if verbose:
        print(f"Using {sampled_individuals} individuals for LD calculation")
    
    # Step 3: Calculate correlation matrix
    try:
        # Handle missing values by excluding them pairwise
        correlation_matrix = np.corrcoef(qtn_subset.T)  # marker-by-marker correlation
        
        # Handle NaN correlations (e.g., from markers with no variation)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
    except Exception as e:
        if verbose:
            warnings.warn(f"Correlation calculation failed: {e}")
        return selected_qtns  # Return original list if correlation fails
    
    # Step 4: Apply LD filtering with exact precedence (keep earlier by p-value order)
    n_qtns = len(selected_qtns)
    keep_mask = np.ones(n_qtns, dtype=bool)

    # Consider only within-chromosome pairs if requested
    if within_chrom_only and map_data is not None:
        map_df = map_data.to_dataframe()
        chrom_vec = map_df['CHROM'].to_numpy()
        chrom_sel = chrom_vec[np.array(selected_qtns, dtype=int)]
        same_chrom = (chrom_sel[:, None] == chrom_sel[None, :])
    else:
        same_chrom = np.ones((n_qtns, n_qtns), dtype=bool)

    # Use absolute correlation as in rMVP's Remove step
    abs_corr = np.abs(correlation_matrix)

    # Iterate in order, drop later ones when highly correlated with any kept earlier one on same chromosome
    for i in range(n_qtns):
        if not keep_mask[i]:
            continue
        for j in range(i):
            if keep_mask[j] and same_chrom[i, j] and abs_corr[i, j] > correlation_threshold:
                keep_mask[i] = False
                if verbose or debug:
                    print(f"Removing QTN {selected_qtns[i]} (chr={chrom_sel[i] if within_chrom_only and map_data is not None else 'NA'}) "
                          f"due to high LD with {selected_qtns[j]} (chr={chrom_sel[j] if within_chrom_only and map_data is not None else 'NA'}): "
                          f"r={correlation_matrix[i, j]:.4f}, |r|={abs_corr[i, j]:.4f}")
                break

    # Step 5: Return filtered QTNs
    filtered_qtns = [selected_qtns[i] for i in range(n_qtns) if keep_mask[i]]
    
    if verbose:
        print(f"LD filtering complete: {len(filtered_qtns)}/{len(selected_qtns)} QTNs retained")
    
    return filtered_qtns
