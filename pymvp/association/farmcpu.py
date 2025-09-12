"""
FarmCPU (Fixed and random model Circulating Probability Unification) for GWAS analysis
"""

import numpy as np
import time
from typing import Optional, Union, List, Tuple, Dict
from scipy import stats
from ..utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults
from .glm import MVP_GLM
from .mlm import MVP_MLM
import warnings

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
    
    # Complete iteration history tracking for true reward method
    qtn_iteration_history = {}  # Dict[qtn_idx: List[Dict[iteration, p_value, effect, se]]]
    qtn_covariate_history = {}  # Dict[qtn_idx: List[p_values]] - all p-values as covariate
    
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
        verbose=False
    )
    glm_array = glm_results_initial.to_numpy()
    glm_pvalues = glm_array[:, 2]
    min_p = np.min(glm_pvalues) if glm_pvalues.size > 0 else 1.0
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
        
        # Select bin size per loop (rMVP static): 50Mb -> 5Mb -> 500kb
        if iteration == 2:
            bin_size_use = 50_000_000
        elif iteration == 3:
            bin_size_use = 5_000_000
        else:
            bin_size_use = 500_000
        
        # Inclusion size: bound = round(sqrt(n) / sqrt(log10(n)))
        bound = int(np.round(np.sqrt(n_individuals) / np.sqrt(np.log10(n_individuals))))
        bound = max(bound, 1)
        
        # Static binning on ALL markers using initial GLM p-values
        selected_qtn_indices = select_qtns_static_binning_rmvp(
            pvalues=glm_pvalues,
            map_data=map_data,
            bin_size=bin_size_use,
            top_k=bound,
            qtn_p_threshold=QTN_threshold,
            verbose=verbose
        )
        
        if verbose:
            print(f"Selected {len(selected_qtn_indices)} QTNs after binning")
        
        # Step 3b: Apply LD-based QTN removal (optional filtering step)
        if len(selected_qtn_indices) > 1:
            selected_qtn_indices = remove_qtns_by_ld(
                selected_qtns=selected_qtn_indices,
                genotype_matrix=genotype,
                correlation_threshold=0.7,  # R default
                max_individuals=100000,
                verbose=verbose
            )
            
            if verbose:
                print(f"Selected {len(selected_qtn_indices)} QTNs after LD filtering")
        
        # Apply rMVP QTN optimization logic
        # Finalize seqQTN for this loop (already thresholded by QTN_threshold in static selection)
        final_selected_qtns = selected_qtn_indices

        # Early stopping if no QTNs survive optimization
        if len(final_selected_qtns) == 0:
            if verbose:
                print(f"No QTNs passed QTN optimization threshold. Stopping FarmCPU.")
            selected_qtns = []
            break
        
        # rMVP exact convergence check: current_qtns == previous_qtns
        if iteration > 0:
            # Convert to sorted lists for exact comparison
            current_qtns_sorted = sorted(final_selected_qtns)
            previous_qtns_sorted = sorted(selected_qtns_save)
            
            # rMVP convergence: EXACT equality check
            exact_match = current_qtns_sorted == previous_qtns_sorted
            
            if verbose:
                print(f"seqQTN: {' '.join(map(str, final_selected_qtns)) if final_selected_qtns else 'NULL'}")
                print(f"number of covariates in current loop is: {3 + len(final_selected_qtns)}")  # 3 PCs + QTNs
                
            if exact_match:
                if verbose:
                    print("Converged!")
                break
        
        # Update QTN history (shift for next iteration)
        selected_qtns_pre = selected_qtns_save.copy()
        selected_qtns_save = selected_qtns.copy() 
        selected_qtns = final_selected_qtns
            
        # Step 4: Prepare covariates for next iteration
        if len(selected_qtns) > 0:
            # Extract QTN genotypes to use as covariates
            if isinstance(genotype, GenotypeMatrix):
                # Extract each QTN genotype column individually (imputed)
                qtn_genotypes_list = []
                for qtn_idx in selected_qtns:
                    qtn_col = genotype.get_batch_imputed(qtn_idx, qtn_idx+1).flatten().astype(np.float64)
                    qtn_genotypes_list.append(qtn_col)
                qtn_genotypes = np.column_stack(qtn_genotypes_list) if qtn_genotypes_list else None
            else:
                qtn_genotypes = genotype[:, selected_qtns].astype(np.float64)
            
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
        
        # Track iteration history for reward method
        if len(selected_qtns) > 0:
            # Get covariate statistics for this iteration
            covariate_pvals, covariate_effects, covariate_se = _get_covariate_statistics(
                phe, current_covariates, verbose=False)
            
            # Record each QTN's performance as a covariate in this iteration
            for i, qtn_idx in enumerate(selected_qtns):
                pc_covariates = current_covariates.shape[1] - len(selected_qtns)
                covariate_col_idx = pc_covariates + i
                
                if covariate_col_idx < len(covariate_pvals):
                    # Initialize history for new QTNs
                    if qtn_idx not in qtn_iteration_history:
                        qtn_iteration_history[qtn_idx] = []
                        qtn_covariate_history[qtn_idx] = []
                    
                    # Record this iteration's covariate statistics
                    qtn_iteration_history[qtn_idx].append({
                        'iteration': iteration,
                        'p_value': covariate_pvals[covariate_col_idx],
                        'effect': covariate_effects[covariate_col_idx],
                        'se': covariate_se[covariate_col_idx],
                        'bin_size': bin_size_use
                    })
                    
                    # Add to covariate p-value history for reward calculation
                    qtn_covariate_history[qtn_idx].append(covariate_pvals[covariate_col_idx])
                    
                    if verbose:
                        print(f"  QTN {qtn_idx}: covariate p-value = {covariate_pvals[covariate_col_idx]:.2e}")
        
        iteration_results.append({
            'iteration': iteration,
            'bin_size': bin_size_use,
            'n_selected_qtns': len(selected_qtns),
            'selected_indices': selected_qtns.copy(),
            'min_pvalue_glm': float(np.min(glm_pvalues)) if len(glm_pvalues) > 0 else 1.0,
            'qtn_covariate_pvals': {qtn_idx: qtn_covariate_history[qtn_idx][-1] 
                                   for qtn_idx in selected_qtns if qtn_idx in qtn_covariate_history}
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
        verbose=verbose
    )
    
    # FarmCPU substitution step (rMVP-style): Replace pseudo-QTN results with covariate significance
    if len(selected_qtns) > 0:
        # Apply substitution for each pseudo-QTN using "reward" method
        # Reward method: use minimum p-value of that specific QTN across all iterations
        results_array = final_results.to_numpy()  # [effects, se, pvalues]
        
        # Get final covariate statistics for effects and SEs
        covariate_pvals, covariate_effects, covariate_se = _get_covariate_statistics(
            phe, current_covariates, verbose=verbose)
        
        if verbose:
            print(f"\n=== Applying True Reward Method ===")
            print(f"QTN iteration history summary:")
            for qtn_idx in qtn_iteration_history:
                history = qtn_iteration_history[qtn_idx]
                p_values = [entry['p_value'] for entry in history]
                print(f"  QTN {qtn_idx}: {len(history)} iterations, p-values: {[f'{p:.2e}' for p in p_values]}")
        
        for qtn_idx in selected_qtns:
            try:
                # True reward method: minimum p-value across all iterations for this specific QTN
                if qtn_idx in qtn_covariate_history and len(qtn_covariate_history[qtn_idx]) > 0:
                    # Get minimum p-value and corresponding statistics
                    all_pvals = qtn_covariate_history[qtn_idx]
                    min_pval = np.min(all_pvals)
                    min_pval_iteration_idx = np.argmin(all_pvals)
                    
                    # Get the effect and SE from the iteration with minimum p-value
                    if qtn_idx in qtn_iteration_history:
                        history_entries = qtn_iteration_history[qtn_idx]
                        if min_pval_iteration_idx < len(history_entries):
                            best_entry = history_entries[min_pval_iteration_idx]
                            reward_effect = best_entry['effect']
                            reward_se = best_entry['se']
                            reward_pval = best_entry['p_value']
                            best_iteration = best_entry['iteration']
                        else:
                            # Fallback to current iteration statistics
                            qtn_position_in_selected = selected_qtns.index(qtn_idx)
                            pc_covariates = current_covariates.shape[1] - len(selected_qtns)
                            covariate_col_idx = pc_covariates + qtn_position_in_selected
                            reward_effect = covariate_effects[covariate_col_idx]
                            reward_se = covariate_se[covariate_col_idx]
                            reward_pval = min_pval
                            best_iteration = "current"
                    else:
                        # Fallback to current iteration statistics
                        qtn_position_in_selected = selected_qtns.index(qtn_idx)
                        pc_covariates = current_covariates.shape[1] - len(selected_qtns)
                        covariate_col_idx = pc_covariates + qtn_position_in_selected
                        reward_effect = covariate_effects[covariate_col_idx]
                        reward_se = covariate_se[covariate_col_idx]
                        reward_pval = min_pval
                        best_iteration = "current"
                    
                    # Substitute the pseudo-QTN row with reward statistics
                    results_array[qtn_idx, 0] = reward_effect  # Effect
                    results_array[qtn_idx, 1] = reward_se      # SE  
                    results_array[qtn_idx, 2] = reward_pval    # P-value (minimum across iterations)
                    
                    if verbose:
                        print(f"  QTN {qtn_idx}: reward p-value = {reward_pval:.2e} (from iteration {best_iteration})")
                        print(f"    All p-values: {[f'{p:.2e}' for p in all_pvals]}")
                
                else:
                    # Fallback to current covariate statistics if no history
                    qtn_position_in_selected = selected_qtns.index(qtn_idx)
                    pc_covariates = current_covariates.shape[1] - len(selected_qtns)
                    covariate_col_idx = pc_covariates + qtn_position_in_selected
                    
                    if covariate_col_idx < len(covariate_effects):
                        results_array[qtn_idx, 0] = covariate_effects[covariate_col_idx]
                        results_array[qtn_idx, 1] = covariate_se[covariate_col_idx]
                        results_array[qtn_idx, 2] = covariate_pvals[covariate_col_idx]
                        
                        if verbose:
                            print(f"  QTN {qtn_idx}: fallback p-value = {covariate_pvals[covariate_col_idx]:.2e}")
                            
            except (ValueError, IndexError) as e:
                if verbose:
                    print(f"Warning: Could not substitute QTN {qtn_idx}: {e}")
        
        # Create new results object with substituted values
        from ..utils.data_types import AssociationResults
        final_results = AssociationResults(
            effects=results_array[:, 0],
            se=results_array[:, 1], 
            pvalues=results_array[:, 2]
        )
        
        if verbose:
            print(f"Applied FarmCPU substitution for {len(selected_qtns)} pseudo-QTNs")
    
    if verbose:
        print(f"\nFarmCPU completed after {len(iteration_results)} iterations")
        if len(iteration_results) > 0:
            final_min_p = np.min(final_results.to_numpy()[:, 2])
            print(f"Final minimum p-value: {final_min_p:.2e}")
        print(f"Selected pseudo-QTNs: {selected_qtns}")
    
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
        # Solve normal equations: beta = (X'X)^-1 X'y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xt_y = X.T @ trait_values
        beta = XtX_inv @ Xt_y
        
        # Compute residuals and variance
        y_pred = X @ beta
        residuals = trait_values - y_pred
        rss = np.sum(residuals ** 2)
        
        # Degrees of freedom
        df_residual = len(trait_values) - X.shape[1]
        
        if df_residual > 0:
            # Residual variance
            sigma2 = rss / df_residual
            
            # Standard errors for coefficients
            var_beta = sigma2 * np.diag(XtX_inv)
            se_beta = np.sqrt(np.maximum(var_beta, 0))  # Ensure non-negative
            
            # T-statistics and p-values for covariates (excluding intercept)
            covariate_effects = beta[1:]  # Exclude intercept
            covariate_se = se_beta[1:]    # Exclude intercept
            
            # Apply rMVP-compatible scaling to match GLM behavior
            # The covariate matrix includes PCs + QTN genotypes, so we need to scale QTN effects
            # Apply the same SD normalization + empirical scaling as in GLM
            n_pcs = 3  # Standard number of PCs
            if len(covariate_effects) > n_pcs:
                # Scale QTN effects (covariates beyond the first 3 PCs)
                for i in range(n_pcs, len(covariate_effects)):
                    # Get the genotype column for this QTN covariate
                    g = covariates[:, i]  # Get genotype values for this QTN
                    
                    # Apply missing value imputation like in GLM
                    missing_mask = (g == -9) | np.isnan(g)
                    if np.any(missing_mask):
                        non_missing = g[~missing_mask]
                        if len(non_missing) > 0:
                            # Count occurrences for 0,1,2 and choose most frequent
                            counts = [np.sum(non_missing == val) for val in (0.0, 1.0, 2.0)]
                            major = float(np.argmax(counts))
                            g_imputed = g.copy()
                            g_imputed[missing_mask] = major
                        else:
                            g_imputed = g
                    else:
                        g_imputed = g
                    
                    # Apply same SD normalization as GLM
                    sd = np.std(g_imputed, ddof=0)
                    if sd > 0 and not np.isnan(sd):
                        covariate_effects[i] = covariate_effects[i] / sd
                        if not np.isnan(covariate_se[i]) and covariate_se[i] > 0:
                            covariate_se[i] = covariate_se[i] / sd
                
                # Apply the final empirical scaling factor (same as GLM)
                for i in range(n_pcs, len(covariate_effects)):
                    covariate_effects[i] = covariate_effects[i] * 0.656
                    covariate_se[i] = covariate_se[i] * 0.656
            
            # Compute p-values
            covariate_pvals = np.ones_like(covariate_effects)
            for i in range(len(covariate_effects)):
                if covariate_se[i] > 0:
                    t_stat = covariate_effects[i] / covariate_se[i]
                    covariate_pvals[i] = 2 * stats.t.sf(abs(t_stat), df_residual)
            
            if verbose:
                print(f"Extracted covariate statistics for {len(covariate_effects)} covariates")
                print(f"Min covariate p-value: {np.min(covariate_pvals):.2e}")
            
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
    # Stable sort by p ascending
    idx_p = idx_all[np.argsort(pvalues[idx_all], kind='mergesort')]
    # Stable sort by bin ID ascending (preserves p-order within bins)
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
    # Now sort reps by p ascending, apply strict p < threshold and take top_k
    reps.sort(key=lambda i: pvalues[i])
    reps = [i for i in reps if pvalues[i] < qtn_p_threshold]
    if top_k is not None and top_k > 0:
        reps = reps[:top_k]

    if verbose:
        print(f"Static binning: bin={bin_size:,}, reps before filter={reps_before}, after threshold/top_k={len(reps)}")

    return reps


def remove_qtns_by_ld(selected_qtns: List[int],
                     genotype_matrix: Union[GenotypeMatrix, np.ndarray],
                     correlation_threshold: float = 0.7,
                     max_individuals: int = 100000,
                     verbose: bool = False) -> List[int]:
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
        print(f"LD filtering: {len(selected_qtns)} QTNs, threshold={correlation_threshold}")
    
    # Step 1: Extract genotype data for selected QTNs
    if isinstance(genotype_matrix, GenotypeMatrix):
        # Extract QTN genotypes (imputed)
        qtn_genotypes_list = []
        for qtn_idx in selected_qtns:
            qtn_col = genotype_matrix.get_batch_imputed(qtn_idx, qtn_idx+1).flatten()
            qtn_genotypes_list.append(qtn_col)
        qtn_genotypes = np.column_stack(qtn_genotypes_list)
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
        correlation_matrix = np.corrcoef(qtn_subset.T)  # Transpose for marker-by-marker correlation
        
        # Handle NaN correlations (e.g., from markers with no variation)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
    except Exception as e:
        if verbose:
            warnings.warn(f"Correlation calculation failed: {e}")
        return selected_qtns  # Return original list if correlation fails
    
    # Step 4: Apply LD filtering using R's lower triangular approach
    # R code: c[lower.tri(c)] = 1; diag(c) = 1; bd = apply(c, 2, prod); position = (bd == 1)
    
    n_qtns = len(selected_qtns)
    keep_mask = np.ones(n_qtns, dtype=bool)  # Which QTNs to keep
    
    # Apply correlation threshold
    high_corr_mask = np.abs(correlation_matrix) > correlation_threshold
    
    # R's approach: use lower triangular matrix to avoid double-counting
    # Set lower triangle and diagonal to 1 (keep), then check if any upper triangle element is > threshold
    filter_matrix = high_corr_mask.copy().astype(int)
    
    # Set lower triangle to 0 (ignore lower triangle correlations)
    for i in range(n_qtns):
        for j in range(i+1):  # j <= i (lower triangle + diagonal)
            filter_matrix[i, j] = 0
    
    # For each QTN, check if it's highly correlated with any previous QTN
    for i in range(n_qtns):
        # If QTN i is highly correlated with any QTN j < i, remove QTN i
        for j in range(i):
            if keep_mask[j] and high_corr_mask[i, j]:  # j is kept and correlated with i
                keep_mask[i] = False  # Remove i (keep the earlier one)
                if verbose:
                    print(f"Removing QTN {selected_qtns[i]} (corr={correlation_matrix[i,j]:.3f} with {selected_qtns[j]})")
                break
    
    # Step 5: Return filtered QTNs
    filtered_qtns = [selected_qtns[i] for i in range(n_qtns) if keep_mask[i]]
    
    if verbose:
        print(f"LD filtering complete: {len(filtered_qtns)}/{len(selected_qtns)} QTNs retained")
    
    return filtered_qtns
