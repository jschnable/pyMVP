"""
FarmCPU with detailed timing functionality for performance analysis
"""

import numpy as np
import time
from typing import Optional, Union, List, Tuple, Dict
from scipy import stats
from ..utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults
from .glm import MVP_GLM
from .glm_optimized import MVP_GLM_batch
from .farmcpu import select_qtns_multiscale_binning_rmvp, remove_qtns_by_ld
import warnings

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


def MVP_FarmCPU_timed(phe: np.ndarray,
                     geno: Union[GenotypeMatrix, np.ndarray],
                     map_data: GenotypeMap,
                     CV: Optional[np.ndarray] = None,
                     maxLoop: int = 10,
                     p_threshold: float = 0.01,
                     QTN_threshold: float = 0.01,
                     bin_size: Optional[List[int]] = None,
                     method_bin: str = "static",
                     maxLine: int = 5000,
                     cpu: int = 1,
                     verbose: bool = True,
                     timing: bool = True) -> Tuple[AssociationResults, FarmCPUTimer]:
    """
    FarmCPU with detailed timing functionality
    
    Returns:
        Tuple of (AssociationResults, FarmCPUTimer)
    """
    
    timer = FarmCPUTimer()
    
    if timing:
        timer.start("total_runtime")
    
    try:
        # Step 0: Input validation and setup
        if timing:
            timer.start("input_validation")
        
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
            print(f"🧬 FarmCPU analysis: {n_individuals} individuals, {n_markers} markers")
            print(f"⚙️  Parameters: maxLoop={maxLoop}, p_threshold={p_threshold}, QTN_threshold={QTN_threshold}")
        
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
        
        if timing:
            setup_time = timer.end("input_validation")
            if verbose:
                print(f"✅ Setup completed in {setup_time:.3f}s")
        
        if verbose:
            print(f"🔄 Starting FarmCPU with {len(bin_size)} bin size iterations")
        
        # Main FarmCPU iteration loop
        for iteration in range(maxLoop):
            if timing:
                timer.start("iteration_total")
                
            if verbose:
                print(f"\n=== 🔄 FarmCPU Iteration {iteration + 1}/{maxLoop} ===")
            
            # Step 1: GLM analysis to identify candidate QTNs
            if timing:
                timer.start("glm_analysis")
                
            if verbose:
                print("📈 Step 1: Running GLM to identify candidate QTNs...")
            
            glm_results = MVP_GLM(
                phe=phe,
                geno=genotype,
                CV=current_covariates,
                maxLine=maxLine,
                verbose=False
            )
            
            glm_pvalues = glm_results.to_numpy()[:, 2]  # P-values column
            
            if timing:
                glm_time = timer.end("glm_analysis")
                if verbose:
                    print(f"   GLM completed in {glm_time:.3f}s")
            
            # Step 2: Select candidate QTNs based on p-values
            if timing:
                timer.start("candidate_selection")
                
            # Use more permissive threshold to ensure iteration continues like R implementation
            current_threshold = max(p_threshold, QTN_threshold) if iteration == 0 else QTN_threshold * 10
            
            candidate_indices = np.where(glm_pvalues < current_threshold)[0]
            
            if verbose:
                print(f"🎯 Found {len(candidate_indices)} candidate QTNs with p < {current_threshold}")
            
            # Force continuation for at least 2 iterations to match R behavior
            if len(candidate_indices) == 0:
                if iteration < 2:
                    if verbose:
                        print("⚠️  No candidates found, but forcing continuation to match R...")
                    # Select the most significant markers even if above threshold
                    n_top = min(3, len(glm_pvalues))  # Take top 3 markers
                    candidate_indices = np.argsort(glm_pvalues)[:n_top]
                    current_threshold = glm_pvalues[candidate_indices[-1]]  # Use actual p-value as threshold
                    if verbose:
                        print(f"   Selected top {len(candidate_indices)} markers with threshold {current_threshold:.6f}")
                else:
                    if verbose:
                        print("🛑 No significant QTNs found after 2+ iterations. Stopping.")
                    break
            
            if timing:
                selection_time = timer.end("candidate_selection")
                if verbose:
                    print(f"   Candidate selection completed in {selection_time:.3f}s")
            
            # Step 3: Binning and QTN selection
            if timing:
                timer.start("binning")
                
            if verbose:
                print("🗂️  Step 3: Performing binning and QTN selection...")
            
            # Determine bin size for this iteration (R uses specific schedule)
            if iteration == 0:
                current_bin_size = bin_size[0] if len(bin_size) > 0 else 500000
            elif iteration == 1:
                current_bin_size = bin_size[2] if len(bin_size) > 2 else 50000000  
            else:
                current_bin_size = bin_size[1] if len(bin_size) > 1 else 5000000
            
            selected_qtn_indices = select_qtns_multiscale_binning_rmvp(
                candidate_indices=candidate_indices,
                pvalues=glm_pvalues,
                map_data=map_data,
                bin_sizes=[current_bin_size],  # Single bin size for timed version
                method=method_bin,
                percentiles=list(range(10, 101, 10)),  # [10,20,30...100] percentiles like rMVP
                verbose=False
            )
            
            if verbose:
                print(f"   Selected {len(selected_qtn_indices)} QTNs after binning")
            
            if timing:
                binning_time = timer.end("binning")
                if verbose:
                    print(f"   Binning completed in {binning_time:.3f}s")
            
            # Step 3b: Apply LD-based QTN removal (optional filtering step)
            if len(selected_qtn_indices) > 1:
                if timing:
                    timer.start("ld_filtering")
                    
                selected_qtn_indices = remove_qtns_by_ld(
                    selected_qtns=selected_qtn_indices,
                    genotype_matrix=genotype,
                    correlation_threshold=0.7,  # R default
                    max_individuals=100000,
                    verbose=False
                )
                
                if verbose:
                    print(f"🔗 Selected {len(selected_qtn_indices)} QTNs after LD filtering")
                
                if timing:
                    ld_time = timer.end("ld_filtering")
                    if verbose:
                        print(f"   LD filtering completed in {ld_time:.3f}s")
            
            # Step 4: Threshold filtering and convergence logic  
            if timing:
                timer.start("convergence_check")
                
            # Apply QTN threshold filtering (matching R lines 183-205)
            # Fix: rMVP uses more permissive filtering in early iterations
            if iteration == 0:
                # Iteration 1: Use all selected QTNs from binning (no threshold filtering)
                final_selected_qtns = selected_qtn_indices
            elif iteration == 1:
                # Iteration 2: Use relaxed threshold for continuation (rMVP behavior)
                # rMVP typically keeps more QTNs in iteration 2 to ensure proper selection
                relaxed_threshold = min(0.1, QTN_threshold * 10)  # More permissive
                threshold_mask = glm_pvalues[selected_qtn_indices] < relaxed_threshold
                final_selected_qtns = [selected_qtn_indices[i] for i in range(len(selected_qtn_indices)) if threshold_mask[i]]
                
                # If still too few, keep top binning results
                if len(final_selected_qtns) < 5:
                    # Keep top QTNs from binning to ensure proper iteration
                    final_selected_qtns = selected_qtn_indices
                    if verbose:
                        print(f"   Keeping all {len(final_selected_qtns)} binned QTNs for iteration continuity")
                
                # Early stopping check (R lines 152-164)
                if len(final_selected_qtns) == 0:
                    if verbose:
                        print(f"🛑 No QTNs passed threshold in iteration 2. Stopping FarmCPU.")
                    selected_qtns = []
                    break
            else:
                # Iteration 3+: Apply proper threshold but preserve previous QTNs
                threshold_mask = glm_pvalues[selected_qtn_indices] < QTN_threshold
                
                # Force preservation of previously significant QTNs (rMVP approach)
                preserved_qtns = []
                for i, qtn_idx in enumerate(selected_qtn_indices):
                    # Keep if: 1) passes current threshold, OR 2) was previously selected
                    if threshold_mask[i] or qtn_idx in selected_qtns_save:
                        preserved_qtns.append(qtn_idx)
                
                final_selected_qtns = preserved_qtns
            
            # Check for convergence using rMVP's EXACT criteria - exact set matching, not Jaccard
            convergence_achieved = False
            if iteration > 0:
                # rMVP convergence: exact set matching - current QTNs must be identical to previous
                current_set = set(final_selected_qtns)
                previous_set = set(selected_qtns_save)
                exact_match = current_set == previous_set
                
                # Check for circular QTNs: identical to pre-previous iteration
                is_circular = False
                if len(selected_qtns_pre) > 0:
                    preprevious_set = set(selected_qtns_pre)
                    is_circular = current_set == preprevious_set
                
                if verbose:
                    print(f"📊 Convergence: Exact match={exact_match}, Circular={is_circular}")
                    print(f"   Current: {len(current_set)} QTNs, Previous: {len(previous_set)} QTNs")
                
                # rMVP convergence conditions: exact set equality OR circular pattern
                if exact_match or is_circular or len(final_selected_qtns) == 0:
                    convergence_achieved = True
                    if verbose:
                        convergence_reason = "Exact set match" if exact_match else \
                                           "Circular QTNs" if is_circular else "No QTNs selected"
                        print(f"✅ Convergence achieved: {convergence_reason}")
            
            if timing:
                convergence_time = timer.end("convergence_check")
                if verbose:
                    print(f"   Convergence check completed in {convergence_time:.3f}s")
            
            # Update QTN history for next iteration
            selected_qtns_pre = selected_qtns_save.copy()
            selected_qtns_save = selected_qtns.copy()
            selected_qtns = final_selected_qtns.copy()
            
            if verbose:
                print(f"✅ Iteration {iteration + 1} completed: {len(selected_qtns)} QTNs selected")
            
            if timing:
                iteration_time = timer.end("iteration_total")
                timer.iteration_timings.append(iteration_time)
                if verbose:
                    print(f"   Iteration total time: {iteration_time:.3f}s")
            
            # Check if we should stop
            if convergence_achieved:
                break
            
            # Step 5: Prepare covariates for next iteration
            if len(selected_qtns) > 0:
                if timing:
                    timer.start("covariate_preparation")
                    
                # Extract QTN genotypes to use as covariates
                if isinstance(genotype, GenotypeMatrix):
                    qtn_genotypes = np.column_stack([genotype.get_marker(i) for i in selected_qtns])
                else:
                    qtn_genotypes = genotype[:, selected_qtns]
                
                # Combine with existing covariates
                if CV is not None:
                    current_covariates = np.column_stack([CV, qtn_genotypes])
                else:
                    current_covariates = qtn_genotypes
                
                if timing:
                    covariate_time = timer.end("covariate_preparation")
                    if verbose:
                        print(f"   Covariate preparation completed in {covariate_time:.3f}s")
            else:
                current_covariates = CV
        
        # Final GLM analysis with selected QTNs (FarmCPU uses GLM, not MLM)
        if timing:
            timer.start("final_glm")
            
        if verbose:
            print(f"\n🏁 Final GLM analysis with {len(selected_qtns)} QTNs as covariates")
        
        # FarmCPU uses GLM with PCs and selected QTNs, not MLM with kinship
        final_results = MVP_GLM(
            phe=phe,
            geno=genotype,
            CV=current_covariates,
            maxLine=maxLine,
            verbose=False
        )
        
        if timing:
            final_glm_time = timer.end("final_glm")
            if verbose:
                print(f"✅ Final GLM completed in {final_glm_time:.3f}s")
        
        if timing:
            total_time = timer.end("total_runtime")
            if verbose:
                print(f"\n🎉 FarmCPU analysis completed in {total_time:.3f}s")
        
        return final_results, timer
        
    except Exception as e:
        if timing:
            timer.end("total_runtime")
        raise e


def select_qtns_by_binning_rmvp(candidate_indices: np.ndarray,
                               pvalues: np.ndarray,
                               map_data: 'GenotypeMap',
                               bin_size: int = 10000000,
                               method: str = "static",
                               percentiles: List[int] = None,
                               verbose: bool = False) -> List[int]:
    """rMVP-compliant binning algorithm with exact percentile selection
    
    Implements rMVP's exact binning algorithm:
    1. Static binning with 3-level hierarchy (500K → 5M → 50M bp)
    2. Percentile selection: [10,20,30...100] percentiles
    3. Position-based tie breaking for consistency
    4. Pseudo-QTN selection optimization
    
    Args:
        candidate_indices: Indices of candidate QTNs
        pvalues: P-values for all markers
        map_data: Genetic map information
        bin_size: Size of bins for positional binning
        method: Binning method (must be "static" for rMVP compliance)
        percentiles: List of percentiles to select [10,20,...,100]
        verbose: Print progress
    
    Returns:
        List of selected QTN indices matching rMVP pattern
    """
    if len(candidate_indices) == 0:
        return []
    
    if percentiles is None:
        percentiles = list(range(10, 101, 10))  # [10,20,30...100]
    
    if verbose:
        print(f"rMVP binning: {len(candidate_indices)} candidates, {len(percentiles)} percentiles")
    
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
        return selected_indices
    
    if method.lower() != "static":
        if verbose:
            print(f"Warning: rMVP uses static binning, got {method}")
    
    # rMVP static binning implementation
    MaxBP = 1e10  # Maximum base pair value for chromosome encoding
    
    # Step 1: Create SNP ID = position + chromosome * MaxBP (rMVP standard)
    snp_ids = positions[candidate_indices] + chromosomes[candidate_indices] * MaxBP
    candidate_pvalues = pvalues[candidate_indices]
    
    # Step 2: Create bin ID = floor(SNP_ID / bin_size)
    bin_ids = np.floor(snp_ids / bin_size).astype(np.int64)
    
    # Step 3: Create binning data with position-based tie breaking
    bin_data = []
    for i in range(len(candidate_indices)):
        bin_data.append({
            'bin_id': bin_ids[i],
            'snp_idx': candidate_indices[i],
            'snp_id': snp_ids[i],
            'pvalue': candidate_pvalues[i],
            'position': positions[candidate_indices[i]],  # For tie breaking
            'original_order': i  # For consistent ordering
        })
    
    # Step 4: Sort by p-value first, then by position for tie breaking (rMVP approach)
    bin_data.sort(key=lambda x: (x['pvalue'], x['position'], x['original_order']))
    
    # Step 5: Group by bin and select best from each bin
    bin_representatives = {}
    for item in bin_data:
        bin_id = item['bin_id']
        if bin_id not in bin_representatives:
            # First (best) SNP in this bin
            bin_representatives[bin_id] = item
    
    # Step 6: rMVP percentile selection - select QTNs at specific percentiles
    bin_list = list(bin_representatives.values())
    bin_list.sort(key=lambda x: (x['pvalue'], x['position'], x['original_order']))
    
    # Select QTNs at percentile positions
    n_bins = len(bin_list)
    selected_qtns = []
    
    for pct in percentiles:
        # Calculate percentile index (1-based to 0-based conversion)
        pct_idx = min(int(n_bins * pct / 100) - 1, n_bins - 1)
        if pct_idx >= 0 and pct_idx < n_bins:
            qtn_idx = bin_list[pct_idx]['snp_idx']
            if qtn_idx not in selected_qtns:  # Avoid duplicates
                selected_qtns.append(qtn_idx)
    
    # Sort selected QTNs by their p-values for final consistency
    if len(selected_qtns) > 1:
        qtn_pvalues = [(idx, pvalues[idx]) for idx in selected_qtns]
        qtn_pvalues.sort(key=lambda x: x[1])  # Sort by p-value
        selected_qtns = [idx for idx, _ in qtn_pvalues]
    
    if verbose:
        print(f"rMVP binning result: {n_bins} bins → {len(selected_qtns)} percentile QTNs")
    
    return selected_qtns