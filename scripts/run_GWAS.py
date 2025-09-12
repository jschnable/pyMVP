#!/usr/bin/env python3
"""
Comprehensive GWAS Analysis Script using pyMVP

This script demonstrates the full functionality of the pyMVP package for
conducting Genome-Wide Association Studies (GWAS) using GLM, MLM, and FarmCPU methods.
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add the parent directory to the path to import pyMVP
sys.path.insert(0, str(Path(__file__).parent.parent))

import pymvp
from pymvp.data.loaders import (
    load_phenotype_file, load_genotype_file, load_map_file, 
    match_individuals, detect_file_format
)
from pymvp.utils.stats import (
    bonferroni_correction, calculate_maf_from_genotypes, 
    genomic_inflation_factor
)
from pymvp.utils.data_types import GenotypeMatrix, AssociationResults
from pymvp.association.glm import MVP_GLM
from pymvp.association.mlm import MVP_MLM
from pymvp.association.farmcpu import MVP_FarmCPU
from pymvp.matrix.pca import MVP_PCA
from pymvp.matrix.kinship import MVP_K_VanRaden
from pymvp.visualization.manhattan import MVP_Report

def print_header():
    """Print script header with version info"""
    print("="*80)
    print("pyMVP Comprehensive GWAS Analysis Script")
    print(f"pyMVP Version: {pymvp.__version__}")
    print("="*80)

def print_step(step_name: str, start_time: Optional[float] = None):
    """Print step information with timing"""
    if start_time is not None:
        elapsed = time.time() - start_time
        print(f"{step_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"{step_name}...")

def load_and_validate_data(phenotype_file: str, 
                          genotype_file: str,
                          map_file: Optional[str] = None,
                          genotype_format: Optional[str] = None,
                          trait_columns: Optional[List[str]] = None,
                          loader_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load and validate input data files"""
    
    step_start = time.time()
    print_step("Loading phenotype data")
    
    # Load phenotype data
    try:
        phenotype_df = load_phenotype_file(phenotype_file, trait_columns=trait_columns)
        print(f"   Loaded {len(phenotype_df)} individuals with {len(phenotype_df.columns)-1} traits")
    except Exception as e:
        raise ValueError(f"Error loading phenotype file: {e}")
    
    print_step("Loading genotype data")
    
    # Detect genotype format if not specified
    if genotype_format is None:
        genotype_format = detect_file_format(genotype_file)
        print(f"   Detected genotype format: {genotype_format}")
        if genotype_format == 'unknown':
            raise ValueError("Could not detect genotype format. Supported: csv/tsv, vcf (vcf, vcf.gz, vcf.bgz, bcf), plink (.bed+.bim+.fam), hapmap (.hmp, .hmp.txt)")
    
    # Load genotype data
    try:
        loader_kwargs = loader_kwargs or {}
        genotype_matrix, individual_ids, geno_map = load_genotype_file(
            genotype_file, file_format=genotype_format, **loader_kwargs
        )
        print(f"   Loaded {genotype_matrix.n_individuals} individuals x {genotype_matrix.n_markers} markers")
    except Exception as e:
        raise ValueError(f"Error loading genotype file: {e}")
    
    # Load map file if provided
    if map_file:
        print_step("Loading genetic map")
        try:
            geno_map = load_map_file(map_file)
            print(f"   Loaded map for {geno_map.n_markers} markers")
        except Exception as e:
            print(f"   Warning: Could not load map file: {e}")
    
    print_step("Data loading", step_start)
    
    return {
        'phenotype_df': phenotype_df,
        'genotype_matrix': genotype_matrix,
        'individual_ids': individual_ids,
        'geno_map': geno_map
    }

def match_and_filter_individuals(phenotype_df: pd.DataFrame,
                               genotype_matrix: GenotypeMatrix,
                               individual_ids: List[str]) -> Dict[str, Any]:
    """Match individuals between phenotype and genotype data"""
    
    step_start = time.time()
    print_step("Matching individuals between phenotype and genotype data")
    
    # Match individuals
    matched_phenotype, matched_indices, summary = match_individuals(
        phenotype_df, individual_ids
    )
    
    # Filter genotype matrix to matched individuals
    matched_genotype_data = genotype_matrix[matched_indices, :]
    matched_genotype_matrix = GenotypeMatrix(matched_genotype_data)
    
    # Print summary
    print(f"   Original phenotype individuals: {summary['n_phenotype_original']}")
    print(f"   Original genotype individuals: {summary['n_genotype_original']}")
    print(f"   Common individuals: {summary['n_common']}")
    
    if summary['n_phenotype_dropped'] > 0:
        print(f"   Dropped from phenotype: {summary['n_phenotype_dropped']}")
    if summary['n_genotype_dropped'] > 0:
        print(f"   Dropped from genotype: {summary['n_genotype_dropped']}")
    
    print_step("Individual matching", step_start)
    
    return {
        'phenotype_df': matched_phenotype,
        'genotype_matrix': matched_genotype_matrix,
        'summary': summary
    }

def calculate_population_structure(genotype_matrix: GenotypeMatrix,
                                 n_pcs: int = 3,
                                 calculate_kinship: bool = False) -> Dict[str, Any]:
    """Calculate principal components and kinship matrix"""
    
    results = {}
    
    # Calculate PCs
    step_start = time.time()
    print_step(f"Calculating {n_pcs} principal components")
    
    try:
        pcs = MVP_PCA(M=genotype_matrix, pcs_keep=n_pcs, verbose=False)
        results['pcs'] = pcs
        print(f"   Calculated {n_pcs} PCs explaining population structure")
        print_step("PCA calculation", step_start)
    except Exception as e:
        raise ValueError(f"Error calculating PCs: {e}")
    
    # Calculate kinship matrix if needed
    if calculate_kinship:
        step_start = time.time()
        print_step("Calculating kinship matrix")
        
        try:
            kinship = MVP_K_VanRaden(genotype_matrix, verbose=False)
            results['kinship'] = kinship
            print(f"   Calculated kinship matrix ({kinship.shape[0]}x{kinship.shape[1]})")
            print_step("Kinship calculation", step_start)
        except Exception as e:
            raise ValueError(f"Error calculating kinship: {e}")
    
    return results

def run_gwas_analysis(phenotype_df: pd.DataFrame,
                     genotype_matrix: GenotypeMatrix,
                     geno_map,
                     population_structure: Dict[str, Any],
                     trait_name: str,
                     methods: List[str] = ['GLM', 'MLM', 'FarmCPU'],
                     max_iterations: int = 10) -> Dict[str, AssociationResults]:
    """Run GWAS analysis using specified methods for a single trait"""
    
    results = {}
    
    # Prepare data for the specified trait
    if trait_name not in phenotype_df.columns:
        raise ValueError(f"Trait '{trait_name}' not found in phenotype data")
    phe_array = phenotype_df[['ID', trait_name]].values
    pcs = population_structure['pcs']
    
    print(f"Running GWAS for trait: {trait_name}")
    print(f"Using {pcs.shape[1]} principal components as covariates")
    
    # GLM Analysis
    if 'GLM' in methods:
        step_start = time.time()
        print_step("Running GLM analysis")
        
        try:
            glm_results = MVP_GLM(
                phe=phe_array,
                geno=genotype_matrix,
                CV=pcs,
                verbose=False
            )
            results['GLM'] = glm_results
            
            # Calculate genomic inflation factor
            lambda_gc = genomic_inflation_factor(glm_results.pvalues)
            print(f"   GLM lambda (genomic inflation): {lambda_gc:.3f}")
            print_step("GLM analysis", step_start)
            
        except Exception as e:
            print(f"   GLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # MLM Analysis
    if 'MLM' in methods:
        if 'kinship' not in population_structure:
            print("   Skipping MLM: kinship matrix not calculated")
        else:
            step_start = time.time()
            print_step("Running MLM analysis")
            
            try:
                mlm_results = MVP_MLM(
                    phe=phe_array,
                    geno=genotype_matrix,
                    K=population_structure['kinship'],
                    CV=pcs,
                    verbose=False
                )
                results['MLM'] = mlm_results
                
                # Calculate genomic inflation factor
                lambda_gc = genomic_inflation_factor(mlm_results.pvalues)
                print(f"   MLM lambda (genomic inflation): {lambda_gc:.3f}")
                print_step("MLM analysis", step_start)
                
            except Exception as e:
                print(f"   MLM analysis failed: {e}")
                import traceback
                traceback.print_exc()
    
    # FarmCPU Analysis
    if 'FARMCPU' in methods:
        step_start = time.time()
        print_step("Running FarmCPU analysis")
        
        try:
            farmcpu_results = MVP_FarmCPU(
                phe=phe_array,
                geno=genotype_matrix,
                map_data=geno_map,
                CV=pcs,
                maxLoop=max_iterations,
                verbose=False
            )
            results['FarmCPU'] = farmcpu_results
            
            # Calculate genomic inflation factor
            lambda_gc = genomic_inflation_factor(farmcpu_results.pvalues)
            print(f"   FarmCPU lambda (genomic inflation): {lambda_gc:.3f}")
            print_step("FarmCPU analysis", step_start)
            
        except Exception as e:
            print(f"   FarmCPU analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def analyze_results_and_save(results: Dict[str, AssociationResults],
                           genotype_matrix: GenotypeMatrix,
                           geno_map,
                           output_dir: str,
                           significance_threshold: Optional[float] = None,
                           trait_label: Optional[str] = None,
                           save_all_results: bool = True,
                           save_significant: bool = True,
                           bonferroni_alpha: float = 0.05,
                           n_eff: Optional[int] = None) -> Dict[str, Any]:
    """Analyze results and conditionally save output files for a single trait"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    step_start = time.time()
    print_step("Analyzing results and calculating significance")
    
    # Calculate MAF for all markers
    genotype_data = genotype_matrix._data if hasattr(genotype_matrix, '_data') else genotype_matrix
    maf = calculate_maf_from_genotypes(genotype_data)
    
    # Prepare base results dataframe
    base_df = geno_map.to_dataframe() if hasattr(geno_map, 'to_dataframe') else pd.DataFrame({
        'SNP': [f'SNP_{i:05d}' for i in range(genotype_matrix.n_markers)],
        'CHROM': [1] * genotype_matrix.n_markers,
        'POS': list(range(1, genotype_matrix.n_markers + 1))
    })
    base_df['MAF'] = maf
    
    # Add results from each method
    all_results_df = base_df.copy()
    significant_snps = []
    summary_stats = {}
    
    for method, result in results.items():
        # Add method results to main dataframe
        all_results_df[f'{method}_Effect'] = result.effects
        all_results_df[f'{method}_SE'] = result.se
        all_results_df[f'{method}_Pvalue'] = result.pvalues
        
        # Determine threshold and correction behavior
        n_tests_true = len(result.pvalues)
        if significance_threshold is not None:
            # Fixed threshold overrides alpha and n_eff
            n_tests_used = n_tests_true
            used_thresh = float(significance_threshold)
        else:
            # Bonferroni with alpha and either true number of tests or n_eff if provided
            n_tests_used = int(n_eff) if (n_eff and n_eff > 0) else n_tests_true
            used_thresh = bonferroni_alpha / max(n_tests_used, 1)

        # Compute Bonferroni-corrected p-values with chosen n_tests
        corrected = np.minimum(result.pvalues * n_tests_used, 1.0)
        all_results_df[f'{method}_Pvalue_Bonf'] = corrected

        # Find significant SNPs based on used threshold (uncorrected p-value compared to used_thresh)
        sig_mask = result.pvalues < used_thresh
        n_significant = np.sum(sig_mask)
        
        summary_stats[method] = {
            'n_significant': n_significant,
            'min_pvalue': np.min(result.pvalues),
            'threshold_used': used_thresh,
            'alpha': bonferroni_alpha,
            'n_tests_used': n_tests_used,
            'n_bonf_significant': np.sum(result.pvalues < (bonferroni_alpha / max(n_tests_true, 1)))
        }
        
        print(f"   {method}: {n_significant} significant SNPs (p < {used_thresh:.2e})")
        if significance_threshold is None:
            print(f"       Bonferroni: alpha={bonferroni_alpha}, n_tests_used={n_tests_used} => threshold {used_thresh:.2e}")
        else:
            print(f"       Fixed threshold (overrides alpha/n_eff): {used_thresh:.2e}")
        print(f"       {summary_stats[method]['n_bonf_significant']} SNPs pass Bonferroni correction")
        
        if n_significant > 0:
            sig_snps = all_results_df[sig_mask].copy()
            sig_snps['Method'] = method
            significant_snps.append(sig_snps)
    
    # Save all results
    suffix = f"_{trait_label}" if trait_label else ""
    if save_all_results:
        all_results_file = output_path / f"GWAS{suffix}_all_results.csv"
        all_results_df.to_csv(all_results_file, index=False)
        print(f"   Saved all results to: {all_results_file}")
    
    # Save significant SNPs if any found
    if save_significant and significant_snps:
        sig_df = pd.concat(significant_snps, ignore_index=True)
        sig_results_file = output_path / f"GWAS{suffix}_significant_SNPs_p{significance_threshold}.csv"
        sig_df.to_csv(sig_results_file, index=False)
        print(f"   Saved significant SNPs to: {sig_results_file}")
    else:
        print(f"   No significant SNPs found at threshold p < {significance_threshold}")
    
    print_step("Results analysis", step_start)
    
    return {
        'all_results_df': all_results_df,
        'significant_snps': significant_snps,
        'summary_stats': summary_stats
    }

def generate_plots(results: Dict[str, AssociationResults],
                  geno_map,
                  output_dir: str,
                  trait_label: Optional[str] = None,
                  plot_manhattan: bool = True,
                  plot_qq: bool = True) -> None:
    """Generate Manhattan and Q-Q plots for each method for a single trait"""
    
    output_path = Path(output_dir)
    
    step_start = time.time()
    print_step("Generating Manhattan and Q-Q plots")
    
    if not (plot_manhattan or plot_qq):
        print("   Plot generation skipped by user request")
        print_step("Plot generation", step_start)
        return

    for method, result in results.items():
        try:
            # Generate plots using MVP_Report
            plot_types = []
            if plot_manhattan:
                plot_types.append("manhattan")
            if plot_qq:
                plot_types.append("qq")
            plot_results = MVP_Report(
                results=result,
                map_data=geno_map,
                output_prefix=str(output_path / f"GWAS_{trait_label}_{method}" if trait_label else output_path / f"GWAS_{method}"),
                plot_types=plot_types,
                verbose=False,
                save_plots=True
            )
            print(f"   Generated plots for {method}")
            
        except Exception as e:
            print(f"   Could not generate plots for {method}: {e}")
    
    print_step("Plot generation", step_start)

def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(
        description="Comprehensive GWAS Analysis using pyMVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Supported genotype formats:\n"
            "  - csv/tsv: numeric matrices (individuals × markers)\n"
            "  - vcf: .vcf, .vcf.gz, .vcf.bgz (built-in parser); .bcf requires cyvcf2\n"
            "  - plink: .bed with .bim and .fam (requires bed-reader)\n"
            "  - hapmap: .hmp or .hmp.txt (also supports gzipped variants)\n\n"
            "Examples:\n"
            "  python scripts/run_GWAS.py -p phe.csv -g geno.vcf.gz --methods GLM,MLM\n"
            "  python scripts/run_GWAS.py -p phe.csv -g plink_prefix --format plink\n"
            "  python scripts/run_GWAS.py -p phe.csv -g geno.hmp.txt --format hapmap\n"
        )
    )
    
    # Required arguments
    parser.add_argument("--phenotype", "-p", required=True,
                       help="Phenotype file (CSV/TSV with ID column and trait columns)")
    parser.add_argument("--genotype", "-g", required=True,
                       help=(
                           "Genotype file: CSV/TSV numeric, VCF/BCF (.vcf/.vcf.gz/.vcf.bgz/.bcf),\n"
                           "PLINK prefix or .bed (with .bim/.fam), or HapMap (.hmp/.hmp.txt)"
                       ))
    
    # Optional arguments
    parser.add_argument("--map", "-m", default=None,
                       help="Genetic map file (CSV/TSV with SNP, CHROM, POS columns)")
    parser.add_argument("--output", "-o", default="./GWAS_results",
                       help="Output directory for results")
    parser.add_argument("--format", "-f", default=None, 
                       choices=['csv', 'tsv', 'numeric', 'vcf', 'plink', 'hapmap'],
                       help="Genotype file format (auto-detect if not specified)")
    parser.add_argument("--methods", default="GLM,MLM,FarmCPU",
                       help="GWAS methods to run (comma-separated: GLM,MLM,FarmCPU)")
    parser.add_argument("--n-pcs", type=int, default=3,
                       help="Number of principal components to use")
    parser.add_argument("--significance", type=float, default=None,
                       help="Fixed p-value threshold; if set, overrides --alpha and --n-eff")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Base alpha for Bonferroni (default: 0.05)")
    parser.add_argument("--n-eff", type=int, default=None,
                       help="Effective number of markers to use for Bonferroni denominator (optional)")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum iterations for FarmCPU")
    parser.add_argument("--traits", default=None,
                       help="Comma-separated list of trait column names (auto-detect if not specified)")
    # Output selection
    parser.add_argument("--outputs", nargs='+',
                       choices=['all_marker_pvalues','significant_marker_pvalues','manhattan','qq'],
                       default=['all_marker_pvalues','significant_marker_pvalues','manhattan','qq'],
                       help=(
                           "Which outputs to generate: 'all_marker_pvalues' (full p-values), 'significant_marker_pvalues' (only significant SNPs),\n"
                           "'manhattan' (Manhattan plot), 'qq' (Q-Q plot). Choose any combination."
                       ))
    # Genotype loader/QC options
    parser.add_argument("--vcf-backend", dest="vcf_backend", choices=['auto','cyvcf2','builtin'], default='auto',
                       help="VCF/BCF parsing backend (auto selects cyvcf2 if available)")
    parser.add_argument("--no-split-multiallelic", dest="no_split_multiallelic", action='store_true',
                       help="Do not split multi-allelic VCF sites into multiple markers")
    parser.add_argument("--snps-only", dest="snps_only", action='store_true',
                       help="Restrict to SNPs only (exclude indels where applicable)")
    parser.add_argument("--drop-monomorphic", dest="drop_monomorphic", action='store_true', default=False,
                       help="Drop monomorphic markers (all non-missing calls are the same homozygote)")
    parser.add_argument("--max-missing", dest="max_missing", type=float, default=1.0,
                       help="Max allowed missingness per marker (fraction 0..1]")
    parser.add_argument("--min-maf", dest="min_maf", type=float, default=0.0,
                       help="Minimum minor allele frequency per marker (0..0.5]")
    
    args = parser.parse_args()
    
    # Parse methods and traits
    methods = [m.strip().upper() for m in args.methods.split(',')]
    trait_columns = [t.strip() for t in args.traits.split(',')] if args.traits else None
    
    # Validate methods
    valid_methods = ['GLM', 'MLM', 'FARMCPU']
    for method in methods:
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods: {valid_methods}")
    
    print_header()
    
    try:
        # Prepare loader kwargs based on CLI
        loader_kwargs = {
            'drop_monomorphic': args.drop_monomorphic,
            'max_missing': args.max_missing,
            'min_maf': args.min_maf,
        }
        if args.format == 'vcf' or (args.format is None and detect_file_format(args.genotype) == 'vcf'):
            loader_kwargs.update({
                'backend': args.vcf_backend,
                'split_multiallelic': (not args.no_split_multiallelic),
                'include_indels': (not args.snps_only),
            })
        if args.format == 'hapmap' or (args.format is None and detect_file_format(args.genotype) == 'hapmap'):
            loader_kwargs.update({
                'include_indels': (not args.snps_only),
            })

        # Step 1: Load and validate data
        print("Step 1: Loading and validating input data")
        data = load_and_validate_data(
            phenotype_file=args.phenotype,
            genotype_file=args.genotype,
            map_file=args.map,
            genotype_format=args.format,
            trait_columns=trait_columns,
            loader_kwargs=loader_kwargs,
        )
        
        # Step 2: Match individuals
        print()
        print("Step 2: Matching individuals between datasets")
        matched_data = match_and_filter_individuals(
            data['phenotype_df'],
            data['genotype_matrix'],
            data['individual_ids']
        )
        
        # Step 3: Calculate population structure
        print()
        print("Step 3: Calculating population structure")
        population_structure = calculate_population_structure(
            matched_data['genotype_matrix'],
            n_pcs=args.n_pcs,
            calculate_kinship=('MLM' in methods)
        )
        
        # Step 4: Determine traits to analyze
        phe_df = matched_data['phenotype_df']
        if trait_columns:
            selected_traits = [t for t in trait_columns if t in phe_df.columns]
            missing_traits = [t for t in trait_columns if t not in phe_df.columns]
            if missing_traits:
                print(f"   Warning: Traits not found and will be skipped: {missing_traits}")
        else:
            selected_traits = [c for c in phe_df.columns if c != 'ID' and pd.api.types.is_numeric_dtype(phe_df[c])]
            if not selected_traits:
                raise ValueError("No numeric trait columns found in phenotype data")

        # Sanitize and deduplicate trait labels
        import re
        def sanitize(label: str) -> str:
            s = re.sub(r"[^A-Za-z0-9]+", "_", label).strip('_')
            return s or "Trait"

        raw_labels = selected_traits
        base_labels = [sanitize(x) for x in raw_labels]
        counts: Dict[str, int] = {}
        final_labels: List[str] = []
        collisions = set()
        # Number left-to-right: first occurrence gets _1
        for b in base_labels:
            counts[b] = counts.get(b, 0) + 1
            final_labels.append(f"{b}_{counts[b]}")
            if counts[b] == 2:
                collisions.add(b)
        if collisions:
            print("   ⚠️  Detected duplicate trait names (original or after sanitization). Renumbering left-to-right:")
            for b in collisions:
                print(f"      - '{b}' -> '{b}_1', '{b}_2', ... by column order")

        # Step 5: Run GWAS analysis for each trait
        print()
        print("Step 4: Running GWAS analysis (per trait)")
        summary_rows = []
        for trait_name, trait_label in zip(raw_labels, final_labels):
            print()
            print(f"-- Trait: {trait_name} (label: {trait_label}) --")
            gwas_results = run_gwas_analysis(
                phe_df,
                matched_data['genotype_matrix'],
                data['geno_map'],
                population_structure,
                trait_name=trait_name,
                methods=methods,
                max_iterations=args.max_iterations
            )
            if not gwas_results:
                print(f"   ❌ No GWAS results for trait '{trait_name}' — skipping saving/plots")
                continue

            # Analyze and save
            print()
            print("Step 5: Analyzing results and saving output")
            # Warn if fixed significance overrides alpha/n_eff
            if args.significance is not None and (args.n_eff is not None or (args.alpha is not None and args.alpha != 0.05)):
                print("   Note: --significance was provided and will override --alpha and --n-eff for significance calls.")

            analysis_results = analyze_results_and_save(
                gwas_results,
                matched_data['genotype_matrix'],
                data['geno_map'],
                args.output,
                args.significance,
                trait_label=trait_label,
                save_all_results=('all_marker_pvalues' in args.outputs),
                save_significant=('significant_marker_pvalues' in args.outputs),
                bonferroni_alpha=args.alpha,
                n_eff=args.n_eff,
            )
            for method, stats in analysis_results['summary_stats'].items():
                summary_rows.append({
                    'Trait': trait_name,
                    'Trait_Label': trait_label,
                    'Method': method,
                    **stats,
                })

            # Plots
            print()
            print("Step 6: Generating visualization plots")
            generate_plots(
                gwas_results,
                data['geno_map'],
                args.output,
                trait_label=trait_label,
                plot_manhattan=('manhattan' in args.outputs),
                plot_qq=('qq' in args.outputs),
            )

        # Write aggregated summary across traits/methods
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = Path(args.output) / "GWAS_summary_by_trait_method.csv"
            summary_df.to_csv(summary_path, index=False)
            print()
            print(f"Saved summary of all traits/methods to: {summary_path}")

        # Final summary banner
        print()
        print("="*80)
        print("GWAS ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {Path(args.output).absolute()}")
        
    except Exception as e:
        print()
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
