"""
Manhattan plot and Q-Q plot visualization for GWAS results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Optional, Union, Dict, List, Tuple
import warnings
import re

from ..utils.data_types import AssociationResults, GenotypeMap
from ..association.farmcpu_resampling import FarmCPUResamplingResults
from ..utils.stats import genomic_inflation_factor

def PANICLE_Report(results: Union[AssociationResults, Dict],
               map_data: Optional[GenotypeMap] = None,
               threshold: float = 5e-8,
               suggestive_threshold: float = 1e-5,
               plot_types: List[str] = ["manhattan", "qq"],
               output_prefix: str = "PANICLE_results",
               dpi: int = 300,
               figsize: Tuple[int, int] = (8, 4),
               colors: Optional[List[str]] = None,
               point_size: float = 12.0,
               verbose: bool = True,
               save_plots: bool = True,
               true_qtns: Optional[List[str]] = None,
               multi_panel: bool = False) -> Dict:
    """Generate comprehensive GWAS visualization report

    Creates Manhattan plots, Q-Q plots, and summary statistics for GWAS results.

    Args:
        results: AssociationResults object or dict of results from different methods
        map_data: Genetic map information for positioning markers
        threshold: Genome-wide significance threshold
        suggestive_threshold: Suggestive significance threshold
        plot_types: Types of plots to generate ["manhattan", "qq", "density"]
        output_prefix: Prefix for output files
        dpi: Plot resolution
        figsize: Figure size (width, height)
        colors: Custom colors for chromosomes
        point_size: Size of points in plots
        verbose: Print progress information
        save_plots: Save plots to files
        true_qtns: List of SNP names that are known true QTNs to highlight
        multi_panel: Create multi-panel Manhattan plot with all methods

    Returns:
        Dictionary with plot objects and summary statistics
    """

    if verbose:
        print("Generating GWAS visualization report...")

    report = {
        'plots': {},
        'summary': {},
        'files_created': []
    }

    # Handle different input types
    if isinstance(results, AssociationResults):
        results_dict = {'GWAS': results}
    elif isinstance(results, FarmCPUResamplingResults):
        results_dict = {'FarmCPUResampling': results}
    elif isinstance(results, dict):
        results_dict = results
    else:
        raise ValueError("Results must be AssociationResults, FarmCPUResamplingResults, or dictionary")

    contains_resampling = any(
        isinstance(result_obj, FarmCPUResamplingResults)
        for result_obj in results_dict.values()
    )

    if contains_resampling and multi_panel:
        if verbose:
            print("FarmCPU resampling results excluded from multi-panel Manhattan; generating individual plots instead.")
        multi_panel = False

    # Handle multi-panel Manhattan plot
    if multi_panel and "manhattan" in plot_types and isinstance(results, dict) and len(results_dict) > 1:
        if verbose:
            print("Creating multi-panel Manhattan plot...")
        
        multi_panel_fig = create_multi_panel_manhattan(
            results_dict=results_dict,
            map_data=map_data,
            threshold=threshold,
            true_qtns=true_qtns,
            figsize=(12, 4 * len(results_dict)),
            colors=colors,
            point_size=point_size
        )
        report['plots']['multi_panel_manhattan'] = multi_panel_fig
        
        if save_plots:
            filename = f"{output_prefix}_multi_panel_manhattan.png"
            multi_panel_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            report['files_created'].append(filename)
        
        # For multi-panel mode, skip individual method plotting for Manhattan plots
        # but still process other plot types and summary stats
        for method_name, result_obj in results_dict.items():
            # Extract data for summary statistics
            if hasattr(result_obj, 'to_numpy'):
                results_array = result_obj.to_numpy()
                effects = results_array[:, 0]
                se = results_array[:, 1]
                pvalues = results_array[:, 2]
            else:
                continue
            
            # Filter out invalid p-values
            valid_mask = ~np.isnan(pvalues) & (pvalues > 0) & (pvalues <= 1)
            valid_pvalues = pvalues[valid_mask]
            valid_effects = effects[valid_mask]
            
            if len(valid_pvalues) == 0:
                continue
            
            # Generate non-Manhattan plots
            method_plots = {}
            
            if "qq" in plot_types:
                if verbose:
                    print(f"Creating Q-Q plot for {method_name}...")
                qq_fig = create_qq_plot(
                    pvalues=valid_pvalues,
                    title=f"Q-Q Plot - {method_name}",
                    figsize=(6, 6)
                )
                method_plots['qq'] = qq_fig
                
                if save_plots:
                    filename = f"{output_prefix}_{method_name}_qq.png"
                    qq_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                    report['files_created'].append(filename)
            
            if "density" in plot_types:
                if verbose:
                    print(f"Creating density plot for {method_name}...")
                density_fig = create_pvalue_density_plot(
                    pvalues=valid_pvalues,
                    title=f"P-value Distribution - {method_name}",
                    figsize=(8, 4)
                )
                method_plots['density'] = density_fig
                
                if save_plots:
                    filename = f"{output_prefix}_{method_name}_density.png"
                    density_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                    report['files_created'].append(filename)
            
            report['plots'][method_name] = method_plots
            
            # Calculate summary statistics
            method_summary = calculate_gwas_summary(
                pvalues=valid_pvalues,
                effects=valid_effects,
                threshold=threshold,
                suggestive_threshold=suggestive_threshold
            )
            report['summary'][method_name] = method_summary
            
            if verbose:
                print(f"Summary for {method_name}:")
                print(f"  Total markers: {method_summary['n_markers']}")
                print(f"  Significant hits: {method_summary['n_significant']}")
                print(f"  Suggestive hits: {method_summary['n_suggestive']}")
                print(f"  Minimum p-value: {method_summary['min_pvalue']:.2e}")
        
        if verbose:
            print(f"Multi-panel report complete. Created {len(report['files_created'])} plot files.")
        
        return report

    # Generate plots for each result set (non multi-panel mode)
    for method_name, result_obj in results_dict.items():
        if verbose:
            print(f"Processing results for method: {method_name}")

        if isinstance(result_obj, FarmCPUResamplingResults):
            method_plots = {}
            rmip_values = result_obj.rmip_values

            if len(rmip_values) == 0:
                if verbose:
                    print(f"No markers identified by FarmCPU resampling for {method_name}")
                report['summary'][method_name] = {
                    'n_markers': 0,
                    'max_rmip': 0.0,
                    'mean_rmip': 0.0,
                    'cluster_mode': result_obj.cluster_mode,
                    'total_runs': result_obj.total_runs
                }
                report['plots'][method_name] = method_plots
                continue

            if "manhattan" in plot_types:
                if verbose:
                    print(f"Creating RMIP Manhattan plot for {method_name}...")

                rmip_fig = create_rmip_manhattan_plot(
                    result=result_obj,
                    map_data=map_data,
                    title=f"FarmCPU Resampling - RMIP ({method_name})",
                    figsize=figsize,
                    colors=colors,
                    point_size=point_size
                )
                method_plots['manhattan'] = rmip_fig

                if save_plots:
                    filename = f"{output_prefix}_{method_name}_rmip_manhattan.png"
                    rmip_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                    report['files_created'].append(filename)

            report['plots'][method_name] = method_plots

            rmip_summary = {
                'n_markers': len(result_obj.entries),
                'max_rmip': float(np.max(rmip_values)),
                'mean_rmip': float(np.mean(rmip_values)),
                'cluster_mode': result_obj.cluster_mode,
                'total_runs': result_obj.total_runs
            }
            report['summary'][method_name] = rmip_summary

            if verbose:
                print(f"Summary for {method_name}:")
                print(f"  Entries: {rmip_summary['n_markers']}")
                print(f"  Max RMIP: {rmip_summary['max_rmip']:.3f}")
                print(f"  Mean RMIP: {rmip_summary['mean_rmip']:.3f}")
                print(f"  Clustering enabled: {rmip_summary['cluster_mode']}")
            continue

        # Extract data
        if hasattr(result_obj, 'to_numpy'):
            results_array = result_obj.to_numpy()
            effects = results_array[:, 0]
            se = results_array[:, 1]
            pvalues = results_array[:, 2]
        else:
            raise ValueError(f"Invalid result object for method {method_name}")

        # Filter out invalid p-values
        valid_mask = ~np.isnan(pvalues) & (pvalues > 0) & (pvalues <= 1)
        valid_pvalues = pvalues[valid_mask]
        valid_effects = effects[valid_mask]

        if len(valid_pvalues) == 0:
            warnings.warn(f"No valid p-values found for method {method_name}")
            continue

        # Generate requested plots
        method_plots = {}

        if "manhattan" in plot_types:
            if verbose:
                print(f"Creating Manhattan plot for {method_name}...")

            manhattan_fig = create_manhattan_plot(
                pvalues=valid_pvalues,
                map_data=map_data,
                threshold=threshold,
                suggestive_threshold=0,  # Disable suggestive threshold
                title="",  # Remove title
                figsize=figsize,
                colors=colors,
                point_size=point_size,
                true_qtns=true_qtns
            )
            method_plots['manhattan'] = manhattan_fig

            if save_plots:
                filename = f"{output_prefix}_{method_name}_manhattan.png"
                manhattan_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                report['files_created'].append(filename)

        if "qq" in plot_types:
            if verbose:
                print(f"Creating Q-Q plot for {method_name}...")

            qq_fig = create_qq_plot(
                pvalues=valid_pvalues,
                title=f"Q-Q Plot - {method_name}",
                figsize=(6, 6)
            )
            method_plots['qq'] = qq_fig

            if save_plots:
                filename = f"{output_prefix}_{method_name}_qq.png"
                qq_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                report['files_created'].append(filename)

        if "density" in plot_types:
            if verbose:
                print(f"Creating density plot for {method_name}...")

            density_fig = create_pvalue_density_plot(
                pvalues=valid_pvalues,
                title=f"P-value Distribution - {method_name}",
                figsize=(8, 4)
            )
            method_plots['density'] = density_fig

            if save_plots:
                filename = f"{output_prefix}_{method_name}_density.png"
                density_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                report['files_created'].append(filename)

        report['plots'][method_name] = method_plots

        # Calculate summary statistics
        method_summary = calculate_gwas_summary(
            pvalues=valid_pvalues,
            effects=valid_effects,
            threshold=threshold,
            suggestive_threshold=suggestive_threshold
        )
        report['summary'][method_name] = method_summary

        if verbose:
            print(f"Summary for {method_name}:")
            print(f"  Total markers: {method_summary['n_markers']}")
            print(f"  Significant hits: {method_summary['n_significant']}")
            print(f"  Suggestive hits: {method_summary['n_suggestive']}")
            print(f"  Minimum p-value: {method_summary['min_pvalue']:.2e}")

    if verbose:
        print(f"Report generation complete. Created {len(report['files_created'])} plot files.")

    return report


def create_manhattan_plot(pvalues: np.ndarray,
                         map_data: Optional[GenotypeMap] = None,
                         threshold: float = 5e-8,
                         suggestive_threshold: float = 1e-5,
                         title: str = "Manhattan Plot",
                         figsize: Tuple[int, int] = (12, 6),
                         colors: Optional[List[str]] = None,
                         point_size: float = 10.0,
                         true_qtns: Optional[List[str]] = None) -> plt.Figure:
    """Create Manhattan plot for GWAS results

    Args:
        pvalues: Array of p-values
        map_data: Genetic map with chromosome and position information
        threshold: Genome-wide significance threshold
        suggestive_threshold: Suggestive significance threshold
        title: Plot title
        figsize: Figure size
        colors: Custom chromosome colors
        point_size: Point size
        true_qtns: List of SNP names that are known true QTNs to highlight

    Returns:
        matplotlib Figure object
    """

    # Convert p-values to -log10 scale
    log_pvalues = -np.log10(pvalues)

    fig, ax = plt.subplots(figsize=figsize)

    if map_data is not None and hasattr(map_data, 'to_dataframe'):
        # Use actual chromosomal positions
        try:
            map_df = map_data.to_dataframe()
            chromosomes = map_df['CHROM'].values[:len(pvalues)]
            positions = map_df['POS'].values[:len(pvalues)]

            # Create Manhattan plot with chromosomal positions
            plot_manhattan_with_positions(
                ax, chromosomes, positions, log_pvalues,
                colors=colors, point_size=point_size, 
                map_data=map_data, true_qtns=true_qtns
            )
        except Exception as e:
            warnings.warn(f"Could not use map data for positioning: {e}")
            # Fallback to sequential plotting
            plot_manhattan_sequential(ax, log_pvalues, point_size=point_size)
    else:
        # Sequential plotting without chromosomal information
        plot_manhattan_sequential(ax, log_pvalues, point_size=point_size)

    # Add only significance threshold (no suggestive threshold)
    if threshold > 0:
        threshold_line = -np.log10(threshold)
        ax.axhline(y=threshold_line, color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    # Set labels and formatting
    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel(r'$-\log_{10}(P)$', fontsize=12)

    # Set title only if provided and not empty
    if title and title.strip():
        ax.set_title(title)

    # Remove grid (no gray grid marks)

    plt.tight_layout()
    return fig


def create_rmip_manhattan_plot(result: FarmCPUResamplingResults,
                               map_data: Optional[GenotypeMap] = None,
                               title: str = "",
                               figsize: Tuple[int, int] = (12, 6),
                               colors: Optional[List[str]] = None,
                               point_size: float = 10.0) -> plt.Figure:
    """Create Manhattan-style plot using RMIP values on the y-axis.

    When map information is available we expand the RMIP vector so that
    chromosomes with zero-RMIP markers are still rendered, avoiding the
    impression that they were omitted from the analysis. Identified markers
    are highlighted with a secondary overlay.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if result.total_runs == 0:
        ax.set_xlabel('Chromosome', fontsize=12)
        ax.set_ylabel('RMIP', fontsize=12)
        ax.text(0.5, 0.5, 'No FarmCPU resampling runs recorded',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig

    highlight_mask: Optional[np.ndarray]

    try:
        if map_data is not None:
            map_df = map_data.to_dataframe()
            chromosomes = map_df['CHROM'].to_numpy()
            positions = map_df['POS'].to_numpy(dtype=np.float64)

            n_markers = len(map_df)
            rmip_values = np.zeros(n_markers, dtype=np.float64)
            scale = 1.0 / float(result.total_runs)
            for marker_index, count in result.per_marker_counts.items():
                idx = int(marker_index)
                if 0 <= idx < n_markers:
                    rmip_values[idx] = float(count) * scale

            highlight_mask = rmip_values > 0
        else:
            chromosomes = result.chromosomes
            positions = result.positions
            rmip_values = result.rmip_values
            highlight_mask = np.ones_like(rmip_values, dtype=bool)

        if rmip_values.size == 0:
            ax.set_xlabel('Chromosome', fontsize=12)
            ax.set_ylabel('RMIP', fontsize=12)
            if title and title.strip():
                ax.set_title(title)
            ax.text(0.5, 0.5, 'No markers identified', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            plt.tight_layout()
            return fig

        plot_manhattan_with_positions(
            ax,
            np.asarray(chromosomes),
            np.asarray(positions, dtype=np.float64),
            np.asarray(rmip_values, dtype=np.float64),
            colors=colors,
            point_size=point_size,
            highlight_mask=highlight_mask,
            highlight_kwargs={'s': point_size * 1.6,
                              'facecolors': 'none',
                              'edgecolors': 'black',
                              'linewidths': 0.5,
                              'zorder': 9}
        )
    except Exception as exc:
        warnings.warn(f"Falling back to sequential RMIP plotting: {exc}")
        rmip_values = np.asarray(result.rmip_values, dtype=np.float64)
        plot_manhattan_sequential(ax, rmip_values, point_size=point_size)

    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel('RMIP', fontsize=12)
    if title and title.strip():
        ax.set_title(title)

    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


def _natural_sort_key(value) -> List[Union[int, str]]:
    """Return a key for natural sorting of chromosome labels."""

    text = str(value).strip()
    if not text:
        return [""]
    parts = re.split(r'(\d+)', text)
    key: List[Union[int, str]] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def plot_manhattan_with_positions(ax, chromosomes: np.ndarray, positions: np.ndarray,
                                 log_pvalues: np.ndarray, colors: Optional[List[str]] = None,
                                 point_size: float = 3.0, map_data: Optional[GenotypeMap] = None,
                                 true_qtns: Optional[List[str]] = None,
                                 highlight_mask: Optional[np.ndarray] = None,
                                 highlight_kwargs: Optional[Dict] = None):
    """Plot Manhattan plot with actual chromosomal positions"""

    unique_chromosomes = np.unique(chromosomes)
    unique_chromosomes = sorted(unique_chromosomes, key=_natural_sort_key)

    # Default colors
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange alternating

    # Calculate cumulative positions for plotting (no gaps between chromosomes)
    cumulative_pos = np.zeros_like(positions, dtype=np.float64)
    tick_positions = []
    tick_labels = []

    current_pos = 0
    highlight_kwargs = highlight_kwargs or {}
    if highlight_mask is not None and highlight_mask.shape != log_pvalues.shape:
        raise ValueError("highlight_mask must match the shape of the plotted values")

    for i, chrom in enumerate(unique_chromosomes):
        chrom_mask = chromosomes == chrom
        if not np.any(chrom_mask):
            continue

        chrom_indices = np.where(chrom_mask)[0]
        chrom_positions = positions[chrom_indices]
        chrom_log_pvals = log_pvalues[chrom_indices]

        order = np.argsort(chrom_positions, kind='mergesort')
        chrom_positions = chrom_positions[order]
        chrom_log_pvals = chrom_log_pvals[order]
        indices_ordered = chrom_indices[order]

        # Normalize positions within chromosome
        min_pos = np.min(chrom_positions)
        max_pos = np.max(chrom_positions)
        if max_pos > min_pos:
            # Scale chromosome length proportionally
            chrom_length = (max_pos - min_pos) / 1e6  # Convert to Mb
            norm_positions = (chrom_positions - min_pos) / (max_pos - min_pos) * chrom_length
        else:
            norm_positions = np.zeros_like(chrom_positions)
            chrom_length = 1.0

        # Add to cumulative position (no gaps)
        plot_positions = current_pos + norm_positions
        cumulative_pos[indices_ordered] = plot_positions

        cumulative_pos[indices_ordered] = plot_positions
        
        color = colors[i % len(colors)]

        # Decimation Logic: Reduce noise points (p > 0.01) by 95%
        # -log10(0.01) = 2.0
        decimate_log_thresh = 1.0
        keep_fraction = 0.10  # Keep 5% of noise points
        
        sig_mask = chrom_log_pvals >= decimate_log_thresh
        noise_mask = ~sig_mask
        
        # Plot all significant points
        if np.any(sig_mask):
            ax.scatter(plot_positions[sig_mask], chrom_log_pvals[sig_mask],
                      c=color, s=point_size, alpha=0.8, edgecolors='none')
            
        # Plot subsampled noise points
        if np.any(noise_mask):
            noise_indices = np.where(noise_mask)[0]
            n_noise = len(noise_indices)
            n_keep = max(1000, int(n_noise * keep_fraction))
            
            if n_keep < n_noise:
                # Deterministic subsampling for reproducibility (visuals shouldn't flicker)
                np.random.seed(42 + i) 
                keep_indices = np.random.choice(noise_indices, n_keep, replace=False)
                ax.scatter(plot_positions[keep_indices], chrom_log_pvals[keep_indices],
                          c=color, s=point_size, alpha=0.8, edgecolors='none')
            else:
                 ax.scatter(plot_positions[noise_mask], chrom_log_pvals[noise_mask],
                          c=color, s=point_size, alpha=0.8, edgecolors='none')

        if highlight_mask is not None:
            chrom_highlight = highlight_mask[indices_ordered]
            if np.any(chrom_highlight):
                highlight_positions = plot_positions[chrom_highlight]
                highlight_values = chrom_log_pvals[chrom_highlight]
                ax.scatter(highlight_positions,
                           highlight_values,
                           s=highlight_kwargs.get('s', point_size * 1.5),
                           facecolors=highlight_kwargs.get('facecolors', 'none'),
                           edgecolors=highlight_kwargs.get('edgecolors', 'black'),
                           linewidths=highlight_kwargs.get('linewidths', 0.5),
                           zorder=highlight_kwargs.get('zorder', 9))

        # Add tick for chromosome center
        tick_positions.append(current_pos + chrom_length / 2)
        tick_labels.append(str(chrom))

        current_pos += chrom_length  # No gap between chromosomes

    # Highlight true QTNs if provided
    if true_qtns is not None and map_data is not None:
        try:
            map_df = map_data.to_dataframe()
            snp_names = map_df['SNP'].values[:len(log_pvalues)]
            
            # Find true QTNs in the data
            qtn_mask = np.isin(snp_names, true_qtns)
            if np.any(qtn_mask):
                qtn_positions = cumulative_pos[qtn_mask]
                qtn_log_pvals = log_pvalues[qtn_mask]
                
                # Plot QTNs with larger red triangles
                ax.scatter(qtn_positions, qtn_log_pvals, 
                          marker='^', s=point_size*3, c='red', 
                          alpha=0.9, edgecolors='black', linewidth=0.5,
                          zorder=10, label='True QTNs')
                
                # Add legend if QTNs are shown
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            warnings.warn(f"Could not highlight true QTNs: {e}")

    # Set x-axis ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Chromosome', fontsize=12)


def plot_manhattan_sequential(ax, log_pvalues: np.ndarray, point_size: float = 3.0):
    """Plot Manhattan plot with sequential marker positions"""

    positions = np.arange(len(log_pvalues))
    ax.scatter(positions, log_pvalues, c='blue', s=point_size, alpha=0.8, edgecolors='none')
    ax.set_xlabel('Chromosome', fontsize=12)


def create_qq_plot(pvalues: np.ndarray,
                  title: str = "Q-Q Plot",
                  figsize: Tuple[int, int] = (6, 6)) -> plt.Figure:
    """Create Q-Q plot for GWAS p-values

    Args:
        pvalues: Array of p-values
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Remove any invalid p-values
    valid_pvals = pvalues[(pvalues > 0) & (pvalues <= 1) & ~np.isnan(pvalues)]

    if len(valid_pvals) == 0:
        ax.text(0.5, 0.5, 'No valid p-values for Q-Q plot',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Sort p-values
    observed_pvals = np.sort(valid_pvals)
    n = len(observed_pvals)

    # Expected p-values under null hypothesis
    expected_pvals = np.arange(1, n + 1) / (n + 1)

    # Convert to -log10 scale
    obs_log = -np.log10(observed_pvals)
    exp_log = -np.log10(expected_pvals)

    # Plot observed vs expected
    ax.scatter(exp_log, obs_log, alpha=0.6, s=2, edgecolors='none')

    # Add diagonal line (null hypothesis)
    max_val = max(np.max(exp_log), np.max(obs_log))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Null hypothesis')

    # Calculate lambda (genomic inflation factor) using shared utility
    lambda_gc = genomic_inflation_factor(observed_pvals)

    ax.set_xlabel(r'Expected $-\log_{10}(P)$')
    ax.set_ylabel(r'Observed $-\log_{10}(P)$')
    ax.set_title(f'{title}\nλ = {lambda_gc:.3f}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def create_pvalue_density_plot(pvalues: np.ndarray,
                              title: str = "P-value Distribution",
                              figsize: Tuple[int, int] = (8, 4)) -> plt.Figure:
    """Create density plot of p-values

    Args:
        pvalues: Array of p-values
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Remove invalid p-values
    valid_pvals = pvalues[(pvalues > 0) & (pvalues <= 1) & ~np.isnan(pvalues)]

    if len(valid_pvals) == 0:
        ax1.text(0.5, 0.5, 'No valid p-values', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No valid p-values', ha='center', va='center', transform=ax2.transAxes)
        fig.suptitle(title)
        return fig

    # Histogram of p-values
    ax1.hist(valid_pvals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Uniform (null)')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Density')
    ax1.set_title('P-value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of -log10 p-values
    log_pvals = -np.log10(valid_pvals)
    ax2.hist(log_pvals, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel(r'$-\log_{10}(P)$')
    ax2.set_ylabel('Density')
    ax2.set_title(r'$-\log_{10}(P)$ Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.suptitle(title, y=1.02)
    return fig


def calculate_gwas_summary(pvalues: np.ndarray,
                          effects: np.ndarray,
                          threshold: float = 5e-8,
                          suggestive_threshold: float = 1e-5) -> Dict:
    """Calculate summary statistics for GWAS results

    Args:
        pvalues: Array of p-values
        effects: Array of effect sizes
        threshold: Significance threshold
        suggestive_threshold: Suggestive threshold

    Returns:
        Dictionary with summary statistics
    """

    # Filter valid values
    valid_mask = ~np.isnan(pvalues) & (pvalues > 0) & (pvalues <= 1)
    valid_pvalues = pvalues[valid_mask]
    valid_effects = effects[valid_mask]

    if len(valid_pvalues) == 0:
        return {
            'n_markers': 0,
            'n_significant': 0,
            'n_suggestive': 0,
            'min_pvalue': np.nan,
            'median_pvalue': np.nan,
            'lambda_gc': np.nan,
            'mean_effect': np.nan,
            'effect_range': (np.nan, np.nan)
        }

    # Count significant hits
    n_significant = np.sum(valid_pvalues < threshold)
    n_suggestive = np.sum(valid_pvalues < suggestive_threshold) - n_significant

    # Calculate genomic inflation factor (lambda)
    lambda_gc = genomic_inflation_factor(valid_pvalues)

    return {
        'n_markers': len(valid_pvalues),
        'n_significant': n_significant,
        'n_suggestive': n_suggestive,
        'min_pvalue': np.min(valid_pvalues),
        'median_pvalue': np.median(valid_pvalues),
        'lambda_gc': lambda_gc,
        'mean_effect': np.mean(valid_effects),
        'effect_range': (np.min(valid_effects), np.max(valid_effects))
    }


def create_multi_panel_manhattan(results_dict: Dict,
                                map_data: Optional[GenotypeMap] = None,
                                threshold: float = 5e-8,
                                true_qtns: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (12, 12),
                                colors: Optional[List[str]] = None,
                                point_size: float = 8.0) -> plt.Figure:
    """Create multi-panel Manhattan plot showing results from multiple GWAS methods
    
    Args:
        results_dict: Dictionary with method names as keys and AssociationResults as values
        map_data: Genetic map information for positioning markers
        threshold: Genome-wide significance threshold
        true_qtns: List of SNP names that are known true QTNs to highlight
        figsize: Figure size (width, height)
        colors: Custom colors for chromosomes
        point_size: Size of points in plots
        
    Returns:
        matplotlib Figure object
    """
    
    n_methods = len(results_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)
    
    # Ensure axes is always a list
    if n_methods == 1:
        axes = [axes]
    
    method_names = list(results_dict.keys())
    
    for i, (method_name, result_obj) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # Extract data
        if hasattr(result_obj, 'to_numpy'):
            results_array = result_obj.to_numpy()
            effects = results_array[:, 0]
            se = results_array[:, 1] 
            pvalues = results_array[:, 2]
        else:
            continue
        
        # Filter out invalid p-values
        valid_mask = ~np.isnan(pvalues) & (pvalues > 0) & (pvalues <= 1)
        valid_pvalues = pvalues[valid_mask]
        
        if len(valid_pvalues) == 0:
            ax.text(0.5, 0.5, f'No valid p-values for {method_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(f'{method_name}\n' + r'$-\log_{10}(P)$', fontsize=10)
            continue
        
        # Convert p-values to -log10 scale
        log_pvalues = -np.log10(valid_pvalues)
        
        if map_data is not None and hasattr(map_data, 'to_dataframe'):
            try:
                map_df = map_data.to_dataframe()
                chromosomes = map_df['CHROM'].values[:len(pvalues)][valid_mask]
                positions = map_df['POS'].values[:len(pvalues)][valid_mask]
                
                # Create Manhattan plot with chromosomal positions
                plot_manhattan_with_positions(
                    ax, chromosomes, positions, log_pvalues,
                    colors=colors, point_size=point_size,
                    map_data=map_data, true_qtns=true_qtns
                )
            except Exception as e:
                warnings.warn(f"Could not use map data for {method_name}: {e}")
                # Fallback to sequential plotting
                plot_manhattan_sequential(ax, log_pvalues, point_size=point_size)
        else:
            # Sequential plotting without chromosomal information
            plot_manhattan_sequential(ax, log_pvalues, point_size=point_size)
        
        # Add significance threshold
        if threshold > 0:
            threshold_line = -np.log10(threshold)
            ax.axhline(y=threshold_line, color='red', linestyle='--',
                      alpha=0.8, linewidth=1.5)

        # Set y-label with method name
        ax.set_ylabel(f'{method_name}\n' + r'$-\log_{10}(P)$', fontsize=10)
        
        # Only show x-axis label on bottom plot
        if i == n_methods - 1:
            ax.set_xlabel('Chromosome', fontsize=12)
        
        # Add method name as title for top plot only
        if i == 0 and true_qtns is not None:
            ax.set_title('GWAS Results with True QTNs Highlighted', fontsize=12, pad=10)
    
    plt.tight_layout()
    return fig
