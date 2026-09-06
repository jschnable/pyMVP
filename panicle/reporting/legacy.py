"""Legacy one-call API output writer (preserves filenames and schemas)."""
import json
import warnings
from typing import Any, Dict, List
from .tables import association_table, json_default


def save_results_to_files(results: Dict[str, Any],
                         output_prefix: str,
                         verbose: bool = True) -> List[str]:
    """Save analysis results to files"""

    saved_files = []

    try:
        # Save summary statistics
        summary_file = f"{output_prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PANICLE GWAS Analysis Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Methods run: {', '.join(results['summary']['methods_run'])}\n")
            f.write(f"Total individuals: {results['summary']['total_individuals']}\n")
            f.write(f"Total markers: {results['summary']['total_markers']}\n")
            n_traits = results['summary'].get('n_traits', 1)
            trait_names = results['summary'].get('trait_names', ['Trait'])
            f.write(f"Traits analyzed: {n_traits} ({', '.join(trait_names)})\n")
            f.write("\nSignificant markers by trait and method:\n")
            for trait_name, methods in results['summary']['significant_markers'].items():
                f.write(f"  {trait_name}:\n")
                for method, count in methods.items():
                    f.write(f"    {method}: {count}\n")
            f.write("\nRuntimes (seconds):\n")
            for phase, time_val in results['summary']['runtime'].items():
                f.write(f"  {phase}: {time_val:.2f}s\n")

        saved_files.append(summary_file)

        # Get map data once for reuse
        map_df = None
        if 'map' in results['data']:
            map_obj = results['data']['map']
            if hasattr(map_obj, 'to_dataframe'):
                map_df = map_obj.to_dataframe()
            elif hasattr(map_obj, 'data'):
                map_df = map_obj.data

        # Save association results as CSV files (nested by trait)
        for trait_name, trait_results in results['results'].items():
            for method_name, result_obj in trait_results.items():
                result_file = f"{output_prefix}_{trait_name}_{method_name}_results.csv"
                result_df = association_table(result_obj, map_df)

                result_df.to_csv(result_file, index=False)
                saved_files.append(result_file)

                metadata = getattr(result_obj, "metadata", None)
                if isinstance(metadata, dict) and metadata:
                    meta_file = f"{output_prefix}_{trait_name}_{method_name}_metadata.json"
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, sort_keys=True, default=json_default)
                    saved_files.append(meta_file)

        if verbose:
            print(f"Saved {len(saved_files)} result files")

    except Exception as e:
        warnings.warn(f"Failed to save some results files: {e}")

    return saved_files
