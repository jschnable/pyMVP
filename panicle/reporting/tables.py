"""Pure result assembly helpers. No files, plots, or pipeline state."""
import numpy as np
from ..utils.data_types import MARKER_ID_COLUMN, LEGACY_MARKER_ID_COLUMN, infer_marker_id_column


def json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def flatten_results(results, *, single_trait):
    return {
        method if single_trait else f'{trait}_{method}': result
        for trait, methods in results.items() for method, result in methods.items()
    }


def association_table(result, map_df=None):
    """Assemble the legacy per-method schema, preserving marker order/aliases."""
    frame = result.to_dataframe()
    if map_df is not None:
        marker_col = infer_marker_id_column(map_df.columns)
        if marker_col is not None:
            if MARKER_ID_COLUMN not in frame.columns:
                frame[MARKER_ID_COLUMN] = map_df[marker_col].values[:len(frame)]
            if LEGACY_MARKER_ID_COLUMN not in frame.columns:
                frame[LEGACY_MARKER_ID_COLUMN] = frame[MARKER_ID_COLUMN].astype(str)
        if 'Chr' not in frame.columns and 'CHROM' in map_df.columns:
            frame['Chr'] = map_df['CHROM'].values[:len(frame)]
        if 'Pos' not in frame.columns and 'POS' in map_df.columns:
            frame['Pos'] = map_df['POS'].values[:len(frame)]
    return frame
