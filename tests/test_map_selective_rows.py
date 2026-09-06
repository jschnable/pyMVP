"""Selective map extraction works without materializing unselected lazy rows."""
import numpy as np
import pandas as pd
import pytest

from panicle.utils.data_types import GenotypeMap
from panicle.utils.map_cache import save_genotype_map_cache, load_genotype_map_cache


@pytest.mark.parametrize('backing', ['numpy', 'series', 'list', 'cache'])
@pytest.mark.parametrize('indices', [[], [2, 0, 2]])
def test_selective_rows_preserve_order_aliases_and_lazy_storage(tmp_path, backing, indices):
    frame = GenotypeMap(pd.DataFrame({
        'MARKER': ['a', 'β', 'c'], 'CHROM': ['chr1', 'chr1', 'chr2'],
        'POS': [10, 20, 30], 'REF': ['A', 'G', 'T'],
    })).to_dataframe()
    if backing == 'numpy':
        gmap = GenotypeMap(frame)
    elif backing == 'cache':
        path = tmp_path / 'map.npz'
        save_genotype_map_cache(path, frame)
        gmap = load_genotype_map_cache(path)
    else:
        columns = {name: frame[name] if backing == 'series' else frame[name].tolist()
                   for name in frame}
        gmap = GenotypeMap.from_columns(columns)
    selected = gmap.to_dataframe_at(np.asarray(indices, dtype=np.int64))
    pd.testing.assert_frame_equal(selected, frame.iloc[indices].reset_index(drop=True), check_dtype=False)
    if backing == 'cache':
        assert gmap._dataframe_cache is None
        for column in gmap._column_data.values():
            if hasattr(column, '_decoded'):
                assert column._decoded is None
