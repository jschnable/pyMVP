"""Format-independent cache validity and fallback behavior."""
import logging
import os

import numpy as np
import pandas as pd
import pytest

from panicle.data.genotype_cache import GenotypeCache
from panicle.utils import data_types, map_cache

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def cached(tmp_path):
    sources = [tmp_path / ('input.' + suffix) for suffix in ['bed', 'bim', 'fam']]
    for source in sources:
        source.write_bytes(b'fixture')
        os.utime(source, (100, 100))
    cache = GenotypeCache(str(tmp_path / 'input'), sources, {'min_maf': 0.01})
    genotype = np.array([[0, 1], [1, 2]], dtype=np.int8)
    frame = pd.DataFrame(dict(MARKER=['a', 'b'], CHROM=['chr1', 'chr2'], POS=[1, 2]))
    cache.save(genotype, ['s1', 's2'], frame, logger=LOGGER)
    return cache, genotype


def test_roundtrip_and_old_imports(cached):
    cache, expected = cached
    genotype, ids, gmap = cache.load(logger=LOGGER)
    assert isinstance(genotype, np.memmap)
    np.testing.assert_array_equal(genotype, expected)
    assert ids == ['s1', 's2']
    assert gmap.n_markers == 2
    assert data_types.load_genotype_map_cache is map_cache.load_genotype_map_cache
    assert data_types.save_genotype_map_cache is map_cache.save_genotype_map_cache


@pytest.mark.parametrize('source_index', [0, 1, 2])
@pytest.mark.parametrize('offset', [0, 1])
def test_every_source_invalidates_at_equal_or_newer_mtime(cached, source_index, offset):
    cache, _ = cached
    stamp = max(os.path.getmtime(cache.path(s)) for s in ['geno.npy', 'ind.txt', 'map.npz']) + offset
    os.utime(cache.sources[source_index], (stamp, stamp))
    assert cache.load(logger=LOGGER) is None


def test_force_and_filter_invalidation(cached):
    cache, _ = cached
    assert cache.load(force=True, logger=LOGGER) is None
    changed = GenotypeCache(cache.base, cache.sources, {'min_maf': 0.1})
    assert changed.load(logger=LOGGER) is None


@pytest.mark.parametrize('suffix', ['geno.npy', 'ind.txt', 'map.npz'])
def test_missing_component_falls_back(cached, suffix):
    cache, _ = cached
    os.unlink(cache.path(suffix))
    assert cache.load(logger=LOGGER) is None


def test_corrupt_cache_falls_back(cached):
    cache, _ = cached
    with open(cache.path('geno.npy'), 'wb') as handle:
        handle.write(b'not a numpy file')
    assert cache.load(logger=LOGGER) is None


def test_legacy_csv_map_migrates(cached):
    cache, _ = cached
    os.unlink(cache.path('map.npz'))
    pd.DataFrame(dict(SNP=['a', 'b'], CHROM=[1, 2], POS=[1, 2])).to_csv(cache.path('map.csv'), index=False)
    _, _, gmap = cache.load(logger=LOGGER)
    assert gmap.n_markers == 2
    assert os.path.exists(cache.path('map.npz'))


def test_save_failure_is_nonfatal(cached, monkeypatch):
    cache, genotype = cached
    def fail(*args, **kwargs):
        raise OSError('disk unavailable')
    monkeypatch.setattr(np, 'save', fail)
    cache.save(genotype, ['s1', 's2'], [], logger=LOGGER)
