import numpy as np
import pytest

from panicle.data import load_genotype_vcf as vcf
from panicle.data import vcf_storage
from panicle.data.vcf_metrics import VCFMetrics, timed, count
from scripts.vcf_observability import StageRecorder, resources, cache_disk_usage
from scripts.benchmark_vcf import fingerprint


def test_nested_timing_exception_and_context_restore():
    def inner():
        count('bytes', 4)
        raise ValueError('expected')
    with VCFMetrics() as metrics:
        with pytest.raises(ValueError):
            timed('outer', timed, 'inner', inner)
    assert metrics.counters == {'bytes': 4}
    assert metrics.stages['outer']['inclusive_seconds'] >= metrics.stages['inner']['inclusive_seconds']
    assert sum(x['exclusive_seconds'] for x in metrics.stages.values()) == pytest.approx(metrics.stages['outer']['inclusive_seconds'])
    count('bytes', 9)
    assert metrics.counters == {'bytes': 4}


@pytest.mark.parametrize('platform_name,scale', [('darwin', 1), ('linux', 1024), ('other', None)])
def test_rss_units(monkeypatch, platform_name, scale):
    import sys
    pytest.importorskip('resource')
    monkeypatch.setattr(sys, 'platform', platform_name)
    result = resources()
    assert result['peak_rss_bytes'] == (result['peak_rss_native'] * scale if scale else None)


def test_spool_counts_storage_bytes_not_source_dtype():
    writer = vcf_storage._DynamicInt8MatrixWriter(2)
    try:
        with VCFMetrics() as metrics:
            writer.append(np.array([0, 1], dtype=np.int64))
            writer.append_block(np.ones((2, 3), dtype=np.float64))
        assert metrics.counters['spool_append_bytes'] == 8
    finally:
        writer.discard()


@pytest.mark.parametrize('direct', [False, True])
def test_instrumentation_preserves_output_and_restores_patches(tmp_path, monkeypatch, direct):
    import gzip
    monkeypatch.setattr(vcf_storage, '_DIRECT_CACHE_MIN_BYTES', 1 if direct else 1000000)
    monkeypatch.setattr(vcf_storage, '_TRANSPOSE_BLOCK_MARKERS', 1)
    path = tmp_path / 'test.vcf.gz'
    header = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n'
    row = '1\t1\tm1\tA\tG\t.\tPASS\t.\tGT:DP\t0/0:2\t1/1:22\t./.:.\n'
    path.write_bytes(gzip.compress((header + row).encode()))
    expected = fingerprint(vcf.load_genotype_vcf(path, backend='builtin', force_recache=True))
    original = vcf._try_load_simple_biallelic_gt_vcf_bulk
    with StageRecorder() as metrics:
        actual = metrics.call('load_total', vcf.load_genotype_vcf, path, backend='builtin', force_recache=True)
    assert fingerprint(actual) == expected
    assert vcf._try_load_simple_biallelic_gt_vcf_bulk is original
    assert metrics.counters['spool_append_bytes'] == 3
    assert metrics.counters['output_tile_write_bytes'] == 3
    assert metrics.stages['transpose_tiles']['calls'] == 1
    assert sum(x['exclusive_seconds'] for x in metrics.stages.values()) == pytest.approx(metrics.stages['load_total']['inclusive_seconds'])
    assert bool(metrics.stages.get('cache_flush')) == direct
    assert cache_disk_usage(path)
    assert 'available' in resources()
    with pytest.raises(RuntimeError):
        with StageRecorder():
            raise RuntimeError('failed')
    assert vcf._try_load_simple_biallelic_gt_vcf_bulk is original
