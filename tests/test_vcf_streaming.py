"""Streaming FORMAT decoding and bounded cache finalization regressions."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.data import load_genotype_vcf as vcf
from panicle.data import vcf_storage as storage
from panicle.utils.data_types import impute_major_allele_inplace


def fixture(tmp_path, records, compressed=False, samples=4):
    path = tmp_path / ('stream.vcf.gz' if compressed else 'stream.vcf')
    text = ('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' +
            '\t'.join(f's{i}' for i in range(samples)) + '\n')
    for i, (fmt, calls) in enumerate(records):
        text += f'1\t{i + 1}\tm{i}\tA\tG\t.\tPASS\t.\t{fmt}\t{calls}\n'
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    return path


@pytest.mark.parametrize('extractor', ['compiled', 'python', 'unavailable'])
@pytest.mark.parametrize('batch_size', [1, 3, 100])
def test_variable_format_matches_general(tmp_path, monkeypatch, extractor, batch_size):
    from panicle.data.vcf_format import _extract
    if extractor != 'compiled':
        monkeypatch.setattr(vcf, 'extract_gt_fields', _extract if extractor == 'python' else None)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', batch_size)
    path = fixture(tmp_path, [
        ('GT', '0/0\t0/1\t1/1\t./.'),
        ('GT:DP', '0/0:1\t0|1:123456\t1/1:.\t./.:0'),
        ('DP:GT:AD', '1:1/1:0,1\t999:0/1:50,49\t.:0/0:.\t0:0/.:.'),
        ('GT:AD:DP', '1/1:0,20:20\t0/0:2,0:2\t./.:.:.\t0/1:1,3:4'),
        # Rejected extraction replays just the pending batch, preserving GT/DS policy.
        ('GT:DP', '0:4\t1:99\t.:.\t0:100'),
        ('GT:DS', './.:1.2\t0/1:2\t1/1:0\t./.:.'),
        ('GT:DP', '1/1:5\t0/1:0\t0/0:8\t./.:9'),
        ('GT:DP', './.:.\t./.:.\t./.:.\t./.:.'),
    ])
    actual = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True, max_missing=.5)
    monkeypatch.setattr(vcf, '_try_load_simple_biallelic_gt_vcf_bulk', lambda *a, **kw: None)
    expected = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True, max_missing=.5)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    pd.testing.assert_frame_equal(actual[2], expected[2])


@pytest.mark.parametrize('compressed', [False, True])
def test_late_unsupported_does_not_reopen_and_resumes_bulk(tmp_path, monkeypatch, compressed):
    records = [('GT', '0/0\t0/1\t1/1\t./.')] * 5
    records += [('GT:DS', './.:2\t0/1:0\t1/1:0\t0/0:2')]
    records += [('GT', '1/1\t0/1\t0/0\t./.')] * 4
    path = fixture(tmp_path, records, compressed)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', 2)
    opened = []
    general = []
    original_open, original_general = vcf._open_binary, vcf._decode_builtin_records
    def open_once(path):
        opened.append(path)
        return original_open(path)
    def decode(*args, **kwargs):
        lines = list(kwargs['lines'])
        general.extend(lines)
        return original_general(*args, **dict(kwargs, lines=lines))
    monkeypatch.setattr(vcf, '_open_binary', open_once)
    monkeypatch.setattr(vcf, '_decode_builtin_records', decode)
    result = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    assert len(opened) == 1
    assert len(general) == 1 and b'GT:DS' in general[0]
    assert result[0].shape == (4, 10)
    assert result[2]['MARKER'].tolist() == [f'm{i}' for i in range(10)]
    np.testing.assert_array_equal(result[0][:, 5], [2, 1, 2, 0])


@pytest.mark.parametrize('late', [False, True])
def test_late_fallback_preserves_legacy_qc_rounding(tmp_path, monkeypatch, late):
    records = [('GT', '\t'.join(['./.'] * 3 + ['0/1'] * 7))] * 3
    if late:
        records += [('GT:DP', '\t'.join(['0/1:8'] * 10))]
    path = fixture(tmp_path, records, samples=10)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', 2)
    result = vcf.load_genotype_vcf(path, backend='builtin', max_missing=.3, force_recache=True)
    # Original bulk: 3/10 <= .3; original general: 1 - 7/10 > .3.
    assert result[2]['MARKER'].tolist() == (['m3'] if late else ['m0', 'm1', 'm2'])


@pytest.mark.parametrize('direct', [False, True])
def test_tiled_finalize_selection_and_imputation(tmp_path, monkeypatch, direct):
    monkeypatch.setattr(storage, '_TRANSPOSE_BLOCK_MARKERS', 2)
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1 if direct else 100000)
    original = np.array([[0, -9, 2, 0, -9], [1, 1, -9, 0, 2], [2, 1, 1, 1, 2]], dtype=np.int8)
    selected = np.array([0, 2, 4])
    expected = original[:, selected].copy()
    missing = impute_major_allele_inplace(expected)
    writer = storage._DynamicInt8MatrixWriter(3, initial_capacity=1)
    writer.append_block(original)
    path = tmp_path / 'geno.npy'
    published = []
    result = writer.finalize(cache_path=path, before_publish=lambda: published.append(True),
                             keep_indices=selected, impute=True)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous and result.flags.writeable
    assert writer.imputed_count == missing
    assert not Path(writer.path).exists()
    assert bool(published) == direct
    if direct:
        assert isinstance(result, np.memmap) and result.mode == 'c'
        result[0, 0] = 99
        np.testing.assert_array_equal(np.load(path), expected)


@pytest.mark.parametrize('failure', ['create', 'publish'])
def test_direct_cache_failure_falls_back_without_leaks(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1)
    def fail(*args, **kwargs):
        raise OSError('injected disk failure')
    if failure == 'create':
        monkeypatch.setattr(np.lib.format, 'open_memmap', fail)
    else:
        monkeypatch.setattr(storage.os, 'replace', fail)
    writer = storage._DynamicInt8MatrixWriter(2)
    writer.append(np.array([0, 2], dtype=np.int8))
    result = writer.finalize(cache_path=tmp_path / 'geno.npy')
    np.testing.assert_array_equal(result, [[0], [2]])
    assert not isinstance(result, np.memmap)
    assert not list(tmp_path.iterdir())
    assert not Path(writer.path).exists()


def test_direct_cache_load_does_not_rewrite_genotype(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1)
    path = fixture(tmp_path, [('GT:DP', '0/0:4\t0/1:500\t1/1:2\t./.:.')])
    original_save = np.save
    def checked_save(path, *args, **kwargs):
        assert not str(path).endswith('geno.npy'), 'Finalized cache must not be copied again'
        return original_save(path, *args, **kwargs)
    monkeypatch.setattr(np, 'save', checked_save)
    fresh = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    assert isinstance(fresh[0], np.memmap)
    expected = fresh[0].copy()
    fresh[0][0, 0] = 99
    cached = vcf.load_genotype_vcf(path, backend='builtin')
    np.testing.assert_array_equal(cached[0], expected)
    assert fresh[1] == cached[1]
    assert fresh[2]['MARKER'].tolist() == cached[2].to_dataframe()['MARKER'].tolist()


def test_failed_metadata_publish_invalidates_old_cache(tmp_path, monkeypatch):
    from panicle.data import genotype_cache
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1)
    path = fixture(tmp_path, [('GT', '0/0\t0/1\t1/1\t./.')])
    vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    fingerprint = Path(str(path) + '.panicle.v2.filters.json')
    assert fingerprint.exists()
    def fail(*args, **kwargs):
        raise OSError('metadata write failed')
    monkeypatch.setattr(genotype_cache, 'save_genotype_map_cache', fail)
    result = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    np.testing.assert_array_equal(result[0][:, 0], [0, 1, 2, 0])
    assert not fingerprint.exists()


@pytest.mark.parametrize('valid_prefix', [False, True])
def test_general_sanity_check_state_across_batches(tmp_path, monkeypatch, valid_prefix):
    records = [('GT:DP', '0/0:4\t0/1:5\t1/1:6\t./.:.')] if valid_prefix else []
    records += [('GT:DP', '0/0/1:4\t0/1/1:5\t1/1/1:6\t./.:.')]
    path = fixture(tmp_path, records)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', 1)
    if not valid_prefix:
        with pytest.raises(ValueError, match='ploidy greater than diploid'):
            vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    else:
        actual = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
        monkeypatch.setattr(vcf, '_try_load_simple_biallelic_gt_vcf_bulk', lambda *a, **kw: None)
        expected = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
        np.testing.assert_array_equal(actual[0], expected[0])
