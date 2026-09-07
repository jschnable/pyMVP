"""VCF QC compatibility, memory layout, and resource lifecycle contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from panicle.data import load_genotype_vcf as vcf
from panicle.data.vcf_records import VCFFilters


def write_fixture(tmp_path, rows):
    path = tmp_path / 'input.vcf'
    path.write_text('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\td\n' +
                    ''.join(f'1\t{i + 1}\tm{i}\tA\tG\t.\tPASS\t.\tGT\t{row}\n' for i, row in enumerate(rows)))
    return path


@pytest.mark.parametrize('numba', [True, False])
@pytest.mark.parametrize('batch_size', [2, 100])
@pytest.mark.parametrize('options', [{}, {'drop_monomorphic': True, 'max_missing': .5, 'min_maf': .1}])
def test_bulk_and_general_outputs_match(tmp_path, monkeypatch, numba, batch_size, options):
    path = write_fixture(tmp_path, ['0/0\t0/1\t1/1\t./.', '0/0\t0/0\t0/0\t0/0',
                                    '0|1\t./.\t1|1\t0|0', './.\t./.\t./.\t./.',
                                    '1/1\t1/1\t1/1\t1/1', '0/1\t0/1\t0/1\t0/1'])
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', batch_size)
    if not numba:
        monkeypatch.setattr(vcf, '_decode_simple_gt_matrix_numba', None)
    bulk = vcf.load_genotype_vcf(path, force_recache=True, backend='builtin', **options)
    monkeypatch.setattr(vcf, '_try_load_simple_biallelic_gt_vcf_bulk', lambda *a, **k: None)
    general = vcf.load_genotype_vcf(path, force_recache=True, backend='builtin', **options)
    assert bulk[0].flags.c_contiguous and general[0].flags.c_contiguous
    np.testing.assert_array_equal(bulk[0], general[0])
    assert bulk[1] == general[1]
    pd.testing.assert_frame_equal(bulk[2], general[2])


@pytest.mark.parametrize('simple', [True, False])
def test_scalar_qc_boundaries(simple):
    col = np.array([0, 1, 2, -9], dtype=np.int8)
    assert VCFFilters(max_missing=.25, min_maf=.375).accepts(col, 'A', 'G', 2, simple_diploid=simple)
    assert not VCFFilters(max_missing=.249).accepts(col, 'A', 'G', 2, simple_diploid=simple)
    assert not VCFFilters(min_maf=.376).accepts(col, 'A', 'G', 2, simple_diploid=simple)
    assert not VCFFilters(include_indels=False).accepts(col, 'AT', 'A', 2, simple_diploid=simple)
    assert VCFFilters(drop_monomorphic=True).accepts(np.ones(4, dtype=np.int8), 'A', 'G', 2, simple_diploid=simple)


def track_writers(monkeypatch):
    writers = []
    original = vcf._DynamicInt8MatrixWriter
    def factory(*args, **kwargs):
        writer = original(*args, **kwargs)
        writers.append(writer)
        return writer
    monkeypatch.setattr(vcf, '_DynamicInt8MatrixWriter', factory)
    return writers


def test_general_failure_discards_temporary_matrix(tmp_path, monkeypatch):
    path = write_fixture(tmp_path, ['0/0\t0/1\t1/1\t0/0'] * 2)
    writers = track_writers(monkeypatch)
    monkeypatch.setattr(vcf, '_try_load_simple_biallelic_gt_vcf_bulk', lambda *a, **k: None)
    original = vcf._parse_simple_biallelic_gt_line
    def fail_second(line, n):
        if b'\tm1\t' in line:
            raise ValueError('decode failed')
        return original(line, n)
    monkeypatch.setattr(vcf, '_parse_simple_biallelic_gt_line', fail_second)
    with pytest.raises(ValueError, match='decode failed'):
        vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    assert writers
    assert all(w.memmap is None and not Path(w.path).exists() for w in writers)


def test_final_bulk_batch_failure_discards_temporary_matrix(tmp_path, monkeypatch):
    if vcf._decode_simple_gt_matrix_numba is None:
        pytest.skip('Numba unavailable')
    path = write_fixture(tmp_path, ['0/0\t0/1\t1/1\t0/0'] * 3)
    writers = track_writers(monkeypatch)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', 2)
    original = vcf._decode_simple_gt_matrix_numba
    def fail_tail(raw, count, samples):
        if count == 1:
            raise ValueError('tail failed')
        return original(raw, count, samples)
    monkeypatch.setattr(vcf, '_decode_simple_gt_matrix_numba', fail_tail)
    with pytest.raises(ValueError, match='tail failed'):
        vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    assert writers
    assert all(w.memmap is None and not Path(w.path).exists() for w in writers)


def test_late_bulk_fallback_reuses_writer(tmp_path, monkeypatch):
    path = write_fixture(tmp_path, ['0/0\t0/1\t1/1\t0/0'] * 3)
    with path.open('a') as handle:
        handle.write('1\t4\tlate\tA\tG\t.\tPASS\t.\tGT:DS\t0/0:0\t0/1:1\t1/1:2\t0/0:0\n')
    writers = track_writers(monkeypatch)
    monkeypatch.setattr(vcf, '_SIMPLE_BULK_BATCH_MARKERS', 2)
    result = vcf.load_genotype_vcf(path, backend='builtin', force_recache=True)
    assert result[0].shape == (4, 4)
    assert len(writers) == 1
    assert all(w.memmap is None and not Path(w.path).exists() for w in writers)


@pytest.mark.parametrize('fail', [False, True])
def test_htslib_reader_and_writer_close_on_exit(tmp_path, monkeypatch, fail):
    path = write_fixture(tmp_path, ['0/0\t0/1\t1/1\t0/0'])
    class Reader:
        samples = ['a', 'b', 'c', 'd']
        closed = False
        def __iter__(self):
            yield SimpleNamespace(CHROM='1', POS=1, ID='m0', REF='A', ALT=['G'],
                                  genotype=SimpleNamespace(array=lambda: np.array(
                                      [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]])))
            if fail:
                raise ValueError('reader failed')
        def close(self):
            self.closed = True
    reader = Reader()
    monkeypatch.setattr(vcf, '_select_vcf_reader', lambda *args: (reader, True, False))
    writers = track_writers(monkeypatch)
    if fail:
        with pytest.raises(ValueError, match='reader failed'):
            vcf.load_genotype_vcf(path, backend='cyvcf2', force_recache=True)
    else:
        assert vcf.load_genotype_vcf(path, backend='cyvcf2', force_recache=True)[0].shape == (4, 1)
    assert reader.closed
    assert writers
    assert all(w.memmap is None and not Path(w.path).exists() for w in writers)
