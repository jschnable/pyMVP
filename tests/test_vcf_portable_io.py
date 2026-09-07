import gzip
import io
import struct
import zlib

import numpy as np
import pytest

from panicle.data import vcf_compression as compression
from panicle.data import vcf_storage as storage


def bgzf(data):
    compressor = zlib.compressobj(wbits=-15)
    payload = compressor.compress(data) + compressor.flush()
    size = 18 + len(payload) + 8
    return (b'\x1f\x8b\x08\x04' + b'\x00' * 4 + b'\x00\xff' +
            struct.pack('<H2sHH', 6, b'BC', 2, size - 1) + payload +
            struct.pack('<II', zlib.crc32(data), len(data)))


@pytest.mark.parametrize('backend', ['stdlib', 'isal'])
@pytest.mark.parametrize('kind', ['gzip', 'members', 'bgzf'])
def test_compressed_streams_and_long_lines(tmp_path, monkeypatch, backend, kind):
    if backend == 'isal':
        pytest.importorskip('isal')
    monkeypatch.setenv('PANICLE_VCF_GZIP_BACKEND', backend)
    text = b'header\r\n' + b'x' * 300000 + b'\nend\n'
    if kind == 'gzip':
        encoded = gzip.compress(text)
    elif kind == 'members':
        encoded = gzip.compress(text[:100]) + gzip.compress(text[100:])
    else:
        encoded = b''.join(bgzf(text[i:i+32000]) for i in range(0, len(text), 32000)) + bgzf(b'')
    path = tmp_path / 'input.bgz'
    path.write_bytes(encoded)
    with compression.open_compressed(path) as handle:
        assert list(handle) == text.splitlines(keepends=True)
    assert handle.closed
    handle.close()


@pytest.mark.parametrize('backend', ['stdlib', 'isal'])
@pytest.mark.parametrize('damage', ['truncated', 'crc'])
def test_corrupt_gzip_is_not_silently_retried(tmp_path, monkeypatch, backend, damage):
    if backend == 'isal':
        pytest.importorskip('isal')
    monkeypatch.setenv('PANICLE_VCF_GZIP_BACKEND', backend)
    encoded = bytearray(gzip.compress(b'0/0\t1/1\n' * 100))
    if damage == 'truncated':
        encoded = encoded[:-4]
    else:
        encoded[-8] ^= 1
    path = tmp_path / 'bad.gz'
    path.write_bytes(encoded)
    with pytest.raises((OSError, EOFError)):
        with compression.open_compressed(path) as handle:
            handle.read()


def test_short_writes():
    class ShortWriter(io.BytesIO):
        def write(self, value):
            return super().write(value[:3])
    handle = ShortWriter()
    storage._write_all(handle, b'abcdefgh')
    assert handle.getvalue() == b'abcdefgh'
    class FailedWriter:
        def write(self, value):
            return 0
    with pytest.raises(OSError, match='Incomplete'):
        storage._write_all(FailedWriter(), b'data')


@pytest.mark.parametrize('failure', [None, 'write', 'sync'])
def test_buffered_cache_small_slabs_and_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1)
    monkeypatch.setattr(storage, '_CACHE_WRITE_BUFFER_BYTES', 12)
    monkeypatch.setattr(storage, '_TRANSPOSE_BLOCK_MARKERS', 2)
    values = np.arange(35, dtype=np.int8).reshape(5, 7)
    writer = storage._DynamicInt8MatrixWriter(5)
    writer.append_block(values)
    def fail(*a, **kw):
        raise OSError('disk failure')
    if failure == 'write':
        monkeypatch.setattr(storage, '_write_slab', fail)
    if failure == 'sync':
        monkeypatch.setattr(storage.os, 'fsync', fail)
    path = tmp_path / 'out.npy'
    result = writer.finalize(cache_path=path, keep_indices=[0, 2, 3, 5, 6])
    np.testing.assert_array_equal(result, values[:, [0, 2, 3, 5, 6]])
    assert result.flags.c_contiguous and result.flags.writeable
    assert not list(tmp_path.glob('.panicle-geno-*'))
    if failure is None:
        expected = result.copy()
        result[0, 0] = 99
        np.testing.assert_array_equal(np.load(path), expected)
    else:
        assert not isinstance(result, np.memmap)
        assert not path.exists()


def test_backend_import_fallback(monkeypatch):
    import builtins
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == 'isal':
            raise ImportError('unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', unavailable)
    monkeypatch.delenv('PANICLE_VCF_GZIP_BACKEND', raising=False)
    assert compression.gzip_backend()[0] == 'stdlib'
    monkeypatch.setenv('PANICLE_VCF_GZIP_BACKEND', 'auto')
    assert compression.gzip_backend()[0] == 'stdlib'
    monkeypatch.setenv('PANICLE_VCF_GZIP_BACKEND', 'isal')
    with pytest.raises(ImportError, match='vcf-fast'):
        compression.gzip_backend()
    monkeypatch.setenv('PANICLE_VCF_GZIP_BACKEND', 'typo')
    with pytest.raises(ValueError, match='PANICLE_VCF_GZIP_BACKEND'):
        compression.gzip_backend()


def test_cache_writer_imputation(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, '_DIRECT_CACHE_MIN_BYTES', 1)
    monkeypatch.setattr(storage, '_CACHE_WRITE_BUFFER_BYTES', 8)
    values = np.array([[0, 2, -9], [0, -9, 1], [-9, 2, 1]], dtype=np.int8)
    writer = storage._DynamicInt8MatrixWriter(3)
    writer.append_block(values)
    result = writer.finalize(cache_path=tmp_path / 'imputed.npy', impute=True)
    np.testing.assert_array_equal(result, [[0, 2, 1], [0, 2, 1], [0, 2, 1]])
    assert writer.imputed_count == 3
