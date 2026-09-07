"""Optional accelerated gzip reading, with an explicit stdlib fallback.

Only import failures trigger fallback. Corrupt/truncated input is never retried
or silently accepted. Both readers validate gzip checksums and support members
such as BGZF. No subprocesses or worker threads are required.
"""
import gzip
import io
import os

from .vcf_metrics import timed


def gzip_backend():
    # Installing an optional library must not silently change performance.
    backend = os.environ.get('PANICLE_VCF_GZIP_BACKEND', 'stdlib').lower()
    if backend not in ('auto', 'stdlib', 'isal'):
        raise ValueError('PANICLE_VCF_GZIP_BACKEND must be auto, stdlib, or isal')
    if backend != 'stdlib':
        try:
            from isal import igzip
            return 'isal', igzip.GzipFile
        except (ImportError, OSError):
            if backend == 'isal':
                raise ImportError('Accelerated gzip requires the optional panicle[vcf-fast] extra')
    return 'stdlib', gzip.GzipFile


class _CompressedInput(io.RawIOBase):
    """Measure both C and Python readers at the same buffered-read boundary."""
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.closed:
            raise ValueError('read of closed file')
        return timed('gzip_read_decompress', self.reader.readinto, buffer)

    def close(self):
        try:
            self.reader.close()
        finally:
            super().close()


def open_compressed(path):
    _, reader_type = gzip_backend()
    reader = reader_type(filename=str(path), mode='rb')
    try:
        return io.BufferedReader(_CompressedInput(reader), buffer_size=128 * 1024)
    except BaseException:
        reader.close()
        raise
