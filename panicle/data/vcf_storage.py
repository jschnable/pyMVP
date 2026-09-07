"""Marker-major temporary storage shared by VCF decoding paths."""
import os
import tempfile
import logging

import numpy as np
from .vcf_metrics import timed, count

_DIRECT_CACHE_MIN_BYTES = 64 * 1024 * 1024
_TRANSPOSE_BLOCK_MARKERS = 8192
_CACHE_WRITE_BUFFER_BYTES = 64 * 1024 * 1024


def _write_all(handle, data):
    """FileIO writes may be short; never publish a partially written cache."""
    view = memoryview(data).cast('B')
    while view:
        written = handle.write(view)
        if written is None or written <= 0:
            raise OSError('Incomplete genotype cache write')
        view = view[written:]


def _write_slab(handle, block, header_bytes, total_cols, start):
    for sample, row in enumerate(block):
        handle.seek(header_bytes + sample * total_cols + start)
        _write_all(handle, row)


class _DynamicInt8MatrixWriter:
    """Append-only int8 matrix builder backed by a temporary memmap."""

    def __init__(self, n_rows, initial_capacity=4096):
        self.n_rows = int(n_rows)
        if self.n_rows <= 0:
            raise ValueError("Writer requires a positive number of rows")
        self.capacity = max(int(initial_capacity), 1)
        tmp = tempfile.NamedTemporaryFile(prefix="panicle_geno_", suffix=".tmp", delete=False)
        self.path = tmp.name
        tmp.close()
        # Store as marker-major while building so each appended marker is a
        # contiguous write. Finalize transposes back to the public sample-major
        # shape: (n_individuals, n_markers).
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='w+', shape=(self.capacity, self.n_rows))
        self.count = 0
        count('spool_capacity_peak_bytes', self.capacity * self.n_rows, maximum=True)

    def _grow(self, min_capacity):
        new_capacity = self.capacity
        while new_capacity < min_capacity:
            new_capacity = max(new_capacity * 2, min_capacity)
        # Extending the file preserves its prefix. Close the old mapping before
        # resizing (also required on Windows), without copying/re-writing it.
        timed('spool_flush', self.memmap.flush)
        self.memmap._mmap.close()
        self.memmap = None
        with open(self.path, 'r+b') as fh:
            fh.truncate(self.n_rows * new_capacity)
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='r+', shape=(new_capacity, self.n_rows))
        self.capacity = new_capacity
        count('spool_capacity_peak_bytes', self.capacity * self.n_rows, maximum=True)

    def append(self, column):
        if column.shape != (self.n_rows,):
            raise ValueError(f"Column shape mismatch: expected ({self.n_rows},), got {column.shape}")
        if self.count >= self.capacity:
            self._grow(self.count + 1)
        self.memmap[self.count, :] = column
        self.count += 1
        count('spool_append_bytes', self.n_rows)

    def append_block(self, columns):
        if columns.ndim != 2 or columns.shape[0] != self.n_rows:
            raise ValueError(
                f"Block shape mismatch: expected ({self.n_rows}, n_columns), got {columns.shape}"
            )
        n_columns = int(columns.shape[1])
        if n_columns == 0:
            return
        new_count = self.count + n_columns
        if new_count > self.capacity:
            self._grow(new_count)
        self.memmap[self.count:new_count, :] = columns.T
        self.count = new_count
        count('spool_append_bytes', self.n_rows * n_columns)

    def finalize(self, *, cache_path=None, before_publish=None, keep_indices=None, impute=False):
        """Finalize in bounded tiles, optionally publishing the existing C-order cache.

        Large cached loads return a writable copy-on-write memmap: edits made by
        callers do not alter the reusable disk cache. Small/uncached loads retain
        the ordinary ndarray return. Cache write failures fall back to RAM.
        """
        mm = self.memmap
        if mm is None:
            return np.zeros((self.n_rows, 0), dtype=np.int8)
        indices = None if keep_indices is None else np.asarray(keep_indices, dtype=np.int64)
        total_cols = self.count if indices is None else len(indices)
        timed('spool_flush', mm.flush)
        self.imputed_count = 0
        self.published_cache_path = None
        shape = (self.n_rows, total_cols)
        count('spool_plus_output_logical_peak_bytes', self.capacity * self.n_rows + self.n_rows * total_cols, maximum=True)

        def fill(destination, first=0, last=None, stage='output_tile_writes'):
            last = total_cols if last is None else last
            missing_count = 0
            for start in range(first, last, _TRANSPOSE_BLOCK_MARKERS):
                stop = min(start + _TRANSPOSE_BLOCK_MARKERS, last)
                source = mm[start:stop, :] if indices is None else mm[indices[start:stop], :]
                block = timed('transpose_tiles', np.array, source.T, dtype=np.int8, copy=True, order='C')
                if impute:
                    from ..utils.data_types import impute_major_allele_inplace
                    missing_count += timed('tile_imputation', impute_major_allele_inplace, block)
                timed(stage, destination.__setitem__, (slice(None), slice(start - first, stop - first)), block)
            return missing_count

        result = None
        if cache_path is not None and self.n_rows * total_cols >= _DIRECT_CACHE_MIN_BYTES:
            temporary = None
            mapped = None
            try:
                fd, temporary = tempfile.mkstemp(prefix='.panicle-geno-', suffix='.npy',
                                                 dir=os.path.dirname(os.path.abspath(cache_path)))
                os.close(fd)
                mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.int8, shape=shape)
                # Small transpose tiles preserve cache locality. Accumulate
                # them in a bounded heap slab, then issue larger row writes.
                # Do not keep the final file mapped while scattering writes:
                # that can trigger expensive page faults under memory pressure.
                header_bytes = mapped.offset
                mapped._mmap.close()
                mapped = None
                slab_cols = max(1, _CACHE_WRITE_BUFFER_BYTES // self.n_rows)
                slab_buffer = np.empty((self.n_rows, min(slab_cols, total_cols)), dtype=np.int8)
                with open(temporary, 'r+b', buffering=0) as handle:
                    for first in range(0, total_cols, slab_cols):
                        last = min(first + slab_cols, total_cols)
                        slab = slab_buffer[:, :last - first]
                        self.imputed_count += fill(slab, first, last, stage='tile_buffer_copies')
                        timed('cache_file_writes', _write_slab, handle, slab, header_bytes, total_cols, first)
                        count('output_tile_write_bytes', slab.nbytes)
                    timed('cache_flush', os.fsync, handle.fileno())
                if before_publish is not None:
                    before_publish()
                os.replace(temporary, cache_path)
                temporary = None
                result = np.load(cache_path, mmap_mode='c')
                self.published_cache_path = os.path.abspath(cache_path)
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).warning('Direct genotype cache write failed; falling back to RAM: %s', exc)
            finally:
                if mapped is not None:
                    mapped._mmap.close()
                if temporary is not None:
                    os.remove(temporary)
        if total_cols == 0:
            result = np.zeros((self.n_rows, 0), dtype=np.int8)
        elif result is None:
            result = np.empty(shape, dtype=np.int8)
            self.imputed_count = fill(result)
            count('output_tile_write_bytes', result.nbytes)
        mm._mmap.close()
        self.memmap = None
        try:
            os.remove(self.path)
        except OSError:
            pass
        del mm
        return result

    def discard(self):
        if self.memmap is not None:
            timed('spool_flush', self.memmap.flush)
            self.memmap._mmap.close()
            self.memmap = None
        try:
            os.remove(self.path)
        except OSError:
            pass
