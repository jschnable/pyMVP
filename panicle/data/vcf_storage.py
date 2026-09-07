"""Marker-major temporary storage shared by VCF decoding paths."""
import os
import tempfile
import logging

import numpy as np

_DIRECT_CACHE_MIN_BYTES = 64 * 1024 * 1024
_TRANSPOSE_BLOCK_MARKERS = 8192


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

    def _grow(self, min_capacity):
        new_capacity = self.capacity
        while new_capacity < min_capacity:
            new_capacity = max(new_capacity * 2, min_capacity)
        # Extending the file preserves its prefix. Close the old mapping before
        # resizing (also required on Windows), without copying/re-writing it.
        self.memmap.flush()
        self.memmap._mmap.close()
        self.memmap = None
        with open(self.path, 'r+b') as fh:
            fh.truncate(self.n_rows * new_capacity)
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='r+', shape=(new_capacity, self.n_rows))
        self.capacity = new_capacity

    def append(self, column):
        if column.shape != (self.n_rows,):
            raise ValueError(f"Column shape mismatch: expected ({self.n_rows},), got {column.shape}")
        if self.count >= self.capacity:
            self._grow(self.count + 1)
        self.memmap[self.count, :] = column
        self.count += 1

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
        mm.flush()
        self.imputed_count = 0
        self.published_cache_path = None
        shape = (self.n_rows, total_cols)

        def fill(destination):
            missing_count = 0
            for start in range(0, total_cols, _TRANSPOSE_BLOCK_MARKERS):
                stop = min(start + _TRANSPOSE_BLOCK_MARKERS, total_cols)
                source = mm[start:stop, :] if indices is None else mm[indices[start:stop], :]
                block = np.array(source.T, dtype=np.int8, copy=True, order='C')
                if impute:
                    from ..utils.data_types import impute_major_allele_inplace
                    missing_count += impute_major_allele_inplace(block)
                destination[:, start:stop] = block
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
                self.imputed_count = fill(mapped)
                mapped.flush()
                mapped._mmap.close()
                mapped = None
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
            self.memmap.flush()
            self.memmap._mmap.close()
            self.memmap = None
        try:
            os.remove(self.path)
        except OSError:
            pass
