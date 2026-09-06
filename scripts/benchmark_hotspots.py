"""Isolated experiments for the performance report (no production code edits).

Run after profile_stages.py, not concurrently with it. Uses warm filesystem
cache; comparisons measure application overhead, not sustained device throughput.
"""
import io
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from panicle.association.glm import PANICLE_GLM, PANICLE_GLM_MULTI
from panicle.association.blink import PANICLE_BLINK
from panicle.data.load_genotype_vcf import _DynamicInt8MatrixWriter
from panicle.utils import compact
from panicle.utils.data_types import GenotypeMatrix, GenotypeMap


def main():
    root = Path(tempfile.mkdtemp(prefix='panicle-hotspots-'))
    rows = []

    def bench(name, fn):
        result = fn()  # warm-up excluded
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            result = fn()
            elapsed.append(time.perf_counter()-start)
        row = dict(stage=name, seconds=elapsed, median=float(np.median(elapsed)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root / 'timings.json').write_text(json.dumps(rows, indent=2))
        return result

    rng = np.random.default_rng(20260906)
    n, m = 738, 100_000
    raw = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    g = GenotypeMatrix(raw, is_imputed=True, precompute_alleles=False)
    cv = rng.normal(size=(n, 3))
    phe = np.column_stack([np.arange(n), rng.normal(size=n)+.5*raw[:, 17]+.3*raw[:, 100]])
    gm = GenotypeMap(pd.DataFrame(dict(MARKER=[f'm{i}' for i in range(m)],
                                    CHROM=np.arange(m)//10_000+1, POS=np.arange(m)*1000+1)))
    for width in [1000, 5000, 20000]:
        block = raw[:, :width]
        a = bench(f'cast_{width}_current', lambda: compact.int8_to_float32(block))
        b = bench(f'cast_{width}_direct', lambda: np.asarray(block, dtype=np.float32, order='C'))
        np.testing.assert_array_equal(a, b)
    originals = {}
    for cpu in [1, 4]:
        originals[cpu] = bench(f'glm_cpu{cpu}', lambda: PANICLE_GLM(phe, g, CV=cv, cpu=cpu, verbose=False))
    original_blink = bench('blink_current', lambda: PANICLE_BLINK(phe, g, gm, CV=cv, verbose=False))

    def direct_cast(src, out=None):
        if out is None:
            return np.asarray(src, dtype=np.float32, order='C')
        np.copyto(out, src, casting='unsafe')
        return out

    with patch.object(compact, 'int8_to_float32', direct_cast):
        candidate = bench('glm_direct_cast', lambda: PANICLE_GLM(phe, g, CV=cv, verbose=False))
        candidate_blink = bench('blink_direct_cast', lambda: PANICLE_BLINK(phe, g, gm, CV=cv, verbose=False))
    np.testing.assert_array_equal(originals[1].to_numpy(), candidate.to_numpy())
    np.testing.assert_array_equal(original_blink.to_numpy(), candidate_blink.to_numpy())

    np.save(root / 'genotype.npy', raw)
    mm = GenotypeMatrix(np.load(root / 'genotype.npy', mmap_mode='r'), is_imputed=True, precompute_alleles=False)
    bench('glm_warm_memmap', lambda: PANICLE_GLM(phe, mm, CV=cv, verbose=False))

    class RemapWriter(_DynamicInt8MatrixWriter):
        def _grow(self, min_capacity):
            capacity = self.capacity
            while capacity < min_capacity:
                capacity = max(capacity*2, min_capacity)
            self.memmap.flush()
            del self.memmap
            with open(self.path, 'r+b') as f:
                f.truncate(self.n_rows*capacity)
            self.memmap = np.memmap(self.path, dtype=np.int8, mode='r+', shape=(capacity, self.n_rows))
            self.capacity = capacity

    def write(cls):
        writer = cls(n)
        try:
            for start in range(0, m, 4096):
                writer.append_block(raw[:, start:start+4096])
            return writer.finalize()
        finally:
            writer.discard()
    a = bench('writer_current', lambda: write(_DynamicInt8MatrixWriter))
    b = bench('writer_remap_only', lambda: write(RemapWriter))
    np.testing.assert_array_equal(a, raw)
    np.testing.assert_array_equal(b, raw)

    table = pd.DataFrame(dict(MARKER=[f'm{i}' for i in range(m)],
                              P=originals[1].pvalues, Effect=originals[1].effects))
    bench('csv_disk', lambda: table.to_csv(root / 'results.csv', index=False))
    bench('csv_stringio', lambda: table.to_csv(io.StringIO(), index=False))
    bench('hdf5_disk', lambda: table_to_hdf5(root / 'results.h5', table))
    for marker_count in [100_000, 1_000_000]:
        matrix = GenotypeMatrix(rng.integers(0, 3, size=(n, marker_count), dtype=np.int8),
                                is_imputed=True, precompute_alleles=False)
        traits = rng.normal(size=(n, 8))
        separate = bench(f'joint_{marker_count}_separate', lambda: [PANICLE_GLM(
            np.column_stack([np.arange(n), traits[:, j]]), matrix, CV=cv, verbose=False) for j in range(8)])
        joint = bench(f'joint_{marker_count}_joint', lambda: PANICLE_GLM_MULTI(
            traits, matrix, CV=cv, trait_names=[f'T{i}' for i in range(8)], verbose=False))
        for j in range(8):
            np.testing.assert_allclose(separate[j].to_numpy(), joint[f'T{j}'].to_numpy(), rtol=1e-4, atol=1e-6)
    (root / 'file_sizes.json').write_text(json.dumps({p.name: p.stat().st_size for p in root.iterdir() if p.is_file()}, indent=2))
    print('Exact conversion/association/writer checks and joint-GLM tolerance checks passed. Artifacts:', root, flush=True)


def table_to_hdf5(path, table):
    import h5py
    with h5py.File(path, 'w') as f:
        f.create_dataset('MARKER', data=table.MARKER.to_numpy(dtype='S16'))
        f.create_dataset('P', data=table.P.to_numpy())
        f.create_dataset('Effect', data=table.Effect.to_numpy())


if __name__ == '__main__':
    main()
