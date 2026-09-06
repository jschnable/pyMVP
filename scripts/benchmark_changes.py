"""Compare performance changes against an independent checkout.

Select the checkout with PYTHONPATH; run baseline and candidate sequentially.
Imports, data generation, warmup, and result validation are excluded from timing.
Example:
  PYTHONPATH=/tmp/baseline python scripts/benchmark_changes.py --output /tmp/before
  PYTHONPATH=. python scripts/benchmark_changes.py --output /tmp/after --compare /tmp/before

Warm filesystem cache only. Top-level API timing excludes plotting (stubbed for
both revisions); single-method scans omit result output. No production monkeypatches
other than that plotting stub. Results are saved for numerical comparisons.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

import panicle
from panicle.association.blink import PANICLE_BLINK
from panicle.association.glm import PANICLE_GLM
from panicle.core import mvp
from panicle.data.load_genotype_vcf import _DynamicInt8MatrixWriter
from panicle.utils.compact import int8_to_float32
from panicle.utils.data_types import GenotypeMatrix, GenotypeMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--markers', type=int, default=1_000_000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.markers < 101 or args.repeats < 1:
        parser.error('Require at least 101 markers and one repetition')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def bench(name, fn, *, exact=True):
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        elapsed = []
        for _ in range(args.repeats):
            with contextlib.redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                result = fn()
                elapsed.append(time.perf_counter() - start)
        np.save(args.output / (name + '.npy'), result)
        if args.compare:
            previous = np.load(args.compare / (name + '.npy'), mmap_mode='r')
            if exact:
                np.testing.assert_array_equal(result, previous)
            else:
                np.testing.assert_allclose(result, previous, rtol=1e-5, atol=1e-6, equal_nan=True)
        row = dict(stage=name, seconds=elapsed, median=float(np.median(elapsed)), exact=exact)
        rows.append(row)
        (args.output / 'timings.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)

    rng = np.random.default_rng(20260906)
    n, m = 738, args.markers
    raw = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    g = GenotypeMatrix(raw, is_imputed=True, precompute_alleles=False)
    cv = rng.normal(size=(n, 3))
    y = rng.normal(size=n) + .5 * raw[:, 17] + .3 * raw[:, 100]
    phe = np.column_stack([np.arange(n), y])
    gm = GenotypeMap(pd.DataFrame(dict(MARKER=[f'm{i}' for i in range(m)],
        CHROM=np.arange(m)//max(1, (m+9)//10)+1, POS=np.arange(m)*1000+1)))
    for width in [5000, 20000]:
        bench(f'cast_{width}', lambda: int8_to_float32(raw[:, :width]))
    bench('glm', lambda: PANICLE_GLM(phe, g, CV=cv, verbose=False).to_numpy())
    bench('glm_prefetch', lambda: PANICLE_GLM(phe, g, CV=cv, cpu=4, verbose=False).to_numpy())
    bench('blink', lambda: PANICLE_BLINK(phe, g, gm, CV=cv, verbose=False).to_numpy())
    bench('blink_filtered', lambda: PANICLE_BLINK(phe, g, gm, CV=cv, maf_threshold=.49, verbose=False).to_numpy())
    traits = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f'T{i}' for i in range(8)])
    traits.insert(0, 'ID', np.arange(n))
    with patch.object(mvp, 'PANICLE_Report', return_value={'files_created': []}):
        def api():
            results = mvp.PANICLE(traits, g, gm, CV=cv, method=['GLM'], file_output=False, verbose=False)
            return np.stack([results['results'][f'T{i}']['GLM'].to_numpy() for i in range(8)])
        bench('api_8_traits', api, exact=False)

    writer_input = raw[:, :min(m, 100_000)]
    def write():
        writer = _DynamicInt8MatrixWriter(n)
        try:
            for start in range(0, writer_input.shape[1], 4096):
                writer.append_block(writer_input[:, start:start+4096])
            return writer.finalize()
        finally:
            writer.discard()
    bench('writer_100k', write)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    (args.output / 'environment.json').write_text(json.dumps(dict(
        source=panicle.__file__, markers=m, samples=n, numpy=np.__version__,
        numpy_config=config.getvalue(), comparison=str(args.compare)), indent=2))
    print('Completed; comparisons passed.' if args.compare else 'Baseline completed.', flush=True)


if __name__ == '__main__':
    main()
