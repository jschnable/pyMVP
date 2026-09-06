"""Cross-revision output check; select the revision with PYTHONPATH.

Run once with --output BEFORE, then with --output AFTER --compare BEFORE.
Exercises real solvers and writers; plotting is stubbed equally in both runs.
Each public interface is compared with its own prior behavior, not with the
other interface (their defaults and output conventions intentionally differ).
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from panicle.core import mvp
from panicle.pipelines.gwas import GWASPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(821)
    n, m = 64, 120
    raw = rng.integers(0, 3, (n, m), dtype=np.int8)
    raw[:, 0] = 0  # exercise MAC filtering and full-map padding
    ids = [f's{i}' for i in range(n)]
    phe = pd.DataFrame({'ID': ids, 'a': rng.normal(size=n), 'b': rng.normal(size=n),
                        'c': rng.normal(size=n)})
    phe.loc[[1, 3], ['a', 'b']] = np.nan  # joint scan plus a distinct sample mask
    cv = rng.normal(size=(n, 1))
    cv[5] = np.nan
    gmap = pd.DataFrame({'MARKER': [f'm{i}' for i in range(m)],
                         'CHROM': np.repeat([1, 2, 3], m // 3), 'POS': np.arange(m) + 1})
    phe.to_csv(args.output / 'phenotype.csv', index=False)
    genoframe = pd.DataFrame(raw, columns=gmap.MARKER)
    genoframe.insert(0, 'ID', ids)
    genoframe.to_csv(args.output / 'genotype.csv', index=False)
    gmap.to_csv(args.output / 'map.csv', index=False)
    pd.DataFrame({'ID': ids, 'environment': cv[:, 0]}).to_csv(args.output / 'covariates.csv', index=False)
    for mode in ['global', 'loco']:
        # The baseline one-call global-MLM path cannot subset KinshipMatrix
        # objects for missing samples. Exercise global with complete data and
        # missing-data grouping with LOCO; pipeline covers both with missingness.
        core_phe = phe.fillna(0) if mode == 'global' else phe
        core_cv = np.nan_to_num(cv) if mode == 'global' else cv
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            mvp, 'PANICLE_Report', return_value={'files_created': [], 'plots': {}},
        ):
            result = mvp.PANICLE(
                core_phe, raw, gmap, CV=core_cv, method=['GLM', 'MLM', 'FarmCPU', 'BLINK'],
                mlm_mode=mode, n_pcs=2, min_mac=2, maxLoop=2, verbose=False,
                output_prefix=str(args.output / ('core_' + mode)),
            )
        for trait, methods in result['results'].items():
            assert set(methods) == {'GLM', 'MLM', 'FarmCPU', 'BLINK'}
            for method, value in methods.items():
                np.save(args.output / f'core_{mode}_{trait}_{method}.npy', value.to_numpy())
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline = GWASPipeline(output_dir=str(args.output / ('pipeline_' + mode)))
            pipeline.load_data(str(args.output / 'phenotype.csv'), str(args.output / 'genotype.csv'),
                               map_file=str(args.output / 'map.csv'),
                               covariate_file=str(args.output / 'covariates.csv'))
            pipeline.align_samples()
            pipeline.compute_population_structure(n_pcs=2, calculate_kinship=mode == 'global')
            pipeline.run_analysis(
                methods=['GLM', 'MLM', 'FARMCPU', 'BLINK'], mlm_mode=mode, min_mac=2,
                max_iterations=2, use_effective_tests=False,
                outputs=['all_marker_pvalues', 'significant_marker_pvalues'],
                include_standard_errors=True,
            )
        for trait in ['a', 'b', 'c']:
            table = pd.read_csv(args.output / ('pipeline_' + mode) / f'GWAS_{trait}_all_results.csv')
            for method in ['GLM', 'MLM', 'FarmCPU', 'BLINK']:
                assert any(method in column for column in table.columns), (method, list(table.columns))

    files = sorted(p.relative_to(args.output) for p in args.output.rglob('*')
                   if p.suffix in {'.npy', '.csv', '.json'} and not p.name.endswith('log.json'))
    if args.compare:
        previous_files = sorted(p.relative_to(args.compare) for p in args.compare.rglob('*')
                                if p.suffix in {'.npy', '.csv', '.json'} and not p.name.endswith('log.json'))
        assert files == previous_files, 'Output filenames changed'
        for relative in files:
            before, after = args.compare / relative, args.output / relative
            if relative.suffix == '.npy':
                np.testing.assert_array_equal(np.load(before), np.load(after))
            elif relative.suffix == '.csv':
                pd.testing.assert_frame_equal(pd.read_csv(before), pd.read_csv(after), check_exact=True)
            else:
                assert json.loads(before.read_text()) == json.loads(after.read_text()), relative
        for after in args.output.glob('core_*_summary.txt'):
            # Only phase runtimes may differ; compare the complete scientific summary.
            before = args.compare / after.name
            assert before.read_text().split('Runtimes (seconds):')[0] == after.read_text().split('Runtimes (seconds):')[0]
        print(f'Exact parity: {len(files)} array/table/metadata files and core scientific summaries.')
    else:
        print(f'Baseline generated: {len(files)} array/table/metadata files.')


if __name__ == '__main__':
    main()
