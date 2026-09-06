"""Reproducible stage benchmark; writes only to a fresh output directory.

Run from the repository: PYTHONPATH=. python scripts/profile_stages.py
Timings exclude imports and fixture generation. First calls are reported separately
from three warm repetitions; an extra cProfile run is excluded from timings.
Filesystem cache is not evicted. File sizes are logical bytes, not physical I/O.
"""
import argparse
import contextlib
import cProfile
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from panicle.association.glm import PANICLE_GLM, PANICLE_GLM_MULTI
from panicle.association.mlm import PANICLE_MLM
from panicle.association.mlm_loco import PANICLE_MLM_LOCO
from panicle.association.farmcpu import PANICLE_FarmCPU
from panicle.association.blink import PANICLE_BLINK
from panicle.data.load_genotype_vcf import load_genotype_vcf
from panicle.matrix.kinship import PANICLE_K_VanRaden
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA
from panicle.pipelines.gwas import GWASPipeline
from panicle.utils.data_types import GenotypeMatrix, GenotypeMap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--samples', type=int, default=738)
    ap.add_argument('--markers', type=int, default=100_000)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.samples < 10 or args.markers < 101 or args.repeats < 1:
        ap.error('Require at least 10 samples, 101 markers, and one warm repetition')
    root = args.output or Path(tempfile.mkdtemp(prefix='panicle-benchmark-'))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    rows = []
    meta = dict(platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, cpus=os.cpu_count(),
                samples=args.samples, markers=args.markers, repeats=args.repeats,
                environment={k: os.environ.get(k) for k in
                             ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']})
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    meta['numpy_config'] = config.getvalue()
    (root / 'environment.json').write_text(json.dumps(meta, indent=2))

    def bench(name, fn, repeats=None):
        result = None
        for rep in range(1 + (args.repeats if repeats is None else repeats)):
            before = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter()
            with (root / 'run.log').open('a') as log, contextlib.redirect_stdout(log):
                result = fn()
            after = resource.getrusage(resource.RUSAGE_SELF)
            row = dict(stage=name, repetition=rep, seconds=time.perf_counter()-start,
                       cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                       peak_rss=after.ru_maxrss, major_faults=after.ru_majflt-before.ru_majflt,
                       in_blocks=after.ru_inblock-before.ru_inblock,
                       out_blocks=after.ru_oublock-before.ru_oublock)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (root / 'timings.json').write_text(json.dumps(rows, indent=2))
        profile = cProfile.Profile()
        with (root / 'run.log').open('a') as log, contextlib.redirect_stdout(log):
            profile.runcall(fn)
        profile.dump_stats(str(root / (name + '.prof')))
        with (root / (name + '.txt')).open('w') as f:
            pstats.Stats(profile, stream=f).strip_dirs().sort_stats('cumulative').print_stats(35)
        return result

    repo = Path(__file__).resolve().parents[1]
    vcf = root / 'demo.vcf.gz'
    shutil.copyfile(repo / 'examples/example_genotypes.vcf.gz', vcf)
    for backend in ['builtin', 'cyvcf2']:
        bench('demo_parse_' + backend, lambda: load_genotype_vcf(
            str(vcf), backend=backend, force_recache=True))
    bench('demo_cache_load', lambda: load_genotype_vcf(str(vcf)))
    for outputs, label in [([], 'none'), (['all_marker_pvalues', 'significant_marker_pvalues'], 'tables'),
                           (['all_marker_pvalues', 'significant_marker_pvalues', 'manhattan', 'qq'], 'all')]:
        def pipeline():
            p = GWASPipeline(str(root / ('demo_' + label)))
            p.load_data(str(repo / 'examples/example_phenotypes.csv'), str(vcf), trait_columns=['PlantHeight'])
            p.align_samples()
            p.compute_population_structure(n_pcs=3, calculate_kinship=False)
            p.run_analysis(traits=['PlantHeight'], methods=['GLM'], outputs=outputs)
            return p
        bench('demo_pipeline_' + label, pipeline)

    rng = np.random.default_rng(20260906)
    n, m = args.samples, args.markers
    raw = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    g = GenotypeMatrix(raw, is_imputed=True, precompute_alleles=False)
    cv = rng.normal(size=(n, 3))
    y = rng.normal(size=n) + .5 * raw[:, 17] + .3 * raw[:, 100]
    phe = np.column_stack([np.arange(n), y])
    gm = GenotypeMap(pd.DataFrame(dict(MARKER=[f'm{i}' for i in range(m)],
                 CHROM=np.arange(m)//max(1, (m+9)//10)+1, POS=np.arange(m)*1000+1)))
    pcs = bench('synthetic_pca', lambda: PANICLE_PCA(M=g, pcs_keep=3, verbose=False))
    kin = bench('synthetic_kinship', lambda: PANICLE_K_VanRaden(g, verbose=False))
    loco = bench('synthetic_loco_kinship', lambda: PANICLE_K_VanRaden_LOCO(g, gm, verbose=False))
    result = bench('synthetic_glm', lambda: PANICLE_GLM(phe, g, CV=cv, verbose=False))
    bench('synthetic_mlm_global', lambda: PANICLE_MLM(phe, g, K=kin, CV=cv, verbose=False))
    bench('synthetic_mlm_loco', lambda: PANICLE_MLM_LOCO(phe, g, gm, loco_kinship=loco, CV=cv, verbose=False))
    bench('synthetic_farmcpu', lambda: PANICLE_FarmCPU(phe, g, gm, CV=cv, verbose=False))
    bench('synthetic_blink', lambda: PANICLE_BLINK(phe, g, gm, CV=cv, verbose=False))
    ys = np.column_stack([np.arange(n), y, rng.normal(size=(n, 7))])
    bench('synthetic_glm_8_separate', lambda: [PANICLE_GLM(ys[:, [0, j]], g, CV=cv, verbose=False) for j in range(1, 9)])
    bench('synthetic_glm_8_joint', lambda: PANICLE_GLM_MULTI(ys[:, 1:], g, CV=cv, verbose=False))
    for outputs, label in [([], 'none'), (['all_marker_pvalues', 'significant_marker_pvalues'], 'tables'),
                          (['manhattan', 'qq'], 'plots')]:
        p = GWASPipeline(str(root / ('synthetic_' + label)))
        p.genotype_matrix, p.geno_map = g, gm
        bench('synthetic_output_' + label, lambda: p._save_trait_results(
            'Trait', {'GLM': result}, .05/m, .05, m, 2., outputs, 'Bonferroni',
            n_samples=n, n_markers=m, geno_for_maf=g))
    (root / 'file_sizes.json').write_text(json.dumps({str(p.relative_to(root)): p.stat().st_size
        for p in root.rglob('*') if p.is_file() and p.suffix not in ['.prof', '.txt', '.log']}, indent=2))
    print('Artifacts:', root, flush=True)


if __name__ == '__main__':
    main()
