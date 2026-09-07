"""Reproducible VCF loading benchmark. Select a checkout with PYTHONPATH.

Generate once, then run before/after sequentially on the same input. Timed loads
force cache rebuilds and include imputation, map construction and cache output.
Imports and checksums are excluded. By default a full workload warmup excludes
JIT warmup too; --skip-warmup includes first-load/JIT effects in the first run.
This is not a controlled cold-storage benchmark. Peak RSS is process-wide.
"""
import argparse
import cProfile
import gzip
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
import time

import numpy as np

from panicle.data.load_genotype_vcf import load_genotype_vcf
if __package__:
    from .vcf_observability import StageRecorder, resources, environment, cache_disk_usage
else:
    from vcf_observability import StageRecorder, resources, environment, cache_disk_usage


def generate(path, samples, markers, extra_format, late_general=False):
    rng = np.random.default_rng(7319)
    tokens = np.array([b'0/0\t', b'0/1\t', b'1/1\t', b'./.\t'], dtype='S4')
    if extra_format:
        tokens = np.array([b'0/0:8\t', b'0/1:8\t', b'1/1:8\t', b'./.:8\t'], dtype='S6')
    opener = gzip.open if path.suffix == '.gz' else open
    compression = {'compresslevel': 1} if path.suffix == '.gz' else {}
    with opener(path, 'xb', **compression) as handle:
        handle.write(b'##fileformat=VCFv4.2\n##contig=<ID=1>\n')
        handle.write(b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        if extra_format:
            handle.write(b'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n')
        if late_general:
            handle.write(b'##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n')
        handle.write(('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' +
                      '\t'.join(f's{i}' for i in range(samples)) + '\n').encode())
        fmt = 'GT:DP' if extra_format else 'GT'
        for start in range(0, markers, 2048):
            count = min(2048, markers - start)
            codes = rng.choice(4, size=(count, samples), p=[.45, .35, .19, .01])
            calls = tokens[codes]
            for row in range(count):
                marker = start + row + 1
                row_fmt = fmt
                row_calls = calls[row].tobytes()[:-1]
                if late_general and marker == markers:
                    row_fmt = 'GT:DS'
                    ds_tokens = np.array([b'0/0:0\t', b'0/1:1\t', b'1/1:2\t', b'./.:.\t'], dtype='S6')
                    row_calls = ds_tokens[codes[row]].tobytes()[:-1]
                handle.write(f'1\t{marker}\tm{marker}\tA\tG\t.\tPASS\t.\t{row_fmt}\t'.encode())
                handle.write(row_calls + b'\n')


def fingerprint(result):
    genotype, ids, gmap = result
    digest = hashlib.sha256()
    for row in genotype:
        for start in range(0, row.size, 8 * 1024 * 1024):
            digest.update(row[start:start + 8 * 1024 * 1024].tobytes(order='C'))
    frame = gmap.to_dataframe() if hasattr(gmap, 'to_dataframe') else gmap
    metadata = hashlib.sha256()
    if len(frame) == 0:
        metadata.update(frame.to_csv(index=False).encode())
    for start in range(0, len(frame), 8192):
        metadata.update(frame.iloc[start:start + 8192].to_csv(index=False, header=start == 0).encode())
    return dict(shape=list(genotype.shape), dtype=str(genotype.dtype),
                genotype_sha256=digest.hexdigest(), map_sha256=metadata.hexdigest(),
                ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--markers', type=int, default=100000)
    parser.add_argument('--extra-format', action='store_true')
    parser.add_argument('--late-general', action='store_true',
                        help='Generate a final GT:DS record to exercise late fallback')
    parser.add_argument('--backend', choices=['builtin', 'cyvcf2'], default='builtin')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--skip-warmup', action='store_true', help='Avoid an extra full load of production-scale files')
    parser.add_argument('--max-missing', type=float, default=.2)
    parser.add_argument('--min-maf', type=float, default=.01)
    parser.add_argument('--stages', action='store_true', help='Opt-in stage timings; no tile/thread tuning')
    args = parser.parse_args()
    if args.generate:
        generate(args.input, args.samples, args.markers, args.extra_format, args.late_general)
        print(f'Generated {args.input} ({args.input.stat().st_size:,} bytes)')
        return
    if args.output is None or args.repeats < 1:
        parser.error('Loading requires --output and --repeats >= 1')
    # Warm on the actual workload: includes both QC and imputation kernels.
    options = dict(backend=args.backend, drop_monomorphic=True,
                   max_missing=args.max_missing, min_maf=args.min_maf)
    expected = None
    if not args.skip_warmup:
        result = load_genotype_vcf(args.input, force_recache=True, **options)
        expected = fingerprint(result)
        del result
    timings = []
    measurements = []
    for _ in range(args.repeats):
        before_resources = resources()
        started = time.perf_counter()
        with StageRecorder() if args.stages else nullcontext() as metrics:
            if metrics is None:
                result = load_genotype_vcf(args.input, force_recache=True, **options)
            else:
                result = metrics.call('load_total', load_genotype_vcf, args.input, force_recache=True, **options)
        timings.append(time.perf_counter() - started)
        measurements.append(dict(resources_before=before_resources, resources_after_load=resources(),
                                 timing=metrics.report() if metrics else None))
        if metrics:
            print(json.dumps(metrics.report()), flush=True)
        print(f'Run {len(timings)}: load finished in {timings[-1]:.3f} s; hashing output', flush=True)
        actual = fingerprint(result)
        first_measurement = expected is None
        if expected is None:
            expected = actual
        assert actual == expected
        print('Fingerprints recorded' if first_measurement else 'Fingerprints verified', flush=True)
        del result
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(load_genotype_vcf, args.input, force_recache=True, **options)
        profiler.dump_stats(args.profile)
    started = time.perf_counter()
    result = load_genotype_vcf(args.input, **options)
    cache_seconds = time.perf_counter() - started
    # Map cache representation may canonicalize chromosome dtypes; genotype/IDs
    # and metadata are already compared on the freshly built output above.
    np.testing.assert_equal(result[0].shape, expected['shape'])
    row = dict(input=str(args.input), input_bytes=args.input.stat().st_size,
               backend=args.backend, options=options, warmup=not args.skip_warmup,
               seconds=timings, median_seconds=float(np.median(timings)),
               cache_seconds=cache_seconds, peak_rss_native=resources().get('peak_rss_native'),
               peak_rss_bytes=resources().get('peak_rss_bytes'), environment=environment(),
               measurements=measurements, cache_files=cache_disk_usage(args.input), stages_enabled=args.stages,
               fingerprint=expected)
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert expected == before['fingerprint'], 'Decoded output changed'
        row['speedup'] = before['median_seconds'] / row['median_seconds']
    args.output.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
