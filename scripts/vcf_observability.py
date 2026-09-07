"""Benchmark-only stage wrappers and portable resource metadata.

Wrappers are scoped/restored and intended for a single benchmark process, not
concurrent application calls. No per-record tracing or cProfile is required.
"""
from contextlib import ExitStack
from functools import wraps
import gzip
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

from panicle.data import load_genotype_vcf as vcf
from panicle.data import genotype_cache
from panicle.data.vcf_storage import _DynamicInt8MatrixWriter


class StageRecorder:
    def __enter__(self):
        from panicle.data.vcf_metrics import VCFMetrics, timed
        self.context = ExitStack()
        self.metrics = self.context.enter_context(VCFMetrics())
        targets = [
            (vcf, '_try_load_simple_biallelic_gt_vcf_bulk', 'builtin_stream'),
            (vcf, '_decode_builtin_records', 'general_decode'),
            (vcf, '_decode_cyvcf2_records', 'cyvcf2_decode'),
            (vcf, 'canonicalize_genotype_map_dataframe', 'map_canonicalization'),
            (genotype_cache.GenotypeCache, 'save', 'cache_save'),
            (genotype_cache, 'save_genotype_map_cache', 'map_serialization'),
            (_DynamicInt8MatrixWriter, 'append_block', 'spool_append'),
            (_DynamicInt8MatrixWriter, 'append', 'spool_append'),
            (_DynamicInt8MatrixWriter, '_grow', 'spool_growth'),
            (_DynamicInt8MatrixWriter, 'finalize', 'finalization'),
        ]
        # Optional implementation detail: unavailable readers leave time in the
        # stream residual; do not pretend that residual is pure parsing time.
        if not hasattr(vcf, 'open_compressed') and hasattr(gzip, '_GzipReader') and hasattr(gzip._GzipReader, 'read'):
            targets.append((gzip._GzipReader, 'read', 'gzip_read_decompress'))
        try:
            for owner, attribute, stage in targets:
                function = getattr(owner, attribute)
                @wraps(function)
                def wrapper(*args, _function=function, _stage=stage, **kwargs):
                    return timed(_stage, _function, *args, **kwargs)
                self.context.enter_context(patch.object(owner, attribute, wrapper))
        except BaseException:
            self.context.close()
            raise
        return self.metrics

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


def resources():
    try:
        import resource
    except ImportError:
        return dict(available=False)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024 if sys.platform.startswith('linux') else None
    result = dict(available=True, peak_rss_native=rss, peak_rss_bytes=rss_bytes,
                  user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                  major_faults=usage.ru_majflt, minor_faults=usage.ru_minflt,
                  input_blocks=usage.ru_inblock, output_blocks=usage.ru_oublock)
    try:
        entries = Path('/proc/self/io').read_text().splitlines()
        result['linux_process_io_bytes'] = {key: int(value) for key, value in
                                            (line.split(':', 1) for line in entries)
                                            if key in ('read_bytes', 'write_bytes', 'cancelled_write_bytes')}
    except (OSError, ValueError):
        result['linux_process_io_bytes'] = None
    return result


def environment():
    import numpy
    import pandas
    result = dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                  processor=platform.processor(), python=platform.python_version(), numpy=numpy.__version__,
                  pandas=pandas.__version__, logical_cpus=os.cpu_count(),
                  thread_environment={k: os.environ.get(k) for k in
                                      ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    if hasattr(vcf, 'open_compressed'):
        from panicle.data.vcf_compression import gzip_backend
        result['vcf_gzip_backend'] = gzip_backend()[0]
        result['vcf_cache_writer'] = 'buffered'
    try:
        import numba
        result.update(numba=numba.__version__, numba_threads=numba.get_num_threads())
    except ImportError:
        result.update(numba=None, numba_threads=None)
    try:
        import psutil
        result['ram_bytes'] = psutil.virtual_memory().total
    except (ImportError, OSError):
        result['ram_bytes'] = None
    return result


def cache_disk_usage(path):
    files = []
    for suffix in ('geno.npy', 'ind.txt', 'map.npz', 'filters.json'):
        item = Path(str(path) + '.panicle.v2.' + suffix)
        if item.exists():
            stat = item.stat()
            files.append(dict(name=item.name, logical_bytes=stat.st_size,
                              allocated_bytes=stat.st_blocks * 512 if hasattr(stat, 'st_blocks') else None))
    return files
