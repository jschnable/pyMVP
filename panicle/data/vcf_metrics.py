"""Opt-in, context-local VCF timing hooks; no clocks/counters when disabled.

Times are wall time, not CPU or physical I/O time. Nested stages report both
inclusive and exclusive time. Logical bytes describe algorithmic work, not
storage-device traffic. Used by the benchmark harness, not enabled by loaders.
"""
from contextvars import ContextVar
from time import perf_counter

_active = ContextVar('vcf_metrics', default=None)


class VCFMetrics:
    def __init__(self):
        self.stages = {}
        self.counters = {}
        self.stack = []

    def __enter__(self):
        self.token = _active.set(self)
        return self

    def __exit__(self, *exc):
        _active.reset(self.token)

    def call(self, name, function, *args, **kwargs):
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.stages.setdefault(name, dict(calls=0, inclusive_seconds=0.0, exclusive_seconds=0.0))
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += elapsed - frame[1]

    def report(self):
        return dict(stages=self.stages, logical_counters=self.counters)


def timed(name, function, *args, **kwargs):
    metrics = _active.get()
    if metrics is None:
        return function(*args, **kwargs)
    return metrics.call(name, function, *args, **kwargs)


def count(name, value, *, maximum=False):
    metrics = _active.get()
    if metrics is not None:
        previous = metrics.counters.get(name, 0)
        metrics.counters[name] = max(previous, int(value)) if maximum else previous + int(value)
