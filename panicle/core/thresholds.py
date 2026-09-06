"""Pure pipeline threshold policy. Preserve historical precedence explicitly.

The one-call API has its own fixed-threshold policy and does not use this module.
Reporting correction is distinct from FarmCPU's solver QTN input.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

from .methods import METHODS


@dataclass(frozen=True)
class Threshold:
    value: float
    source: str
    n_tests: float = float('nan')


@dataclass(frozen=True)
class TraitThresholds:
    base: Threshold
    methods: Mapping[str, Threshold]
    effective_n: Optional[int]


def base_threshold(*, n_markers, alpha=0.05, significance=None, n_eff=None,
                   use_effective_tests=True, estimated_me=None) -> Threshold:
    if significance is not None:
        return Threshold(significance, 'Fixed p-value')
    denominator = float(n_markers)
    source = 'Bonferroni (markers)'
    if n_eff:
        denominator = float(n_eff)
        source = 'Bonferroni (effective tests)'
    elif use_effective_tests and estimated_me:
        denominator = float(estimated_me)
        source = 'Bonferroni (effective tests)'
    return Threshold(alpha / max(denominator, 1.0), source, denominator)


def trait_thresholds(*, base, methods, n_tested, mac_filtered=False,
                     significance=None, alpha=0.05, n_eff=None,
                     use_effective_tests=True, estimated_me=None, farmcpu_params=None) -> TraitThresholds:
    params = farmcpu_params or {}
    # Historical solver precedence differs from base Bonferroni precedence:
    # an enabled LD estimate wins over explicit n_eff for FarmCPU.
    effective_n = int(estimated_me) if use_effective_tests and estimated_me else (n_eff or None)
    if significance is None and mac_filtered:
        base = Threshold(alpha / max(n_tested, 1), 'Bonferroni (markers, post-MAC)', float(n_tested))
    qtn = params.get('QTN_threshold', 0.01)
    if params.get('QTN_threshold_is_corrected'):
        farmcpu = Threshold(qtn, 'FarmCPU QTN threshold (corrected)')
    else:
        farmcpu = Threshold(qtn / (effective_n if effective_n else n_tested), 'FarmCPU QTN threshold')
    if 'resampling_significance_threshold' in params:
        resampling = Threshold(params['resampling_significance_threshold'], 'Resampling significance threshold')
    else:
        resampling = Threshold(farmcpu.value, 'FarmCPU QTN threshold (default)')
    selected = {method.upper() for method in methods}
    thresholds = {}
    for key, definition in METHODS.items():
        if key in selected:
            thresholds[definition.display_name] = (
                farmcpu if key == 'FARMCPU' else resampling if key == 'FARMCPURESAMPLING' else base
            )
    return TraitThresholds(base, thresholds, effective_n)
