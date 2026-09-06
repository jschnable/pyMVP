"""Shared preparation and execution primitives for both public GWAS interfaces.

Interface adapters retain their defaults, logging and error policies. These
operations neither write files nor plot and do not change numerical kernels.
"""
from dataclasses import dataclass
import time
from typing import Any, Callable, Optional

import numpy as np

from ..utils.data_types import GenotypeMatrix, GenotypeMap, ensure_eager_genotype
from ..utils.stats import compute_mac_keep_indices


@dataclass
class PreparedTrait:
    name: str
    phenotype: np.ndarray
    genotype: GenotypeMatrix
    covariates: Optional[np.ndarray]
    kinship: Any
    sample_indices: np.ndarray
    geno_map: Optional[GenotypeMap]
    keep_indices: Optional[np.ndarray] = None

    def legacy_tuple(self):
        """Compatibility for the old private pipeline preparation API."""
        return (self.phenotype, self.genotype, self.covariates, self.kinship,
                self.sample_indices, self.geno_map, self.keep_indices)


@dataclass(frozen=True)
class TraitCacheKey:
    samples: bytes
    n_pcs: int
    need_kinship: bool
    min_mac: int
    max_dosage: float

    @classmethod
    def create(cls, indices, n_pcs=0, need_kinship=False, min_mac=0, max_dosage=2.0):
        return cls(np.asarray(indices, dtype=np.int64).tobytes(), int(n_pcs),
                   bool(need_kinship), int(min_mac or 0), float(max_dosage))


@dataclass
class TraitPreparation:
    key: TraitCacheKey
    genotype: GenotypeMatrix
    pcs: Optional[np.ndarray]
    kinship: Any
    geno_map: Optional[GenotypeMap]
    keep_indices: Optional[np.ndarray]


@dataclass
class MarkerSelection:
    genotype: GenotypeMatrix
    geno_map: Optional[GenotypeMap]
    keep_indices: Optional[np.ndarray]


@dataclass
class MethodRunResult:
    name: str
    result: Any = None
    seconds: float = 0.0
    lambda_gc: Optional[float] = None
    lambda_gc_is_approx: bool = False
    error: Optional[str] = None

    def legacy_tuple(self):
        return (self.name, self.result, self.lambda_gc, self.lambda_gc_is_approx, self.error)


def retained_samples(values, covariates=None):
    mask = np.isfinite(values)
    if covariates is not None:
        mask = mask & np.isfinite(covariates).all(axis=1)
    return mask


def group_sample_indices(sample_sets):
    """Stable groups of positions with exactly identical ordered sample sets."""
    groups = {}
    for position, samples in enumerate(sample_sets):
        key = np.asarray(samples, dtype=np.int64).tobytes()
        groups.setdefault(key, []).append(position)
    return groups


def select_markers(genotype, geno_map, min_mac=0, max_dosage=2.0, *,
                   materialize=False, filter_fn=compute_mac_keep_indices):
    keep = filter_fn(genotype, int(min_mac or 0), max_dosage=max_dosage)
    if keep is None or keep.size == genotype.n_markers:
        return MarkerSelection(genotype, geno_map, None)
    selected_map = geno_map.subset_markers(keep) if geno_map is not None else None
    selected = genotype.subset_markers(keep) if materialize else genotype
    return MarkerSelection(selected, selected_map, keep)


def association_genotype(genotype, keep_indices=None):
    if keep_indices is not None:
        return genotype.subset_markers(keep_indices)
    return ensure_eager_genotype(genotype)


_MAP_METHODS = {'MLM_LOCO', 'FARMCPU', 'BLINK', 'BAYESLOCO', 'FARMCPURESAMPLING'}
_DISPLAY_NAMES = {'FARMCPU': 'FarmCPU', 'FARMCPURESAMPLING': 'FarmCPUResampling', 'MLM_LOCO': 'MLM'}


def run_method(method: str, trait: PreparedTrait, *, runner: Callable, options=None) -> MethodRunResult:
    """Execute one solver using the same input contract in either interface.

    Exceptions propagate: the one-call API raises, while the pipeline's adapter
    records failures and continues as before. Options contain interface defaults
    and method-specific settings, never a second copy of phenotype/genotype data.
    """
    key = method.upper()
    kwargs = dict(phe=trait.phenotype, geno=trait.genotype, CV=trait.covariates)
    if key in _MAP_METHODS:
        kwargs['map_data'] = trait.geno_map
    if key == 'MLM':
        kwargs['K'] = trait.kinship
    kwargs.update(options or {})
    started = time.time()
    result = runner(**kwargs)
    return MethodRunResult(_DISPLAY_NAMES.get(key, key), result, time.time() - started)


def run_trait_group(traits, genotype, *, runner, options=None):
    """Shared phenotype assembly and execution for compatible multi-trait scans."""
    first = traits[0]
    kwargs = dict(phe=np.column_stack([t.phenotype[:, 1].astype(np.float64) for t in traits]),
                  geno=genotype, CV=first.covariates, trait_names=[t.name for t in traits])
    kwargs.update(options or {})
    started = time.time()
    results = runner(**kwargs)
    elapsed = time.time() - started
    return {name: MethodRunResult(name, result, elapsed / len(traits)) for name, result in results.items()}
