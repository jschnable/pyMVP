"""Named reporting inputs; numerical arrays are referenced, never copied."""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from ..core.thresholds import Threshold
from ..core.workflow import MethodRunResult
from ..utils.data_types import GenotypeMatrix


@dataclass(frozen=True)
class MethodReport:
    run: MethodRunResult
    threshold: Threshold


@dataclass(frozen=True)
class ReportOptions:
    outputs: Sequence[str]
    alpha: float
    max_dosage: float
    include_standard_errors: bool = False


@dataclass(frozen=True)
class TraitReport:
    name: str
    methods: Mapping[str, MethodReport]
    base_threshold: Threshold
    n_samples: Optional[int] = None
    n_markers: Optional[int] = None
    runtime_seconds: Optional[float] = None
    genotype_for_maf: Optional[GenotypeMatrix] = None
    maf_keep_indices: Optional[np.ndarray] = None
