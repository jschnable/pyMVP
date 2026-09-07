"""Shared v2 genotype cache policy; parsers retain format-specific decoding/QC."""
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .io_utils import genotype_cache_filters_match, save_genotype_cache_filters
from ..utils.map_cache import load_genotype_map_cache, save_genotype_map_cache


@dataclass(frozen=True)
class GenotypeCache:
    base: str
    sources: Sequence[Union[str, Path]]
    filters: Mapping[str, Any]

    def path(self, suffix: str) -> str:
        return self.base + '.panicle.v2.' + suffix

    def load(self, *, force: bool = False, logger: logging.Logger):
        """Return the cached triple, or None on a stale, absent or unreadable cache.

        Preserve the legacy strict mtime comparison and CSV-map migration. All
        source files participate (including BIM/FAM for PLINK).
        """
        if force:
            return None
        try:
            maps = [p for p in (self.path('map.npz'), self.path('map.csv')) if os.path.exists(p)]
            if not maps or not all(os.path.exists(self.path(s)) for s in ('geno.npy', 'ind.txt')):
                return None
            source_mtime = max(os.path.getmtime(p) for p in self.sources)
            if not (
                os.path.getmtime(self.path('geno.npy')) > source_mtime
                and os.path.getmtime(self.path('ind.txt')) > source_mtime
                and max(os.path.getmtime(p) for p in maps) > source_mtime
            ):
                return None
            if not genotype_cache_filters_match(self.base, self.filters):
                logger.info('[Cache] Filter fingerprint mismatch or missing for %s; rebuilding cache.', self.base)
                return None
            logger.info('[Cache] Loading binary cache for %s...', self.base)
            geno = np.load(self.path('geno.npy'), mmap_mode='r')
            with open(self.path('ind.txt'), 'r') as handle:
                ids = [line.strip() for line in handle]
            gmap = load_genotype_map_cache(
                self.path('map.npz'), legacy_csv_path=self.path('map.csv'),
                migrate_legacy=True, legacy_is_imputed=True,
            )
            return geno, ids, gmap
        except Exception as exc:
            # Caches are optional accelerators; retain the parsers' fallback.
            logger.warning('[Cache] Failed to load cache: %s', exc)
            return None

    def invalidate(self):
        """Invalidate the old fingerprint before publishing a rebuilt genotype."""
        try:
            os.remove(self.path('filters.json'))
        except FileNotFoundError:
            pass

    def save(self, geno, ids, gmap, *, logger: logging.Logger, genotype_written=False) -> None:
        """Write the existing cache format; failures do not discard parsed data."""
        try:
            logger.info('[Cache] Saving binary cache to %s.panicle.v2.*', self.base)
            if not genotype_written:
                np.save(self.path('geno.npy'), geno)
            with open(self.path('ind.txt'), 'w') as handle:
                for individual in ids:
                    handle.write(f'{individual}\n')
            frame = pd.DataFrame(gmap) if isinstance(gmap, list) else gmap
            if hasattr(frame, 'attrs'):
                frame.attrs['is_imputed'] = True
            save_genotype_map_cache(self.path('map.npz'), frame)
            save_genotype_cache_filters(self.base, self.filters)
        except Exception as exc:
            logger.warning('[Cache] Failed to save cache: %s', exc)
