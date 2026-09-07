"""QC and accumulation for general VCF records, outside the bulk hot path.

Scalar filtering deliberately preserves the loader's pre-imputation MAF rule
and its definition of monomorphic (all valid calls 0 or all 2). Bulk decoding
retains a vectorized implementation: never route marker batches through here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VCFFilters:
    include_indels: bool = True
    drop_monomorphic: bool = False
    max_missing: float = 1.0
    min_maf: float = 0.0

    def accepts(self, col, ref, alt, ploidy, *, simple_diploid=False):
        if not self.include_indels and (len(ref) != 1 or len(alt) != 1):
            return False
        valid = col != -9
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            return False
        # Keep the general/fast paths' original comparison and evaluation order.
        if not simple_diploid and self.drop_monomorphic:
            values = np.unique(col[valid])
            if values.size == 1 and (values[0] == 0 or values[0] == 2):
                return False
        if not simple_diploid or self.max_missing < 1.0:
            if 1.0 - n_valid / float(col.size) > self.max_missing:
                return False
        valid_col = None
        if simple_diploid and self.drop_monomorphic:
            valid_col = col[valid]
            if np.all(valid_col == 0) or np.all(valid_col == 2):
                return False
        if self.min_maf > 0.0:
            if valid_col is None:
                valid_col = col[valid]
            dosage_sum = float(np.sum(valid_col))
            ploidy = max(ploidy, 1)
            minor_count = min(dosage_sum, ploidy * n_valid - dosage_sum)
            if minor_count / max(ploidy * col.size, 1.0) < self.min_maf:
                return False
        return True


@dataclass
class BuiltinDecodeState:
    sanity_checked: bool = False


class VariantAccumulator:
    """Filter decoded columns and append them to a marker-major writer.

    Column-oriented map lists avoid allocating a dictionary per marker. The
    public map/list representation is constructed once, after decoding.
    """
    def __init__(self, filters, writer_factory):
        self.filters = filters
        self.writer_factory = writer_factory
        self.n_samples = 0
        self.writer = None
        markers = []
        self.map_columns = dict(MARKER=markers, SNP=markers, CHROM=[], POS=[], REF=[], ALT=[])

    def consider(self, col, chrom, pos, vid, ref, alt, ploidy, *, simple_diploid=False):
        if not simple_diploid:
            col = np.asarray(col, dtype=np.int16)
        if not self.filters.accepts(col, ref, alt, ploidy, simple_diploid=simple_diploid):
            return
        col = col.astype(np.int8, copy=False)
        if self.writer is None:
            self.writer = self.writer_factory(self.n_samples)
        self.writer.append(col)
        marker = vid if vid and vid != '.' else "%s:%s:%s:%s" % (chrom, pos, ref, alt)
        columns = self.map_columns
        columns['MARKER'].append(marker)
        columns['CHROM'].append(str(chrom))
        columns['POS'].append(int(pos))
        columns['REF'].append(ref)
        columns['ALT'].append(alt)

    def finalize(self, **kwargs):
        if self.writer is None:
            return np.zeros((self.n_samples, 0), dtype=np.int8)
        return self.writer.finalize(**kwargs)

    def close(self):
        if self.writer is not None:
            self.writer.discard()
