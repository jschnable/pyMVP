"""Compiled extraction of diploid GT from variable-width FORMAT sample fields.

DS-containing records deliberately use the general decoder so GT/DS precedence
is unchanged. Unsupported GTs return a rejection flag, never guessed dosages.
"""
import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None


def _extract(raw, offsets, gt_indices, n_samples):
    n_markers = len(gt_indices)
    out = np.empty((n_markers, n_samples * 4 - 1), dtype=np.uint8)
    invalid = np.zeros(n_markers, dtype=np.bool_)
    for marker in prange(n_markers):
        cursor = offsets[marker]
        end = offsets[marker + 1]
        for sample in range(n_samples):
            field = 0
            gt_start = cursor
            gt_stop = cursor
            while cursor < end and raw[cursor] != 9:
                if raw[cursor] == 58:
                    if field == gt_indices[marker]:
                        gt_stop = cursor
                    field += 1
                    if field == gt_indices[marker]:
                        gt_start = cursor + 1
                cursor += 1
            if field == gt_indices[marker]:
                gt_stop = cursor
            if field < gt_indices[marker] or gt_stop - gt_start != 3:
                invalid[marker] = True
                a, sep, b = 46, 47, 46
            else:
                a, sep, b = raw[gt_start], raw[gt_start + 1], raw[gt_start + 2]
                if (a != 48 and a != 49 and a != 46) or (b != 48 and b != 49 and b != 46) or (sep != 47 and sep != 124):
                    invalid[marker] = True
            out[marker, sample * 4] = a
            out[marker, sample * 4 + 1] = sep
            out[marker, sample * 4 + 2] = b
            if sample < n_samples - 1:
                out[marker, sample * 4 + 3] = 9
                if cursor == end:
                    invalid[marker] = True
            elif cursor != end:
                invalid[marker] = True
            cursor += 1
    return out, invalid


if njit is not None:
    extract_gt_fields = njit(cache=True, parallel=True)(_extract)
else:  # pragma: no cover
    prange = range
    # Python byte-by-byte extraction is slower than the general record parser.
    extract_gt_fields = None
