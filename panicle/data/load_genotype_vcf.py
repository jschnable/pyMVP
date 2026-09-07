#!/usr/bin/env python
"""
VCF loader for GWAS: builds (geno_matrix, individual_ids, geno_map).

Key features:
- Streaming parsing of VCF text (supports .vcf and .vcf.gz)
- Supports arbitrary ploidy by counting ALT alleles in each GT field
- Optional multi-allelic splitting into per-ALT pseudo-biallelic markers
- GT-based coding with DS fallback; missing as -9 (int8)
- Optional basic QC filters: monomorphic, missingness, MAF

Return signature:
    (geno_matrix: np.ndarray[int8], individual_ids: List[str], geno_map: DataFrame-like)

The geno_map is a pandas DataFrame if pandas is installed; otherwise a list of dict rows.

Simple GT text uses batched decoding; general text and cyvcf2/BCF retain separate
record decoders. QC and accumulation are shared outside the bulk hot path.
"""
from __future__ import print_function
from panicle.data.genotype_cache import GenotypeCache
from .vcf_records import VCFFilters, VariantAccumulator, BuiltinDecodeState
from .vcf_format import extract_gt_fields
from contextlib import nullcontext

import logging
import sys
import gzip
import io
import os

logger = logging.getLogger(__name__)
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise ImportError("NumPy is required: pip install numpy")
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional accelerator
    _NUMBA_AVAILABLE = False
    njit = None
    prange = range
from panicle.utils.data_types import (
    CHROM_COLUMN,
    LEGACY_MARKER_ID_COLUMN,
    MARKER_ID_COLUMN,
    POS_COLUMN,
    canonicalize_genotype_map_dataframe,
    impute_major_allele_inplace,
)


MISSING = -9
_SIMPLE_BULK_BATCH_MARKERS = 32768

_GT_TOKEN_CACHE_SENTINEL = object()
_GT_TOKEN_CACHE: Dict[str, Optional[Tuple[str, ...]]] = {}
_BIALLELIC_DOSAGE_CACHE: Dict[Tuple[str, ...], Tuple[int, int]] = {}
_BIALLELIC_GT_CACHE: Dict[str, Tuple[int, int]] = {}
_BIALLELIC_GT_DIRECT: Dict[str, Tuple[int, int]] = {
    '0/0': (0, 2),
    '0|0': (0, 2),
    '0/1': (1, 2),
    '1/0': (1, 2),
    '0|1': (1, 2),
    '1|0': (1, 2),
    '1/1': (2, 2),
    '1|1': (2, 2),
    '0': (0, 1),
    '1': (1, 1),
    './.': (MISSING, 0),
    '.|.': (MISSING, 0),
    '.': (MISSING, 0),
}

_FORMAT_CACHE: Dict[str, Tuple[Tuple[str, ...], Dict[str, int]]] = {}

if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _decode_simple_gt_matrix_numba(raw, n_markers, n_samples):
        # Each worker writes one contiguous marker; transpose is a view.
        out = np.empty((n_markers, n_samples), dtype=np.int8).T
        missing_counts = np.zeros(n_markers, dtype=np.int64)
        invalid = 0
        for marker_idx in prange(n_markers):
            c0 = 0
            c1 = 0
            c2 = 0
            marker_missing = 0
            marker_invalid = 0
            for sample_idx in range(n_samples):
                offset = sample_idx * 4
                allele_a = raw[marker_idx, offset]
                separator = raw[marker_idx, offset + 1]
                allele_b = raw[marker_idx, offset + 2]
                if separator != 47 and separator != 124:  # '/' or '|'
                    marker_invalid = 1
                if sample_idx < n_samples - 1 and raw[marker_idx, offset + 3] != 9:
                    marker_invalid = 1

                value = np.int8(0)
                if allele_a == 46 or allele_b == 46:  # '.'
                    value = np.int8(-9)
                    marker_missing += 1
                else:
                    if allele_a == 49:  # '1'
                        value += 1
                    elif allele_a != 48:  # not '0'
                        marker_invalid = 1
                    if allele_b == 49:
                        value += 1
                    elif allele_b != 48:
                        marker_invalid = 1
                    if value == 0:
                        c0 += 1
                    elif value == 1:
                        c1 += 1
                    elif value == 2:
                        c2 += 1
                out[sample_idx, marker_idx] = value

            major = np.int8(0)
            if c1 > c0 and c1 >= c2:
                major = np.int8(1)
            elif c2 > c0 and c2 > c1:
                major = np.int8(2)
            if marker_missing > 0:
                for sample_idx in range(n_samples):
                    if out[sample_idx, marker_idx] == -9:
                        out[sample_idx, marker_idx] = major
            missing_counts[marker_idx] = marker_missing
            invalid += marker_invalid

        return out, missing_counts, invalid

    @njit(cache=True, parallel=True)
    def _decode_simple_gt_matrix_stats_numba(raw, n_markers, n_samples):
        out = np.empty((n_markers, n_samples), dtype=np.int8).T
        missing_counts = np.zeros(n_markers, dtype=np.int64)
        counts_0 = np.zeros(n_markers, dtype=np.int64)
        counts_1 = np.zeros(n_markers, dtype=np.int64)
        counts_2 = np.zeros(n_markers, dtype=np.int64)
        invalid = 0
        for marker_idx in prange(n_markers):
            c0 = 0
            c1 = 0
            c2 = 0
            marker_missing = 0
            marker_invalid = 0
            for sample_idx in range(n_samples):
                offset = sample_idx * 4
                allele_a = raw[marker_idx, offset]
                separator = raw[marker_idx, offset + 1]
                allele_b = raw[marker_idx, offset + 2]
                if separator != 47 and separator != 124:  # '/' or '|'
                    marker_invalid = 1
                if sample_idx < n_samples - 1 and raw[marker_idx, offset + 3] != 9:
                    marker_invalid = 1

                value = np.int8(0)
                if allele_a == 46 or allele_b == 46:  # '.'
                    value = np.int8(-9)
                    marker_missing += 1
                else:
                    if allele_a == 49:  # '1'
                        value += 1
                    elif allele_a != 48:  # not '0'
                        marker_invalid = 1
                    if allele_b == 49:
                        value += 1
                    elif allele_b != 48:
                        marker_invalid = 1
                    if value == 0:
                        c0 += 1
                    elif value == 1:
                        c1 += 1
                    elif value == 2:
                        c2 += 1
                out[sample_idx, marker_idx] = value

            major = np.int8(0)
            if c1 > c0 and c1 >= c2:
                major = np.int8(1)
            elif c2 > c0 and c2 > c1:
                major = np.int8(2)
            if marker_missing > 0:
                for sample_idx in range(n_samples):
                    if out[sample_idx, marker_idx] == -9:
                        out[sample_idx, marker_idx] = major
            missing_counts[marker_idx] = marker_missing
            counts_0[marker_idx] = c0
            counts_1[marker_idx] = c1
            counts_2[marker_idx] = c2
            invalid += marker_invalid

        return out, missing_counts, counts_0, counts_1, counts_2, invalid
else:
    _decode_simple_gt_matrix_numba = None
    _decode_simple_gt_matrix_stats_numba = None


# Compatibility import for callers/tests of the former local writer.
from .vcf_storage import _DynamicInt8MatrixWriter


def _open_text(path):
    """Open VCF text transparently from plain or gzip-compressed files.

    Accepts string or Path-like, and handles .vcf, .vcf.gz, and .vcf.bgz.
    """
    p = str(path)
    pl = p.lower()
    if pl.endswith('.gz') or pl.endswith('.bgz'):
        return io.TextIOWrapper(gzip.open(p, 'rb'))
    return open(p, 'r')


def _open_binary(path):
    """Open VCF bytes transparently from plain or gzip-compressed files."""
    p = str(path)
    pl = p.lower()
    if pl.endswith('.gz') or pl.endswith('.bgz'):
        raw = gzip.GzipFile(filename=p, mode='rb')
        return io.BufferedReader(raw, buffer_size=128 * 1024)
    return open(p, 'rb')


def _parse_samples(header_line):
    # header line starts with #CHROM
    cols = header_line.strip().split('\t')
    if len(cols) < 9 or cols[0] != '#CHROM':
        raise ValueError('Malformed VCF header line: missing #CHROM ... FORMAT ...')
    samples = cols[9:]
    return samples


def _build_snp_id(chrom, pos, vid, ref, alt):
    if vid and vid != '.':
        return vid
    return "%s:%s:%s:%s" % (chrom, pos, ref, alt)


def _find_first_tabs(line: bytes, count: int) -> Optional[Tuple[int, ...]]:
    tabs = []
    start = 0
    for _ in range(count):
        pos = line.find(b'\t', start)
        if pos < 0:
            return None
        tabs.append(pos)
        start = pos + 1
    return tuple(tabs)


def _parse_simple_biallelic_gt_line(
    line: bytes,
    n_samples: int,
) -> Optional[Tuple[np.ndarray, str, int, str, str, str]]:
    """Fast parse ``FORMAT=GT`` biallelic diploid records with fixed 3-byte calls.

    The general parser below handles all other compatible VCF shapes. This
    helper only accepts rows like ``...\\tGT\\t0/0\\t0/1\\t1/1`` (phased or
    unphased, with ``./.``/``.|.`` missing calls). It deliberately returns
    ``None`` for unusual allele codes, extra FORMAT fields, multiallelic rows,
    variable ploidy, or non-fixed-width sample fields so existing behavior is
    preserved by the fallback path.
    """
    if line.endswith(b'\n'):
        line = line[:-1]
    if line.endswith(b'\r'):
        line = line[:-1]

    tabs = _find_first_tabs(line, 9)
    if tabs is None:
        return None

    t0, t1, t2, t3, t4, _t5, _t6, _t7, t8 = tabs
    fmt = line[tabs[7] + 1 : t8]
    if fmt != b'GT':
        return None

    alt = line[t3 + 1 : t4]
    if not alt or alt == b'.' or b',' in alt:
        return None

    sample_blob = line[t8 + 1 :]
    expected_len = n_samples * 4 - 1
    if n_samples <= 0 or len(sample_blob) != expected_len:
        return None

    raw = np.frombuffer(sample_blob, dtype=np.uint8)
    if raw.size != expected_len:
        return None
    if n_samples > 1 and not np.all(raw[3::4] == 9):  # tab separators
        return None

    separators = raw[1::4]
    if not np.all((separators == 47) | (separators == 124)):  # '/' or '|'
        return None

    allele_a = raw[0::4]
    allele_b = raw[2::4]
    valid_a = (allele_a == 48) | (allele_a == 49) | (allele_a == 46)
    valid_b = (allele_b == 48) | (allele_b == 49) | (allele_b == 46)
    if not (np.all(valid_a) and np.all(valid_b)):
        return None

    col = ((allele_a == 49).astype(np.int8) + (allele_b == 49).astype(np.int8))
    missing = (allele_a == 46) | (allele_b == 46)
    if np.any(missing):
        col = col.copy()
        col[missing] = MISSING

    chrom = line[:t0].decode()
    pos = int(line[t0 + 1 : t1])
    vid = line[t1 + 1 : t2].decode()
    ref = line[t2 + 1 : t3].decode()
    alt_str = alt.decode()
    return col, chrom, pos, vid, ref, alt_str


def _try_load_simple_biallelic_gt_vcf_bulk(
    vcf_path,
    include_indels=True,
    drop_monomorphic=False,
    max_missing=1.0,
    min_maf=0.0,
    *, split_multiallelic=True, cache=None,
) -> Optional[Tuple[np.ndarray, list, dict, int]]:
    """Stream bulk-compatible records and decode unsupported records in place.

    Pending batches can fall back without rereading the completed prefix. Keep
    legacy bulk/general QC rounding by deferring differing selections to EOF.
    """
    individual_ids = None
    sample_blobs = []
    batch_marker_ids = []
    batch_chrom_values = []
    batch_pos_values = []
    batch_ref_values = []
    batch_alt_values = []
    marker_ids = []
    chrom_values = []
    pos_values = []
    ref_values = []
    alt_values = []
    n_samples = 0
    expected_len = 0
    writer = None
    total_n_missing = 0
    batch_prefixes = []
    batch_gt_indices = []
    batch_has_extra = False
    used_general = False
    needs_imputation = False
    batch_bytes = 0
    bulk_drops = []
    general_drops = []
    state = BuiltinDecodeState()
    filters = VCFFilters(include_indels, drop_monomorphic, max_missing, min_maf)

    def process_batch(blobs, batch_marker_ids, batch_chrom_values, batch_pos_values, batch_ref_values, batch_alt_values):
        if not blobs:
            return None
        n_markers = len(blobs)
        raw = np.frombuffer(b''.join(blobs), dtype=np.uint8)
        if batch_has_extra:
            if extract_gt_fields is None:
                return False
            offsets = np.empty(n_markers + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum([len(blob) for blob in blobs], out=offsets[1:])
            raw, invalid_rows = extract_gt_fields(
                raw, offsets, np.asarray(batch_gt_indices, dtype=np.int64), n_samples,
            )
            if np.any(invalid_rows):
                return False
        else:
            try:
                raw = raw.reshape(n_markers, expected_len)
            except ValueError:
                return False

        need_counts = drop_monomorphic or min_maf > 0.0
        if _decode_simple_gt_matrix_numba is not None:
            if need_counts:
                decoded = _decode_simple_gt_matrix_stats_numba(raw, n_markers, n_samples)
                geno, missing_counts, counts_0, counts_1, counts_2, invalid = decoded
            else:
                geno, missing_counts, invalid = _decode_simple_gt_matrix_numba(raw, n_markers, n_samples)
                counts_0 = counts_1 = counts_2 = None
            if invalid:
                return False
        else:
            if n_samples > 1 and not np.all(raw[:, 3::4] == 9):  # tab separators
                return False
            separators = raw[:, 1::4]
            if not np.all((separators == 47) | (separators == 124)):  # '/' or '|'
                return False

            allele_a = raw[:, 0::4]
            allele_b = raw[:, 2::4]
            valid_a = (allele_a == 48) | (allele_a == 49) | (allele_a == 46)
            valid_b = (allele_b == 48) | (allele_b == 49) | (allele_b == 46)
            if not (np.all(valid_a) and np.all(valid_b)):
                return False

            geno_marker_major = (
                (allele_a == 49).astype(np.int8)
                + (allele_b == 49).astype(np.int8)
            )
            missing = (allele_a == 46) | (allele_b == 46)
            if np.any(missing):
                geno_marker_major[missing] = MISSING

            if need_counts or np.any(missing):
                counts_0 = np.sum(geno_marker_major == 0, axis=1)
                counts_1 = np.sum(geno_marker_major == 1, axis=1)
                counts_2 = np.sum(geno_marker_major == 2, axis=1)
            else:
                counts_0 = counts_1 = counts_2 = None
            missing_counts = np.sum(missing, axis=1)

            if int(np.sum(missing_counts)) > 0:
                if counts_0 is None:
                    counts_0 = np.sum(geno_marker_major == 0, axis=1)
                    counts_1 = np.sum(geno_marker_major == 1, axis=1)
                    counts_2 = np.sum(geno_marker_major == 2, axis=1)
                major = np.argmax(np.stack([counts_0, counts_1, counts_2], axis=0), axis=0)
                major = major.astype(np.int8, copy=False)
                geno_marker_major[missing] = np.broadcast_to(
                    major[:, np.newaxis],
                    geno_marker_major.shape,
                )[missing]

            geno = geno_marker_major.T

        keep_rows = missing_counts != n_samples
        if not include_indels:
            is_snp = np.fromiter(
                (len(ref) == 1 and len(alt) == 1 for ref, alt in zip(batch_ref_values, batch_alt_values)),
                dtype=np.bool_,
                count=n_markers,
            )
            keep_rows &= is_snp
        if drop_monomorphic:
            monomorphic_ref_alt = (
                ((counts_0 > 0) & (counts_1 == 0) & (counts_2 == 0))
                | ((counts_2 > 0) & (counts_0 == 0) & (counts_1 == 0))
            )
            keep_rows &= ~monomorphic_ref_alt
        if min_maf > 0.0:
            n_valid = counts_0 + counts_1 + counts_2
            sum_dosage = counts_1 + (2 * counts_2)
            valid_alleles = 2 * n_valid
            minor_count = np.minimum(sum_dosage, valid_alleles - sum_dosage)
            maf = minor_count / float(2 * n_samples)
            keep_rows &= maf >= min_maf

        bulk_keep = keep_rows.copy()
        general_keep = keep_rows.copy()
        if max_missing < 1.0:
            bulk_keep &= missing_counts / float(n_samples) <= max_missing
            general_keep &= ~(1.0 - (n_samples - missing_counts) / float(n_samples) > max_missing)
        keep_rows = bulk_keep | general_keep
        bulk_flags = bulk_keep[keep_rows]
        general_flags = general_keep[keep_rows]

        if not np.all(keep_rows):
            geno = geno[:, keep_rows]
            missing_counts = missing_counts[keep_rows]
            kept = keep_rows.tolist()
            kept_marker_ids = [value for value, keep in zip(batch_marker_ids, kept) if keep]
            kept_chrom_values = [value for value, keep in zip(batch_chrom_values, kept) if keep]
            kept_pos_values = [value for value, keep in zip(batch_pos_values, kept) if keep]
            kept_ref_values = [value for value, keep in zip(batch_ref_values, kept) if keep]
            kept_alt_values = [value for value, keep in zip(batch_alt_values, kept) if keep]
        else:
            kept_marker_ids = batch_marker_ids
            kept_chrom_values = batch_chrom_values
            kept_pos_values = batch_pos_values
            kept_ref_values = batch_ref_values
            kept_alt_values = batch_alt_values
        return (
            geno,
            int(np.sum(missing_counts)),
            kept_marker_ids,
            kept_chrom_values,
            kept_pos_values,
            kept_ref_values,
            kept_alt_values,
            bulk_flags, general_flags, missing_counts,
        )

    def clear_batch():
        nonlocal batch_has_extra, batch_bytes
        batch_has_extra = False
        batch_bytes = 0
        batch_prefixes.clear()
        batch_gt_indices.clear()
        del sample_blobs[:]
        del batch_marker_ids[:]
        del batch_chrom_values[:]
        del batch_pos_values[:]
        del batch_ref_values[:]
        del batch_alt_values[:]

    def append_result(result):
        nonlocal writer, total_n_missing
        if result is None:
            return True
        if result is False:
            return False
        geno, n_missing, kept_marker_ids, kept_chrom_values, kept_pos_values, kept_ref_values, kept_alt_values, bulk_flags, general_flags, missing_counts = result
        offset = 0 if writer is None else writer.count
        bulk_drops.extend((offset + int(i), int(missing_counts[i])) for i in np.flatnonzero(~bulk_flags))
        general_drops.extend((offset + int(i), int(missing_counts[i])) for i in np.flatnonzero(~general_flags))
        if geno.shape[1] > 0:
            if writer is None:
                writer = _DynamicInt8MatrixWriter(n_samples)
            writer.append_block(geno)
        total_n_missing += n_missing
        marker_ids.extend(kept_marker_ids)
        chrom_values.extend(kept_chrom_values)
        pos_values.extend(kept_pos_values)
        ref_values.extend(kept_ref_values)
        alt_values.extend(kept_alt_values)
        if batch_has_extra:
            state.sanity_checked = True
        return True

    def decode_general(lines):
        nonlocal writer, used_general, needs_imputation
        used_general = True
        needs_imputation = True
        sink = VariantAccumulator(filters, _DynamicInt8MatrixWriter)
        sink.n_samples = n_samples
        sink.writer = writer
        sink.map_columns = dict(MARKER=marker_ids, SNP=marker_ids, CHROM=chrom_values,
                                POS=pos_values, REF=ref_values, ALT=alt_values)
        try:
            _decode_builtin_records(
                vcf_path, sink, split_multiallelic, lines=lines,
                individual_ids=individual_ids, state=state,
            )
        finally:
            writer = sink.writer

    def flush_batch():
        if not sample_blobs:
            return
        result = process_batch(sample_blobs, batch_marker_ids, batch_chrom_values,
                               batch_pos_values, batch_ref_values, batch_alt_values)
        if result is False:
            decode_general(prefix + blob + b'\n' for prefix, blob in zip(batch_prefixes, sample_blobs))
        else:
            append_result(result)
        clear_batch()

    try:
        with _open_binary(vcf_path) as fh:
            for raw_line in fh:
                if not raw_line or raw_line.startswith(b'##'):
                    continue
                if raw_line.startswith(b'#CHROM'):
                    individual_ids = _parse_samples(raw_line.decode())
                    n_samples = len(individual_ids)
                    if not n_samples:
                        raise ValueError('VCF contains no sample columns')
                    expected_len = n_samples * 4 - 1
                    continue
                if individual_ids is None:
                    raise ValueError('VCF header not found before data lines')

                line = raw_line.rstrip(b'\r\n')
                parts = line.split(b'\t', 9)
                eligible = len(parts) == 10
                gt_index = 0
                if eligible:
                    chrom_b, pos_b, vid_b, ref_b, alt, _, _, _, fmt, blob = parts
                    if fmt == b'GT':
                        eligible = len(blob) == expected_len
                    else:
                        fields = fmt.split(b':')
                        eligible = extract_gt_fields is not None and b'GT' in fields and b'DS' not in fields
                        if eligible:
                            gt_index = len(fields) - 1 - fields[::-1].index(b'GT')
                    eligible = eligible and bool(alt) and alt != b'.' and b',' not in alt
                if not eligible:
                    flush_batch()
                    decode_general([raw_line])
                    continue

                if fmt != b'GT':
                    batch_has_extra = True
                    used_general = True
                sample_blobs.append(blob)
                batch_gt_indices.append(gt_index)
                batch_prefixes.append(line[:len(line) - len(blob)])
                batch_bytes += len(blob)
                chrom, pos, ref, alt_str, vid = chrom_b.decode(), int(pos_b), ref_b.decode(), alt.decode(), vid_b.decode()
                batch_marker_ids.append(vid if vid and vid != '.' else "%s:%s:%s:%s" % (chrom, pos, ref, alt_str))
                batch_chrom_values.append(chrom)
                batch_pos_values.append(pos)
                batch_ref_values.append(ref)
                batch_alt_values.append(alt_str)
                if len(sample_blobs) >= _SIMPLE_BULK_BATCH_MARKERS or (batch_has_extra and batch_bytes >= 128 * 1024 * 1024):
                    flush_batch()
            flush_batch()

        if individual_ids is None:
            raise ValueError('No header line found; invalid VCF')
        drops = general_drops if used_general else bulk_drops
        keep_indices = None
        if drops:
            keep = np.ones(writer.count, dtype=np.bool_)
            for index, missing in drops:
                keep[index] = False
                total_n_missing -= missing
            keep_indices = np.flatnonzero(keep)
            marker_ids = [value for value, flag in zip(marker_ids, keep) if flag]
            chrom_values = [value for value, flag in zip(chrom_values, keep) if flag]
            pos_values = [value for value, flag in zip(pos_values, keep) if flag]
            ref_values = [value for value, flag in zip(ref_values, keep) if flag]
            alt_values = [value for value, flag in zip(alt_values, keep) if flag]
        if writer is None:
            geno = np.zeros((n_samples, 0), dtype=np.int8)
        else:
            geno = writer.finalize(
                cache_path=cache.path('geno.npy') if cache is not None else None,
                before_publish=cache.invalidate if cache is not None else None,
                keep_indices=keep_indices, impute=needs_imputation,
            )
            total_n_missing += writer.imputed_count
        map_rows = dict(MARKER=marker_ids, SNP=marker_ids, CHROM=chrom_values,
                        POS=pos_values, REF=ref_values, ALT=alt_values)
        return geno, individual_ids, map_rows, total_n_missing
    finally:
        if writer is not None:
            writer.discard()


def _parse_format_keys(fmt_str: str) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    cached = _FORMAT_CACHE.get(fmt_str)
    if cached is not None:
        return cached
    if not fmt_str:
        result: Tuple[Tuple[str, ...], Dict[str, int]] = (tuple(), {})
    else:
        keys = tuple(fmt_str.split(':'))
        key_to_idx = {k: i for i, k in enumerate(keys)}
        result = (keys, key_to_idx)
    _FORMAT_CACHE[fmt_str] = result
    return result


def _split_gt_tokens(gt):
    # Accept phased or unphased; return list of allele indices as strings
    if gt is None or gt == '.' or gt == './.' or gt == '.|.':
        return None
    cached = _GT_TOKEN_CACHE.get(gt, _GT_TOKEN_CACHE_SENTINEL)
    if cached is not _GT_TOKEN_CACHE_SENTINEL:
        return cached
    sep = '/' if '/' in gt else '|' if '|' in gt else None
    if sep is None:
        _GT_TOKEN_CACHE[gt] = None
        return None
    toks = tuple(gt.split(sep))
    if any(token == '' for token in toks):
        _GT_TOKEN_CACHE[gt] = None
        return None
    _GT_TOKEN_CACHE[gt] = toks
    return toks


def _code_dosage_biallelic(gt_tokens):
    # gt_tokens like ['0','1', ...]; returns (dosage, ploidy)
    if not gt_tokens:
        return MISSING, 0
    if isinstance(gt_tokens, tuple):
        key = gt_tokens
    else:
        key = tuple(gt_tokens)
    cached = _BIALLELIC_DOSAGE_CACHE.get(key)
    if cached is not None:
        return cached
    alt_count = 0
    ploidy = 0
    for token in gt_tokens:
        if token == '.':
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        try:
            allele = int(token)
        except ValueError:
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        if allele not in (0, 1):
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        ploidy += 1
        if allele == 1:
            alt_count += 1
    result = (alt_count, ploidy)
    _BIALLELIC_DOSAGE_CACHE[key] = result
    return result


def _code_dosage_split(gt_tokens, alt_index):
    # alt_index is the 1-based index of ALT for the split
    if not gt_tokens:
        return MISSING, 0
    allowed = {0, alt_index}
    alt_count = 0
    ploidy = 0
    for token in gt_tokens:
        if token == '.':
            return MISSING, 0
        try:
            allele = int(token)
        except ValueError:
            return MISSING, 0
        if allele not in allowed:
            return MISSING, 0
        ploidy += 1
        if allele == alt_index:
            alt_count += 1
    return alt_count, ploidy


def _decode_biallelic_gt(gt: Optional[str]) -> Tuple[int, int]:
    if gt is None:
        return MISSING, 0
    direct = _BIALLELIC_GT_DIRECT.get(gt)
    if direct is not None:
        return direct
    cached = _BIALLELIC_GT_CACHE.get(gt)
    if cached is not None:
        return cached
    alt_count = 0
    ploidy = 0
    result: Tuple[int, int]
    reading_digit = False
    allele_value = 0
    for ch in gt:
        if ch == '.':
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
        if ch in '/|':
            if reading_digit:
                if allele_value == 1:
                    alt_count += 1
                elif allele_value not in (0,):
                    result = (MISSING, 0)
                    _BIALLELIC_GT_CACHE[gt] = result
                    return result
                ploidy += 1
                reading_digit = False
                allele_value = 0
            continue
        if '0' <= ch <= '9':
            reading_digit = True
            allele_value = allele_value * 10 + (ord(ch) - 48)
        else:
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
    if reading_digit:
        if allele_value == 1:
            alt_count += 1
        elif allele_value not in (0,):
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
        ploidy += 1
    if ploidy == 0:
        result = (MISSING, 0)
    else:
        result = (alt_count, ploidy)
    _BIALLELIC_GT_CACHE[gt] = result
    return result


def _ds_to_int(ds_val):
    try:
        x = float(ds_val)
    except Exception:
        return MISSING
    if np.isnan(x):
        return MISSING
    xi = int(round(x))
    if xi < 0:
        return MISSING
    return xi


def _select_vcf_reader(vcf_path, backend, threads):
    """Resolve backend/thread policy and open an htslib reader when needed."""
    # Standard loading proceeds...
    vcf_lower = str(vcf_path).lower()
    is_bcf = vcf_lower.endswith('.bcf')

    if backend == 'auto':
        if is_bcf:
            try:
                import cyvcf2  # type: ignore
                use_cyvcf2 = True
            except ImportError:
                raise ImportError(
                    'Loading .bcf requires cyvcf2. Install with "pip install cyvcf2" or convert to .vcf/.vcf.gz.'
                )
        else:
            use_cyvcf2 = False
    elif backend == 'cyvcf2':
        try:
            import cyvcf2  # type: ignore
            use_cyvcf2 = True
        except Exception:
            raise ImportError('cyvcf2 requested but not available')
    elif backend == 'builtin':
        use_cyvcf2 = False
    else:
        raise ValueError("backend must be one of {'auto', 'cyvcf2', 'builtin'}")

    # Determine thread count for cyvcf2/htslib. This mainly helps compressed
    # input decompression; variant decoding below still happens in this process.
    from ..utils.perf import available_cpu_count
    cpu_count = available_cpu_count()
    if threads is None:
        n_threads = min(4, cpu_count)
    else:
        try:
            requested_threads = int(threads)
        except (TypeError, ValueError):
            raise ValueError("threads must be an integer or None")
        if requested_threads < 0:
            raise ValueError("threads must be >= 0")
        n_threads = cpu_count if requested_threads == 0 else max(1, requested_threads)
    
    # Initialize VCF reader based on backend and file type
    vcf = None
    if is_bcf:
        # BCF requires cyvcf2
        try:
            from cyvcf2 import VCF
            vcf = VCF(vcf_path, threads=n_threads)
        except ImportError:
            raise ImportError('cyvcf2 is required for BCF files')
    elif use_cyvcf2:
        # Optimized VCF path
        from cyvcf2 import VCF
        vcf = VCF(vcf_path, threads=n_threads)

    return vcf, use_cyvcf2, is_bcf


def _decode_cyvcf2_records(vcf, sink, split_multiallelic):
    """Decode htslib records; the sink owns QC, map accumulation, and storage.

    Retain this backend's existing GT-only dosage/ploidy interpretation. Do not
    replace it with the builtin GT/DS policy as part of a performance refactor.
    """
    individual_ids = list(vcf.samples)
    if not individual_ids:
        raise ValueError('VCF contains no sample columns')
    sink.n_samples = len(individual_ids)
    consider_variant = sink.consider
    for var in vcf:
        chrom = var.CHROM
        pos = int(var.POS)
        vid = var.ID if var.ID else '.'
        ref = var.REF
        alts = var.ALT or []
        if not alts or (len(alts) > 1 and not split_multiallelic):
            continue
        try:
            # Final column is the phase flag, not an allele.
            gt_arr = np.array(var.genotype.array())
        except Exception:
            continue  # Preserve legacy behavior for undecodable records.
        alleles = gt_arr[:, :-1]
        if len(alts) == 1:
            missing_mask = np.any(alleles < 0, axis=1)
            dosages = np.sum(alleles, axis=1)
            col = dosages.astype(np.int16)
            col[missing_mask] = MISSING
            col[dosages > 2] = MISSING
            consider_variant(col, chrom, pos, vid, ref, alts[0], 2)
        else:
            for ai, alt_base in enumerate(alts, start=1):
                # Other ALTs invalidate this pseudo-biallelic call; -2 is
                # padding and -1 is missing in cyvcf2's variable-ploidy array.
                invalid_alleles = (alleles != 0) & (alleles != ai) & (alleles >= 0)
                row_invalid_mask = np.any(invalid_alleles, axis=1)
                counts = np.sum(alleles == ai, axis=1)
                sample_ploidy = np.sum(alleles != -2, axis=1)
                var_ploidy = int(np.max(sample_ploidy)) if len(sample_ploidy) > 0 else 2
                col = counts.astype(np.int16)
                has_missing = np.any(alleles == -1, axis=1)
                col[row_invalid_mask | has_missing] = MISSING
                consider_variant(col, chrom, pos, vid, ref, alt_base, var_ploidy)
    return individual_ids


def _decode_builtin_records(vcf_path, sink, split_multiallelic, *, lines=None,
                            individual_ids=None, state=None):
    """Decode general text records; keep per-line NumPy fast parsing available."""
    state = BuiltinDecodeState() if state is None else state
    consider_variant = sink.consider
    with (_open_binary(vcf_path) if lines is None else nullcontext(lines)) as fh:
        for raw_line in fh:
            if not raw_line:
                continue
            if raw_line.startswith(b'##'):
                continue
            if raw_line.startswith(b'#CHROM'):
                individual_ids = _parse_samples(raw_line.decode())
                n = len(individual_ids)
                sink.n_samples = n
                # Edge: no samples
                if n == 0:
                    raise ValueError('VCF contains no sample columns')
                continue
            if individual_ids is None:
                raise ValueError('VCF header not found before data lines')

            fast_record = _parse_simple_biallelic_gt_line(raw_line, len(individual_ids))
            if fast_record is not None:
                col, chrom, pos, vid, ref, alt_base = fast_record
                consider_variant(col, chrom, pos, vid, ref, alt_base, 2, simple_diploid=True)
                continue

            line = raw_line.decode()
            if not line:
                continue
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                individual_ids = _parse_samples(line)
                n = len(individual_ids)
                sink.n_samples = n
                # Edge: no samples
                if n == 0:
                    raise ValueError('VCF contains no sample columns')
                continue
            # Data line
            if individual_ids is None:
                raise ValueError('VCF header not found before data lines')
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 8:
                continue  # malformed
            chrom, pos_str, vid, ref, alt_str = parts[0], parts[1], parts[2], parts[3], parts[4]
            pos = int(pos_str)

            # Determine ALT alleles
            alt_alleles = alt_str.split(',') if alt_str and alt_str != '.' else []
            if not alt_alleles:
                continue  # no ALT

            fmt = parts[8] if len(parts) >= 9 else ''
            sample_fields = parts[9:] if len(parts) >= 10 else []
            fmt_keys, key_to_idx = _parse_format_keys(fmt)

            gt_index = key_to_idx.get('GT')
            ds_index = key_to_idx.get('DS')
            gt_primary = gt_index == 0

            ds_array: Optional[np.ndarray]
            if ds_index is None:
                ds_array = None
            else:
                ds_array = np.full(len(individual_ids), MISSING, dtype=np.int16)
                if ds_index == 0:
                    for si, field in enumerate(sample_fields):
                        token = field.partition(':')[0]
                        if token:
                            ds_array[si] = _ds_to_int(token)
                elif ds_index == 1 and gt_primary:
                    for si, field in enumerate(sample_fields):
                        head, sep, tail = field.partition(':')
                        if sep:
                            token, _, _ = tail.partition(':')
                            if token:
                                ds_array[si] = _ds_to_int(token)
                else:
                    for si, field in enumerate(sample_fields):
                        toks = field.split(':')
                        if ds_index < len(toks):
                            token = toks[ds_index]
                            if token:
                                ds_array[si] = _ds_to_int(token)

            is_biallelic = len(alt_alleles) == 1

            if gt_index is not None:
                if gt_primary:
                    gt_values = [
                        field.partition(':')[0] if field else '' for field in sample_fields
                    ]
                else:
                    gt_values = []
                    for field in sample_fields:
                        if not field:
                            gt_values.append('')
                            continue
                        toks = field.split(':')
                        gt_values.append(toks[gt_index] if gt_index < len(toks) else '')
            else:
                gt_values = [''] * len(sample_fields)
            gt_array = np.array(gt_values, dtype='<U8') if gt_values else np.empty(len(sample_fields), dtype='<U8')

            split_tokens = _split_gt_tokens

            # Helper: build column(s) for this site
            def build_columns_for_alt(alt_index, alt_base):
                col = np.full(len(individual_ids), MISSING, dtype=np.int16)
                missing_mask = np.ones(len(individual_ids), dtype=bool)
                variant_ploidy = 0
                if is_biallelic and gt_index is not None:
                    if not state.sanity_checked:
                        if len(alt_alleles) != 1:
                            raise ValueError(
                                "Multi-allelic variants are not supported by the fast builtin loader. "
                                "Please switch to the cyvcf2 backend."
                            )
                        subset = gt_array[: min(10, gt_array.size)]
                        if subset.size:
                            subset = subset[(subset != '') & (np.char.find(subset, '.') == -1)]
                            if subset.size:
                                cleaned_subset = np.char.replace(np.char.replace(subset, '/', ''), '|', '')
                                if np.any(np.char.find(cleaned_subset, '2') != -1) or np.any(np.char.find(cleaned_subset, '3') != -1):
                                    raise ValueError(
                                        "Detected genotype allele codes greater than 1. "
                                        "Polyploid genotypes require the cyvcf2 backend."
                                    )
                                lengths = np.char.str_len(cleaned_subset)
                                if np.any(lengths > 2):
                                    raise ValueError(
                                        "Detected genotypes with ploidy greater than diploid. "
                                        "Please use the cyvcf2 backend for polyploid datasets."
                                    )
                        state.sanity_checked = True

                    unique_gts = np.unique(gt_array)
                    for gt_code in unique_gts:
                        mask = gt_array == gt_code
                        if not gt_code:
                            if ds_array is not None:
                                ds_mask = mask & (ds_array != MISSING)
                                if np.any(ds_mask):
                                    col[ds_mask] = ds_array[ds_mask]
                                    missing_mask[ds_mask] = False
                            continue
                        dosage, ploidy = _decode_biallelic_gt(gt_code)
                        if dosage != MISSING:
                            col[mask] = dosage
                            missing_mask[mask] = False
                            variant_ploidy = max(variant_ploidy, ploidy)
                        elif ds_array is not None:
                            ds_mask = mask & (ds_array != MISSING)
                            if np.any(ds_mask):
                                col[ds_mask] = ds_array[ds_mask]
                                missing_mask[ds_mask] = False
                else:
                    for si, gt in enumerate(gt_values):
                        ds_val = ds_array[si] if ds_array is not None else MISSING
                        gt_tokens = split_tokens(gt) if gt else None
                        if gt_tokens is not None:
                            dosage, ploidy = _code_dosage_split(gt_tokens, alt_index)
                            if dosage != MISSING:
                                col[si] = dosage
                                variant_ploidy = max(variant_ploidy, ploidy)
                                missing_mask[si] = False
                            elif ds_val != MISSING:
                                col[si] = ds_val
                                missing_mask[si] = False
                        elif ds_val != MISSING:
                            col[si] = ds_val
                            missing_mask[si] = False
                if ds_array is not None:
                    ds_mask = missing_mask & (ds_array != MISSING)
                    if np.any(ds_mask):
                        col[ds_mask] = ds_array[ds_mask]
                        missing_mask[ds_mask] = False
                return col, variant_ploidy

            if len(alt_alleles) == 1:
                col, ploidy = build_columns_for_alt(1, alt_alleles[0])
                if ploidy == 0:
                    ploidy = 2
                consider_variant(col, chrom, pos, vid, ref, alt_alleles[0], ploidy)
            else:
                if not split_multiallelic:
                    # Skip multi-allelic sites entirely in non-split mode
                    continue
                for ai, alt_base in enumerate(alt_alleles, start=1):
                    col, ploidy = build_columns_for_alt(ai, alt_base)
                    if ploidy == 0:
                        ploidy = 2
                    consider_variant(col, chrom, pos, vid, ref, alt_base, ploidy)

    return individual_ids


def load_genotype_vcf(
    vcf_path,
    split_multiallelic=True,
    include_indels=True,
    drop_monomorphic=False,
    max_missing=1.0,
    min_maf=0.0,
    return_pandas=True,
    backend='auto',  # 'auto', 'cyvcf2', 'builtin'
    threads=None,
    force_recache=False,
):
    """
    Load a VCF file and return (geno_matrix, individual_ids, geno_map).

    Parameters
    - vcf_path: path to .vcf or .vcf.gz
    - split_multiallelic: if True, split multi-ALT variants into separate entries
    - include_indels: include biallelic indels (if False, only include SNPs)
    - drop_monomorphic: drop variants with all non-missing 0 or all 2
    - max_missing: drop variants with missing rate > threshold (0..1]
    - min_maf: drop variants with minor allele frequency < threshold
      (missing calls treated as major allele for filtering)
    - force_recache: if True, ignore any existing cache and overwrite it
    - return_pandas: return geno_map as pandas.DataFrame if pandas is available
    - backend: 'auto' (uses the builtin parser for VCF text and cyvcf2 for BCF),
      'cyvcf2', or 'builtin'
    - threads: cyvcf2/htslib worker threads. None uses min(4, cpu_count);
      0 uses all detected CPUs. Ignored by the builtin text parser.

    Fresh matrices are C-contiguous sample-major int8 arrays. When a matrix is
    large enough for direct cache finalization, it is a writable copy-on-write
    np.memmap instead of a heap ndarray; caller edits never update the cache.
    Existing cache hits retain their read-only memmap behavior.
    """

    # Backend selection: auto uses the builtin text parser for VCF so the
    # simple-GT bulk path gets first shot. BCF is binary and requires cyvcf2.
    # --- CACHING LOGIC START ---
    # Cache version 2: pre-imputes missing values (-9) at cache time for faster downstream.
    # Filter fingerprint sidecar (*.panicle.v2.filters.json) invalidates the cache
    # when QC parameters that change the marker set differ from the build config.
    cache_base = str(vcf_path)
    cache_filters = {
        'cache_version': 2,
        'drop_monomorphic': bool(drop_monomorphic),
        'include_indels': bool(include_indels),
        'max_missing': float(max_missing),
        'min_maf': float(min_maf),
        'split_multiallelic': bool(split_multiallelic),
    }
    cache = GenotypeCache(cache_base, (vcf_path,), cache_filters)
    cached = cache.load(force=force_recache, logger=logger)
    if cached is not None:
        return cached

    filters = VCFFilters(include_indels, drop_monomorphic, max_missing, min_maf)
    vcf, use_cyvcf2, is_bcf = _select_vcf_reader(vcf_path, backend, threads)
    sink = None
    direct_n_missing = None
    try:
        if not use_cyvcf2 and is_bcf:
            raise ImportError('Builtin VCF parser does not support .bcf. Please install cyvcf2 or use .vcf/.vcf.gz.')
        bulk_result = None
        if not use_cyvcf2:
            bulk_result = _try_load_simple_biallelic_gt_vcf_bulk(
                vcf_path, include_indels=include_indels, drop_monomorphic=drop_monomorphic,
                max_missing=max_missing, min_maf=min_maf,
                split_multiallelic=split_multiallelic, cache=cache,
            )
        if bulk_result is not None:
            geno, individual_ids, map_rows, direct_n_missing = bulk_result
        else:
            sink = VariantAccumulator(filters, _DynamicInt8MatrixWriter)
            if use_cyvcf2:
                individual_ids = _decode_cyvcf2_records(vcf, sink, split_multiallelic)
            else:
                individual_ids = _decode_builtin_records(vcf_path, sink, split_multiallelic)
            geno = sink.finalize(cache_path=cache.path('geno.npy'),
                                 before_publish=cache.invalidate, impute=True)
            direct_n_missing = sink.writer.imputed_count if sink.writer is not None else 0
            map_rows = sink.map_columns
    finally:
        if sink is not None:
            sink.close()
        if vcf is not None and hasattr(vcf, 'close'):
            vcf.close()

    # Build geno_map output
    if return_pandas:
        try:
            import pandas as pd  # type: ignore
            geno_map = pd.DataFrame(
                map_rows,
                columns=[
                    MARKER_ID_COLUMN,
                    LEGACY_MARKER_ID_COLUMN,
                    CHROM_COLUMN,
                    POS_COLUMN,
                    'REF',
                    'ALT',
                ],
            )
            geno_map = canonicalize_genotype_map_dataframe(geno_map)
        except Exception:
            geno_map = map_rows
    else:
        if isinstance(map_rows, dict):
            row_count = len(next(iter(map_rows.values()), []))
            geno_map = [
                {column: values[row_idx] for column, values in map_rows.items()}
                for row_idx in range(row_count)
            ]
        else:
            geno_map = map_rows

    # Integrity checks
    if individual_ids is None:
        raise ValueError('No header line found; invalid VCF')
    if geno.shape[0] != len(individual_ids):
        raise AssertionError('Row count mismatch: %d vs %d' % (geno.shape[0], len(individual_ids)))
    n_markers = geno.shape[1]
    if isinstance(geno_map, list):
        if len(geno_map) != n_markers:
            raise AssertionError('Map length mismatch: %d vs %d' % (len(geno_map), n_markers))
    else:
        if int(getattr(geno_map, 'shape', (0, 0))[0]) != n_markers:
            raise AssertionError('Map length mismatch: %d vs %d' % (int(geno_map.shape[0]), n_markers))

    # --- CACHING LOGIC SAVE START ---
    try:
        # Impute missing values (-9) before caching
        # This avoids repeated -9 checks in downstream kinship/MLM code
        if direct_n_missing is None:
            n_missing = impute_major_allele_inplace(geno, missing_value=MISSING)
        else:
            n_missing = direct_n_missing
        if n_missing > 0:
            logger.info("[Cache] Imputed %s missing values (%.2f%%)", f"{n_missing:,}", 100*n_missing/geno.size)
        if hasattr(geno_map, "attrs"):
            geno_map.attrs["is_imputed"] = True

        # Save only if successful
        genotype_written = (
            isinstance(geno, np.memmap)
            and os.path.abspath(str(geno.filename)) == os.path.abspath(cache.path('geno.npy'))
        )
        cache.save(geno, individual_ids, geno_map, logger=logger, genotype_written=genotype_written)

    except Exception as e:
        logger.warning("[Cache] Failed to save cache: %s", e)
    # --- CACHING LOGIC SAVE END ---

    return geno, individual_ids, geno_map


def _main(argv):  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description='Load VCF into genotype matrix for GWAS')
    p.add_argument('vcf')
    p.add_argument('--no-split', action='store_true', help='Do not split multi-allelic sites')
    p.add_argument('--snps-only', action='store_true', help='Restrict to SNPs only')
    p.add_argument('--drop-monomorphic', action='store_true')
    p.add_argument('--max-missing', type=float, default=1.0)
    p.add_argument('--min-maf', type=float, default=0.0)
    p.add_argument('--force-recache', action='store_true', help='Rebuild and overwrite cache files')
    p.add_argument('--no-pandas', action='store_true', help='Return map as list instead of DataFrame')
    p.add_argument('--backend', choices=['auto','cyvcf2','builtin'], default='auto',
                   help='Choose parsing backend (auto uses builtin for VCF text; cyvcf2 required for BCF)')
    p.add_argument('--threads', type=int, default=None,
                   help='cyvcf2/htslib worker threads; 0 uses all detected CPUs')
    args = p.parse_args(argv)

    geno, ids, gmap = load_genotype_vcf(
        args.vcf,
        split_multiallelic=not args.no_split,
        include_indels=not args.snps_only,
        drop_monomorphic=args.drop_monomorphic,
        max_missing=args.max_missing,
        min_maf=args.min_maf,
        return_pandas=not args.no_pandas,
        backend=args.backend,
        threads=args.threads,
        force_recache=args.force_recache,
    )
    print('Samples:', len(ids))
    print('Markers:', geno.shape[1])
    print('Genotype dtype:', geno.dtype)
    # Spot-check MAF for first few markers
    if geno.shape[1] > 0:
        col = geno[:, 0]
        mask = col != MISSING
        if mask.any():
            n_total = col.size
            n_valid = int(np.count_nonzero(mask))
            total_alleles = 2 * n_total
            valid_alleles = 2 * n_valid
            sum_dos = float(np.sum(col[mask]))
            minor_count = min(sum_dos, valid_alleles - sum_dos)
            maf = minor_count / max(total_alleles, 1.0)
            print('First marker MAF ~', round(maf, 4))
    # Print first few map rows
    if hasattr(gmap, 'head'):
        print(gmap.head())
    else:
        print(gmap[:3])


if __name__ == '__main__':  # pragma: no cover
    _main(sys.argv[1:])
