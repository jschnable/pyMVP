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

This module avoids external VCF libraries for portability. For very large VCFs,
consider replacing the parser with cyvcf2/pysam keeping the same coding logic.
"""
from __future__ import print_function
import sys
import gzip
import io
import os
import tempfile

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise ImportError("NumPy is required: pip install numpy")


MISSING = -9


class _DynamicInt8MatrixWriter:
    """Append-only int8 matrix builder backed by a temporary memmap."""

    def __init__(self, n_rows, initial_capacity=4096):
        self.n_rows = int(n_rows)
        if self.n_rows <= 0:
            raise ValueError("Writer requires a positive number of rows")
        self.capacity = max(int(initial_capacity), 1)
        tmp = tempfile.NamedTemporaryFile(prefix="pymvp_geno_", suffix=".tmp", delete=False)
        self.path = tmp.name
        tmp.close()
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='w+', shape=(self.n_rows, self.capacity))
        self.count = 0

    def _grow(self, min_capacity):
        new_capacity = self.capacity
        while new_capacity < min_capacity:
            new_capacity = max(new_capacity * 2, min_capacity)
        old_columns = self.count
        if old_columns > 0:
            preserved = np.empty((self.n_rows, old_columns), dtype=np.int8)
            np.copyto(preserved, self.memmap[:, :old_columns])
        else:
            preserved = None
        self.memmap.flush()
        del self.memmap
        with open(self.path, 'r+b') as fh:
            fh.truncate(self.n_rows * new_capacity)
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='r+', shape=(self.n_rows, new_capacity))
        if preserved is not None:
            self.memmap[:, :old_columns] = preserved
        self.capacity = new_capacity

    def append(self, column):
        if column.shape != (self.n_rows,):
            raise ValueError(f"Column shape mismatch: expected ({self.n_rows},), got {column.shape}")
        if self.count >= self.capacity:
            self._grow(self.count + 1)
        self.memmap[:, self.count] = column
        self.count += 1

    def finalize(self):
        mm = self.memmap
        if mm is None:
            return np.zeros((self.n_rows, 0), dtype=np.int8)
        total_cols = self.count
        mm.flush()
        if total_cols == 0:
            result = np.zeros((self.n_rows, 0), dtype=np.int8)
        else:
            result = np.array(mm[:, :total_cols], dtype=np.int8, copy=True, order='C')
        try:
            os.remove(self.path)
        except OSError:
            pass
        self.memmap = None
        del mm
        return result


def _open_text(path):
    """Open VCF text transparently from plain or gzip-compressed files.

    Accepts string or Path-like, and handles .vcf, .vcf.gz, and .vcf.bgz.
    """
    p = str(path)
    pl = p.lower()
    if pl.endswith('.gz') or pl.endswith('.bgz'):
        return io.TextIOWrapper(gzip.open(p, 'rb'))
    return open(p, 'r')


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


def _parse_format_keys(fmt_str):
    return fmt_str.split(':') if fmt_str else []


def _split_gt_tokens(gt):
    # Accept phased or unphased; return list of allele indices as strings
    if gt is None or gt == '.' or gt == './.' or gt == '.|.':
        return None
    sep = '/' if '/' in gt else '|' if '|' in gt else None
    if sep is None:
        # Non-standard single allele or missing
        return None
    toks = gt.split(sep)
    return toks


def _code_dosage_biallelic(gt_tokens):
    # gt_tokens like ['0','1', ...]; returns (dosage, ploidy)
    if not gt_tokens:
        return MISSING, 0
    alt_count = 0
    ploidy = 0
    for token in gt_tokens:
        if token == '.':
            return MISSING, 0
        try:
            allele = int(token)
        except ValueError:
            return MISSING, 0
        if allele not in (0, 1):
            return MISSING, 0
        ploidy += 1
        if allele == 1:
            alt_count += 1
    return alt_count, ploidy


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


def load_genotype_vcf(
    vcf_path,
    split_multiallelic=True,
    include_indels=True,
    drop_monomorphic=False,
    max_missing=1.0,
    min_maf=0.0,
    return_pandas=True,
    backend='auto',  # 'auto', 'cyvcf2', 'builtin'
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
    - return_pandas: return geno_map as pandas.DataFrame if pandas is available
    """
    import re

    # Optional fast backend
    use_cyvcf2 = False
    vcf_lower = str(vcf_path).lower()
    is_bcf = vcf_lower.endswith('.bcf')
    if backend in ('auto', 'cyvcf2'):
        try:
            import cyvcf2  # type: ignore
            use_cyvcf2 = True
        except Exception:
            use_cyvcf2 = False
            if backend == 'cyvcf2':
                raise ImportError('cyvcf2 requested but not available')
    # BCF requires cyvcf2 backend
    if is_bcf and not use_cyvcf2:
        raise ImportError('Loading .bcf requires cyvcf2. Install with "pip install cyvcf2" or convert to .vcf/.vcf.gz.')

    # Initialize
    individual_ids = None
    writer = None  # lazy initialised streaming writer
    map_rows = []  # dict rows

    # Helper to finalize a candidate variant column with QC
    def consider_variant(col, chrom, pos, vid, ref, alt, ploidy):
        nonlocal writer, map_rows
        col = np.asarray(col, dtype=np.int16)  # temp safe range
        # If requested, restrict to SNPs
        if not include_indels:
            if len(ref) != 1 or len(alt) != 1:
                return
        # Skip if all missing
        valid = col != MISSING
        if not np.any(valid):
            return
        # Optional monomorphic filter
        if drop_monomorphic:
            vals = np.unique(col[valid])
            if vals.size == 1 and (vals[0] == 0 or vals[0] == 2):
                return
        # Missingness filter
        miss_rate = 1.0 - (np.count_nonzero(valid) / float(col.size))
        if miss_rate > max_missing:
            return
        # MAF filter
        if min_maf > 0.0:
            mean_dos = float(np.mean(col[valid]))
            denom = max(ploidy, 1)
            maf = min(mean_dos / denom, 1.0 - (mean_dos / denom))
            if maf < min_maf:
                return
        # Finalize dtype and append to streaming writer
        col = col.astype(np.int8, copy=False)
        if writer is None:
            writer = _DynamicInt8MatrixWriter(len(individual_ids))
        writer.append(col)
        map_rows.append({
            'SNP': _build_snp_id(chrom, pos, vid, ref, alt),
            'CHROM': str(chrom),
            'POS': int(pos),
            'REF': ref,
            'ALT': alt,
        })

    if use_cyvcf2:
        # Fast path using cyvcf2
        from cyvcf2 import VCF  # type: ignore
        vcf = VCF(vcf_path)
        individual_ids = list(vcf.samples)
        n = len(individual_ids)
        if n == 0:
            raise ValueError('VCF contains no sample columns')

        def consider_variant(col, chrom, pos, vid, ref, alt, ploidy):
            nonlocal writer, map_rows
            col = np.asarray(col, dtype=np.int16)
            if not include_indels and (len(ref) != 1 or len(alt) != 1):
                return
            valid = col != MISSING
            if not np.any(valid):
                return
            if drop_monomorphic:
                vals = np.unique(col[valid])
                if vals.size == 1 and (vals[0] == 0 or vals[0] == 2):
                    return
            miss_rate = 1.0 - (np.count_nonzero(valid) / float(col.size))
            if miss_rate > max_missing:
                return
            if min_maf > 0.0:
                mean_dos = float(np.mean(col[valid]))
                denom = max(ploidy, 1)
                maf = min(mean_dos / denom, 1.0 - (mean_dos / denom))
                if maf < min_maf:
                    return
            col = col.astype(np.int8, copy=False)
            if writer is None:
                writer = _DynamicInt8MatrixWriter(len(individual_ids))
            writer.append(col)
            map_rows.append({
                'SNP': _build_snp_id(chrom, pos, vid, ref, alt),
                'CHROM': str(chrom),
                'POS': int(pos),
                'REF': ref,
                'ALT': alt,
            })

        # Pre-fetch DS if needed per site; cyvcf2 provides per-variant format arrays
        for var in vcf:
            chrom = var.CHROM
            pos = int(var.POS)
            vid = var.ID if var.ID else '.'
            ref = var.REF
            alts = var.ALT or []
            if not alts:
                continue
            genos = var.genotypes  # list of [a, b, phased]
            n_samples = len(genos)
            # Ensure sample count alignment
            if n_samples != len(individual_ids):
                raise AssertionError('Sample count mismatch within VCF record')

            # Optional DS fallback vector
            try:
                ds = var.format('DS')  # shape (n_samples,)
            except Exception:
                ds = None

            if len(alts) == 1:
                col = np.full(n, MISSING, dtype=np.int16)
                variant_ploidy = 0
                for i, g in enumerate(genos):
                    if g is None:
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    alleles = g[:-1] if len(g) > 2 else g[:2]
                    if not alleles:
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    if any(a == -1 for a in alleles):
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    if any(a not in (0, 1) for a in alleles):
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    alt_count = sum(1 for a in alleles if a == 1)
                    col[i] = alt_count
                    variant_ploidy = max(variant_ploidy, len(alleles))
                if variant_ploidy == 0:
                    variant_ploidy = 2
                consider_variant(col, chrom, pos, vid, ref, alts[0], variant_ploidy)
            else:
                if not split_multiallelic:
                    continue
                # Build a column per ALT, recoding non-focal ALTs as missing
                for ai, alt_base in enumerate(alts, start=1):
                    col = np.full(n, MISSING, dtype=np.int16)
                    variant_ploidy = 0
                    for i, g in enumerate(genos):
                        if g is None:
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        alleles = g[:-1] if len(g) > 2 else g[:2]
                        if not alleles:
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        if any(a == -1 for a in alleles):
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        allowed = {0, ai}
                        if any(a not in allowed for a in alleles):
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        alt_count = sum(1 for a in alleles if a == ai)
                        col[i] = alt_count
                        variant_ploidy = max(variant_ploidy, len(alleles))
                    if variant_ploidy == 0:
                        variant_ploidy = 2
                    consider_variant(col, chrom, pos, vid, ref, alt_base, variant_ploidy)
    else:
        # Built-in text parser
        # Guard: .bcf is binary and not supported by builtin parser
        if is_bcf:
            raise ImportError('Builtin VCF parser does not support .bcf. Please install cyvcf2 or use .vcf/.vcf.gz.')
        fh = _open_text(vcf_path)
        try:
            for line in fh:
                if not line:
                    continue
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    individual_ids = _parse_samples(line)
                    n = len(individual_ids)
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
                fmt_keys = _parse_format_keys(fmt)
                key_to_idx = {k: i for i, k in enumerate(fmt_keys)}

                # Helper: build column(s) for this site
                def build_columns_for_alt(alt_index, alt_base):
                    col = np.full(len(individual_ids), MISSING, dtype=np.int16)
                    variant_ploidy = 0
                    # For each sample, parse its field
                    for si, field in enumerate(sample_fields):
                        if not field:
                            continue
                        toks = field.split(':')
                        gt = None
                        if 'GT' in key_to_idx and key_to_idx['GT'] < len(toks):
                            gt = toks[key_to_idx['GT']]
                        gt_tokens = _split_gt_tokens(gt) if gt is not None else None
                        if gt_tokens is not None:
                            if len(alt_alleles) > 1:
                                dosage, ploidy = _code_dosage_split(gt_tokens, alt_index)
                            else:
                                dosage, ploidy = _code_dosage_biallelic(gt_tokens)
                            if dosage != MISSING:
                                col[si] = dosage
                                variant_ploidy = max(variant_ploidy, ploidy)
                            elif 'DS' in key_to_idx and key_to_idx['DS'] < len(toks):
                                col[si] = _ds_to_int(toks[key_to_idx['DS']])
                        else:
                            if 'DS' in key_to_idx and key_to_idx['DS'] < len(toks):
                                col[si] = _ds_to_int(toks[key_to_idx['DS']])
                            else:
                                # Keep missing
                                pass
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
        finally:
            try:
                fh.close()
            except Exception:
                pass

    if writer is None:
        # No variants passed filters
        geno = np.zeros((len(individual_ids or []), 0), dtype=np.int8)
    else:
        geno = writer.finalize()

    # Build geno_map output
    if return_pandas:
        try:
            import pandas as pd  # type: ignore
            geno_map = pd.DataFrame(map_rows, columns=['SNP', 'CHROM', 'POS', 'REF', 'ALT'])
        except Exception:
            geno_map = map_rows
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
    p.add_argument('--no-pandas', action='store_true', help='Return map as list instead of DataFrame')
    p.add_argument('--backend', choices=['auto','cyvcf2','builtin'], default='auto', help='Choose parsing backend')
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
    )
    print('Samples:', len(ids))
    print('Markers:', geno.shape[1])
    print('Genotype dtype:', geno.dtype)
    # Spot-check MAF for first few markers
    if geno.shape[1] > 0:
        col = geno[:, 0]
        mask = col != MISSING
        if mask.any():
            maf = min(float(col[mask].mean()) / 2.0, 1.0 - float(col[mask].mean()) / 2.0)
            print('First marker MAF ~', round(maf, 4))
    # Print first few map rows
    if hasattr(gmap, 'head'):
        print(gmap.head())
    else:
        print(gmap[:3])


if __name__ == '__main__':  # pragma: no cover
    _main(sys.argv[1:])
