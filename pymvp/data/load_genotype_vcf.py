#!/usr/bin/env python
"""
VCF loader for GWAS: builds (geno_matrix, individual_ids, geno_map).

Key features:
- Streaming parsing of VCF text (supports .vcf and .vcf.gz)
- Diploid-only, biallelic by default; optional multi-allelic splitting
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

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise ImportError("NumPy is required: pip install numpy")


MISSING = -9


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
    # gt_tokens like ['0','1'] etc.; returns 0/1/2 or MISSING
    if not gt_tokens or len(gt_tokens) != 2:
        return MISSING
    a, b = gt_tokens[0], gt_tokens[1]
    if a == '.' or b == '.':
        return MISSING
    # Non-diploid or non-integer
    try:
        ia, ib = int(a), int(b)
    except ValueError:
        return MISSING
    # Only permit alleles 0/1 for biallelic
    if ia not in (0, 1) or ib not in (0, 1):
        return MISSING
    return ia + ib


def _code_dosage_split(gt_tokens, alt_index):
    # alt_index is the 1-based index of ALT for the split
    if not gt_tokens or len(gt_tokens) != 2:
        return MISSING
    a, b = gt_tokens[0], gt_tokens[1]
    if a == '.' or b == '.':
        return MISSING
    try:
        ia, ib = int(a), int(b)
    except ValueError:
        return MISSING
    # Only 0 and alt_index allowed; any other ALT makes sample missing in this split
    allowed = (0, alt_index)
    if ia not in allowed or ib not in allowed:
        return MISSING
    # Count copies of alt_index
    return (1 if ia == alt_index else 0) + (1 if ib == alt_index else 0)


def _ds_to_int(ds_val):
    try:
        x = float(ds_val)
    except Exception:
        return MISSING
    # Round to nearest int and clip to [0,2]
    xi = int(round(x))
    if xi < 0:
        xi = 0
    elif xi > 2:
        xi = 2
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
    columns = []  # list of np.ndarray (int8) per marker
    map_rows = []  # dict rows

    # Helper to finalize a candidate variant column with QC
    def consider_variant(col, chrom, pos, vid, ref, alt):
        nonlocal columns, map_rows
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
            maf = min(mean_dos / 2.0, 1.0 - (mean_dos / 2.0))
            if maf < min_maf:
                return
        # Finalize dtype
        col = col.astype(np.int8, copy=False)
        columns.append(col)
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

        def consider_variant(col, chrom, pos, vid, ref, alt):
            nonlocal columns, map_rows
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
                maf = min(mean_dos / 2.0, 1.0 - (mean_dos / 2.0))
                if maf < min_maf:
                    return
            columns.append(col.astype(np.int8, copy=False))
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
                for i, g in enumerate(genos):
                    # cyvcf2 uses -1 for missing
                    if g is None or len(g) < 2:
                        # attempt DS fallback
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    a, b = g[0], g[1]
                    if a == -1 or b == -1:
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                        continue
                    # Only accept 0/1 alleles
                    if (a in (0, 1)) and (b in (0, 1)):
                        col[i] = a + b
                    else:
                        # multi-allelic allele present but site says 1 alt; treat as missing
                        if ds is not None:
                            col[i] = _ds_to_int(ds[i])
                consider_variant(col, chrom, pos, vid, ref, alts[0])
            else:
                if not split_multiallelic:
                    continue
                # Build a column per ALT, recoding non-focal ALTs as missing
                for ai, alt_base in enumerate(alts, start=1):
                    col = np.full(n, MISSING, dtype=np.int16)
                    for i, g in enumerate(genos):
                        if g is None or len(g) < 2:
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        a, b = g[0], g[1]
                        if a == -1 or b == -1:
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                            continue
                        allowed = (0, ai)
                        if a in allowed and b in allowed:
                            col[i] = (1 if a == ai else 0) + (1 if b == ai else 0)
                        else:
                            # non-focal ALT present -> missing (optionally DS)
                            if ds is not None:
                                col[i] = _ds_to_int(ds[i])
                    consider_variant(col, chrom, pos, vid, ref, alt_base)
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
                            # If multi-allelic split, only allow 0 and alt_index
                            if len(alt_alleles) > 1:
                                col[si] = _code_dosage_split(gt_tokens, alt_index)
                            else:
                                col[si] = _code_dosage_biallelic(gt_tokens)
                        else:
                            # Fallback to DS if available
                            if 'DS' in key_to_idx and key_to_idx['DS'] < len(toks):
                                col[si] = _ds_to_int(toks[key_to_idx['DS']])
                            else:
                                # Keep missing
                                pass
                    return col

                if len(alt_alleles) == 1:
                    col = build_columns_for_alt(1, alt_alleles[0])
                    consider_variant(col, chrom, pos, vid, ref, alt_alleles[0])
                else:
                    if not split_multiallelic:
                        # Skip multi-allelic sites entirely in non-split mode
                        continue
                    for ai, alt_base in enumerate(alt_alleles, start=1):
                        col = build_columns_for_alt(ai, alt_base)
                        consider_variant(col, chrom, pos, vid, ref, alt_base)
        finally:
            try:
                fh.close()
            except Exception:
                pass

    if not columns:
        # No variants passed filters
        geno = np.zeros((len(individual_ids or []), 0), dtype=np.int8)
    else:
        geno = np.column_stack(columns).astype(np.int8, copy=False)

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
