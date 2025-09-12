#!/usr/bin/env python
"""
PLINK .bed loader for GWAS: builds (geno_matrix, individual_ids, geno_map).

Dependencies:
    - bed-reader (pip install bed-reader)

Conventions:
    - Genotypes coded 0/1/2 for copies of ALT allele; missing as -9 (int8)
    - .bim columns used to build map with columns ['SNP','CHROM','POS','REF','ALT']
      where REF = A2 and ALT = A1 from BIM (PLINK convention)
    - .fam provides sample IDs (IID by default)

QC options mirror VCF loader: monomorphic, missingness, MAF filters.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except Exception:
    raise ImportError("NumPy is required: pip install numpy")

MISSING = -9


def _resolve_plink_paths(prefix_or_bed: str | Path, bim: str | Path | None, fam: str | Path | None) -> Tuple[Path, Path, Path]:
    p = Path(prefix_or_bed)
    if p.suffix.lower() == '.bed':
        bed = p
        pref = p.with_suffix('')
    else:
        pref = p
        bed = pref.with_suffix('.bed')
    bim_path = Path(bim) if bim else pref.with_suffix('.bim')
    fam_path = Path(fam) if fam else pref.with_suffix('.fam')
    for fp, ext in ((bed, '.bed'), (bim_path, '.bim'), (fam_path, '.fam')):
        if not fp.exists():
            raise FileNotFoundError(f"Missing PLINK file: {fp} ({ext})")
    return bed, bim_path, fam_path


def _read_fam_ids(fam_path: Path) -> List[str]:
    ids: List[str] = []
    with fam_path.open('r') as fh:
        for line in fh:
            if not line:
                continue
            parts = line.rstrip('\n').split()
            if not parts:
                continue
            # FID IID ...; use IID (col 2)
            if len(parts) < 2:
                # degrade gracefully: use first token
                ids.append(parts[0])
            else:
                ids.append(parts[1])
    if not ids:
        raise ValueError('FAM file contained no individuals')
    return ids


def _read_bim_map(bim_path: Path, return_pandas: bool):
    rows = []
    with bim_path.open('r') as fh:
        for line in fh:
            if not line:
                continue
            parts = line.rstrip('\n').split()
            if len(parts) < 6:
                continue
            chrom, snp, _cm, pos, a1, a2 = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            rows.append({
                'SNP': snp,
                'CHROM': str(chrom),
                'POS': int(float(pos)),
                'REF': a2,
                'ALT': a1,
            })
    if return_pandas:
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(rows, columns=['SNP', 'CHROM', 'POS', 'REF', 'ALT'])
        except Exception:
            pass
    return rows


def load_genotype_plink(
    prefix_or_bed: str | Path,
    bim: str | Path | None = None,
    fam: str | Path | None = None,
    drop_monomorphic: bool = False,
    max_missing: float = 1.0,
    min_maf: float = 0.0,
    return_pandas: bool = True,
):
    """
    Load PLINK 1 .bed genotype data.

    Parameters
    - prefix_or_bed: path to PLINK prefix or .bed file
    - bim, fam: optional explicit paths; by default inferred from prefix
    - drop_monomorphic: drop variants with all non-missing 0 or all 2
    - max_missing: drop variants with missing rate > threshold (0..1]
    - min_maf: drop variants with minor allele frequency < threshold
    - return_pandas: return geno_map as pandas.DataFrame if pandas is available
    """
    bed_path, bim_path, fam_path = _resolve_plink_paths(prefix_or_bed, bim, fam)

    # Read sample IDs and map first (for integrity checks)
    individual_ids = _read_fam_ids(fam_path)
    geno_map = _read_bim_map(bim_path, return_pandas=return_pandas)

    # Load genotype matrix with bed-reader
    try:
        from bed_reader import open_bed  # type: ignore
    except Exception as e:
        raise ImportError("bed-reader is required for PLINK .bed loading. pip install bed-reader") from e

    b = open_bed(str(bed_path))
    X = b.read()  # (n_individuals, n_markers); dtype may be float with NaN or integers

    # Normalize to int8 with -9 missing
    if np.issubdtype(X.dtype, np.floating):
        # Round to nearest int, replace NaN with -9
        missing_mask = np.isnan(X)
        Xi = np.rint(X).astype(np.int16, copy=False)
        Xi[missing_mask] = MISSING
    else:
        Xi = X.astype(np.int16, copy=False)
        bad = (Xi != 0) & (Xi != 1) & (Xi != 2)
        if np.any(bad):
            Xi[bad] = MISSING

    # QC and filtering, column-wise
    n_ind, n_mark = Xi.shape
    if len(individual_ids) != n_ind:
        # Some readers can infer fam/iid; but we rely on fam
        # We still enforce consistent dimensions
        raise AssertionError(f"Sample count mismatch: FAM {len(individual_ids)} vs BED {n_ind}")

    keep_mask = np.ones(n_mark, dtype=bool)
    # Missingness
    if max_missing < 1.0:
        valid = Xi != MISSING
        call_rate = valid.sum(axis=0) / float(n_ind)
        keep_mask &= (1.0 - call_rate) <= max_missing
    # MAF
    if min_maf > 0.0 or drop_monomorphic:
        valid = Xi != MISSING
        # To avoid division by zero, set mean to 0 where no valid calls
        sums = np.where(valid.any(axis=0), Xi.clip(min=0).sum(axis=0), 0)
        counts = valid.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_dos = np.where(counts > 0, sums / counts, 0.0)
        maf = np.minimum(mean_dos / 2.0, 1.0 - (mean_dos / 2.0))
        if min_maf > 0.0:
            keep_mask &= maf >= min_maf
        if drop_monomorphic:
            # monomorphic: all non-missing are 0 or all are 2
            all0 = (Xi == 0) | (Xi == MISSING)
            all2 = (Xi == 2) | (Xi == MISSING)
            mono = all0.all(axis=0) | all2.all(axis=0)
            keep_mask &= ~mono

    # Apply filter
    Xi = Xi[:, keep_mask].astype(np.int8, copy=False)

    # Filter geno_map accordingly
    if isinstance(geno_map, list):
        geno_map = [row for (row, k) in zip(geno_map, keep_mask) if k]
    else:
        geno_map = geno_map.loc[np.where(keep_mask)[0]].reset_index(drop=True)

    return Xi, individual_ids, geno_map

