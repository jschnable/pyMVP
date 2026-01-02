import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.data import load_genotype_plink


def test_resolve_plink_paths_requires_all_files(tmp_path) -> None:
    prefix = tmp_path / "study"
    bed_path = prefix.with_suffix(".bed")
    bed_path.write_bytes(b"bed")

    with pytest.raises(FileNotFoundError):
        load_genotype_plink._resolve_plink_paths(prefix, None, None)

    for suffix in (".bim", ".fam"):
        prefix.with_suffix(suffix).write_text("placeholder\n", encoding="utf-8")

    bed, bim, fam = load_genotype_plink._resolve_plink_paths(prefix, None, None)
    assert bed == bed_path
    assert bim == prefix.with_suffix(".bim")
    assert fam == prefix.with_suffix(".fam")

    # Accept explicit .bed path as well
    bed_explicit = tmp_path / "explicit.bed"
    bed_explicit.write_bytes(b"bed")
    (tmp_path / "explicit.bim").write_text("bim\n", encoding="utf-8")
    (tmp_path / "explicit.fam").write_text("fam iid\n", encoding="utf-8")
    bed, bim, fam = load_genotype_plink._resolve_plink_paths(bed_explicit, None, None)
    assert bed == bed_explicit
    assert fam.name.endswith(".fam")


def test_read_fam_ids_handles_short_and_empty(tmp_path) -> None:
    fam_path = tmp_path / "mixed.fam"
    fam_path.write_text("fam1 iid1 rest\nlonely\n\n \n", encoding="utf-8")
    ids = load_genotype_plink._read_fam_ids(fam_path)
    assert ids == ["iid1", "lonely"]

    empty_path = tmp_path / "empty.fam"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_genotype_plink._read_fam_ids(empty_path)


def test_read_bim_map_dataframe_and_list(tmp_path) -> None:
    bim_path = tmp_path / "map.bim"
    bim_path.write_text("1 snp1 0 100 A G\ninvalid\n2 snp2 0 200 T C extra\n", encoding="utf-8")

    df = load_genotype_plink._read_bim_map(bim_path, return_pandas=True)
    assert list(df["SNP"]) == ["snp1", "snp2"]
    assert df.loc[0, "REF"] == "G"
    assert df.loc[1, "ALT"] == "T"

    rows = load_genotype_plink._read_bim_map(bim_path, return_pandas=False)
    assert rows[0]["POS"] == 100
    assert rows[1]["CHROM"] == "2"


def test_load_genotype_plink_imports_bed_reader(monkeypatch, tmp_path) -> None:
    bed_path = tmp_path / "importcheck.bed"
    bed_path.write_bytes(b"bed")
    bed_path.with_suffix(".bim").write_text("", encoding="utf-8")
    bed_path.with_suffix(".fam").write_text("fam1 iid1\n", encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bed_reader":
            raise ImportError("no bed_reader")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError):
        load_genotype_plink.load_genotype_plink(bed_path)


def test_load_genotype_plink_filters_missing_maf_and_mono(monkeypatch, tmp_path) -> None:
    bed_path = tmp_path / "filtered.bed"
    bed_path.write_bytes(b"bed")
    bed_path.with_suffix(".bim").write_text(
        "1 snp1 0 100 A G\n1 snp2 0 200 A G\n1 snp3 0 300 A G\n1 snp4 0 400 A G\n",
        encoding="utf-8",
    )
    bed_path.with_suffix(".fam").write_text("f1 i1\nf2 i2\nf3 i3\n", encoding="utf-8")

    geno_matrix = np.array(
        [
            [0.0, 1.0, 0.0, np.nan],
            [0.0, 1.0, 0.0, np.nan],
            [2.0, 1.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    class FakeBed:
        def read(self):
            return geno_matrix

    fake_mod = types.ModuleType("bed_reader")
    fake_mod.open_bed = lambda path: FakeBed()
    monkeypatch.setitem(sys.modules, "bed_reader", fake_mod)

    with np.errstate(invalid="ignore"):
        geno, ids, geno_map = load_genotype_plink.load_genotype_plink(
            bed_path, max_missing=0.3, min_maf=0.4, drop_monomorphic=True, return_pandas=True
        )

    assert ids == ["i1", "i2", "i3"]
    # Only snp2 survives filters (maf >= 0.4, not monomorphic, missingness OK)
    np.testing.assert_array_equal(geno, np.array([[1], [1], [1]], dtype=np.int8))
    assert list(geno_map["SNP"]) == ["snp2"]


def test_load_genotype_plink_marks_invalid_and_checks_sample_count(monkeypatch, tmp_path) -> None:
    bed_path = tmp_path / "badcount.bed"
    bed_path.write_bytes(b"bed")
    bed_path.with_suffix(".bim").write_text("1 snp1 0 100 A G\n1 snp2 0 200 A G\n", encoding="utf-8")
    bed_path.with_suffix(".fam").write_text("f1 iid1\n", encoding="utf-8")

    matrix_with_invalid = np.array([[0, 5], [2, 2]], dtype=np.int16)

    class FakeBed:
        def read(self):
            return matrix_with_invalid

    fake_mod = types.ModuleType("bed_reader")
    fake_mod.open_bed = lambda path: FakeBed()
    monkeypatch.setitem(sys.modules, "bed_reader", fake_mod)

    with pytest.raises(AssertionError):
        load_genotype_plink.load_genotype_plink(bed_path, return_pandas=False)
