import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from panicle.tools import convert_genotype as cg


def test_coerce_value_handles_bool_int_float_and_string() -> None:
    assert cg._coerce_value("True") is True
    assert cg._coerce_value("false") is False
    assert cg._coerce_value("7") == 7
    assert cg._coerce_value("3.5") == pytest.approx(3.5)
    assert cg._coerce_value("NaNish") == "NaNish"


def test_parse_loader_options_converts_and_validates() -> None:
    opts = ["max_missing=0.1", "precompute_alleles=true", "threads=4"]
    parsed = cg._parse_loader_options(opts)
    assert parsed == {"max_missing": 0.1, "precompute_alleles": True, "threads": 4}

    with pytest.raises(ValueError):
        cg._parse_loader_options(["badoption"])


def test_main_invokes_loader_and_writer(monkeypatch, tmp_path, capsys) -> None:
    input_file = tmp_path / "geno.csv"
    input_file.write_text("dummy", encoding="utf-8")
    output_prefix = tmp_path / "out" / "cache"
    map_df = pd.DataFrame({"SNP": ["s1", "s2", "s3"], "CHROM": ["1"] * 3, "POS": [1, 2, 3]})
    recorded = {}

    def fake_detect(path):
        recorded["detected_path"] = path
        return "csv"

    def fake_load(path, file_format, **kwargs):
        recorded["load_args"] = {"path": path, "file_format": file_format, "kwargs": kwargs}
        geno = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int8)
        return geno, ["A", "B"], map_df

    def fake_save(genotype, sample_ids, geno_map, output_prefix, dtype, batch_size):
        recorded["save_args"] = {
            "genotype": genotype.copy(),
            "sample_ids": sample_ids,
            "geno_map": geno_map.copy(),
            "output_prefix": output_prefix,
            "dtype": dtype,
            "batch_size": batch_size,
        }
        return {
            "shape": genotype.shape,
            "memmap_path": output_prefix.with_suffix(".mmap"),
            "metadata_path": output_prefix.with_suffix(".json"),
            "samples_path": output_prefix.with_suffix(".samples"),
            "map_path": output_prefix.with_suffix(".map"),
        }

    monkeypatch.setattr(cg, "detect_file_format", fake_detect)
    monkeypatch.setattr(cg, "load_genotype_file", fake_load)
    monkeypatch.setattr(cg, "save_genotype_to_memmap", fake_save)

    ret = cg.main(
        [
            "-i",
            str(input_file),
            "-o",
            str(output_prefix),
            "--dtype",
            "int16",
            "--batch-size",
            "10",
            "--load-option",
            "max_missing=0.2",
            "--precompute-alleles",
        ]
    )

    assert ret == 0
    assert recorded["detected_path"] == input_file
    assert recorded["load_args"]["file_format"] == "csv"
    assert recorded["load_args"]["kwargs"]["max_missing"] == 0.2
    assert recorded["load_args"]["kwargs"]["precompute_alleles"] is True
    assert recorded["save_args"]["dtype"] == np.dtype("int16")
    assert recorded["save_args"]["batch_size"] == 10
    assert recorded["save_args"]["output_prefix"] == output_prefix

    out = capsys.readouterr().out
    assert "Cached genotype matrix" in out
    assert "Memmap:" in out
    assert "Map:" in out


def test_main_errors_on_unknown_format(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "geno.txt"
    input_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cg, "detect_file_format", lambda path: "unknown")

    with pytest.raises(SystemExit):
        cg.main(["-i", str(input_file), "-o", str(tmp_path / "out")])


def test_main_rejects_invalid_dtype(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "geno.txt"
    input_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cg, "load_genotype_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not load")))
    monkeypatch.setattr(cg, "detect_file_format", lambda path: "csv")
    monkeypatch.setattr(cg, "save_genotype_to_memmap", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        cg.main(["-i", str(input_file), "-o", str(tmp_path / "out"), "--dtype", "not-a-dtype", "--format", "csv"])


def test_main_errors_when_input_missing() -> None:
    with pytest.raises(SystemExit):
        cg.main(["-i", "nonexistent.file", "-o", "out"])


def test_main_handles_missing_map(monkeypatch, tmp_path, capsys) -> None:
    infile = tmp_path / "g.txt"
    infile.write_text("x", encoding="utf-8")
    out_prefix = tmp_path / "out"

    monkeypatch.setattr(cg, "detect_file_format", lambda path: "csv")
    monkeypatch.setattr(cg, "load_genotype_file", lambda *a, **k: (np.zeros((1, 1), dtype=np.int8), ["id"], None))

    def fake_save(**kwargs):
        return {
            "shape": (1, 1),
            "memmap_path": out_prefix.with_suffix(".mmap"),
            "metadata_path": out_prefix.with_suffix(".json"),
            "samples_path": out_prefix.with_suffix(".samples"),
            "map_path": None,
        }

    monkeypatch.setattr(cg, "save_genotype_to_memmap", fake_save)

    ret = cg.main(["-i", str(infile), "-o", str(out_prefix), "--format", "csv"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Map: not provided" in out
