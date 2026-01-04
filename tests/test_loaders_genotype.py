import numpy as np
import pandas as pd
import pytest

from panicle.data import loaders
from panicle.utils.memmap_utils import save_genotype_to_memmap
from panicle.utils.data_types import GenotypeMatrix


def test_load_genotype_csv_effective_tests_and_dedup(tmp_path) -> None:
    csv_path = tmp_path / "geno.csv"
    csv_path.write_text(
        "ID,snp1,snp2\n"
        "I1,0,1\n"
        "I2,1,0\n"
        "I1,2,2\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning):
        genotype, individual_ids, geno_map = loaders.load_genotype_file(
            csv_path,
            file_format="csv",
            compute_effective_tests=True,
        )

    assert individual_ids == ["I1", "I2"]  # duplicate I1 dropped
    assert genotype.shape == (2, 2)
    assert isinstance(genotype, GenotypeMatrix)
    assert genotype.is_imputed is True
    assert geno_map.metadata.get("effective_tests") is not None
    assert set(geno_map.chromosomes.astype(str).unique()) == {"1"}
    assert geno_map.positions.iloc[-1] == 2


def test_load_genotype_numeric_detects_separator(tmp_path) -> None:
    num_path = tmp_path / "geno_numeric.txt"
    num_path.write_text(
        "ID\tsnp1\tsnp2\n"
        "A\t0\t1\n"
        "B\t1\t2\n",
        encoding="utf-8",
    )

    genotype, individual_ids, geno_map = loaders.load_genotype_file(
        num_path,
        file_format="numeric",
    )

    assert individual_ids == ["A", "B"]
    np.testing.assert_array_equal(genotype[:, :], np.array([[0, 1], [1, 2]], dtype=np.int8))
    assert genotype.is_imputed is True
    assert list(geno_map.snp_ids) == ["snp1", "snp2"]


def test_load_genotype_memmap_metadata_round_trip(tmp_path) -> None:
    geno_array = np.array([[0, 1], [2, 0]], dtype=np.int8)
    geno_map = loaders.GenotypeMap(
        pd.DataFrame({"SNP": ["s1", "s2"], "CHROM": ["1", "1"], "POS": [10, 20]})
    )
    sample_ids = ["S1", "S2"]
    meta = save_genotype_to_memmap(
        geno_array,
        sample_ids=sample_ids,
        geno_map=geno_map,
        output_prefix=tmp_path / "cache" / "geno_cache",
    )

    genotype, ids, loaded_map = loaders.load_genotype_file(meta["metadata_path"], file_format=None)

    np.testing.assert_array_equal(genotype[:, :], geno_array)
    assert ids == sample_ids
    pd.testing.assert_frame_equal(loaded_map.to_dataframe(), geno_map.to_dataframe())


def test_load_genotype_vcf_plink_hapmap_monkeypatched(monkeypatch, tmp_path) -> None:
    geno_np = np.array([[0, 1], [1, 2], [2, 2]], dtype=np.int8)
    ids = ["A", "A", "B"]  # duplicate to trigger dedup
    map_df = pd.DataFrame({"SNP": ["rs1", "rs2"], "CHROM": ["1", "1"], "POS": [1, 2]})
    map_df.attrs["is_imputed"] = True

    def fake_loader(path, **kwargs):
        return geno_np, ids, map_df

    for attr in ("_load_genotype_vcf", "_load_genotype_plink", "_load_genotype_hapmap"):
        monkeypatch.setattr(loaders, attr, fake_loader)

    genotype_vcf, ids_vcf, map_vcf = loaders.load_genotype_file(tmp_path / "file.vcf", file_format="vcf")
    assert genotype_vcf.is_imputed is True
    assert ids_vcf == ["A", "B"]
    assert map_vcf.n_markers == 2

    genotype_plink, ids_plink, _ = loaders.load_genotype_file(tmp_path / "file.bed", file_format="plink")
    assert genotype_plink.is_imputed is True
    assert ids_plink == ["A", "B"]
    np.testing.assert_array_equal(genotype_plink[:, :], np.array([[0, 1], [2, 2]], dtype=np.int8))

    genotype_hmp, ids_hmp, _ = loaders.load_genotype_file(tmp_path / "file.hmp", file_format="hapmap")
    assert genotype_hmp.is_imputed is True
    assert ids_hmp == ["A", "B"]
    np.testing.assert_array_equal(genotype_hmp[:, :], np.array([[0, 1], [2, 2]], dtype=np.int8))


def test_load_genotype_file_unsupported_format(tmp_path) -> None:
    path = tmp_path / "geno.unknown"
    path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError):
        loaders.load_genotype_file(path, file_format="foo")
