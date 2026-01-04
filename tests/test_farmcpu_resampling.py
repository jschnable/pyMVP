import numpy as np
import pandas as pd
import pytest

from panicle.association import farmcpu_resampling as fr
from panicle.utils.data_types import GenotypeMap, GenotypeMatrix


def _map(n: int) -> GenotypeMap:
    return GenotypeMap(
        pd.DataFrame(
            {"SNP": [f"s{i}" for i in range(n)], "CHROM": ["1"] * n, "POS": np.arange(1, n + 1)}
        )
    )


@pytest.mark.parametrize(
    "runs,mask,ld",
    [
        (0, 0.1, 0.7),
        (2, -0.1, 0.7),
        (2, 0.1, 1.1),
    ],
)
def test_validate_inputs_rejects_out_of_range(runs: int, mask: float, ld: float) -> None:
    with pytest.raises(ValueError):
        fr._validate_inputs(runs, mask, ld)


def test_fetch_imputed_genotypes_fills_missing_with_mode() -> None:
    geno = np.array([[-9.0, 0.0], [np.nan, 2.0], [1.0, 2.0]])
    out = fr._fetch_imputed_genotypes(geno, [0, 1])
    expected = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(out, expected)


def test_fetch_imputed_genotypes_accepts_genotype_matrix() -> None:
    gm = GenotypeMatrix(np.array([[0, -9], [2, 2]], dtype=np.int8))
    out = fr._fetch_imputed_genotypes(gm, [1])
    np.testing.assert_array_equal(out, np.array([[2.0], [2.0]]))


def test_build_non_cluster_entries_sorted_by_rmip() -> None:
    counts = np.array([2, 0, 1], dtype=np.int32)
    entries = fr._build_non_cluster_entries(counts, np.array(["a", "b", "c"]), np.array([1, 1, 1]), np.array([10, 20, 30]), total_runs=4)
    assert [e.snp for e in entries] == ["a", "c"]
    assert [e.rmip for e in entries] == [0.5, 0.25]


def test_build_clusters_groups_high_ld_markers() -> None:
    counts = np.array([3, 2, 1], dtype=np.int32)
    run_sets = [{0}, {1}, {0, 1}, {2}]
    geno = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 0],
            [1, 1, 2],
        ],
        dtype=float,
    )
    entries = fr._build_clusters(
        marker_counts=counts,
        run_marker_sets=run_sets,
        genotype=geno,
        snp_ids=np.array(["s0", "s1", "s2"]),
        chroms=np.array(["1", "1", "1"]),
        positions=np.array([10, 20, 30]),
        total_runs=4,
        ld_threshold=0.8,
    )

    assert len(entries) == 2
    primary = entries[0]
    assert primary.cluster_size == 2
    assert primary.snp == "s0"  # higher count wins as representative
    assert pytest.approx(primary.rmip, rel=1e-6) == 0.75  # 3/4 runs hit the LD cluster
    assert set(primary.cluster_members.keys()) == {"s0", "s1"}


def test_panicle_farmcpu_resampling_runs_and_counts(monkeypatch) -> None:
    class DummyResult:
        def __init__(self, pvalues):
            self.pvalues = np.array(pvalues, dtype=float)

    def fake_farmcpu(**kwargs):
        return DummyResult([1e-9, 0.2])

    monkeypatch.setattr(fr, "PANICLE_FarmCPU", fake_farmcpu, raising=False)

    phe = np.column_stack([np.arange(6), np.linspace(0.0, 1.0, 6)])
    geno = np.zeros((6, 2), dtype=np.int8)
    res = fr.PANICLE_FarmCPUResampling(
        phe=phe,
        geno=geno,
        map_data=_map(2),
        runs=3,
        mask_proportion=0.0,
        significance_threshold=5e-8,
        cluster_markers=False,
        random_seed=0,
        verbose=False,
    )

    assert len(res.entries) == 1
    assert res.entries[0].snp == "s0"
    assert res.entries[0].rmip == 1.0
    assert res.per_marker_counts == {0: 3}


def test_panicle_farmcpu_resampling_clusters_and_reports(monkeypatch) -> None:
    class DummyResult:
        def __init__(self):
            self.pvalues = np.array([0.01, 0.02, 0.8], dtype=float)

    def fake_farmcpu(**kwargs):
        return DummyResult()

    monkeypatch.setattr(fr, "PANICLE_FarmCPU", fake_farmcpu, raising=False)

    phe = np.column_stack([np.arange(8), np.linspace(0.0, 1.0, 8)])
    geno = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 0],
            [1, 1, 1],
            [0, 0, 0],
            [2, 2, 1],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=np.int8,
    )
    cv = np.random.default_rng(3).normal(size=(geno.shape[0], 3))

    res = fr.PANICLE_FarmCPUResampling(
        phe=phe,
        geno=geno,
        map_data=_map(3),
        CV=cv,
        runs=4,
        mask_proportion=0.25,
        significance_threshold=0.05,
        cluster_markers=True,
        ld_threshold=0.8,
        random_seed=1,
        verbose=False,
    )

    assert res.cluster_mode is True
    assert res.entries and res.entries[0].cluster_size == 2
    assert res.entries[0].rmip == 1.0
    df = res.to_dataframe()
    assert "ClusterMembers" in df.columns
    assert set(res.entries[0].cluster_members.keys()) == {"s0", "s1"}
