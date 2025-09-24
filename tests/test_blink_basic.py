import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymvp.association.blink import MVP_BLINK, _aggregate_history
from pymvp.utils.data_types import GenotypeMap


def _make_synthetic_data(n_individuals: int = 40, n_markers: int = 25):
    rng = np.random.default_rng(42)

    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int64)
    missing_mask = rng.random(geno.shape) < 0.05
    geno = geno.astype(np.float64)
    geno[missing_mask] = -9

    ids = np.arange(n_individuals)
    trait = rng.normal(loc=0.0, scale=1.0, size=n_individuals)
    phe = np.column_stack([ids, trait])

    chroms = np.repeat(np.arange(1, (n_markers // 5) + 2), 5)[:n_markers]
    positions = np.arange(1, n_markers + 1) * 1_000
    map_df = pd.DataFrame({
        "SNP": [f"SNP_{i}" for i in range(n_markers)],
        "CHROM": chroms,
        "POS": positions,
    })
    return phe, geno, GenotypeMap(map_df)


def test_blink_runs_and_returns_association_results():
    phe, geno, map_data = _make_synthetic_data()

    results = MVP_BLINK(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=None,
        maxLoop=4,
        verbose=False,
    )

    assert results.n_markers == map_data.n_markers
    arr = results.to_numpy()
    assert arr.shape == (map_data.n_markers, 3)
    assert np.all(np.isfinite(arr[:, 2]) | np.isnan(arr[:, 2]))


def test_blink_with_covariates_produces_consistent_output():
    phe, geno, map_data = _make_synthetic_data()
    cov = np.linspace(-1, 1, phe.shape[0]).reshape(-1, 1)

    res1 = MVP_BLINK(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=cov,
        maxLoop=3,
        verbose=False,
    )
    res2 = MVP_BLINK(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=cov,
        maxLoop=3,
        verbose=False,
    )

    np.testing.assert_allclose(res1.to_numpy(), res2.to_numpy())


def test_blink_handles_minimal_marker_set():
    phe, geno, map_data = _make_synthetic_data(n_markers=3)

    res = MVP_BLINK(
        phe=phe,
        geno=geno,
        map_data=map_data,
        maxLoop=2,
        verbose=False,
    )

    arr = res.to_numpy()
    assert arr.shape == (3, 3)


def test_blink_raises_for_missing_trait_values():
    phe, geno, map_data = _make_synthetic_data()
    phe[0, 1] = np.nan

    try:
        MVP_BLINK(
            phe=phe,
            geno=geno,
            map_data=map_data,
            verbose=False,
        )
    except ValueError as exc:
        assert "requires complete trait" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing trait values")


def test_blink_maf_threshold_removing_all_markers():
    phe, geno, map_data = _make_synthetic_data(n_markers=5)

    try:
        MVP_BLINK(
            phe=phe,
            geno=geno,
            map_data=map_data,
            maf_threshold=0.6,
            verbose=False,
        )
    except ValueError as exc:
        assert "removed" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError when all markers are filtered by MAF")


def test_aggregate_history_methods():
    history = [
        {"p": 0.01, "effect": 0.2, "se": 0.05},
        {"p": 0.05, "effect": 0.4, "se": 0.08},
        {"p": 0.03, "effect": 0.3, "se": 0.06},
    ]
    initial = np.array([0.5, 0.4, 0.3])

    reward = _aggregate_history(history, "reward", initial, 0)
    penalty = _aggregate_history(history, "penalty", initial, 0)
    mean_val = _aggregate_history(history, "mean", initial, 0)
    median_val = _aggregate_history(history, "median", initial, 0)
    onsite = _aggregate_history(history, "onsite", initial, 0)

    assert reward[2] == 0.01
    assert penalty[2] == 0.05
    assert np.isclose(mean_val[0], np.mean([h["effect"] for h in history]))
    assert np.isclose(median_val[2], np.median([h["p"] for h in history]))
    assert onsite[2] == initial[0]
