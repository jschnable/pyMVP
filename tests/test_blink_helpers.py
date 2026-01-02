import numpy as np
import pandas as pd
import pytest

from panicle.association import blink
from panicle.association.blink import (
    _aggregate_history,
    _apply_substitution_r_style,
    _bic_positions,
    _build_design_matrix,
    _compute_bic_statistics,
    _compute_maf_mask,
    _ensure_numpy_genotype,
    _farmcpu_prior_simple,
    _jaccard_similarity,
    _prepare_covariates,
    _precompute_map_coordinates,
)
from panicle.utils.data_types import GenotypeMatrix, GenotypeMap


def test_ensure_numpy_genotype_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        _ensure_numpy_genotype(np.ones((2, 2, 1)))

    gm = GenotypeMatrix(np.ones((2, 2), dtype=np.int8), is_imputed=True)
    arr, majors = _ensure_numpy_genotype(gm)
    assert majors is None
    np.testing.assert_array_equal(arr, np.ones((2, 2), dtype=np.int8))


def test_precompute_map_coordinates_handles_non_numeric_chromosomes() -> None:
    df = pd.DataFrame({"SNP": ["a", "b"], "CHROM": ["X", "1"], "POS": [5, 10]})
    chrom_vals, pos_vals = _precompute_map_coordinates(df)
    assert chrom_vals.shape == (2,)
    np.testing.assert_array_equal(pos_vals, np.array([5.0, 10.0]))


def test_prepare_covariates_shape_validation() -> None:
    with pytest.raises(ValueError):
        _prepare_covariates(np.ones((3, 1)), n_individuals=2)

    cov = _prepare_covariates(np.array([1, 2]), n_individuals=2)
    assert cov.shape == (2, 1)


def test_compute_maf_mask_threshold_zero() -> None:
    geno = np.array([[0, 1], [2, 1]], dtype=float)
    mask, maf = _compute_maf_mask(geno, maf_threshold=0.0, max_genotype_dosage=2.0)
    assert mask.tolist() == [True, True]
    np.testing.assert_allclose(maf, np.array([0.5, 0.5]))


def test_farmcpu_prior_simple_applies_weights() -> None:
    map_df = pd.DataFrame({"SNP": ["s1", "s2"], "CHROM": [1, 1], "POS": [10, 20]})
    pvals = np.array([0.1, 0.2])
    prior = pd.DataFrame({"SNP": ["s2"], "Chr": [1], "Pos": [20], "Weight": [0.5]})
    updated = _farmcpu_prior_simple(map_df, pvalues=pvals, prior_info=prior)
    np.testing.assert_allclose(updated, np.array([0.1, 0.1]))


def test_build_design_matrix_and_bic_statistics_paths() -> None:
    trait = np.array([1.0, 2.0, 3.0, 4.0])
    cov = np.array([[0.1], [0.2], [0.3], [0.4]])
    geno_subset = np.array([[0], [1], [-9], [2]], dtype=float)

    X = _build_design_matrix(trait, cov, geno_subset, fill_value=0.0)
    assert X is not None
    bic, stats_arr = _compute_bic_statistics(X, trait)
    assert np.isfinite(bic)
    assert stats_arr.shape[1] == 4

    # df <= 0 branch
    X_bad = np.ones((2, 3))
    bic_bad, stats_bad = _compute_bic_statistics(X_bad, np.array([1.0, 2.0]))
    assert not np.isfinite(bic_bad)
    assert stats_bad.size == 0


def test_bic_positions_methods_and_warning() -> None:
    assert _bic_positions(5, "naive") == [1, 2, 3, 4, 5]
    even = _bic_positions(10, "even")
    assert even[-1] == 10
    lg = _bic_positions(3, "lg")
    assert lg[-1] == 3
    ln = _bic_positions(3, "ln")
    assert ln[-1] == 3
    fixed = _bic_positions(25, "fixed")
    assert fixed[-1] == 25
    defaulted = _bic_positions(3, "unknown")
    assert defaulted == [1, 2, 3]


def test_apply_substitution_r_style_handles_methods_and_empty() -> None:
    result = np.ones((3, 3))
    map_df = GenotypeMap(
        pd.DataFrame({"SNP": ["s0", "s1", "s2"], "CHROM": ["1", "1", "1"], "POS": [1, 2, 3]})
    )
    history = {0: [{"Method": "onsite", "Effect": 1.0, "p": 0.05, "effect": 0.5, "se": 0.1}]}

    _apply_substitution_r_style(result, [0], map_df.to_dataframe().values, history, method_sub="reward", n_covariates=1)
    _apply_substitution_r_style(result, [], map_df.to_dataframe().values, history, method_sub="onsite", n_covariates=1)
    # Unknown method should fall through without raising
    _apply_substitution_r_style(result, [0], map_df.to_dataframe().values, history, method_sub="unknown", n_covariates=0)


def test_jaccard_similarity_and_aggregate_history() -> None:
    assert _jaccard_similarity([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert _jaccard_similarity([], []) == 1.0

    hist_entries = [{"effect": 1.0, "se": 0.1, "p": 0.01}, {"effect": 2.0, "se": 0.2, "p": 0.02}]
    agg_reward = _aggregate_history(hist_entries, method="reward", initial_pvalues=None, marker_idx=0)
    assert agg_reward is not None and agg_reward[0] == 1.0
    agg_onsite = _aggregate_history(hist_entries, method="onsite", initial_pvalues=np.array([0.5]), marker_idx=0)
    assert agg_onsite is not None and agg_onsite[2] == 0.5
