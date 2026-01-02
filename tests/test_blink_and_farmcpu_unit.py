import numpy as np
import pandas as pd
import pytest

from panicle.association.blink import (
    PANICLE_BLINK,
    _get_covariate_statistics,
    remove_qtns_by_ld,
)
from panicle.association.farmcpu import PANICLE_FarmCPU
from panicle.association.glm import PANICLE_GLM
from panicle.utils.data_types import GenotypeMap


def _basic_map(n_markers: int) -> GenotypeMap:
    return GenotypeMap(
        pd.DataFrame(
            {
                "SNP": [f"snp{i}" for i in range(n_markers)],
                "CHROM": ["1"] * n_markers,
                "POS": np.arange(1, n_markers + 1) * 10,
            }
        )
    )


def test_panicle_blink_identifies_qtn_and_handles_missing() -> None:
    rng = np.random.default_rng(42)
    n = 40
    # Marker 0 drives the trait; include a couple of missing calls to exercise imputation.
    g0 = (np.arange(n) % 3).astype(float)
    g0[:2] = -9
    g1 = rng.integers(0, 3, size=n).astype(float)
    g2 = rng.integers(0, 3, size=n).astype(float)
    geno = np.column_stack([g0, g1, g2])

    trait = np.where(g0 == -9, 1.0, g0) + rng.normal(0, 0.05, size=n)
    phe = np.column_stack([np.arange(n), trait])

    res = PANICLE_BLINK(
        phe=phe,
        geno=geno,
        map_data=_basic_map(geno.shape[1]),
        maxLoop=3,
        converge=1.0,
        ld_threshold=0.9,
        verbose=False,
    )

    assert np.isfinite(res.pvalues).all()
    assert PANICLE_BLINK.last_selected_qtns, "expected at least one QTN to be selected"
    assert 0 in PANICLE_BLINK.last_selected_qtns
    assert res.pvalues[0] < 0.05


def test_panicle_blink_errors_when_maf_filters_all_markers() -> None:
    geno = np.zeros((6, 2), dtype=float)
    phe = np.column_stack([np.arange(6), np.ones(6)])
    with pytest.raises(ValueError, match="All markers were removed"):
        PANICLE_BLINK(
            phe=phe,
            geno=geno,
            map_data=_basic_map(geno.shape[1]),
            maf_threshold=0.6,
            verbose=False,
        )


def test_remove_qtns_by_ld_filters_correlated_markers() -> None:
    g0 = np.array([0, 0, 1, 1, 2], dtype=float)
    g1 = g0.copy()  # perfectly correlated with g0
    g2 = np.array([2, 1, 0, 1, 2], dtype=float)
    geno = np.column_stack([g0, g1, g2])
    map_data = GenotypeMap(
        pd.DataFrame(
            {"SNP": ["s0", "s1", "s2"], "CHROM": ["1", "1", "2"], "POS": [1, 2, 3]}
        )
    )

    kept = remove_qtns_by_ld(
        selected_qtns=[0, 1, 2],
        genotype_matrix=geno,
        correlation_threshold=0.8,
        within_chrom_only=True,
        map_data=map_data,
        verbose=False,
    )

    assert kept == [0, 2]


def test_get_covariate_statistics_handles_regular_and_singular_designs() -> None:
    rng = np.random.default_rng(7)
    cov = rng.normal(size=(50, 1))
    trait = 0.5 * cov[:, 0] + rng.normal(0, 0.05, size=50)
    phe = np.column_stack([np.arange(50), trait])

    pvals, effects, se = _get_covariate_statistics(phe, cov)
    assert pvals.shape == (1,)
    assert pvals[0] < 1e-6
    assert effects.shape == (1,)
    assert se.shape == (1,)
    assert effects[0] > 0.3  # effect direction preserved

    singular_cov = np.ones((2, 2))
    singular_phe = np.column_stack([np.arange(2), np.array([1.0, 2.0])])
    pvals_bad, effects_bad, se_bad = _get_covariate_statistics(singular_phe, singular_cov)
    assert np.allclose(pvals_bad, np.ones(2))
    assert np.allclose(effects_bad, np.zeros(2))
    assert np.allclose(se_bad, np.ones(2))


def test_panicle_farmcpu_selects_causal_marker(tmp_path) -> None:
    rng = np.random.default_rng(101)
    n = 40
    g0 = rng.integers(0, 3, size=n)
    g1 = rng.integers(0, 3, size=n)
    g2 = rng.integers(0, 3, size=n)
    geno = np.column_stack([g0, g1, g2]).astype(np.int8)

    trait = g0 * 2.0 + rng.normal(0, 0.05, size=n)
    phe = np.column_stack([np.arange(n), trait])
    cv = rng.normal(size=(n, 3))
    map_data = _basic_map(geno.shape[1])

    res = PANICLE_FarmCPU(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=cv,
        maxLoop=3,
        p_threshold=0.05,
        verbose=False,
    )

    assert np.isfinite(res.pvalues).all()
    assert hasattr(PANICLE_FarmCPU, "last_selected_qtns")
    assert PANICLE_FarmCPU.last_selected_qtns
    assert 0 in PANICLE_FarmCPU.last_selected_qtns
    assert res.pvalues[0] < 0.05


def test_panicle_farmcpu_returns_glm_when_min_p_above_cutoff() -> None:
    rng = np.random.default_rng(202)
    n = 12
    geno = rng.integers(0, 3, size=(n, 4)).astype(np.int8)
    trait = rng.normal(size=n)
    phe = np.column_stack([np.arange(n), trait])
    map_data = _basic_map(geno.shape[1])

    farmcpu_res = PANICLE_FarmCPU(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=None,
        maxLoop=2,
        p_threshold=None,  # forces 0.01/n cutoff and early return
        verbose=False,
    )
    glm_res = PANICLE_GLM(phe=phe, geno=geno, verbose=False)

    np.testing.assert_allclose(farmcpu_res.pvalues, glm_res.pvalues)


def test_remove_qtns_by_ld_truncates_to_ld_max() -> None:
    geno = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    kept = remove_qtns_by_ld(
        selected_qtns=[0, 1, 2, 3],
        genotype_matrix=geno,
        correlation_threshold=0.2,
        ld_max=1,
        verbose=False,
    )
    assert kept == [0]

