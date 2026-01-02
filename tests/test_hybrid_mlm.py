import numpy as np
import pandas as pd
import pytest

from panicle.association import hybrid_mlm as hm
from panicle.utils.data_types import GenotypeMap, GenotypeMatrix


class DummyKinship:
    def __init__(self, eigenvals, eigenvecs):
        self._eigenvals = eigenvals
        self._eigenvecs = eigenvecs

    def get_eigen(self, chrom):
        return {"eigenvals": self._eigenvals, "eigenvecs": self._eigenvecs}


class DummyResults:
    def __init__(self, effects, se, pvals):
        self.effects = effects
        self.se = se
        self.pvalues = pvals


def _inputs():
    phe = np.column_stack([np.arange(6), np.linspace(0.0, 1.0, 6)])
    geno = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 1],
            [2, 0],
        ],
        dtype=np.int8,
    )
    map_df = pd.DataFrame({"SNP": ["s0", "s1"], "CHROM": ["1", "1"], "POS": [1, 2]})
    return phe, GenotypeMatrix(geno), GenotypeMap(map_df)


def test_panicle_mlm_hybrid_requires_map() -> None:
    phe, geno, _ = _inputs()
    with pytest.raises(ValueError):
        hm.PANICLE_MLM_Hybrid(phe, geno, None, verbose=False)


def test_panicle_mlm_hybrid_returns_wald_when_no_candidates(monkeypatch) -> None:
    phe, geno, gmap = _inputs()
    wald = DummyResults(
        effects=np.zeros(2),
        se=np.ones(2),
        pvals=np.array([0.5, 0.6]),
    )

    monkeypatch.setattr(hm, "PANICLE_K_VanRaden_LOCO", lambda *a, **k: DummyKinship(np.ones(6), np.eye(6)))
    monkeypatch.setattr(hm, "PANICLE_MLM_LOCO", lambda **kwargs: wald)

    res = hm.PANICLE_MLM_Hybrid(phe, geno, gmap, screen_threshold=1e-3, verbose=False)
    np.testing.assert_array_equal(res.pvalues, wald.pvalues)


def test_panicle_mlm_hybrid_refines_candidates(monkeypatch) -> None:
    phe, geno, gmap = _inputs()
    wald = DummyResults(
        effects=np.zeros(2),
        se=np.ones(2),
        pvals=np.array([1e-6, 0.5]),
    )
    kin = DummyKinship(np.ones(6), np.eye(6))

    monkeypatch.setattr(hm, "PANICLE_K_VanRaden_LOCO", lambda *a, **k: kin)
    monkeypatch.setattr(hm, "PANICLE_MLM_LOCO", lambda **kwargs: wald)
    monkeypatch.setattr(hm, "estimate_variance_components_brent", lambda *a, **k: (0.5, 0.5, 0.5))
    monkeypatch.setattr(hm, "_calculate_neg_ml_likelihood", lambda *a, **k: 0.1)
    monkeypatch.setattr(hm, "fit_marker_lrt", lambda *a, **k: (1.0, 0.01, 0.4, 0.2))

    res = hm.PANICLE_MLM_Hybrid(
        phe,
        geno,
        gmap,
        loco_kinship=kin,
        screen_threshold=1e-4,
        verbose=False,
    )

    assert res.pvalues[0] == pytest.approx(0.01)
    assert res.effects[0] == pytest.approx(0.4)
    assert res.se[0] == pytest.approx(0.2)
