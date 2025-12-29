import numpy as np
from scipy import optimize

from panicle.association.lrt import fit_marker_lrt
from panicle.association.mlm import _calculate_neg_ml_likelihood


def test_fit_marker_lrt_returns_finite_effects() -> None:
    n = 6
    g = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
    y = 2.0 + 0.8 * g + np.array([0.1, -0.2, 0.05, -0.1, 0.2, -0.05], dtype=np.float64)
    X = np.ones((n, 1), dtype=np.float64)
    eigenvals = np.ones(n, dtype=np.float64)

    def neg_ll(h2: float) -> float:
        return _calculate_neg_ml_likelihood(h2, y, X, eigenvals)

    result = optimize.minimize_scalar(
        neg_ll,
        bounds=(0.001, 0.999),
        method="bounded",
        options={"xatol": 1.22e-4, "maxiter": 100},
    )
    null_neg_loglik = result.fun if result.success else neg_ll(0.5)

    lrt_stat, p_value, beta, se = fit_marker_lrt(y, X, g, eigenvals, null_neg_loglik)

    assert 0.0 <= p_value <= 1.0
    assert np.isfinite(lrt_stat)
    assert np.isfinite(beta)
    assert np.isfinite(se)
    assert se > 0.0
