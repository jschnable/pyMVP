import json
import pathlib
import subprocess

import numpy as np
import pandas as pd
import pytest

from pymvp.association.blink import MVP_BLINK
from pymvp.utils.data_types import GenotypeMap

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
R_SCRIPT = REPO_ROOT / "tests" / "r_helpers" / "run_blink.R"


def _generate_data(n_individuals: int = 60, n_markers: int = 40, seed: int = 2025):
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int64)
    geno = geno.astype(np.float64)
    missing_mask = rng.random(geno.shape) < 0.03
    geno[missing_mask] = -9

    qtn_effects = rng.normal(0, 0.05, size=n_markers)
    qtn_effects[:3] = [0.6, -0.4, 0.5]
    trait = geno @ qtn_effects + rng.normal(0, 1.0, size=n_individuals)
    ids = np.arange(n_individuals)
    phe = np.column_stack([ids, trait])

    chroms = np.repeat(np.arange(1, (n_markers // 4) + 2), 4)[:n_markers]
    positions = np.arange(1, n_markers + 1) * 1000
    map_df = pd.DataFrame({
        "SNP": [f"SNP_{i}" for i in range(n_markers)],
        "CHROM": chroms,
        "POS": positions,
    })

    cov = np.column_stack([
        np.linspace(-1, 1, n_individuals),
        rng.normal(size=n_individuals),
    ])

    return phe, geno, GenotypeMap(map_df), cov


def _run_r_blink(phe, geno, map_data, covariates):
    payload = {
        "phenotype": phe.tolist(),
        "genotype": geno.tolist(),
        "map": {
            "snp_id": map_data.to_dataframe()["SNP"].tolist(),
            "chrom": map_data.to_dataframe()["CHROM"].tolist(),
            "pos": map_data.to_dataframe()["POS"].tolist(),
        },
        "covariates": covariates.tolist(),
        "maxLoop": 4,
        "converge": 1,
        "ld_threshold": 0.7,
        "maf_threshold": 0.0,
        "method_sub": "reward",
        "p_threshold": None,
        "qtn_threshold": 0.01,
    }

    result = subprocess.run(
        ["Rscript", str(R_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "R BLINK script failed",
            result.returncode,
            result.stdout,
            result.stderr,
        )

    parsed = json.loads(result.stdout)
    return np.asarray(parsed["pvalues"], dtype=float)


def test_blink_parity_with_r_reference():
    phe, geno, map_data, cov = _generate_data()

    py_res = MVP_BLINK(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=cov,
        maxLoop=4,
        verbose=False,
    )
    py_pvals = py_res.to_numpy()[:, 2]

    try:
        r_pvals = _run_r_blink(phe, geno, map_data, covariates=cov)
    except RuntimeError as exc:
        pytest.xfail(f"R BLINK reference execution failed: {exc}")

    assert py_pvals.shape[0] == r_pvals.shape[0]
    np.testing.assert_allclose(py_pvals, r_pvals, rtol=1e-5, atol=1e-6)
