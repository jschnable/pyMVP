import json
import pathlib
import subprocess

import numpy as np
import pytest

from pymvp.association.glm import MVP_GLM
from pymvp.data.loaders import load_genotype_file, load_phenotype_file, match_individuals
from pymvp.matrix.pca import MVP_PCA
from pymvp.utils.data_types import GenotypeMatrix

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
R_SCRIPT = REPO_ROOT / "tests" / "r_helpers" / "run_glm.R"


def test_glm_parity_with_gapit():
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    def locate(name: str) -> pathlib.Path:
        candidates = [
            repo_root / name,
            repo_root / "comparison" / name,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not locate required file '{name}' in {candidates}")

    geno_path = locate("mdp_genotype_test.hmp.txt")
    phe_path = locate("mdp_traits.txt")

    geno_matrix, individual_ids, _ = load_genotype_file(geno_path)

    phenotype_df = load_phenotype_file(phe_path, trait_columns=["dpoll"])
    matched_phe, matched_indices, _ = match_individuals(phenotype_df, individual_ids)
    matched_phe = matched_phe.reset_index(drop=True)

    geno_full = geno_matrix.get_batch(0, geno_matrix.n_markers)
    geno_np = geno_full[matched_indices, :]
    trait_values = matched_phe["dpoll"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(trait_values)

    geno_np = geno_np[valid_mask]
    trait_values = trait_values[valid_mask]

    # Use GenotypeMatrix representation so PCA and GLM mirror the production flow
    geno_filtered = geno_np.astype(np.int8, copy=False)
    geno_matrix_filtered = GenotypeMatrix(geno_filtered)
    pcs = MVP_PCA(M=geno_matrix_filtered, pcs_keep=3, verbose=False)

    phe = np.column_stack([
        np.arange(trait_values.shape[0], dtype=float),
        trait_values,
    ])

    py_res = MVP_GLM(
        phe=phe,
        geno=geno_matrix_filtered,
        CV=pcs,
        maxLine=geno_np.shape[1],
        verbose=False,
        missing_fill_value=1.0,
    )
    py_pvals = py_res.to_numpy()[:, 2]

    payload = {
        "phenotype": phe.tolist(),
        "genotype": geno_filtered.astype(int).tolist(),
        "covariates": pcs.tolist(),
        "impute_value": 1.0,
    }

    result = subprocess.run(
        ["Rscript", str(R_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "R GLM script failed",
            result.returncode,
            result.stdout,
            result.stderr,
        )

    r_pvals = np.asarray(json.loads(result.stdout)["pvalues"], dtype=float)

    try:
        np.testing.assert_allclose(py_pvals, r_pvals, rtol=1e-5, atol=1e-6)
    except AssertionError as exc:
        abs_diff = np.abs(py_pvals - r_pvals)
        rel_diff = abs_diff / np.maximum(np.abs(r_pvals), 1e-20)
        top_idx = int(np.argmax(abs_diff))
        mismatch_report = [
            "GLM parity mismatch summary:",
            f"  markers compared     : {py_pvals.size}",
            f"  max abs diff @ {top_idx}: {abs_diff[top_idx]:.6e}",
            f"  max rel diff         : {rel_diff[top_idx]:.6e}",
            f"  python p-values head : {py_pvals[:5]}",
            f"  GAPIT  p-values head : {r_pvals[:5]}",
            f"  abs diff head        : {abs_diff[:5]}",
        ]
        pytest.fail("\n".join(mismatch_report + [str(exc)]), pytrace=False)
