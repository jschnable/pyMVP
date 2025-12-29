#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from panicle.association.mlm import estimate_variance_components_brent  # noqa: E402
from panicle.data.loaders import load_genotype_vcf  # noqa: E402
from panicle.matrix.kinship import MVP_K_VanRaden  # noqa: E402
from panicle.utils.data_types import GenotypeMatrix  # noqa: E402


def _solve_safe(mat, vec):
    try:
        return np.linalg.solve(mat, vec)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat) @ vec


def _fit_gblup_all(y, K_all, idx_pheno):
    n_pheno = y.shape[0]
    X = np.ones((n_pheno, 1), dtype=float)

    K11 = K_all[np.ix_(idx_pheno, idx_pheno)].astype(float, copy=True)
    if not np.all(np.isfinite(K11)):
        K11 = np.nan_to_num(K11, nan=0.0, posinf=0.0, neginf=0.0)
    K11 = (K11 + K11.T) / 2.0

    # Eigendecomposition for variance component estimation.
    eigenvals, eigenvecs = np.linalg.eigh(K11)
    order = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y_t = eigenvecs.T @ y
        X_t = eigenvecs.T @ X
    delta, vg, ve = estimate_variance_components_brent(y_t, X_t, eigenvals, verbose=False)
    delta = max(delta, 1e-6)

    w = 1.0 / (eigenvals + delta)
    XViX = X_t.T @ (w[:, None] * X_t)
    XViY = X_t.T @ (w * y_t)
    beta = _solve_safe(XViX, XViY)
    resid = y - X @ beta

    K11_delta = K11 + np.eye(n_pheno) * delta
    alpha = _solve_safe(K11_delta, resid)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u_hat = K_all[:, idx_pheno] @ alpha

    return u_hat, delta, vg, ve


def main():
    pheno_path = REPO_ROOT / "sorghum_data" / "SbDiv_NE2021_Phenos_spats.csv"
    vcf_path = REPO_ROOT / "sorghum_data" / "SbDiv_RNAseq_GeneticMarkers_Mangal2025.vcf.gz"
    out_path = REPO_ROOT / "sorghum_data" / "SbDiv_NE2021_Phenos_spats_GBLUPs.csv"
    summary_path = REPO_ROOT / "sorghum_data" / "SbDiv_NE2021_Phenos_spats_GBLUPs_summary.csv"

    pheno = pd.read_csv(pheno_path).set_index("Genotype")

    geno_matrix, individual_ids, _geno_map = load_genotype_vcf(vcf_path)
    geno = geno_matrix if isinstance(geno_matrix, GenotypeMatrix) else GenotypeMatrix(geno_matrix)
    kinship = MVP_K_VanRaden(geno, verbose=True).to_numpy()

    id_to_idx = {gid: i for i, gid in enumerate(individual_ids)}
    out_df = pd.DataFrame({"Genotype": individual_ids})
    summary_rows = []

    for trait in pheno.columns:
        y_series = pheno[trait]
        pheno_ids = [gid for gid in y_series.index if gid in id_to_idx and pd.notna(y_series[gid])]
        if not pheno_ids:
            out_df[f"{trait}_GBLUP"] = np.nan
            summary_rows.append(
                {"Trait": trait, "N": 0, "Delta": np.nan, "Vg": np.nan, "Ve": np.nan}
            )
            continue

        idx = np.array([id_to_idx[gid] for gid in pheno_ids], dtype=int)
        y = y_series.loc[pheno_ids].to_numpy().astype(float)
        gblup, delta, vg, ve = _fit_gblup_all(y, kinship, idx)
        out_df[f"{trait}_GBLUP"] = gblup

        summary_rows.append(
            {"Trait": trait, "N": len(pheno_ids), "Delta": delta, "Vg": vg, "Ve": ve}
        )

    out_df.to_csv(out_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote GBLUPs: {out_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
