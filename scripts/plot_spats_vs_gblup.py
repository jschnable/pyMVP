#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _r2_from_corr(x, y):
    if x.size < 2:
        return np.nan
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return np.nan
    corr = np.corrcoef(x, y)[0, 1]
    return corr * corr


def main():
    repo_root = Path(__file__).resolve().parents[1]
    spats_path = repo_root / "sorghum_data" / "SbDiv_NE2021_Phenos_spats.csv"
    gblup_path = repo_root / "sorghum_data" / "SbDiv_NE2021_Phenos_spats_GBLUPs.csv"
    out_dir = repo_root / "sorghum_data" / "spats_vs_gblup_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "spats_vs_gblup_r2.csv"

    spats = pd.read_csv(spats_path)
    gblup = pd.read_csv(gblup_path)

    merged = spats.merge(gblup, on="Genotype", how="inner")
    traits = [c for c in spats.columns if c != "Genotype"]

    summary_rows = []

    for trait in traits:
        gblup_col = f"{trait}_GBLUP"
        if gblup_col not in merged.columns:
            continue
        sub = merged[[trait, gblup_col]].dropna()
        x = sub[trait].to_numpy(dtype=float)
        y = sub[gblup_col].to_numpy(dtype=float)
        r2 = _r2_from_corr(x, y)

        summary_rows.append(
            {"Trait": trait, "N": int(sub.shape[0]), "R2": r2}
        )

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"matplotlib not available: {exc}")
            sys.exit(1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=18, alpha=0.7, edgecolors="none")
        ax.set_xlabel(f"{trait} (SpATS)")
        ax.set_ylabel(f"{trait} (GBLUP)")
        ax.set_title(f"{trait}: SpATS vs GBLUP")

        if x.size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()], dtype=float)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=1, alpha=0.7)

        ax.text(
            0.05,
            0.95,
            f"N={sub.shape[0]}\nR^2={r2:.3f}" if np.isfinite(r2) else f"N={sub.shape[0]}\nR^2=NA",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

        out_file = out_dir / f"spats_vs_gblup_{trait}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote plots to {out_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
