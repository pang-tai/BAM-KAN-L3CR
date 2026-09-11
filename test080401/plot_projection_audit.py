#!/usr/bin/env python3
"""Plot the completed raw-node to common-grid projection audit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(HERE / "14_projection_audit.csv")
    regions = ["global", "hole"]
    fields = ["S1", "U"]
    column_names = {"S1": {"global": "S1_nrmse", "hole": "S1_hole_nrmse"}, "U": {"global": "U_nrmse", "hole": "U_hole_nrmse"}}
    means = np.array([[frame[column_names[field][region]].mean() for region in regions] for field in fields])
    stds = np.array([[frame[column_names[field][region]].std() for region in regions] for field in fields])
    hole_count = int(frame["hole_band_defined"].sum())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5), sharey=False)
    colors = ["#d95f02", "#2b7bba"]
    for index, field in enumerate(fields):
        axes[index].bar(regions, means[index], yerr=stds[index], capsize=4, color=colors[index])
        axes[index].set_title(field)
        axes[index].set_ylabel("Projection NRMSE")
        axes[index].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Raw FEM nodal field projection to 64x64x4 and back (hole-band n={hole_count})")
    fig.tight_layout()
    fig.savefig(HERE / "figures/projection_audit_summary.png", dpi=240)
    fig.savefig(HERE / "figures/projection_audit_summary.pdf")
    plt.close(fig)
    summary = pd.DataFrame({"field": fields, "global_mean_nrmse": means[:, 0], "global_std_nrmse": stds[:, 0], "hole_mean_nrmse": means[:, 1], "hole_std_nrmse": stds[:, 1]})
    summary.to_csv(HERE / "tables/projection_audit_field_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
