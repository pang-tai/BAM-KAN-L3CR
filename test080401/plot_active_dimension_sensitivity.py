#!/usr/bin/env python3
"""Plot the active-dimension accuracy-cost audit with protocol labels."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(HERE / "12_active_dimension_all_summary.csv")
    refined = frame[frame["model"].str.contains("L3CR")].copy()
    colors = {"leakage_free_v1": "#2b7bba", "legacy_test071601": "#d95f02"}
    markers = {"formal_five_seed": "o", "smoke_seed11_one_outer_step": "x", "legacy_formal_five_seed": "s"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for _, row in refined.iterrows():
        x = row["total_wall_seconds"] if "total_wall_seconds" in row else row["refinement_seconds_mean"]
        if pd.isna(x):
            x = row["refinement_seconds_mean"]
        ax.scatter(x, row["balanced_score_mean"], color=colors[row["data_protocol"]], marker=markers[row["result_status"]], s=70)
        ax.annotate(f"r={int(row['active_dimension'])}", (x, row["balanced_score_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean refinement wall time per recorded seed (s)")
    ax.set_ylabel("Balanced score")
    ax.set_title("Active-dimension accuracy-cost audit")
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([], [], marker="o", color="#2b7bba", linestyle="None", label="leakage-free v1"),
        plt.Line2D([], [], marker="s", color="#d95f02", linestyle="None", label="legacy test071601"),
        plt.Line2D([], [], marker="x", color="black", linestyle="None", label="smoke r=128"),
    ]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "12_active_dimension_accuracy_cost.pdf")
    fig.savefig(HERE / "figures/active_dimension_accuracy_cost.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
