#!/usr/bin/env python3
"""Add Holm-adjusted Wilcoxon p-values to the paired-statistics tables."""

from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def holm(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main() -> None:
    tables = []
    for name, experiment in (("15_paired_statistics.csv", "E6_main_architecture"), ("15_paired_statistics_extended.csv", None), ("15_refinement_paired_statistics.csv", "E10_clean_L3CR_PGD")):
        frame = pd.read_csv(HERE / name)
        if "experiment" not in frame:
            frame["experiment"] = experiment
        tables.append(frame)
    result = pd.concat(tables, ignore_index=True).drop_duplicates(subset=["comparison", "metric", "experiment"], keep="last")
    result["holm_group"] = result["experiment"].fillna("all") + "::" + result["metric"]
    result["wilcoxon_p_holm"] = np.nan
    for _, index in result.groupby("holm_group").groups.items():
        result.loc[index, "wilcoxon_p_holm"] = holm(result.loc[index, "wilcoxon_p_less"])
    result.to_csv(HERE / "15_paired_statistics_holm.csv", index=False)
    print(result[["comparison", "metric", "experiment", "wilcoxon_p_less", "wilcoxon_p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
