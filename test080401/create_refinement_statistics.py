#!/usr/bin/env python3
"""Add Wilcoxon statistics for the new clean AdamW-to-PGD continuation."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(HERE / "formal_clean_l3cr_pgd/metrics_per_sample.csv")
    pivot = frame.pivot_table(index=["seed", "sample"], columns="model", values=["global_score", "hole_score", "balanced_score"])
    rows = []
    refined_name = "BAM-KAN-AdamW-L3CR-PGD"
    baseline_name = "BAM-KAN-AdamW"
    for metric in ("global_score", "hole_score", "balanced_score"):
        a = pivot[(metric, refined_name)]
        b = pivot[(metric, baseline_name)]
        valid = a.notna() & b.notna()
        difference = (a[valid] - b[valid]).to_numpy(dtype=float)
        statistic, p_value = wilcoxon(difference, alternative="less", zero_method="wilcox", method="auto")
        rows.append({
            "comparison": f"{refined_name} - {baseline_name}", "metric": metric,
            "mean_difference": float(np.mean(difference)),
            "ci95_lower": np.nan, "ci95_upper": np.nan,
            "win_rate": float(np.mean(difference < 0.0)), "paired_observations": int(valid.sum()),
            "wilcoxon_statistic": float(statistic), "wilcoxon_p_less": float(p_value),
            "experiment": "E10_clean_L3CR_PGD",
            "bootstrap_source": str(HERE / "formal_clean_l3cr_pgd/paired_bootstrap.csv"),
        })
    result = pd.DataFrame(rows)
    bootstrap = pd.read_csv(HERE / "formal_clean_l3cr_pgd/paired_bootstrap.csv")
    result = result.drop(columns=["ci95_lower", "ci95_upper"]).merge(
        bootstrap[["metric", "ci95_lower", "ci95_upper"]], on="metric", how="left"
    )
    result.to_csv(HERE / "10_refinement_paired_statistics.csv", index=False)
    result.to_csv(HERE / "15_refinement_paired_statistics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
