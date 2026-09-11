#!/usr/bin/env python3
"""Create a transparent legacy-input versus leakage-free BAM comparison."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    legacy = pd.read_csv(ROOT / "test071407/formal/metrics_per_seed.csv")
    legacy = legacy[legacy["model"].str.contains("BAM")].copy()
    legacy["source_label"] = "legacy_inputs_test071407"
    clean = pd.read_csv(HERE / "formal_leakage_free/metrics_per_seed.csv")
    clean = clean[clean["model"] == "BAM-KAN"].copy()
    clean["source_label"] = "leakage_free_v1"
    columns = ["source_label", "model", "seed", "parameters", "global_score", "hole_score", "balanced_score"]
    result = pd.concat([legacy[columns], clean[columns]], ignore_index=True)
    result.to_csv(HERE / "legacy_vs_leakage_free_bam.csv", index=False)
    summary = result.groupby("source_label", as_index=False).agg(
        n_seeds=("seed", "count"), global_score_mean=("global_score", "mean"), global_score_std=("global_score", "std"),
        hole_score_mean=("hole_score", "mean"), hole_score_std=("hole_score", "std"),
        balanced_score_mean=("balanced_score", "mean"), balanced_score_std=("balanced_score", "std"),
    )
    summary.to_csv(HERE / "legacy_vs_leakage_free_bam_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
