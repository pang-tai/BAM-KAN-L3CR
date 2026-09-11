#!/usr/bin/env python3
"""Verify that repeated full-BAM rows and checkpoint metadata agree."""

from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    main = pd.read_csv(HERE / "07_main_architecture_results.csv")
    isolation = pd.read_csv(HERE / "08_bam_kan_isolation_results.csv")
    ablation = pd.read_csv(HERE / "09_pathway_ablation_results.csv")
    refine = pd.read_csv(HERE / "10_refinement_results.csv")
    checks = []
    metric_columns = ["parameters", "global_score", "hole_score", "balanced_score"]
    for label, frame, model_name in (("E7_full_BAM", isolation, "BAM-KAN"), ("E8_full_BAM", ablation, "BAM-KAN-full"), ("E10_AdamW_start", refine, "BAM-KAN-AdamW")):
        other = frame[frame["model"] == model_name].sort_values("seed").reset_index(drop=True)
        reference = main[main["model"] == "BAM-KAN"].sort_values("seed").reset_index(drop=True)
        equal = len(other) == len(reference) and all(np.allclose(other[col].to_numpy(dtype=float), reference[col].to_numpy(dtype=float), equal_nan=True) for col in metric_columns)
        checks.append({"check": label, "rows_reference": len(reference), "rows_other": len(other), "metrics_equal": bool(equal)})
    for model in ("CNN", "U-Net", "BAM-KAN"):
        for seed in (11, 23, 37, 51, 73):
            path = HERE / "formal_leakage_free" / f"model_{model.lower().replace('-', '_')}_seed{seed}.pt"
            checks.append({"check": "checkpoint_exists", "model": model, "seed": seed, "metrics_equal": bool(path.exists())})
    result = pd.DataFrame(checks)
    result.to_csv(HERE / "09_checkpoint_consistency_audit.csv", index=False)
    print(result.to_string(index=False))
    if not result["metrics_equal"].fillna(False).all():
        raise SystemExit("checkpoint consistency audit failed")


if __name__ == "__main__":
    main()
