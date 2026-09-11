#!/usr/bin/env python3
"""Combine active-dimension results while retaining protocol/status labels."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent


def normalize(frame: pd.DataFrame, active: int, protocol: str, status: str) -> pd.DataFrame:
    columns = ["seed", "model", "global_score", "hole_score", "balanced_score", "accepted_steps", "refinement_seconds"]
    result = frame[[column for column in columns if column in frame]].copy()
    result["active_dimension"] = active
    result["data_protocol"] = protocol
    result["result_status"] = status
    return result


def main() -> None:
    frames = []
    clean12 = pd.read_csv(HERE / "formal_clean_l3cr_pgd/metrics_per_seed.csv")
    frames.append(normalize(clean12, 12, "leakage_free_v1", "formal_five_seed"))
    clean16 = pd.read_csv(HERE / "formal_clean_l3cr_k16/metrics_per_seed.csv")
    frames.append(normalize(clean16, 16, "leakage_free_v1", "formal_five_seed"))
    clean32 = pd.read_csv(HERE / "formal_clean_l3cr_k32/metrics_per_seed.csv")
    frames.append(normalize(clean32, 32, "leakage_free_v1", "formal_five_seed"))
    clean64 = pd.read_csv(HERE / "formal_clean_l3cr_k64/metrics_per_seed.csv")
    frames.append(normalize(clean64, 64, "leakage_free_v1", "formal_five_seed"))
    smoke128 = pd.read_csv(HERE / "pilot_clean_l3cr_k128/metrics_per_seed.csv")
    frames.append(normalize(smoke128, 128, "leakage_free_v1", "smoke_seed11_one_outer_step"))
    for active in (32, 64):
        legacy = pd.read_csv(HERE.parent / "test071601" / f"formal_k{active}/metrics_per_seed.csv")
        legacy = legacy[legacy["model"].isin(["BAM-PIKAN-AdamW", f"BAM-AdamW-L3CR-PGD-k{active}"])]
        frames.append(normalize(legacy, active, "legacy_test071601", "legacy_formal_five_seed"))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(HERE / "12_active_dimension_all_per_seed.csv", index=False)
    summary = result.groupby(["active_dimension", "model", "data_protocol", "result_status"], as_index=False).agg(
        n_rows=("seed", "count"), n_seeds=("seed", "nunique"),
        global_score_mean=("global_score", "mean"), global_score_std=("global_score", "std"),
        hole_score_mean=("hole_score", "mean"), hole_score_std=("hole_score", "std"),
        balanced_score_mean=("balanced_score", "mean"), balanced_score_std=("balanced_score", "std"),
        accepted_steps_mean=("accepted_steps", "mean"), refinement_seconds_mean=("refinement_seconds", "mean"),
    )
    summary.to_csv(HERE / "12_active_dimension_all_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
