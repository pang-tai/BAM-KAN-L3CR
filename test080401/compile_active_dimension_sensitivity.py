#!/usr/bin/env python3
"""Compile the primary leakage-free active-dimension sensitivity table."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "test071601"


def main() -> None:
    frames = []
    sources = {
        12: (HERE / "formal_clean_l3cr_pgd/metrics_per_seed.csv", "formal_five_seed"),
        16: (HERE / "formal_clean_l3cr_k16/metrics_per_seed.csv", "formal_five_seed"),
        32: (HERE / "formal_clean_l3cr_k32/metrics_per_seed.csv", "formal_five_seed"),
        64: (HERE / "formal_clean_l3cr_k64/metrics_per_seed.csv", "formal_five_seed"),
        128: (HERE / "pilot_clean_l3cr_k128/metrics_per_seed.csv", "smoke_seed11_one_outer_step"),
    }
    for active_dimension, (source, result_status) in sources.items():
        frame = pd.read_csv(source).copy()
        frame.insert(0, "source_label", f"leakage_free_v1_r{active_dimension}")
        frame.insert(1, "active_dimension", active_dimension)
        frame["data_protocol"] = "leakage_free_v1"
        frame["result_status"] = result_status
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(HERE / "12_active_dimension_per_seed.csv", index=False)
    summary = result.groupby(["active_dimension", "model"], as_index=False).agg(
        n_seeds=("seed", "count"), global_score_mean=("global_score", "mean"), global_score_std=("global_score", "std"),
        hole_score_mean=("hole_score", "mean"), hole_score_std=("hole_score", "std"),
        balanced_score_mean=("balanced_score", "mean"), balanced_score_std=("balanced_score", "std"),
        accepted_steps_mean=("accepted_steps", "mean"), refinement_seconds_mean=("refinement_seconds", "mean"),
    )
    summary.to_csv(HERE / "12_active_dimension_sensitivity.csv", index=False)
    legacy_frames = []
    for active_dimension in (32, 64):
        source = LEGACY / f"formal_k{active_dimension}/metrics_per_seed.csv"
        frame = pd.read_csv(source)
        frame = frame[frame["model"].isin(["BAM-PIKAN-AdamW", f"BAM-AdamW-L3CR-PGD-k{active_dimension}"])].copy()
        frame.insert(0, "source_label", f"legacy_test071601_formal_k{active_dimension}")
        frame.insert(1, "active_dimension", active_dimension)
        frame["data_protocol"] = "legacy_test071601"
        frame["result_status"] = "legacy_formal_five_seed"
        legacy_frames.append(frame)
    pd.concat(legacy_frames, ignore_index=True).to_csv(HERE / "12_active_dimension_legacy_per_seed.csv", index=False)
    (HERE / "12_active_dimension_metadata.md").write_text(
        "# Active-dimension sensitivity\n\n"
        "The primary table now contains leakage-free results for `r=12/16/32/64` with five seeds and a `r=128` seed-11 smoke result. "
        "The separate `12_active_dimension_legacy_per_seed.csv` preserves reusable legacy `r=32/64` results for protocol comparison. "
        "The `r=128` row is not treated as a formal five-seed conclusion.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
