#!/usr/bin/env python3
"""Compute diagnostic subgroup means from the saved leakage-free predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label


HERE = Path(__file__).resolve().parent


def has_internal_hole(material: np.ndarray) -> bool:
    for z in range(material.shape[2]):
        void = ~material[:, :, z]
        labels, _ = label(void)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        internal = void.copy()
        for value in border:
            internal &= labels != value
        if internal.any():
            return True
    return False


def main() -> None:
    per_case = pd.read_csv(HERE / "formal_leakage_free/metrics_per_sample.csv")
    metadata = pd.read_csv(HERE / "data_audit/case_metadata.csv")
    test_meta = metadata[metadata["split"] == "test"].sort_values("sample").copy()
    if len(test_meta) != 19 or set(test_meta["sample"]) != set(per_case["sample"]):
        raise RuntimeError("test metadata and saved per-sample predictions do not align")
    raw = np.load(HERE.parent / "data_28_06_2026" / "test.npy").astype(np.float32)
    material = raw[..., 3] > 0.0
    has_opening = []
    for sample in range(len(raw)):
        has_opening.append(has_internal_hole(material[sample]))
    test_meta["has_rebar"] = (test_meta["reinforcement_ratio_pct"].fillna(0.0) > 0.0).astype(int)
    test_meta["opening_detected_from_tensor_mask"] = [int(value) for value in has_opening]
    test_meta["grid_size_group"] = test_meta["grid_size_m"].fillna("missing").astype(str)
    test_meta["temperature_group"] = pd.cut(test_meta["temperature_c"], bins=[-np.inf, 15, 25, np.inf], labels=["<=15C", "15-25C", ">25C"]).astype(str)
    merged = per_case.merge(test_meta[["sample", "case_family", "has_rebar", "opening_detected_from_tensor_mask", "grid_size_group", "temperature_group", "geometry_group"]], on="sample", how="left")
    rows = []
    metrics = ["global_score", "hole_score", "balanced_score", "S1_global_NRMSE", "U_global_NRMSE", "S1_hole_NRMSE", "U_hole_NRMSE"]
    for model in sorted(merged["model"].unique()):
        current = merged[merged["model"] == model]
        for group_by in ("case_family", "has_rebar", "opening_detected_from_tensor_mask", "grid_size_group", "temperature_group", "geometry_group"):
            for group, subset in current.groupby(group_by, dropna=False):
                row = {"source_label": "leakage_free_v1_saved_predictions", "model": model, "group_by": group_by, "group": str(group), "n_test_cases": int(subset["sample"].nunique()), "n_seed_case_observations": len(subset), "inference": "diagnostic if n_test_cases<5"}
                row.update({f"{metric}_mean": float(subset[metric].mean()) for metric in metrics})
                row.update({f"{metric}_std": float(subset[metric].std(ddof=1)) if len(subset) > 1 else float("nan") for metric in metrics})
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "12_subgroup_results_clean.csv", index=False)
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
