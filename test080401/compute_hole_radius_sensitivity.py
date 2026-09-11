#!/usr/bin/env python3
"""Re-evaluate saved leakage-free checkpoints under fixed hole-band radii."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, label


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "test071407"))
sys.path.insert(0, str(HERE.parent / "test071405"))

from run_leakage_free_architecture_experiment import (  # noqa: E402
    Config,
    choose_models,
    load_clean_pack,
    predict,
)


def internal_void(material: np.ndarray) -> np.ndarray:
    void = ~material.astype(bool)
    labels, _ = label(void)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    internal = void.copy()
    for value in border:
        internal &= labels != value
    return internal


def make_masks(raw: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    material = raw[..., 3] > 0.0
    hole = np.zeros_like(material, dtype=bool)
    for sample in range(raw.shape[0]):
        for z in range(raw.shape[3]):
            internal = internal_void(material[sample, :, :, z])
            hole[sample, :, :, z] = binary_dilation(internal, np.ones((2 * radius + 1, 2 * radius + 1), bool)) & material[sample, :, :, z]
    material_ncdhw = material[:, None].transpose(0, 1, 4, 2, 3)
    hole_ncdhw = hole[:, None].transpose(0, 1, 4, 2, 3)
    return material_ncdhw, hole_ncdhw


def main() -> None:
    seeds = (11, 23, 37, 51, 73)
    config = Config(seeds=seeds, epochs=48, batch_size=8, patience=14, bootstrap_repetitions=100, device="cpu", target_parameters=90000)
    data_dir = HERE.parent / "data_28_06_2026"
    pack, _, _ = load_clean_pack(HERE / "data_clean/leakage_free_v1", data_dir)
    raw_test = np.load(data_dir / "test.npy").astype(np.float32)
    y_std = pack.y_std.reshape(1, 2, 1, 1, 1)
    rows = []
    for model_name in ("CNN", "U-Net", "BAM-KAN"):
        for seed in seeds:
            model = choose_models(config, input_channels=11)[0][model_name]
            checkpoint = HERE / "formal_leakage_free" / f"model_{model_name.lower().replace('-', '_')}_seed{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            prediction = predict(model.eval(), pack.test_x, torch.device("cpu"), config.batch_size)
            error = prediction - pack.test_y
            for radius in (1, 2, 3, 4):
                material, hole = make_masks(raw_test, radius)
                global_scores = []
                hole_scores = []
                hole_rmse = []
                for channel in range(2):
                    global_values = error[:, channel : channel + 1][material.astype(bool)]
                    values = error[:, channel : channel + 1][hole.astype(bool)]
                    global_scores.append(float(np.sqrt(np.mean(global_values ** 2))))
                    hole_scores.append(float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan"))
                    hole_rmse.append(float(hole_scores[-1] * pack.y_std[channel]))
                global_score = float(np.mean(global_scores))
                hole_score = float(np.mean(hole_scores))
                rows.append({
                    "source_label": "leakage_free_v1_formal_checkpoints",
                    "model": model_name, "seed": seed, "boundary_radius_voxels": radius,
                    "global_score": float(global_score), "hole_score": hole_score,
                    "balanced_score": float(0.5 * (global_score + hole_score)),
                    "S1_hole_NRMSE": hole_scores[0], "U_hole_NRMSE": hole_scores[1],
                    "S1_hole_RMSE": hole_rmse[0], "U_hole_RMSE": hole_rmse[1],
                    "hole_voxels": int(hole.sum()), "test_cases_with_hole_band": int((hole.sum(axis=(1, 2, 3, 4)) > 0).sum()),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "13_hole_radius_per_seed.csv", index=False)
    summary = frame.groupby(["model", "boundary_radius_voxels"], as_index=False).agg(
        n_seeds=("seed", "count"), global_score_mean=("global_score", "mean"), global_score_std=("global_score", "std"),
        hole_score_mean=("hole_score", "mean"), hole_score_std=("hole_score", "std"),
        balanced_score_mean=("balanced_score", "mean"), balanced_score_std=("balanced_score", "std"),
        S1_hole_RMSE_mean=("S1_hole_RMSE", "mean"), U_hole_RMSE_mean=("U_hole_RMSE", "mean"),
        hole_voxels=("hole_voxels", "first"), test_cases_with_hole_band=("test_cases_with_hole_band", "first"),
    )
    summary.to_csv(HERE / "13_hole_radius_results.csv", index=False)
    (HERE / "13_hole_radius_metadata.json").write_text(json.dumps({"models": ["CNN", "U-Net", "BAM-KAN"], "seeds": list(seeds), "radii": [1, 2, 3, 4], "dataset": "leakage_free_v1", "note": "Global score is invariant to radius; hole and Balanced are re-evaluated from saved predictions."}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
