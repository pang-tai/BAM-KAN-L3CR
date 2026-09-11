#!/usr/bin/env python3
"""Audit the physical non-negativity of the reconstructed displacement magnitude."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_leakage_free_architecture_experiment import (  # noqa: E402
    MODEL_ORDER,
    choose_models,
    load_clean_pack,
)
from run_fair_pikan_unet_comparison import Config, choose_device, predict, set_seed  # noqa: E402


def main() -> None:
    config = Config(seeds=(11, 23, 37, 51, 73), epochs=48, target_parameters=90000, device="mps")
    pack, _, _ = load_clean_pack(HERE / "data_clean/leakage_free_v1", HERE.parent / "data_28_06_2026")
    device = choose_device("mps")
    rows = []
    for seed in config.seeds:
        set_seed(seed)
        models, _ = choose_models(config, input_channels=11)
        for name in MODEL_ORDER:
            checkpoint = HERE / "formal_leakage_free" / f"model_{name.lower().replace('-', '_')}_seed{seed}.pt"
            models[name].load_state_dict(__import__("torch").load(checkpoint, map_location="cpu"))
            prediction = predict(models[name].to(device), pack.test_x, device, config.batch_size)
            u_physical = prediction[:, 1] * pack.y_std[1] + pack.y_mean[1]
            material = pack.test_masks["material"][:, 0].astype(bool)
            selected = u_physical[material]
            negative = selected[selected < 0.0]
            rows.append({
                "model": name,
                "seed": seed,
                "minimum_U_m": float(selected.min()),
                "minimum_U_mm": float(selected.min() * 1.0e3),
                "negative_voxel_count": int(len(negative)),
                "material_voxel_count": int(len(selected)),
                "negative_voxel_fraction": float(len(negative) / max(1, len(selected))),
                "negative_p95_magnitude_m": float(np.quantile(np.abs(negative), 0.95)) if len(negative) else 0.0,
                "negative_max_magnitude_m": float(np.max(np.abs(negative))) if len(negative) else 0.0,
                "negative_max_magnitude_mm": float(np.max(np.abs(negative)) * 1.0e3) if len(negative) else 0.0,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "tables/19_displacement_nonnegative_audit.csv", index=False)
    frame.to_csv(HERE / "19_displacement_nonnegative_audit.csv", index=False)
    frame.groupby("model").agg(["mean", "std"]).to_csv(HERE / "tables/19_displacement_nonnegative_summary.csv")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
