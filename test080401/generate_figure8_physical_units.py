#!/usr/bin/env python3
"""Create a physical-unit Figure 8 from the saved BAM-KAN checkpoints.

The representative case is selected before plotting as the median Balanced
error among the three test cases whose public-grid geometry mask contains a
hole band. No test metric is used to tune a model or a color scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "test071407"
FAIR = HERE.parent / "test071405"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(FAIR))

from run_leakage_free_architecture_experiment import (  # noqa: E402
    Config,
    choose_models,
    evaluate_clean,
    load_clean_pack,
    predict,
)


def load_bam(checkpoint: Path, device: torch.device):
    config = Config(
        seeds=(11,), epochs=48, batch_size=8, patience=14,
        bootstrap_repetitions=100, device="cpu", target_parameters=90000,
    )
    model = choose_models(config, input_channels=11)[0]["BAM-KAN"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def masked_limits(values: np.ndarray, mask: np.ndarray, error: bool = False) -> tuple[float, float]:
    selected = values[mask.astype(bool)] if mask.any() else values.reshape(-1)
    if error:
        selected = np.abs(selected)
        return 0.0, float(np.quantile(selected, 0.99))
    return float(np.quantile(selected, 0.01)), float(np.quantile(selected, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--z-index", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    device = torch.device("cpu")
    clean = HERE / "data_clean/leakage_free_v1"
    raw = HERE.parent / "data_28_06_2026"
    pack, _, test_target_raw = load_clean_pack(clean, raw)
    test_target = np.transpose(test_target_raw, (0, 4, 3, 1, 2))
    baseline = load_bam(HERE / "formal_leakage_free/model_bam_kan_seed11.pt", device)
    refined = load_bam(HERE / "formal_clean_l3cr_pgd/model_bam_kan_l3cr_pgd_seed11.pt", device)
    with torch.no_grad():
        baseline_std = predict(baseline, pack.test_x, device, batch_size=8)
        refined_std = predict(refined, pack.test_x, device, batch_size=8)
    baseline_phys = baseline_std * pack.y_std.reshape(1, 2, 1, 1, 1) + pack.y_mean.reshape(1, 2, 1, 1, 1)
    refined_phys = refined_std * pack.y_std.reshape(1, 2, 1, 1, 1) + pack.y_mean.reshape(1, 2, 1, 1, 1)
    target_phys = test_target
    _, baseline_per_case = evaluate_clean(baseline_std, pack.test_y, pack.test_masks, pack.y_std)
    opening_cases = np.flatnonzero(pack.test_masks["hole"].sum(axis=(1, 2, 3, 4)) > 0)
    if len(opening_cases) == 0:
        raise RuntimeError("No test case has a geometry-defined hole band")
    order = sorted(opening_cases.tolist(), key=lambda i: baseline_per_case[i]["balanced_score"])
    sample = order[len(order) // 2]
    z = min(max(args.z_index, 0), target_phys.shape[2] - 1)
    fields = [(0, "S1", "MPa", 1.0e-6), (1, "U", "mm", 1.0e3)]
    fig, axes = plt.subplots(2, 6, figsize=(19, 7.1), constrained_layout=True)
    column_titles = ["FEM target", "BAM-KAN + AdamW", "L3CR-PGD", "AdamW error", "L3CR error", "difference"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)
    hole = pack.test_masks["hole"][sample, 0, z].astype(bool)
    material = pack.test_masks["material"][sample, 0, z].astype(bool)
    for row, (channel, name, unit, scale) in enumerate(fields):
        target = target_phys[sample, channel, z] * scale
        adamw = baseline_phys[sample, channel, z] * scale
        pgd = refined_phys[sample, channel, z] * scale
        adam_error = np.abs(adamw - target)
        pgd_error = np.abs(pgd - target)
        difference = pgd_error - adam_error
        for arr in (target, adamw, pgd, adam_error, pgd_error, difference):
            arr[~material] = np.nan
        data_limits = masked_limits(target, material)
        error_limit = max(masked_limits(adam_error, material, error=True)[1], masked_limits(pgd_error, material, error=True)[1], 1e-12)
        diff_limit = max(float(np.nanquantile(np.abs(difference), 0.99)), 1e-12)
        images = [target, adamw, pgd, adam_error, pgd_error, difference]
        for col, arr in enumerate(images):
            ax = axes[row, col]
            if col < 3:
                im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=data_limits[0], vmax=data_limits[1])
            elif col < 5:
                im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0.0, vmax=error_limit)
            else:
                im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
            ax.contour(hole, levels=[0.5], colors="white", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{name} ({unit})", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Physical-unit field comparison, test case {sample}, z-slice {z}; hole-band median case", fontsize=12)
    output = args.out_dir / "17_figure8_physical_units.pdf"
    fig.savefig(output)
    fig.savefig(args.out_dir / "figures/figure8_physical_units.png", dpi=240)
    plt.close(fig)
    metadata = {
        "sample": int(sample), "z_index": int(z), "opening_cases": opening_cases.tolist(),
        "selection": "median Balanced error among test cases with a geometry-defined hole band",
        "checkpoint_adamw": str(HERE / "formal_leakage_free/model_bam_kan_seed11.pt"),
        "checkpoint_l3cr_pgd": str(HERE / "formal_clean_l3cr_pgd/model_bam_kan_l3cr_pgd_seed11.pt"),
        "units": {"S1": "MPa", "U": "mm"},
        "hole_mask": "geometry-derived; non-material voxels are NaN",
    }
    (args.out_dir / "figure8_physical_units_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
