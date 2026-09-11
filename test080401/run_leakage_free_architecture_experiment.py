#!/usr/bin/env python3
"""Leakage-free CNN/U-Net/BAM-KAN experiment on the cleaned legacy tensor data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "test071407"
FAIR = HERE.parent / "test071405"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(FAIR))

from run_bampikan_unet_cnn_comparison import BAMPIKAN, FairCNN  # noqa: E402
from run_fair_pikan_unet_comparison import (  # noqa: E402
    Config,
    DataPack,
    FairUNet,
    choose_device,
    normalized_average_mse,
    parameter_count,
    predict,
    set_seed,
    split_train_validation,
    train_model,
    to_ncdhw,
)


MODEL_ORDER = ("CNN", "U-Net", "BAM-KAN")
RETAINED_INPUTS = [2, 3, 5, 6, 7, 8, 9, 11]
TARGETS = [0, 1]


def internal_void_mask(material_2d: np.ndarray) -> np.ndarray:
    void = ~material_2d.astype(bool)
    labels, _ = label(void)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    internal = void.copy()
    for value in border:
        internal &= labels != value
    return internal


def masks_from_raw(raw: np.ndarray) -> Dict[str, np.ndarray]:
    material = raw[..., 3] > 0.0
    near = np.zeros_like(material, dtype=bool)
    for sample in range(raw.shape[0]):
        for z_index in range(raw.shape[3]):
            internal = internal_void_mask(material[sample, :, :, z_index])
            near[sample, :, :, z_index] = binary_dilation(internal, np.ones((5, 5), bool)) & material[sample, :, :, z_index]
    high = np.zeros_like(material, dtype=bool)
    for sample in range(raw.shape[0]):
        values = raw[sample, ..., 0][material[sample]]
        threshold = np.quantile(values, 0.90) if values.size else 0.0
        high[sample] = material[sample] & (raw[sample, ..., 0] >= threshold)
    return {
        "material": material[:, None].transpose(0, 1, 4, 2, 3).astype(np.float32),
        "hole": near[:, None].transpose(0, 1, 4, 2, 3).astype(np.float32),
        "high": high[:, None].transpose(0, 1, 4, 2, 3).astype(np.float32),
    }


def coordinates(count: int, shape: Tuple[int, int, int]) -> np.ndarray:
    xx, yy, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    coord = np.stack([xx, yy, zz], axis=-1)
    return np.broadcast_to(coord[None], (count, *coord.shape)).copy()


def load_clean_pack(clean_dir: Path, raw_dir: Path) -> Tuple[DataPack, np.ndarray, np.ndarray]:
    train_raw = np.load(raw_dir / "train.npy").astype(np.float32)
    test_raw = np.load(raw_dir / "test.npy").astype(np.float32)
    train_x_raw = np.load(clean_dir / "train_inputs.npy").astype(np.float32)
    test_x_raw = np.load(clean_dir / "test_inputs.npy").astype(np.float32)
    train_y_raw = np.load(clean_dir / "train_targets.npy").astype(np.float32)
    test_y_raw = np.load(clean_dir / "test_targets.npy").astype(np.float32)
    stats = json.loads((clean_dir / "normalization.json").read_text(encoding="utf-8"))
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    train_x = (train_x_raw - x_mean) / x_std
    test_x = (test_x_raw - x_mean) / x_std
    train_x = np.concatenate([train_x, coordinates(len(train_x), train_x.shape[1:4])], axis=-1)
    test_x = np.concatenate([test_x, coordinates(len(test_x), test_x.shape[1:4])], axis=-1)
    train_y = (train_y_raw - y_mean) / y_std
    test_y = (test_y_raw - y_mean) / y_std
    train_masks = masks_from_raw(train_raw)
    test_masks = masks_from_raw(test_raw)
    # Keep the name used by the verified L3CR closure implementation while
    # retaining the cleaner ``hole`` name for the architecture metrics.
    train_masks["near_opening"] = train_masks["hole"]
    test_masks["near_opening"] = test_masks["hole"]
    return (
        DataPack(
            train_x=to_ncdhw(train_x),
            train_y=to_ncdhw(train_y),
            test_x=to_ncdhw(test_x),
            test_y=to_ncdhw(test_y),
            train_masks=train_masks,
            test_masks=test_masks,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        ),
        train_y_raw,
        test_y_raw,
    )


def choose_models(config: Config, input_channels: int) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, int]]]:
    candidates_unet = [(abs(parameter_count(FairUNet(input_channels, width)) - config.target_parameters), width) for width in range(8, 33)]
    _, unet_width = min(candidates_unet)
    unet = FairUNet(input_channels, unet_width)
    target = parameter_count(unet)
    candidates_cnn = [(abs(parameter_count(FairCNN(input_channels, width)) - target), width) for width in range(8, 40)]
    _, cnn_width = min(candidates_cnn)
    cnn = FairCNN(input_channels, cnn_width)
    candidates_bam = []
    for spatial in range(8, 25):
        for fine in range(6, 19, 2):
            for coarse in range(8, 23, 2):
                model = BAMPIKAN(8, 3, spatial, fine, coarse)
                candidates_bam.append((abs(parameter_count(model) - target), spatial, fine, coarse))
    _, spatial, fine, coarse = min(candidates_bam)
    bam = BAMPIKAN(8, 3, spatial, fine, coarse)
    models = {"CNN": cnn, "U-Net": unet, "BAM-KAN": bam}
    specs = {
        "CNN": {"width": cnn_width, "parameters": parameter_count(cnn)},
        "U-Net": {"width": unet_width, "parameters": parameter_count(unet)},
        "BAM-KAN": {"spatial_width": spatial, "fine_kan_width": fine, "coarse_kan_width": coarse, "parameters": parameter_count(bam)},
    }
    counts = [item["parameters"] for item in specs.values()]
    mismatch = (max(counts) - min(counts)) / max(counts)
    if mismatch > config.parameter_tolerance:
        raise RuntimeError(f"parameter mismatch {mismatch:.3%} exceeds tolerance")
    return models, specs


def masked_rmse(error: np.ndarray, mask: np.ndarray) -> float:
    selected = error[mask.astype(bool)]
    return float(np.sqrt(np.mean(selected ** 2))) if selected.size else float("nan")


def evaluate_clean(prediction: np.ndarray, target: np.ndarray, masks: Dict[str, np.ndarray], y_std: np.ndarray) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    error = prediction - target
    metrics: Dict[str, float] = {}
    regions = {"global": masks["material"], "hole": masks["hole"], "high": masks["high"]}
    per_sample: List[Dict[str, float]] = []
    for region, mask in regions.items():
        for channel, name in enumerate(("S1", "U")):
            value = masked_rmse(error[:, channel : channel + 1], mask)
            metrics[f"{name}_{region}_NRMSE"] = value
            metrics[f"{name}_{region}_RMSE"] = value * float(y_std[channel])
    metrics["global_score"] = 0.5 * (metrics["S1_global_NRMSE"] + metrics["U_global_NRMSE"])
    metrics["hole_score"] = 0.5 * (metrics["S1_hole_NRMSE"] + metrics["U_hole_NRMSE"])
    metrics["balanced_score"] = 0.5 * (metrics["global_score"] + metrics["hole_score"])
    for sample in range(len(prediction)):
        row = {"sample": sample}
        for region, mask in regions.items():
            for channel, name in enumerate(("S1", "U")):
                row[f"{name}_{region}_NRMSE"] = masked_rmse(error[sample : sample + 1, channel : channel + 1], mask[sample : sample + 1])
        row["global_score"] = 0.5 * (row["S1_global_NRMSE"] + row["U_global_NRMSE"])
        row["hole_score"] = 0.5 * (row["S1_hole_NRMSE"] + row["U_hole_NRMSE"])
        row["balanced_score"] = 0.5 * (row["global_score"] + row["hole_score"])
        per_sample.append(row)
    return metrics, per_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data_28_06_2026")
    parser.add_argument("--clean-dir", type=Path, default=HERE / "data_clean/leakage_free_v1")
    parser.add_argument("--out-dir", type=Path, default=HERE / "smoke")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(seeds=tuple(args.seeds), epochs=args.epochs, batch_size=8, patience=max(2, min(14, args.epochs)), bootstrap_repetitions=100, device=args.device, target_parameters=90000)
    set_seed(config.seeds[0])
    pack, _, _ = load_clean_pack(args.clean_dir, args.data_dir)
    models, specs = choose_models(config, input_channels=11)
    (args.out_dir / "model_specifications.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    device = choose_device(args.device)
    metric_rows: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []
    for seed in config.seeds:
        for name in MODEL_ORDER:
            set_seed(seed)
            current_models, current_specs = choose_models(config, input_channels=11)
            started = time.perf_counter()
            trained, history, diagnostics = train_model(name, current_models[name], seed, pack, config, device)
            prediction = predict(trained, pack.test_x, device, config.batch_size)
            metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
            metric_rows.append({"model": name, "seed": seed, "parameters": current_specs[name]["parameters"], **diagnostics, "wall_seconds": time.perf_counter() - started, **metrics})
            sample_rows.extend({"model": name, "seed": seed, **row} for row in per_sample)
            history_rows.extend(history)
            torch.save(trained.state_dict(), args.out_dir / f"model_{name.lower().replace('-', '_')}_seed{seed}.pt")
            print(f"[result] {name} seed={seed} global={metrics['global_score']:.6f} hole={metrics['hole_score']:.6f} balanced={metrics['balanced_score']:.6f}", flush=True)
    metrics_frame = pd.DataFrame(metric_rows)
    samples_frame = pd.DataFrame(sample_rows)
    history_frame = pd.DataFrame(history_rows)
    metrics_frame.to_csv(args.out_dir / "metrics_per_seed.csv", index=False)
    samples_frame.to_csv(args.out_dir / "metrics_per_sample.csv", index=False)
    history_frame.to_csv(args.out_dir / "training_history.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(args.out_dir / "metrics_summary.csv")
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
