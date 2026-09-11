#!/usr/bin/env python3
"""Formal ablation of the BAM local and multi-scale paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "test071407"
FAIR = HERE.parent / "test071405"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(FAIR))

from run_leakage_free_architecture_experiment import evaluate_clean, load_clean_pack  # noqa: E402
from run_bampikan_unet_cnn_comparison import BAMPIKAN, parameter_count  # noqa: E402
from run_fair_pikan_unet_comparison import Config, choose_device, predict, set_seed, train_model  # noqa: E402


class BAMPathAblation(BAMPIKAN):
    def __init__(self, ablation: str) -> None:
        super().__init__(8, 3, 19, 12, 12, grid_size=5)
        self.ablation = ablation

    def forward(self, inputs: Tensor) -> Tensor:
        physical = self.physical_stem(inputs[:, : self.physical_channels])
        coordinates = self.coordinate_stem(inputs[:, self.physical_channels :])
        concatenated = torch.cat([physical, coordinates], dim=1)
        local = concatenated if self.ablation == "no-local" else self.local(concatenated)

        coarse = F.avg_pool3d(local, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        coarse = self.coarse_spatial(self.coarse_projection(coarse))

        fine = self.fine_projection(local)
        fine = F.avg_pool3d(fine, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        fine = self._apply_kan(fine, self.fine_kan)
        fine = F.interpolate(fine, size=local.shape[-3:], mode="trilinear", align_corners=False)
        if self.ablation == "no-fine":
            fine = torch.zeros_like(fine)

        coarse_kan = self.coarse_kan_projection(coarse)
        coarse_kan = F.avg_pool3d(coarse_kan, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        coarse_kan = self._apply_kan(coarse_kan, self.coarse_kan)
        coarse_kan = F.interpolate(coarse_kan, size=local.shape[-3:], mode="trilinear", align_corners=False)
        if self.ablation == "no-coarse":
            coarse_kan = torch.zeros_like(coarse_kan)

        fused = self.fusion(torch.cat([local, fine, coarse_kan], dim=1))
        s1 = self.s1_head(torch.cat([fused, fine], dim=1))
        displacement = self.u_head(torch.cat([fused, coarse_kan], dim=1))
        return torch.cat([s1, displacement], dim=1)


VARIANTS = ("BAM-KAN-no-local", "BAM-KAN-no-fine", "BAM-KAN-no-coarse")


def variant_key(name: str) -> str:
    return name.removeprefix("BAM-KAN-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data_28_06_2026")
    parser.add_argument("--clean-dir", type=Path, default=HERE / "data_clean/leakage_free_v1")
    parser.add_argument("--out-dir", type=Path, default=HERE / "formal_path_ablation")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 73])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    config = Config(seeds=tuple(args.seeds), epochs=args.epochs, batch_size=8, patience=args.patience, bootstrap_repetitions=100, device=args.device, target_parameters=90000)
    pack, _, _ = load_clean_pack(args.clean_dir.resolve(), args.data_dir.resolve())
    device = choose_device(args.device)
    reference = pd.read_csv(HERE / "formal_leakage_free/metrics_per_seed.csv")
    reference_samples = pd.read_csv(HERE / "formal_leakage_free/metrics_per_sample.csv")
    reference = reference[reference["model"] == "BAM-KAN"].copy()
    reference_samples = reference_samples[reference_samples["model"] == "BAM-KAN"].copy()
    reference["model"] = "BAM-KAN-full"
    reference_samples["model"] = "BAM-KAN-full"
    metric_rows = reference.to_dict("records")
    sample_rows = reference_samples.to_dict("records")
    history_rows: List[Dict[str, float]] = []
    specifications = {name: parameter_count(BAMPathAblation(variant_key(name))) for name in VARIANTS}
    specifications["BAM-KAN-full"] = int(reference["parameters"].iloc[0])
    (outdir / "model_specifications.json").write_text(json.dumps(specifications, indent=2), encoding="utf-8")

    for seed in args.seeds:
        for name in VARIANTS:
            set_seed(seed)
            model = BAMPathAblation(variant_key(name))
            started = time.perf_counter()
            trained, history, diagnostics = train_model(name, model, seed, pack, config, device)
            prediction = predict(trained, pack.test_x, device, config.batch_size)
            metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
            metric_rows.append({"model": name, "seed": seed, "parameters": specifications[name], "wall_seconds": time.perf_counter() - started, **diagnostics, **metrics})
            sample_rows.extend({"model": name, "seed": seed, **row} for row in per_sample)
            history_rows.extend(history)
            torch.save(trained.state_dict(), outdir / f"model_{name.lower().replace('-', '_')}_seed{seed}.pt")
            print(f"[result] {name} seed={seed} global={metrics['global_score']:.6f} hole={metrics['hole_score']:.6f} balanced={metrics['balanced_score']:.6f}", flush=True)

    metrics_frame = pd.DataFrame(metric_rows)
    samples_frame = pd.DataFrame(sample_rows)
    metrics_frame.to_csv(outdir / "metrics_per_seed.csv", index=False)
    samples_frame.to_csv(outdir / "metrics_per_sample.csv", index=False)
    pd.DataFrame(history_rows).to_csv(outdir / "training_history.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(outdir / "metrics_summary.csv")
    (outdir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    summary = metrics_frame.groupby("model")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    (outdir / "experiment_report_zh.md").write_text(
        "# BAM 路径消融实验\n\n"
        "BAM-KAN-full 与三个变体保持相同参数规模和训练协议；每个变体只将对应路径的有效特征置零。\n\n"
        + summary.to_string()
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
