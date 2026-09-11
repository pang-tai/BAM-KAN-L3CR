#!/usr/bin/env python3
"""Formal KAN-isolation experiment under the leakage-free v1 protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "test071407"
FAIR = HERE.parent / "test071405"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(FAIR))

from run_leakage_free_architecture_experiment import (  # noqa: E402
    MODEL_ORDER,
    choose_models,
    evaluate_clean,
    load_clean_pack,
)
from run_bampikan_unet_cnn_comparison import BAMPIKAN, parameter_count  # noqa: E402
from run_fair_pikan_unet_comparison import (  # noqa: E402
    Config,
    choose_device,
    predict,
    set_seed,
    train_model,
)


class ResidualPointwiseMLP(nn.Module):
    """Pointwise MLP replacement with the same residual contract as KAN."""

    def __init__(self, features: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 3 * features
        self.normalization = nn.LayerNorm(features)
        self.first = nn.Linear(features, hidden)
        self.second = nn.Linear(hidden, features)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.second(torch.nn.functional.gelu(self.first(self.normalization(inputs))))
        return inputs + self.residual_scale * torch.tanh(values)


class ResidualPointwiseConv(nn.Module):
    """1-D pointwise convolution replacement for the KAN block."""

    def __init__(self, features: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 3 * features
        self.normalization = nn.LayerNorm(features)
        self.first = nn.Conv1d(features, hidden, kernel_size=1)
        self.second = nn.Conv1d(hidden, features, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.normalization(inputs).unsqueeze(-1)
        values = torch.nn.functional.gelu(self.first(values))
        values = self.second(values).squeeze(-1)
        return inputs + self.residual_scale * torch.tanh(values)


def make_variant(kind: str) -> nn.Module:
    model = BAMPIKAN(
        physical_channels=8,
        coordinate_channels=3,
        spatial_width=19,
        fine_kan_width=12,
        coarse_kan_width=12,
        grid_size=5,
    )
    block = ResidualPointwiseMLP if kind == "BAM-MLP" else ResidualPointwiseConv
    model.fine_kan = nn.Sequential(block(12), block(12))
    model.coarse_kan = nn.Sequential(block(12), block(12))
    return model


def run(args: argparse.Namespace) -> None:
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds)
    config = Config(
        seeds=seeds,
        epochs=args.epochs,
        batch_size=8,
        patience=args.patience,
        bootstrap_repetitions=100,
        device=args.device,
        target_parameters=90000,
    )
    pack, _, _ = load_clean_pack(args.clean_dir.resolve(), args.data_dir.resolve())
    device = choose_device(args.device)
    reference = pd.read_csv(HERE / "formal_leakage_free/metrics_per_seed.csv")
    reference_samples = pd.read_csv(HERE / "formal_leakage_free/metrics_per_sample.csv")
    reference = reference[reference["model"] == "BAM-KAN"].copy()
    reference_samples = reference_samples[reference_samples["model"] == "BAM-KAN"].copy()
    reference["model"] = "BAM-KAN"
    reference_samples["model"] = "BAM-KAN"
    metric_rows = reference.to_dict("records")
    sample_rows = reference_samples.to_dict("records")
    history_rows: List[Dict[str, float]] = []
    specs = {"BAM-KAN": int(reference["parameters"].iloc[0])}
    for kind in ("BAM-MLP", "BAM-Conv"):
        specs[kind] = parameter_count(make_variant(kind))
    (outdir / "model_specifications.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    for seed in seeds:
        for kind in ("BAM-MLP", "BAM-Conv"):
            set_seed(seed)
            model = make_variant(kind)
            started = time.perf_counter()
            trained, history, diagnostics = train_model(kind, model, seed, pack, config, device)
            prediction = predict(trained, pack.test_x, device, config.batch_size)
            metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
            metric_rows.append({"model": kind, "seed": seed, "parameters": specs[kind], "wall_seconds": time.perf_counter() - started, **diagnostics, **metrics})
            sample_rows.extend({"model": kind, "seed": seed, **row} for row in per_sample)
            history_rows.extend(history)
            torch.save(trained.state_dict(), outdir / f"model_{kind.lower().replace('-', '_')}_seed{seed}.pt")
            print(f"[result] {kind} seed={seed} global={metrics['global_score']:.6f} hole={metrics['hole_score']:.6f} balanced={metrics['balanced_score']:.6f}", flush=True)

    metrics_frame = pd.DataFrame(metric_rows)
    sample_frame = pd.DataFrame(sample_rows)
    history_frame = pd.DataFrame(history_rows)
    metrics_frame.to_csv(outdir / "metrics_per_seed.csv", index=False)
    sample_frame.to_csv(outdir / "metrics_per_sample.csv", index=False)
    history_frame.to_csv(outdir / "training_history.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(outdir / "metrics_summary.csv")
    (outdir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    summary = metrics_frame.groupby("model")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    lines = [
        "# BAM-KAN KAN 隔离实验",
        "",
        "BAM-KAN、BAM-MLP 和 BAM-Conv 使用相同 BAM 主干、相同输入、相同数据划分、AdamW、early stopping 和五种子协议。BAM-MLP/BAM-Conv 仅替换 KAN 点映射块，参数量保持在同一数量级。",
        "",
        summary.to_string(),
        "",
        "该表用于判断 KAN 点映射本身是否带来额外收益；它不能替代网络结构总比较。",
    ]
    (outdir / "experiment_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data_28_06_2026")
    parser.add_argument("--clean-dir", type=Path, default=HERE / "data_clean/leakage_free_v1")
    parser.add_argument("--out-dir", type=Path, default=HERE / "formal_kan_isolation")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 73])
    parser.add_argument("--device", default="mps")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
