#!/usr/bin/env python3
"""Fair BAM-PIKAN, U-Net, and CNN comparison on the real hole-plate FEM data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
FAIR_BASE = HERE.parent / "test071405"
sys.path.insert(0, str(FAIR_BASE))

from run_fair_pikan_unet_comparison import (  # noqa: E402
    Config,
    DATA_DIR,
    DataPack,
    FairUNet,
    choose_device,
    evaluate_prediction,
    load_data,
    parameter_count,
    predict,
    set_seed,
    train_model,
)


OUT_DIR = HERE
MODEL_ORDER = ("CNN", "U-Net", "BAM-PIKAN")
DEFAULT_REUSE_UNET_DIR = HERE.parent / "test071405" / "formal_five_seed"


def valid_groups(channels: int) -> int:
    return next(group for group in (4, 3, 2, 1) if channels % group == 0)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        padding = tuple(dilation[index] * (kernel_size[index] // 2) for index in range(3))
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(valid_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class SpatialResidual(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        padding = tuple(dilation[index] * (kernel_size[index] // 2) for index in range(3))
        self.first = ConvNormAct(channels, channels, kernel_size, dilation)
        self.second = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(valid_groups(channels), channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.gelu(inputs + self.second(self.first(inputs)))


class AdaptiveKANLayer(nn.Module):
    """Gaussian-basis KAN with channel-wise widths and a stable linear path."""

    def __init__(self, features: int, grid_size: int = 5) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, grid_size))
        self.log_width = nn.Parameter(torch.full((features,), math.log(0.75)))
        self.coefficients = nn.Parameter(torch.randn(features, features, grid_size) * 0.02)
        self.linear = nn.Linear(features, features)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, inputs: Tensor) -> Tensor:
        widths = self.log_width.exp().clamp(0.20, 2.50).view(1, -1, 1)
        basis = torch.exp(-((inputs.unsqueeze(-1) - self.centers) / widths).square())
        nonlinear = torch.einsum("nig,oig->no", basis, self.coefficients)
        return self.linear(inputs) + nonlinear + self.bias


class ResidualKANBlock(nn.Module):
    def __init__(self, features: int, grid_size: int = 5) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(features)
        self.kan = AdaptiveKANLayer(features, grid_size)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.residual_scale * torch.tanh(self.kan(self.normalization(inputs)))


class FairCNN(nn.Module):
    """Parameter-matched dilated 3D CNN baseline without encoder-decoder skips."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, width)
        self.blocks = nn.Sequential(
            SpatialResidual(width, (3, 3, 3), (1, 1, 1)),
            SpatialResidual(width, (3, 3, 3), (1, 2, 2)),
            SpatialResidual(width, (3, 3, 3), (1, 4, 4)),
        )
        self.head = nn.Conv3d(width, 2, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(inputs)))


class BAMPIKAN(nn.Module):
    """Boundary-aware anisotropic multi-scale Spatial PIKAN.

    No target-derived mask or region label is used. Boundary sensitivity comes
    from the full-resolution anisotropic branch and the fine/coarse KAN scales.
    """

    def __init__(
        self,
        physical_channels: int,
        coordinate_channels: int,
        spatial_width: int,
        fine_kan_width: int,
        coarse_kan_width: int,
        grid_size: int = 5,
    ) -> None:
        super().__init__()
        coordinate_width = max(4, spatial_width // 4)
        physical_width = spatial_width - coordinate_width
        middle_width = spatial_width + spatial_width // 2
        head_width = max(6, spatial_width // 2)

        self.physical_channels = physical_channels
        self.physical_stem = ConvNormAct(
            physical_channels, physical_width, kernel_size=(1, 3, 3)
        )
        self.coordinate_stem = ConvNormAct(
            coordinate_channels, coordinate_width, kernel_size=(1, 1, 1)
        )
        self.local = nn.Sequential(
            SpatialResidual(spatial_width, (1, 3, 3), (1, 1, 1)),
            SpatialResidual(spatial_width, (1, 3, 3), (1, 2, 2)),
        )
        self.coarse_projection = ConvNormAct(spatial_width, middle_width, (3, 3, 3))
        self.coarse_spatial = SpatialResidual(
            middle_width, (3, 3, 3), (1, 1, 1)
        )

        self.fine_projection = ConvNormAct(
            spatial_width, fine_kan_width, kernel_size=(1, 1, 1)
        )
        self.fine_kan = nn.Sequential(
            ResidualKANBlock(fine_kan_width, grid_size),
            ResidualKANBlock(fine_kan_width, grid_size),
        )
        self.coarse_kan_projection = ConvNormAct(
            middle_width, coarse_kan_width, kernel_size=(1, 1, 1)
        )
        self.coarse_kan = nn.Sequential(
            ResidualKANBlock(coarse_kan_width, grid_size),
            ResidualKANBlock(coarse_kan_width, grid_size),
        )

        fused_channels = spatial_width + fine_kan_width + coarse_kan_width
        self.fusion = ConvNormAct(
            fused_channels, spatial_width, kernel_size=(1, 1, 1)
        )
        self.s1_head = nn.Sequential(
            ConvNormAct(spatial_width + fine_kan_width, head_width, (1, 3, 3)),
            nn.Conv3d(head_width, 1, kernel_size=1),
        )
        self.u_head = nn.Sequential(
            ConvNormAct(spatial_width + coarse_kan_width, head_width, (3, 3, 3)),
            nn.Conv3d(head_width, 1, kernel_size=1),
        )

    @staticmethod
    def _apply_kan(features: Tensor, blocks: nn.Module) -> Tensor:
        batch, channels, depth, height, width = features.shape
        values = features.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        values = blocks(values)
        return values.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3)

    def forward(self, inputs: Tensor) -> Tensor:
        physical = self.physical_stem(inputs[:, : self.physical_channels])
        coordinates = self.coordinate_stem(inputs[:, self.physical_channels :])
        local = self.local(torch.cat([physical, coordinates], dim=1))

        coarse = F.avg_pool3d(local, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        coarse = self.coarse_spatial(self.coarse_projection(coarse))

        fine = self.fine_projection(local)
        fine = F.avg_pool3d(fine, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        fine = self._apply_kan(fine, self.fine_kan)
        fine = F.interpolate(fine, size=local.shape[-3:], mode="trilinear", align_corners=False)

        coarse_kan = self.coarse_kan_projection(coarse)
        coarse_kan = F.avg_pool3d(coarse_kan, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        coarse_kan = self._apply_kan(coarse_kan, self.coarse_kan)
        coarse_kan = F.interpolate(
            coarse_kan, size=local.shape[-3:], mode="trilinear", align_corners=False
        )

        fused = self.fusion(torch.cat([local, fine, coarse_kan], dim=1))
        s1 = self.s1_head(torch.cat([fused, fine], dim=1))
        displacement = self.u_head(torch.cat([fused, coarse_kan], dim=1))
        return torch.cat([s1, displacement], dim=1)


def select_models(
    config: Config, input_channels: int
) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, int]]]:
    unet_candidates = []
    for width in range(8, 33):
        model = FairUNet(input_channels, width)
        unet_candidates.append((abs(parameter_count(model) - config.target_parameters), width))
    _, unet_width = min(unet_candidates)
    unet = FairUNet(input_channels, unet_width)
    target = parameter_count(unet)

    cnn_candidates = []
    for width in range(8, 35):
        model = FairCNN(input_channels, width)
        cnn_candidates.append((abs(parameter_count(model) - target), width))
    _, cnn_width = min(cnn_candidates)
    cnn = FairCNN(input_channels, cnn_width)

    bam_candidates = []
    for spatial_width in range(8, 25):
        for fine_width in range(6, 19, 2):
            for coarse_width in range(8, 23, 2):
                model = BAMPIKAN(11, 3, spatial_width, fine_width, coarse_width)
                bam_candidates.append(
                    (
                        abs(parameter_count(model) - target),
                        spatial_width,
                        fine_width,
                        coarse_width,
                    )
                )
    _, spatial_width, fine_width, coarse_width = min(bam_candidates)
    bam = BAMPIKAN(11, 3, spatial_width, fine_width, coarse_width)

    models = {"CNN": cnn, "U-Net": unet, "BAM-PIKAN": bam}
    specifications = {
        "CNN": {"width": cnn_width, "parameters": parameter_count(cnn)},
        "U-Net": {"width": unet_width, "parameters": parameter_count(unet)},
        "BAM-PIKAN": {
            "spatial_width": spatial_width,
            "fine_kan_width": fine_width,
            "coarse_kan_width": coarse_width,
            "grid_size": 5,
            "parameters": parameter_count(bam),
        },
    }
    counts = [int(specifications[name]["parameters"]) for name in MODEL_ORDER]
    mismatch = (max(counts) - min(counts)) / max(counts)
    if mismatch > config.parameter_tolerance:
        raise RuntimeError(f"Parameter mismatch {mismatch:.2%} exceeds tolerance.")
    return models, specifications


def hierarchical_bootstrap(
    per_sample: pd.DataFrame,
    repetitions: int,
    proposed: str = "BAM-PIKAN",
) -> pd.DataFrame:
    rng = np.random.default_rng(20260715)
    metrics = [
        "S1_global_NRMSE",
        "U_global_NRMSE",
        "S1_hole_NRMSE",
        "U_hole_NRMSE",
        "global_score",
        "hole_score",
        "balanced_score",
    ]
    rows = []
    for baseline in ("U-Net", "CNN"):
        for metric in metrics:
            pivot = per_sample.pivot_table(
                index=["seed", "sample"], columns="model", values=metric
            ).dropna(subset=[proposed, baseline])
            matrix = (pivot[proposed] - pivot[baseline]).unstack("sample")
            matrix = matrix.dropna(axis=1, how="any").to_numpy(dtype=np.float64)
            if matrix.size == 0:
                continue
            seed_count, sample_count = matrix.shape
            differences = np.empty(repetitions, dtype=np.float64)
            chunk_size = 2_000
            for start in range(0, repetitions, chunk_size):
                count = min(chunk_size, repetitions - start)
                seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
                sample_indices = rng.integers(0, sample_count, size=(count, sample_count))
                selected = matrix[
                    seed_indices[:, :, None], sample_indices[:, None, :]
                ]
                differences[start : start + count] = selected.mean(axis=(1, 2))
            lower, upper = np.quantile(differences, [0.025, 0.975])
            rows.append(
                {
                    "comparison": f"{proposed} - {baseline}",
                    "metric": metric,
                    "difference": float(matrix.mean()),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "probability_bampikan_better": float(np.mean(differences < 0.0)),
                    "paired_observations": int(seed_count * sample_count),
                }
            )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, history: pd.DataFrame, outdir: Path) -> None:
    selected = ["global_score", "hole_score", "balanced_score"]
    summary = metrics.groupby("model")[selected].agg(["mean", "std"])
    positions = np.arange(len(selected), dtype=float)
    width = 0.24
    colors = {"CNN": "#687078", "U-Net": "#2b7bba", "BAM-PIKAN": "#d95f02"}
    fig, axis = plt.subplots(figsize=(8.6, 5.0))
    for offset, model in zip((-width, 0.0, width), MODEL_ORDER):
        means = [summary.loc[model, (metric, "mean")] for metric in selected]
        errors = [summary.loc[model, (metric, "std")] for metric in selected]
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=model,
            color=colors[model],
        )
    axis.set_xticks(positions, ["Global", "Hole", "Balanced"])
    axis.set_ylabel("Normalized RMSE score (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "architecture_comparison.png", dpi=240)
    fig.savefig(outdir / "architecture_comparison.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.6, 5.0))
    for model in MODEL_ORDER:
        frame = history[history["model"] == model]
        grouped = frame.groupby("epoch")["validation_loss"].agg(["mean", "std"])
        axis.plot(grouped.index, grouped["mean"], label=model, color=colors[model])
        spread = grouped["std"].fillna(0.0)
        axis.fill_between(
            grouped.index,
            grouped["mean"] - spread,
            grouped["mean"] + spread,
            color=colors[model],
            alpha=0.14,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation normalized average MSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=240)
    fig.savefig(outdir / "training_curves.pdf")
    plt.close(fig)


def write_report(
    config: Config,
    specifications: Dict[str, Dict[str, int]],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    outdir: Path,
) -> None:
    means = metrics.groupby("model").mean(numeric_only=True)
    standard_deviations = metrics.groupby("model").std(numeric_only=True)
    lines = [
        "# BAM-PIKAN versus U-Net and CNN",
        "",
        "## Locked fair protocol",
        "",
        f"- Seeds: `{list(config.seeds)}`.",
        f"- Parameters: `{specifications}`.",
        "- Identical inputs: normalized `c02-c12 + x/y/z`.",
        "- Identical loss: normalized average MSE on material voxels.",
        f"- Identical optimizer: AdamW, lr `{config.learning_rate:g}`, weight decay `{config.weight_decay:g}`.",
        f"- Epoch limit `{config.epochs}`, batch size `{config.batch_size}`, identical early stopping.",
        "- No MMSE, TCGN-L3CR, L-BFGS, calibration, Fourier features, or target-derived region inputs.",
        "",
        "## Five-seed test results",
        "",
        "Mean +/- sample standard deviation; lower is better.",
        "",
        "| Metric | CNN | U-Net | BAM-PIKAN |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "S1_global_RMSE",
        "U_global_RMSE",
        "S1_hole_RMSE",
        "U_hole_RMSE",
        "global_score",
        "hole_score",
        "balanced_score",
    ]:
        values = []
        for model in MODEL_ORDER:
            values.append(
                f"{means.loc[model, metric]:.7g} +/- {standard_deviations.loc[model, metric]:.3g}"
            )
        lines.append(f"| `{metric}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Hierarchical paired bootstrap",
            "",
            "Negative BAM-PIKAN-minus-baseline differences favor BAM-PIKAN.",
            "",
            "| Comparison | Metric | Difference | 95% CI | P(BAM-PIKAN better) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['comparison']} | `{row['metric']}` | {row['difference']:.6g} | "
            f"[{row['ci95_lower']:.6g}, {row['ci95_upper']:.6g}] | "
            f"{row['probability_bampikan_better']:.3f} |"
        )
    (outdir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "formal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 73])
    parser.add_argument(
        "--reuse-unet-dir",
        type=Path,
        default=DEFAULT_REUSE_UNET_DIR,
        help="Directory containing compatible U-Net CSV results. Disabled in smoke mode.",
    )
    parser.add_argument(
        "--rerun-unet",
        action="store_true",
        help="Train U-Net instead of reusing the compatible test071405 results.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        seeds=tuple(args.seeds),
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        patience=2 if args.smoke else 14,
        bootstrap_repetitions=2_000 if args.smoke else 20_000,
        device=args.device,
    )
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seeds[0])
    prototypes, specifications = select_models(config, input_channels=14)
    print(json.dumps(specifications, indent=2), flush=True)
    if args.dry_run:
        return

    pack: DataPack = load_data(args.data_dir.resolve())
    device = choose_device(config.device)
    (outdir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (outdir / "model_specifications.json").write_text(
        json.dumps(specifications, indent=2), encoding="utf-8"
    )
    metric_rows: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []
    started = time.perf_counter()

    reuse_unet = not args.smoke and not args.rerun_unet
    training_order = tuple(name for name in MODEL_ORDER if not (reuse_unet and name == "U-Net"))
    if reuse_unet:
        reuse_dir = args.reuse_unet_dir.resolve()
        reused_metrics = pd.read_csv(reuse_dir / "metrics_per_seed.csv")
        reused_samples = pd.read_csv(reuse_dir / "metrics_per_sample.csv")
        reused_history = pd.read_csv(reuse_dir / "training_history.csv")
        reused_metrics = reused_metrics[
            (reused_metrics["model"] == "U-Net")
            & (reused_metrics["seed"].isin(config.seeds))
        ].copy()
        reused_samples = reused_samples[
            (reused_samples["model"] == "U-Net")
            & (reused_samples["seed"].isin(config.seeds))
        ].copy()
        reused_history = reused_history[
            (reused_history["model"] == "U-Net")
            & (reused_history["seed"].isin(config.seeds))
        ].copy()
        found_seeds = set(reused_metrics["seed"].astype(int))
        expected_seeds = set(config.seeds)
        found_parameters = set(reused_metrics["parameters"].astype(int))
        expected_parameters = {int(specifications["U-Net"]["parameters"])}
        if found_seeds != expected_seeds or found_parameters != expected_parameters:
            raise RuntimeError(
                "Reused U-Net results are incompatible: "
                f"seeds={sorted(found_seeds)}, parameters={sorted(found_parameters)}."
            )
        metric_rows.extend(reused_metrics.to_dict("records"))
        sample_rows.extend(reused_samples.to_dict("records"))
        history_rows.extend(reused_history.to_dict("records"))
        provenance = {
            "model": "U-Net",
            "source": str(reuse_dir),
            "seeds": sorted(found_seeds),
            "parameters": next(iter(found_parameters)),
            "reason": "Identical locked data, input, loss, optimizer, epoch, and seed protocol.",
        }
        (outdir / "reused_results.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        print(f"[reuse] model=U-Net source={reuse_dir}", flush=True)

    for seed in config.seeds:
        for model_name in training_order:
            set_seed(seed)
            models, current_specs = select_models(config, input_channels=14)
            model = models[model_name]
            print(
                f"[train] model={model_name} seed={seed} "
                f"params={parameter_count(model)} device={device}",
                flush=True,
            )
            trained, history, diagnostics = train_model(
                model_name, model, seed, pack, config, device
            )
            prediction = predict(trained, pack.test_x, device, config.batch_size)
            metrics, per_sample = evaluate_prediction(
                prediction, pack.test_y, pack.test_masks, pack.y_std
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameters": current_specs[model_name]["parameters"],
                    **diagnostics,
                    **metrics,
                }
            )
            sample_rows.extend(
                {"model": model_name, "seed": seed, **row} for row in per_sample
            )
            history_rows.extend(history)
            torch.save(
                trained.state_dict(),
                outdir / f"model_{model_name.lower().replace('-', '')}_seed{seed}.pt",
            )
            print(
                f"[result] model={model_name} seed={seed} "
                f"global={metrics['global_score']:.6f} hole={metrics['hole_score']:.6f} "
                f"balanced={metrics['balanced_score']:.6f}",
                flush=True,
            )
            del trained, model, models
            if device.type == "mps":
                torch.mps.empty_cache()

    metrics_frame = pd.DataFrame(metric_rows)
    samples_frame = pd.DataFrame(sample_rows)
    history_frame = pd.DataFrame(history_rows)
    bootstrap_frame = hierarchical_bootstrap(
        samples_frame, config.bootstrap_repetitions
    )
    metrics_frame.to_csv(outdir / "metrics_per_seed.csv", index=False)
    samples_frame.to_csv(outdir / "metrics_per_sample.csv", index=False)
    history_frame.to_csv(outdir / "training_history.csv", index=False)
    bootstrap_frame.to_csv(outdir / "paired_bootstrap.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(
        outdir / "metrics_summary.csv"
    )
    plot_results(metrics_frame, history_frame, outdir)
    write_report(config, specifications, metrics_frame, bootstrap_frame, outdir)
    runtime = {"total_seconds": time.perf_counter() - started, "device": str(device)}
    (outdir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(f"[done] results={outdir} runtime={runtime['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
