#!/usr/bin/env python3
"""Parameter-matched PIKAN versus U-Net comparison on the real hole-plate FEM data.

The experiment isolates network design: both models receive the same normalized
c02-c12 fields plus x/y/z coordinates, minimize the same normalized average MSE,
and use the same optimizer, schedule, split, batches, stopping rule, and seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label
from torch import Tensor, nn


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent / "data_28_06_2026"
INPUT_CHANNELS = list(range(2, 13))
TARGET_CHANNELS = [0, 1]
SPLIT_SEED = 20260629


@dataclass
class Config:
    seeds: Tuple[int, ...] = (11, 23, 37)
    epochs: int = 48
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    eta_min_ratio: float = 0.05
    patience: int = 14
    grad_clip: float = 5.0
    target_parameters: int = 90_000
    parameter_tolerance: float = 0.05
    kan_grid_size: int = 5
    kan_downsample: int = 4
    bootstrap_repetitions: int = 20_000
    device: str = "auto"


@dataclass
class DataPack:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_masks: Dict[str, np.ndarray]
    test_masks: Dict[str, np.ndarray]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(10, os.cpu_count() or 1)))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_ncdhw(array: np.ndarray) -> np.ndarray:
    return np.moveaxis(array, -1, 1).transpose(0, 1, 4, 2, 3).astype(np.float32)


def mask_to_ncdhw(mask: np.ndarray) -> np.ndarray:
    return mask[:, None].transpose(0, 1, 4, 2, 3).astype(np.float32)


def internal_void_mask(material_2d: np.ndarray) -> np.ndarray:
    void = ~material_2d.astype(bool)
    labels, _ = label(void)
    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    internal = void.copy()
    for border_label in border_labels:
        internal &= labels != border_label
    return internal


def construct_masks(data: np.ndarray) -> Dict[str, np.ndarray]:
    material = data[..., 3] > 0.0
    near = np.zeros_like(material, dtype=bool)
    structure = np.ones((5, 5), dtype=bool)
    for sample in range(material.shape[0]):
        for z_index in range(material.shape[3]):
            internal_void = internal_void_mask(material[sample, :, :, z_index])
            if internal_void.any():
                near[sample, :, :, z_index] = (
                    binary_dilation(internal_void, structure=structure, iterations=1)
                    & material[sample, :, :, z_index]
                )

    contact_signal = np.abs(data[..., 10]) + np.abs(data[..., 11]) + np.abs(data[..., 12])
    contact_nonzero = material & (contact_signal > 1.0e-12)
    contact_layer = np.zeros_like(material, dtype=bool)
    for sample in range(material.shape[0]):
        layer_scores = []
        for z_index in range(material.shape[3]):
            active = material[sample, :, :, z_index]
            layer_scores.append(float(contact_signal[sample, :, :, z_index][active].mean()) if active.any() else 0.0)
        best_layer = int(np.argmax(layer_scores))
        if layer_scores[best_layer] > 1.0e-12:
            contact_layer[sample, :, :, best_layer] = material[sample, :, :, best_layer]
    contact_active = contact_layer & contact_nonzero

    high_stress = np.zeros_like(material, dtype=bool)
    for sample in range(material.shape[0]):
        values = data[sample, ..., 0][material[sample]]
        if values.size:
            threshold = float(np.quantile(values, 0.90))
            high_stress[sample] = material[sample] & (data[sample, ..., 0] >= threshold)

    return {
        "material": mask_to_ncdhw(material),
        "near_opening": mask_to_ncdhw(near),
        "contact_active": mask_to_ncdhw(contact_active),
        "high_stress": mask_to_ncdhw(high_stress),
    }


def coordinate_channels(sample_count: int, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
    x_size, y_size, z_size = spatial_shape
    x = np.linspace(-1.0, 1.0, x_size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, y_size, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, z_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([xx, yy, zz], axis=-1)
    return np.broadcast_to(coords[None], (sample_count, *coords.shape)).copy()


def load_data(data_dir: Path) -> DataPack:
    train = np.load(data_dir / "train.npy").astype(np.float32)
    test = np.load(data_dir / "test.npy").astype(np.float32)
    if train.shape != (167, 64, 64, 4, 13) or test.shape != (19, 64, 64, 4, 13):
        raise ValueError(f"Unexpected data shapes: train={train.shape}, test={test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Input data contain NaN or Inf.")

    train_masks = construct_masks(train)
    test_masks = construct_masks(test)
    material = train[..., 3] > 0.0
    raw_train_x = train[..., INPUT_CHANNELS]
    raw_test_x = test[..., INPUT_CHANNELS]
    raw_train_y = train[..., TARGET_CHANNELS]
    raw_test_y = test[..., TARGET_CHANNELS]

    x_mean = np.array([raw_train_x[..., i][material].mean() for i in range(len(INPUT_CHANNELS))], dtype=np.float32)
    x_std = np.array([raw_train_x[..., i][material].std() for i in range(len(INPUT_CHANNELS))], dtype=np.float32)
    x_std = np.where(x_std > 1.0e-12, x_std, 1.0).astype(np.float32)
    y_mean = np.array([raw_train_y[..., i][material].mean() for i in range(2)], dtype=np.float32)
    y_std = np.array([raw_train_y[..., i][material].std() for i in range(2)], dtype=np.float32)
    y_std = np.where(y_std > 1.0e-12, y_std, 1.0).astype(np.float32)

    train_x = (raw_train_x - x_mean) / x_std
    test_x = (raw_test_x - x_mean) / x_std
    train_x = np.concatenate([train_x, coordinate_channels(len(train), train.shape[1:4])], axis=-1)
    test_x = np.concatenate([test_x, coordinate_channels(len(test), test.shape[1:4])], axis=-1)
    train_y = (raw_train_y - y_mean) / y_std
    test_y = (raw_test_y - y_mean) / y_std
    return DataPack(
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
    )


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = next(g for g in (4, 3, 2, 1) if out_channels % g == 0)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = next(g for g in (4, 3, 2, 1) if channels % g == 0)
        self.first = ConvGNAct(channels, channels)
        self.second = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.gelu(inputs + self.second(self.first(inputs)))


class FairUNet(nn.Module):
    """One-level 3D U-Net with a tunable base width."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        middle = width + width // 2
        bottleneck = width * 2
        self.enc = nn.Sequential(ConvGNAct(in_channels, width), ResidualBlock(width))
        self.down = nn.Sequential(ConvGNAct(width, middle), ResidualBlock(middle))
        self.mid = ConvGNAct(middle, bottleneck)
        self.dec = nn.Sequential(
            ConvGNAct(bottleneck + width, middle),
            nn.Conv3d(middle, 2, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        high_resolution = self.enc(inputs)
        pooled = F.avg_pool3d(high_resolution, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        low_resolution = self.mid(self.down(pooled))
        upsampled = F.interpolate(low_resolution, size=high_resolution.shape[-3:], mode="trilinear", align_corners=False)
        return self.dec(torch.cat([upsampled, high_resolution], dim=1))


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, grid_size))
        self.log_width = nn.Parameter(torch.tensor(math.log(0.75), dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.025)
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs: Tensor) -> Tensor:
        width = torch.exp(self.log_width).clamp(0.25, 2.0)
        basis = torch.exp(-((inputs.unsqueeze(-1) - self.centers) / width).square())
        return torch.einsum("nig,oig->no", basis, self.coefficients) + self.linear(inputs) + self.bias


class FairSpatialPIKAN(nn.Module):
    """Spatial convolutional encoder with a pointwise KAN output mixer."""

    def __init__(
        self,
        in_channels: int,
        spatial_width: int,
        kan_width: int,
        grid_size: int,
        kan_downsample: int,
    ) -> None:
        super().__init__()
        middle = spatial_width + spatial_width // 2
        self.kan_downsample = kan_downsample
        self.stem = ConvGNAct(in_channels, spatial_width)
        self.local = ResidualBlock(spatial_width)
        self.down = ConvGNAct(spatial_width, middle)
        self.mid = ResidualBlock(middle)
        self.fuse = ConvGNAct(spatial_width + middle, spatial_width, kernel_size=1)
        self.pre_kan = ConvGNAct(spatial_width + in_channels, kan_width, kernel_size=1)
        self.kan1 = KANLayer(kan_width, kan_width, grid_size)
        self.kan2 = KANLayer(kan_width, kan_width, grid_size)
        self.linear_skip = nn.Conv3d(kan_width, 2, kernel_size=1)
        self.head = nn.Linear(kan_width, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.local(self.stem(inputs))
        pooled = F.avg_pool3d(local, (1, 2, 2), (1, 2, 2), ceil_mode=True)
        low_resolution = self.mid(self.down(pooled))
        upsampled = F.interpolate(low_resolution, size=local.shape[-3:], mode="trilinear", align_corners=False)
        fused = self.fuse(torch.cat([local, upsampled], dim=1))
        features = self.pre_kan(torch.cat([fused, inputs], dim=1))
        skip = self.linear_skip(features)

        kan_features = F.avg_pool3d(
            features,
            (1, self.kan_downsample, self.kan_downsample),
            (1, self.kan_downsample, self.kan_downsample),
            ceil_mode=True,
        )
        batch, channels, depth, height, width = kan_features.shape
        values = kan_features.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        values = torch.tanh(self.kan1(values))
        values = torch.tanh(self.kan2(values))
        output = self.head(values).reshape(batch, depth, height, width, 2).permute(0, 4, 1, 2, 3)
        output = F.interpolate(output, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return output + skip


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_matched_models(config: Config, input_channels: int) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, int]]]:
    unet_candidates = []
    for width in range(8, 33):
        model = FairUNet(input_channels, width)
        unet_candidates.append((abs(parameter_count(model) - config.target_parameters), width, model))
    _, unet_width, unet = min(unet_candidates, key=lambda item: item[0])

    pikan_candidates = []
    for spatial_width in range(8, 33):
        for kan_width in range(8, 41, 2):
            model = FairSpatialPIKAN(
                input_channels,
                spatial_width,
                kan_width,
                config.kan_grid_size,
                config.kan_downsample,
            )
            count = parameter_count(model)
            pikan_candidates.append((abs(count - parameter_count(unet)), spatial_width, kan_width, model))
    _, spatial_width, kan_width, pikan = min(pikan_candidates, key=lambda item: item[0])
    models = {"U-Net": unet, "PIKAN": pikan}
    specifications = {
        "U-Net": {"width": unet_width, "parameters": parameter_count(unet)},
        "PIKAN": {
            "spatial_width": spatial_width,
            "kan_width": kan_width,
            "grid_size": config.kan_grid_size,
            "kan_downsample": config.kan_downsample,
            "parameters": parameter_count(pikan),
        },
    }
    counts = [specifications[name]["parameters"] for name in ("U-Net", "PIKAN")]
    mismatch = abs(counts[0] - counts[1]) / max(counts)
    if mismatch > config.parameter_tolerance:
        raise RuntimeError(f"Parameter mismatch {mismatch:.2%} exceeds tolerance.")
    return models, specifications


def split_train_validation(sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(sample_count)
    validation_count = max(1, int(round(0.18 * sample_count)))
    return np.sort(indices[validation_count:]), np.sort(indices[:validation_count])


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    permutation = rng.permutation(indices)
    for start in range(0, len(permutation), batch_size):
        yield permutation[start : start + batch_size]


def tensor_batch(pack: DataPack, split: str, indices: np.ndarray, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    if split == "train":
        x, y, mask = pack.train_x, pack.train_y, pack.train_masks["material"]
    else:
        x, y, mask = pack.test_x, pack.test_y, pack.test_masks["material"]
    return (
        torch.as_tensor(x[indices], device=device),
        torch.as_tensor(y[indices], device=device),
        torch.as_tensor(mask[indices], device=device),
    )


def normalized_average_mse(prediction: Tensor, target: Tensor, material: Tensor) -> Tensor:
    squared_error = (prediction - target).square().mean(dim=1, keepdim=True)
    return (squared_error * material).sum() / material.sum().clamp_min(1.0)


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    pack: DataPack,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(indices), batch_size):
        local = indices[start : start + batch_size]
        x, y, mask = tensor_batch(pack, "train", local, device)
        total += float(normalized_average_mse(model(x), y, mask).cpu()) * len(local)
    return total / len(indices)


def train_model(
    model_name: str,
    model: nn.Module,
    seed: int,
    pack: DataPack,
    config: Config,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, float]]:
    train_indices, validation_indices = split_train_validation(len(pack.train_x))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * config.eta_min_ratio
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        accumulated = 0.0
        seen = 0
        for indices in batches(train_indices, config.batch_size, rng):
            x, y, mask = tensor_batch(pack, "train", indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_average_mse(model(x), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            accumulated += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        scheduler.step()
        current_validation = validation_loss(
            model, pack, validation_indices, device, config.batch_size
        )
        history.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": accumulated / seen,
                "validation_loss": current_validation,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if current_validation < best_loss:
            best_loss = current_validation
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, history, {
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": elapsed,
    }


@torch.no_grad()
def predict(model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        x = torch.as_tensor(inputs[start : start + batch_size], device=device)
        outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs)


def masked_rmse(error: np.ndarray, mask: np.ndarray) -> float:
    selected = error[mask.astype(bool)]
    return float(np.sqrt(np.mean(np.square(selected)))) if selected.size else float("nan")


def evaluate_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    masks: Dict[str, np.ndarray],
    y_std: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    error = prediction - target
    metrics: Dict[str, float] = {}
    regions = {
        "global": masks["material"],
        "hole": masks["near_opening"],
        "contact": masks["contact_active"],
        "high": masks["high_stress"],
    }
    for region, mask in regions.items():
        for channel, label_name in enumerate(("S1", "U")):
            normalized = masked_rmse(error[:, channel : channel + 1], mask)
            metrics[f"{label_name}_{region}_NRMSE"] = normalized
            metrics[f"{label_name}_{region}_RMSE"] = normalized * float(y_std[channel])
    metrics["global_score"] = 0.5 * (metrics["S1_global_NRMSE"] + metrics["U_global_NRMSE"])
    metrics["hole_score"] = 0.5 * (metrics["S1_hole_NRMSE"] + metrics["U_hole_NRMSE"])
    metrics["balanced_score"] = 0.5 * (metrics["global_score"] + metrics["hole_score"])

    per_sample = []
    for sample in range(len(prediction)):
        row: Dict[str, float] = {"sample": sample}
        for region, mask in regions.items():
            for channel, label_name in enumerate(("S1", "U")):
                row[f"{label_name}_{region}_NRMSE"] = masked_rmse(
                    error[sample : sample + 1, channel : channel + 1],
                    mask[sample : sample + 1],
                )
        row["global_score"] = 0.5 * (row["S1_global_NRMSE"] + row["U_global_NRMSE"])
        row["hole_score"] = 0.5 * (row["S1_hole_NRMSE"] + row["U_hole_NRMSE"])
        row["balanced_score"] = 0.5 * (row["global_score"] + row["hole_score"])
        per_sample.append(row)
    return metrics, per_sample


def paired_bootstrap(per_sample: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
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
    for metric in metrics:
        pivot = per_sample.pivot_table(index=["seed", "sample"], columns="model", values=metric)
        pivot = pivot.dropna(subset=["PIKAN", "U-Net"])
        seeds = np.array(sorted(pivot.index.get_level_values("seed").unique()))
        samples = np.array(sorted(pivot.index.get_level_values("sample").unique()))
        differences = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            selected_samples = rng.choice(samples, size=len(samples), replace=True)
            values = []
            for seed in selected_seeds:
                for sample in selected_samples:
                    key = (seed, sample)
                    if key in pivot.index:
                        values.append(float(pivot.loc[key, "PIKAN"] - pivot.loc[key, "U-Net"]))
            differences[repetition] = np.mean(values)
        point = float((pivot["PIKAN"] - pivot["U-Net"]).mean())
        lower, upper = np.quantile(differences, [0.025, 0.975])
        probability_pikan_better = float(np.mean(differences < 0.0))
        rows.append(
            {
                "metric": metric,
                "pikan_minus_unet": point,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "probability_pikan_better": probability_pikan_better,
                "paired_observations": len(pivot),
            }
        )
    return pd.DataFrame(rows)


def save_plots(metrics: pd.DataFrame, history: pd.DataFrame, outdir: Path) -> None:
    selected = ["global_score", "hole_score", "balanced_score"]
    summary = metrics.groupby("model")[selected].agg(["mean", "std"])
    positions = np.arange(len(selected))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, model in zip((-width / 2, width / 2), ("U-Net", "PIKAN")):
        means = [summary.loc[model, (metric, "mean")] for metric in selected]
        errors = [summary.loc[model, (metric, "std")] for metric in selected]
        axis.bar(positions + offset, means, width, yerr=errors, capsize=4, label=model)
    axis.set_xticks(positions, ["Global", "Hole", "Balanced"])
    axis.set_ylabel("Normalized RMSE score (lower is better)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fair_architecture_scores.png", dpi=220)
    fig.savefig(outdir / "fair_architecture_scores.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for model, frame in history.groupby("model"):
        grouped = frame.groupby("epoch")["validation_loss"].agg(["mean", "std"])
        axis.plot(grouped.index, grouped["mean"], label=model)
        axis.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"].fillna(0.0),
            grouped["mean"] + grouped["std"].fillna(0.0),
            alpha=0.18,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation normalized average MSE")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fair_training_curves.png", dpi=220)
    fig.savefig(outdir / "fair_training_curves.pdf")
    plt.close(fig)


def write_report(
    config: Config,
    specifications: Dict[str, Dict[str, int]],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    outdir: Path,
) -> None:
    medians = metrics.groupby("model").median(numeric_only=True)
    lines = [
        "# Fair PIKAN versus U-Net experiment",
        "",
        "## Controlled protocol",
        "",
        f"- Seeds: `{list(config.seeds)}`.",
        f"- Epoch limit: `{config.epochs}`; batch size: `{config.batch_size}`.",
        f"- Optimizer: AdamW, learning rate `{config.learning_rate:g}`, weight decay `{config.weight_decay:g}`.",
        "- Inputs for both models: normalized `c02-c12 + x/y/z`.",
        "- Loss for both models: normalized average MSE over material voxels.",
        "- No MMSE, L-BFGS, TCGN-L3CR, calibration, Fourier features, or architecture-specific region branches.",
        f"- U-Net specification: `{specifications['U-Net']}`.",
        f"- PIKAN specification: `{specifications['PIKAN']}`.",
        "",
        "## Test medians across seeds",
        "",
        "| Metric | U-Net | PIKAN | Relative PIKAN change |",
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
        unet = float(medians.loc["U-Net", metric])
        pikan = float(medians.loc["PIKAN", metric])
        relative = (pikan / unet - 1.0) * 100.0
        lines.append(f"| `{metric}` | {unet:.8g} | {pikan:.8g} | {relative:+.2f}% |")
    lines.extend(["", "## Paired hierarchical bootstrap", ""])
    lines.append("Negative PIKAN-minus-U-Net differences favor PIKAN.")
    lines.extend(["", "| Metric | Difference | 95% CI | P(PIKAN better) |", "|---|---:|---:|---:|"])
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| `{row['metric']}` | {row['pikan_minus_unet']:.6g} | "
            f"[{row['ci95_lower']:.6g}, {row['ci95_upper']:.6g}] | "
            f"{row['probability_pikan_better']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "A network is declared superior only when its global and hole scores are both lower and the paired 95% confidence intervals exclude zero. Split outcomes are reported as a global/local trade-off.",
        ]
    )
    (outdir / "fair_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
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
        bootstrap_repetitions=1_000 if args.smoke else 20_000,
        device=args.device,
    )
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seeds[0])
    prototype_models, specifications = select_matched_models(config, input_channels=14)
    print(json.dumps(specifications, indent=2), flush=True)
    if args.dry_run:
        return

    pack = load_data(args.data_dir.resolve())
    device = choose_device(config.device)
    (outdir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (outdir / "model_specifications.json").write_text(json.dumps(specifications, indent=2), encoding="utf-8")
    (outdir / "normalization.json").write_text(
        json.dumps(
            {
                "input_channels": [f"c{channel:02d}" for channel in INPUT_CHANNELS] + ["x", "y", "z"],
                "x_mean": pack.x_mean.tolist(),
                "x_std": pack.x_std.tolist(),
                "y_mean": pack.y_mean.tolist(),
                "y_std": pack.y_std.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_rows: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []
    for seed in config.seeds:
        for model_name in ("U-Net", "PIKAN"):
            set_seed(seed)
            models, current_specifications = select_matched_models(config, input_channels=14)
            model = models[model_name]
            print(f"[train] model={model_name} seed={seed} params={parameter_count(model)} device={device}", flush=True)
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
                    "parameters": current_specifications[model_name]["parameters"],
                    **diagnostics,
                    **metrics,
                }
            )
            sample_rows.extend({"model": model_name, "seed": seed, **row} for row in per_sample)
            history_rows.extend(history)
            torch.save(trained.state_dict(), outdir / f"model_{model_name.replace('-', '').lower()}_seed{seed}.pt")
            print(
                f"[result] model={model_name} seed={seed} global={metrics['global_score']:.6f} "
                f"hole={metrics['hole_score']:.6f} balanced={metrics['balanced_score']:.6f}",
                flush=True,
            )
            del trained, model, models
            if device.type == "mps":
                torch.mps.empty_cache()

    metrics_frame = pd.DataFrame(metric_rows)
    per_sample_frame = pd.DataFrame(sample_rows)
    history_frame = pd.DataFrame(history_rows)
    bootstrap_frame = paired_bootstrap(per_sample_frame, config.bootstrap_repetitions)
    metrics_frame.to_csv(outdir / "metrics_per_seed.csv", index=False)
    per_sample_frame.to_csv(outdir / "metrics_per_sample.csv", index=False)
    history_frame.to_csv(outdir / "training_history.csv", index=False)
    bootstrap_frame.to_csv(outdir / "paired_bootstrap.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(outdir / "metrics_summary.csv")
    save_plots(metrics_frame, history_frame, outdir)
    write_report(config, specifications, metrics_frame, bootstrap_frame, outdir)
    print(f"[done] results={outdir}", flush=True)


if __name__ == "__main__":
    main()
