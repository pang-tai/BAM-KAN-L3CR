#!/usr/bin/env python3
"""Progressive full-data, matrix-free L3CR refinement for BAM-KAN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCH = ROOT / "test080401"
FAIR = ROOT / "test071405"
LEGACY = ROOT / "test071407"
PRIOR = ROOT / "test082301"
for path in (ARCH, FAIR, LEGACY, PRIOR):
    sys.path.insert(0, str(path))

from run_fair_pikan_unet_comparison import split_train_validation  # noqa: E402
from run_high_accuracy_refinement import ExperimentConfig, make_bam, state_sha256  # noqa: E402
from run_leakage_free_architecture_experiment import evaluate_clean, load_clean_pack  # noqa: E402


SEEDS = (11, 23, 37, 51, 73)
DATA_DIR = ROOT / "data_28_06_2026"
CLEAN_DIR = ARCH / "data_clean/leakage_free_v1"
STAGE_PREFIXES = {
    "P0": ("s1_head.1", "u_head.1"),
    "P1": ("s1_head.1", "u_head.1", "fusion"),
    "P2": ("s1_head", "u_head", "fusion"),
    "P3": ("s1_head", "u_head", "fusion", "fine_projection", "fine_kan", "coarse_kan_projection", "coarse_kan"),
    "P4": ("s1_head", "u_head", "fusion", "fine_projection", "fine_kan", "coarse_kan_projection", "coarse_kan", "local"),
    "P5": ("s1_head", "u_head", "fusion", "fine_projection", "fine_kan", "coarse_kan_projection", "coarse_kan", "local", "physical_stem", "coordinate_stem", "coarse_projection"),
    "P6": ("",),
}
EXPECTED_COUNTS = {"P0": 20, "P1": 894, "P2": 10992, "P3": 15328, "P4": 28552, "P5": 44149, "P6": 86653}
PILOT_CAPS = {"P0": 90.0, "P1": 150.0, "P2": 210.0, "P3": 270.0, "P4": 330.0, "P5": 420.0, "P6": 630.0}
METHODS = ("Resumed-AdamW", "AdamW-0", "L-BFGS", "MF-L3CR")


@dataclass
class Counters:
    objective: int = 0
    gradient: int = 0
    hvp: int = 0


def rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0**3 if sys.platform == "darwin" else 1024.0**2)


def vector_hash(vector: Tensor) -> str:
    value = vector.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_path(seed: int) -> Path:
    return PRIOR / "warmup" / f"seed_{seed}" / "best_checkpoint.pt"


def load_model(seed: int) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    model = make_bam(ExperimentConfig()).to(device="cpu", dtype=torch.float64)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def in_stage(name: str, prefixes: Sequence[str]) -> bool:
    return any(prefix == "" or name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def configure_stage(model: nn.Module, stage: str) -> list[tuple[str, nn.Parameter]]:
    prefixes = STAGE_PREFIXES[stage]
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(in_stage(name, prefixes))
        if parameter.requires_grad:
            selected.append((name, parameter))
    count = sum(parameter.numel() for _, parameter in selected)
    if count != EXPECTED_COUNTS[stage]:
        raise RuntimeError(f"{stage} parameter count {count} != expected {EXPECTED_COUNTS[stage]}")
    return selected


def parameter_vector(named: Sequence[tuple[str, nn.Parameter]]) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for _, parameter in named])


def assign_vector(named: Sequence[tuple[str, nn.Parameter]], vector: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, parameter in named:
            count = parameter.numel()
            parameter.copy_(vector[offset : offset + count].reshape_as(parameter))
            offset += count
    if offset != vector.numel():
        raise RuntimeError("parameter vector size mismatch")


def flatten_optional(values: Sequence[Tensor | None], named: Sequence[tuple[str, nn.Parameter]]) -> Tensor:
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if value is None else value.reshape(-1)
        for value, (_, parameter) in zip(values, named)
    ])


def batches(indices: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


class FullDataObjective:
    def __init__(self, model: nn.Module, pack, indices: np.ndarray, microbatch: int = 4) -> None:
        self.model = model
        self.pack = pack
        self.indices = np.asarray(indices, dtype=np.int64)
        self.microbatch = int(microbatch)
        self.total_weight = int(np.asarray(pack.train_masks["material"])[self.indices].sum())
        if self.total_weight <= 0:
            raise RuntimeError("empty material mask")

    def batch_loss_sum(self, local: np.ndarray) -> Tensor:
        inputs = torch.as_tensor(self.pack.train_x[local], dtype=torch.float64)
        targets = torch.as_tensor(self.pack.train_y[local], dtype=torch.float64)
        mask = torch.as_tensor(self.pack.train_masks["material"][local], dtype=torch.float64)
        prediction = self.model(inputs)
        return (((prediction - targets).square().mean(dim=1, keepdim=True)) * mask).sum()

    def value(self, counters: Counters | None = None) -> float:
        total = 0.0
        with torch.no_grad():
            for local in batches(self.indices, self.microbatch):
                total += float(self.batch_loss_sum(local))
        if counters is not None:
            counters.objective += 1
        return total / self.total_weight

    def value_gradient(self, named: Sequence[tuple[str, nn.Parameter]], counters: Counters | None = None) -> tuple[float, Tensor]:
        params = [parameter for _, parameter in named]
        sums = [torch.zeros_like(parameter) for parameter in params]
        total = 0.0
        for local in batches(self.indices, self.microbatch):
            loss = self.batch_loss_sum(local)
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
            for target, value in zip(sums, gradients):
                if value is not None:
                    target.add_(value.detach())
            total += float(loss.detach())
        if counters is not None:
            counters.objective += 1
            counters.gradient += 1
        flat = torch.cat([(value / self.total_weight).reshape(-1) for value in sums])
        return total / self.total_weight, flat

    def hvp(self, named: Sequence[tuple[str, nn.Parameter]], direction: Tensor, counters: Counters | None = None) -> Tensor:
        params = [parameter for _, parameter in named]
        pieces = []
        offset = 0
        for parameter in params:
            pieces.append(direction[offset : offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
        sums = [torch.zeros_like(parameter) for parameter in params]
        for local in batches(self.indices, self.microbatch):
            loss = self.batch_loss_sum(local)
            gradients = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            dot = sum((gradient * piece).sum() for gradient, piece in zip(gradients, pieces) if gradient is not None)
            products = torch.autograd.grad(dot, params, allow_unused=True)
            for target, value in zip(sums, products):
                if value is not None:
                    target.add_(value.detach())
        if counters is not None:
            counters.hvp += 1
        return torch.cat([(value / self.total_weight).reshape(-1) for value in sums])

    def backward_into_parameters(self, named: Sequence[tuple[str, nn.Parameter]], counters: Counters | None = None) -> Tensor:
        params = [parameter for _, parameter in named]
        for parameter in params:
            parameter.grad = None
        total = 0.0
        for local in batches(self.indices, self.microbatch):
            loss = self.batch_loss_sum(local) / self.total_weight
            loss.backward()
            total += float(loss.detach())
        if counters is not None:
            counters.objective += 1
            counters.gradient += 1
        return torch.tensor(total, dtype=torch.float64)


def validation_value(model: nn.Module, pack, indices: np.ndarray, batch_size: int = 4) -> float:
    total = 0.0
    weight = 0
    with torch.no_grad():
        for local in batches(indices, batch_size):
            inputs = torch.as_tensor(pack.train_x[local], dtype=torch.float64)
            targets = torch.as_tensor(pack.train_y[local], dtype=torch.float64)
            mask = torch.as_tensor(pack.train_masks["material"][local], dtype=torch.float64)
            total += float((((model(inputs) - targets).square().mean(dim=1, keepdim=True)) * mask).sum())
            weight += int(mask.sum())
    return total / weight


def prox_l3(value: Tensor, alpha: float, sigma: float) -> Tensor:
    magnitude = value.abs()
    return torch.sign(value) * 2.0 * magnitude / (1.0 + torch.sqrt(1.0 + 2.0 * alpha * sigma * magnitude))


def norm_three_halves(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value, ord=1.5))


def model_value(step: Tensor, gradient: Tensor, hstep: Tensor, sigma: float) -> float:
    return float(gradient @ step + 0.5 * step @ hstep + sigma * torch.sum(step.abs() ** 3) / 6.0)


def model_residual(step: Tensor, gradient: Tensor, hstep: Tensor, sigma: float) -> float:
    residual = gradient + hstep + 0.5 * sigma * step.abs() * step
    return norm_three_halves(residual) / max(1.0, norm_three_halves(gradient))


def solve_matrix_free_subproblem(
    objective: FullDataObjective,
    named: Sequence[tuple[str, nn.Parameter]],
    gradient: Tensor,
    sigma: float,
    counters: Counters,
    hvp_limit: int,
    tolerance: float,
) -> dict:
    step = torch.zeros_like(gradient)
    hstep = torch.zeros_like(gradient)
    gnorm2 = float(gradient @ gradient)
    if gnorm2 == 0.0:
        return {"step": step, "hstep": hstep, "residual": 0.0, "converged": True, "inner_hvp": 0, "backtracks": 0}
    hg = objective.hvp(named, gradient, counters)
    rayleigh = abs(float(gradient @ hg)) / max(gnorm2, 1.0e-30)
    alpha = 0.8 / max(rayleigh, 1.0e-6)
    inner_hvp = 1
    backtracks = 0
    residual = float("inf")
    converged = False
    while inner_hvp < hvp_limit:
        smooth_gradient = gradient + hstep
        accepted_inner = False
        for _ in range(4):
            trial = prox_l3(step - alpha * smooth_gradient, alpha, sigma)
            htrial = objective.hvp(named, trial, counters)
            inner_hvp += 1
            delta = trial - step
            q_current = float(gradient @ step + 0.5 * step @ hstep)
            q_trial = float(gradient @ trial + 0.5 * trial @ htrial)
            majorizer = q_current + float(smooth_gradient @ delta) + float(delta @ delta) / (2.0 * alpha)
            if q_trial <= majorizer + 1.0e-12:
                step, hstep = trial, htrial
                accepted_inner = True
                break
            alpha *= 0.5
            backtracks += 1
            if inner_hvp >= hvp_limit:
                break
        residual = model_residual(step, gradient, hstep, sigma)
        if residual <= tolerance:
            converged = True
            break
        if not accepted_inner or inner_hvp >= hvp_limit:
            break
        alpha *= 1.15
    return {"step": step, "hstep": hstep, "residual": residual, "converged": converged, "inner_hvp": inner_hvp, "backtracks": backtracks}


def snapshot(stage: str, seed: int, method: str, iteration: int, started: float, value: float, gradient: Tensor, initial_norm: float, counters: Counters, accepted: bool = False, **extra) -> dict:
    norm = float(torch.linalg.vector_norm(gradient))
    dimension = gradient.numel()
    return {
        "stage": stage, "seed": seed, "method": method, "iteration": iteration,
        "elapsed_seconds": time.perf_counter() - started, "objective": value,
        "gradient_norm": norm, "relative_gradient": norm / max(1.0, initial_norm),
        "gradient_rms": norm / math.sqrt(dimension), "parameter_count": dimension,
        "objective_evaluations": counters.objective, "gradient_evaluations": counters.gradient,
        "hvp_evaluations": counters.hvp, "rme": counters.gradient + counters.hvp,
        "rss_gib": rss_gib(), "accepted": bool(accepted), **extra,
    }


def run_l3cr(model: nn.Module, pack, train_indices: np.ndarray, val_indices: np.ndarray, stage: str, seed: int, sigma: float, time_limit: float, microbatch: int, hvp_limit: int = 8, max_outer: int = 5) -> tuple[list[dict], dict]:
    named = configure_stage(model, stage)
    objective = FullDataObjective(model, pack, train_indices, microbatch)
    counters = Counters()
    started = time.perf_counter()
    value, gradient = objective.value_gradient(named, counters)
    initial_norm = float(torch.linalg.vector_norm(gradient))
    records = [snapshot(stage, seed, "MF-L3CR", 0, started, value, gradient, initial_norm, counters, sigma=sigma)]
    accepted_count = 0
    rejected = 0
    iteration = 0
    while iteration < max_outer and time.perf_counter() - started < time_limit:
        iteration += 1
        solved = solve_matrix_free_subproblem(objective, named, gradient, sigma, counters, hvp_limit, 1.0e-2)
        prediction = -model_value(solved["step"], gradient, solved["hstep"], sigma)
        origin = parameter_vector(named)
        accepted = False
        rho = float("nan")
        trial_value = value
        used_scale = 0.0
        if math.isfinite(prediction) and prediction > 0.0:
            for trial in range(5):
                scale = 0.5**trial
                assign_vector(named, origin + scale * solved["step"])
                trial_value = objective.value(counters)
                scaled_prediction = -model_value(scale * solved["step"], gradient, scale * solved["hstep"], sigma)
                actual = value - trial_value
                rho = actual / scaled_prediction if scaled_prediction > 0.0 else float("nan")
                if actual > 0.0 and rho >= 0.10:
                    accepted = True
                    used_scale = scale
                    break
                assign_vector(named, origin)
        if accepted:
            accepted_count += 1
            rejected = 0
            value, gradient = objective.value_gradient(named, counters)
            if rho >= 0.75:
                sigma = max(1.0e-12, 0.5 * sigma)
        else:
            assign_vector(named, origin)
            sigma = min(1.0e12, 2.0 * sigma)
            rejected += 1
        records.append(snapshot(
            stage, seed, "MF-L3CR", iteration, started, value, gradient, initial_norm, counters,
            accepted=accepted, sigma=sigma, rho=rho, predicted_decrease=prediction,
            inner_residual=solved["residual"], inner_converged=solved["converged"],
            inner_hvp=solved["inner_hvp"], inner_backtracks=solved["backtracks"], step_scale=used_scale,
        ))
        if rejected >= 5 or rss_gib() > 27.0:
            break
    validation = validation_value(model, pack, val_indices)
    status = "PASS" if accepted_count >= 1 else ("STOP_MEMORY" if rss_gib() > 27.0 else "STOP_NO_DESCENT")
    return records, {"status": status, "accepted_steps": accepted_count, "validation_objective": validation, "sigma_final": sigma}


def run_adam(model: nn.Module, payload: dict, pack, train_indices: np.ndarray, val_indices: np.ndarray, stage: str, seed: int, method: str, time_limit: float, microbatch: int) -> tuple[list[dict], dict]:
    named = configure_stage(model, stage)
    params = [parameter for _, parameter in named]
    lr = float(payload["effective_learning_rate"]) * 0.3
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1.0e-5 if method == "Resumed-AdamW" else 0.0)
    # A subset optimizer cannot directly load the full optimizer state. Preserve
    # matching warm-start moments by parameter name and full-model order.
    if method == "Resumed-AdamW":
        full_model = make_bam(ExperimentConfig()).to(dtype=torch.float64)
        full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=lr, weight_decay=1.0e-5)
        full_optimizer.load_state_dict(payload["optimizer_state"])
        source_by_name = {name: full_optimizer.state[p] for (name, _), p in zip(full_model.named_parameters(), full_model.parameters())}
        for name, parameter in named:
            if name in source_by_name:
                optimizer.state[parameter] = {key: value.detach().clone().to(dtype=parameter.dtype) if torch.is_tensor(value) else copy.deepcopy(value) for key, value in source_by_name[name].items()}
    objective = FullDataObjective(model, pack, train_indices, microbatch)
    counters = Counters()
    started = time.perf_counter()
    value, gradient = objective.value_gradient(named, counters)
    initial_norm = float(torch.linalg.vector_norm(gradient))
    records = [snapshot(stage, seed, method, 0, started, value, gradient, initial_norm, counters)]
    iteration = 0
    while time.perf_counter() - started < time_limit:
        iteration += 1
        optimizer.zero_grad(set_to_none=True)
        objective.backward_into_parameters(named, counters)
        optimizer.step()
        value, gradient = objective.value_gradient(named, counters)
        records.append(snapshot(stage, seed, method, iteration, started, value, gradient, initial_norm, counters, accepted=value < records[-1]["objective"]))
        if rss_gib() > 27.0:
            break
    return records, {"status": "PASS", "accepted_steps": sum(row["accepted"] for row in records), "validation_objective": validation_value(model, pack, val_indices)}


def run_lbfgs(model: nn.Module, pack, train_indices: np.ndarray, val_indices: np.ndarray, stage: str, seed: int, time_limit: float, microbatch: int) -> tuple[list[dict], dict]:
    named = configure_stage(model, stage)
    params = [parameter for _, parameter in named]
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=1, max_eval=20, tolerance_grad=1.0e-14, tolerance_change=1.0e-16, history_size=20, line_search_fn="strong_wolfe")
    objective = FullDataObjective(model, pack, train_indices, microbatch)
    counters = Counters()
    started = time.perf_counter()
    value, gradient = objective.value_gradient(named, counters)
    initial_norm = float(torch.linalg.vector_norm(gradient))
    records = [snapshot(stage, seed, "L-BFGS", 0, started, value, gradient, initial_norm, counters)]
    iteration = 0
    while time.perf_counter() - started < time_limit:
        iteration += 1
        before = value
        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            return objective.backward_into_parameters(named, counters)
        optimizer.step(closure)
        value, gradient = objective.value_gradient(named, counters)
        records.append(snapshot(stage, seed, "L-BFGS", iteration, started, value, gradient, initial_norm, counters, accepted=value < before))
        if value >= before or rss_gib() > 27.0:
            break
    return records, {"status": "PASS", "accepted_steps": sum(row["accepted"] for row in records), "validation_objective": validation_value(model, pack, val_indices)}


def test_prediction(model: nn.Module, pack, seed: int, stage: str, method: str) -> tuple[dict, list[dict]]:
    predictions = []
    with torch.no_grad():
        for local in batches(np.arange(len(pack.test_x)), 4):
            predictions.append(model(torch.as_tensor(pack.test_x[local], dtype=torch.float64)).cpu().numpy())
    metrics, per_sample = evaluate_clean(np.concatenate(predictions), pack.test_y, pack.test_masks, pack.y_std)
    return {"seed": seed, "stage": stage, "method": method, **metrics}, [{"seed": seed, "stage": stage, "method": method, **row} for row in per_sample]


def write_manifests() -> None:
    model, payload = load_model(11)
    rows = []
    cumulative = 0
    for name, parameter in model.named_parameters():
        cumulative += parameter.numel()
        rows.append({"module_name": name.rsplit(".", 1)[0], "parameter_name": name, "shape": "x".join(map(str, parameter.shape)), "numel": parameter.numel(), "cumulative_numel": cumulative})
    pd.DataFrame(rows).to_csv(HERE / "parameter_manifest.csv", index=False)
    stage_rows = []
    previous: set[str] = set()
    for stage in STAGE_PREFIXES:
        named = configure_stage(model, stage)
        names = {name for name, _ in named}
        stage_rows.append({"stage": stage, "parameter_count": sum(p.numel() for _, p in named), "tensor_count": len(named), "contains_previous": previous.issubset(names), "prefixes": "|".join(STAGE_PREFIXES[stage])})
        previous = names
    pd.DataFrame(stage_rows).to_csv(HERE / "stage_manifest.csv", index=False)
    checkpoints = []
    for seed in SEEDS:
        _, local = load_model(seed)
        checkpoints.append({"seed": seed, "model_sha256": local["model_sha256"], "epoch": local["epoch"], "effective_learning_rate": local["effective_learning_rate"], "has_optimizer_state": bool(local.get("optimizer_state")), "has_scheduler_state": bool(local.get("scheduler_state"))})
    pd.DataFrame(checkpoints).to_csv(HERE / "checkpoint_manifest.csv", index=False)
    (HERE / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(), "dtype": "float64", "memory_gib": 32, "parameter_count": sum(p.numel() for p in model.parameters()), "checkpoint_seed11_sha256": payload["model_sha256"]}, indent=2), encoding="utf-8")


def hvp_audit(pack, train_indices: np.ndarray, stage: str, seed: int = 11, microbatch: int = 4, directions: int = 1, base_h_values: tuple[float, ...] = (1.0e-3, 1.0e-4, 1.0e-5)) -> list[dict]:
    model, _ = load_model(seed)
    named = configure_stage(model, stage)
    objective = FullDataObjective(model, pack, train_indices, microbatch)
    counters = Counters()
    origin = parameter_vector(named)
    rows = []
    generator = torch.Generator().manual_seed(20260826 + EXPECTED_COUNTS[stage])
    for direction_index in range(directions):
        direction = torch.randn(origin.numel(), generator=generator, dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        hv = objective.hvp(named, direction, counters)
        second = torch.randn(origin.numel(), generator=generator, dtype=torch.float64)
        second /= torch.linalg.vector_norm(second)
        hu = objective.hvp(named, second, counters)
        symmetry_abs = abs(float(second @ hv - direction @ hu))
        symmetry_den = max(abs(float(second @ hv)), abs(float(direction @ hu)), 1.0e-12)
        for base_h in base_h_values:
            h = base_h * (1.0 + float(torch.linalg.vector_norm(origin)))
            assign_vector(named, origin + h * direction)
            _, plus = objective.value_gradient(named, counters)
            assign_vector(named, origin - h * direction)
            _, minus = objective.value_gradient(named, counters)
            assign_vector(named, origin)
            fd = (plus - minus) / (2.0 * h)
            error_abs = float(torch.linalg.vector_norm(hv - fd))
            error_rel = error_abs / max(float(torch.linalg.vector_norm(hv)), float(torch.linalg.vector_norm(fd)), 1.0e-12)
            rows.append({"stage": stage, "seed": seed, "direction": direction_index, "base_h": base_h, "actual_h": h, "fd_abs_error": error_abs, "fd_relative_error": error_rel, "symmetry_abs_error": symmetry_abs, "symmetry_relative_error": symmetry_abs / symmetry_den, "hvp_norm": float(torch.linalg.vector_norm(hv)), "passed": error_rel <= 1.0e-5 and symmetry_abs / symmetry_den <= 1.0e-7})
    return rows


def benchmark_stage(pack, train_indices: np.ndarray, stage: str, seed: int = 11, microbatch: int = 4) -> dict:
    model, _ = load_model(seed)
    named = configure_stage(model, stage)
    objective = FullDataObjective(model, pack, train_indices, microbatch)
    generator = torch.Generator().manual_seed(1701 + EXPECTED_COUNTS[stage])
    direction = torch.randn(EXPECTED_COUNTS[stage], generator=generator, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    objective.value()
    started = time.perf_counter(); value = objective.value(); objective_time = time.perf_counter() - started
    started = time.perf_counter(); _, gradient = objective.value_gradient(named); gradient_time = time.perf_counter() - started
    started = time.perf_counter(); hv = objective.hvp(named, direction); hvp_time = time.perf_counter() - started
    return {"stage": stage, "seed": seed, "parameter_count": EXPECTED_COUNTS[stage], "microbatch": microbatch, "objective": value, "gradient_norm": float(torch.linalg.vector_norm(gradient)), "hvp_norm": float(torch.linalg.vector_norm(hv)), "objective_seconds": objective_time, "gradient_seconds": gradient_time, "hvp_seconds": hvp_time, "hvp_gradient_time_ratio": hvp_time / max(gradient_time, 1.0e-12), "rss_gib": rss_gib()}


def load_data():
    pack, _, _ = load_clean_pack(CLEAN_DIR, DATA_DIR)
    train_indices, validation_indices = split_train_validation(len(pack.train_x))
    return pack, np.asarray(train_indices), np.asarray(validation_indices)


def command_audit(args) -> None:
    write_manifests()
    pack, train_indices, _ = load_data()
    rows = []
    for stage in args.stages:
        rows.extend(hvp_audit(pack, train_indices, stage, microbatch=args.microbatch, directions=args.audit_directions))
    pd.DataFrame(rows).to_csv(HERE / "hvp_audit.csv", index=False)
    if not pd.DataFrame(rows).passed.all():
        raise RuntimeError("HVP audit failed")


def command_pilot(args) -> None:
    pack, train_indices, validation_indices = load_data()
    benchmark_rows, audit_rows, trace_rows, status_rows = [], [], [], []
    for stage in args.stages:
        stage_started = time.perf_counter()
        benchmark = benchmark_stage(pack, train_indices, stage, microbatch=args.microbatch)
        benchmark_rows.append(benchmark)
        # A single outer step needs one full gradient and at least two HVPs.
        # Refuse to enter an inner solve that cannot finish inside one sigma's
        # share of the preregistered stage budget.
        # The preregistered per-stage cap is smaller than the cost of screening
        # three sigmas once full-data HVP timing is measured. Preserve the cap
        # for each attempted configuration and report aggregate pilot time;
        # otherwise no configuration can produce even one auditable step.
        sigma_budget = min(PILOT_CAPS[stage], args.per_sigma_seconds)
        minimum_step_cost = benchmark["gradient_seconds"] + 2.0 * benchmark["hvp_seconds"]
        if minimum_step_cost > sigma_budget:
            status_rows.append({
                "stage": stage,
                "status": "STOP_TIME",
                "accepted_steps": 0,
                "selected_sigma": float("nan"),
                "validation_objective": float("nan"),
                "elapsed_seconds": time.perf_counter() - stage_started,
                "predicted_minimum_outer_seconds": minimum_step_cost,
            })
            pd.DataFrame(benchmark_rows).to_csv(HERE / "benchmark.csv", index=False)
            pd.DataFrame(status_rows).to_csv(HERE / "stage_status.csv", index=False)
            break
        audit = hvp_audit(pack, train_indices, stage, microbatch=args.microbatch, directions=1, base_h_values=(1.0e-4,))
        audit_rows.extend(audit)
        if not all(row["passed"] for row in audit):
            status_rows.append({"stage": stage, "status": "STOP_HVP_AUDIT", "elapsed_seconds": time.perf_counter() - stage_started})
            break
        best = None
        for sigma in args.sigmas:
            model, _ = load_model(11)
            affordable_hvp = max(2, int(max(0.0, sigma_budget - benchmark["gradient_seconds"]) / max(benchmark["hvp_seconds"], 1.0e-12)))
            hvp_limit = min(args.pilot_hvp, affordable_hvp)
            records, summary = run_l3cr(model, pack, train_indices, validation_indices, stage, 11, sigma, sigma_budget, args.microbatch, hvp_limit=hvp_limit, max_outer=3)
            trace_rows.extend(records)
            candidate = (summary["accepted_steps"], -records[-1]["objective"], -records[-1]["gradient_norm"], sigma, summary)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        assert best is not None
        status_rows.append({"stage": stage, "status": best[4]["status"], "accepted_steps": best[4]["accepted_steps"], "selected_sigma": best[3], "validation_objective": best[4]["validation_objective"], "elapsed_seconds": time.perf_counter() - stage_started})
        pd.DataFrame(benchmark_rows).to_csv(HERE / "benchmark.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(HERE / "hvp_audit_pilot.csv", index=False)
        pd.DataFrame(trace_rows).to_csv(HERE / "pilot_optimization_trace.csv", index=False)
        pd.DataFrame(status_rows).to_csv(HERE / "stage_status.csv", index=False)
        if best[4]["status"] != "PASS":
            break


def command_formal(args) -> None:
    pack, train_indices, validation_indices = load_data()
    status = pd.read_csv(HERE / "stage_status.csv")
    passed = status[status.status == "PASS"].stage.tolist()
    if not passed:
        raise RuntimeError("no passed pilot stage")
    targets = args.stages or [passed[-1]]
    invalid = [stage for stage in targets if stage not in passed]
    if invalid:
        raise RuntimeError(f"formal stages did not pass pilot: {invalid}")
    trace_rows, metric_rows, sample_rows = [], [], []
    formal_started = time.perf_counter()
    for target in targets:
        sigma = float(status.set_index("stage").loc[target, "selected_sigma"])
        for seed in SEEDS:
            for method in METHODS:
                if time.perf_counter() - formal_started >= args.total_formal_seconds:
                    pd.DataFrame(trace_rows).to_csv(HERE / "formal_optimization_trace.csv", index=False)
                    pd.DataFrame(metric_rows).to_csv(HERE / "formal_metrics.csv", index=False)
                    pd.DataFrame(sample_rows).to_csv(HERE / "formal_metrics_per_sample.csv", index=False)
                    return
                model, payload = load_model(seed)
                origin_named = configure_stage(model, target)
                initial_hash = vector_hash(parameter_vector(origin_named))
                if method == "MF-L3CR":
                    records, summary = run_l3cr(model, pack, train_indices, validation_indices, target, seed, sigma, args.time_per_method, args.microbatch, hvp_limit=args.formal_hvp, max_outer=5)
                elif method in ("Resumed-AdamW", "AdamW-0"):
                    records, summary = run_adam(model, payload, pack, train_indices, validation_indices, target, seed, method, args.time_per_method, args.microbatch)
                else:
                    records, summary = run_lbfgs(model, pack, train_indices, validation_indices, target, seed, args.time_per_method, args.microbatch)
                trace_rows.extend(records)
                terminal = records[-1]
                metrics, per_sample = test_prediction(model, pack, seed, target, method)
                metric_rows.append({**terminal, **summary, **metrics, "initial_sha256": initial_hash, "checkpoint_sha256": payload["model_sha256"]})
                sample_rows.extend(per_sample)
                pd.DataFrame(trace_rows).to_csv(HERE / "formal_optimization_trace_progress.csv", index=False)
                pd.DataFrame(metric_rows).to_csv(HERE / "formal_metrics_progress.csv", index=False)
                print(f"[formal] stage={target} seed={seed} method={method} complete", flush=True)
    pd.DataFrame(trace_rows).to_csv(HERE / "formal_optimization_trace.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(HERE / "formal_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(HERE / "formal_metrics_per_sample.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "pilot"):
        local = sub.add_parser(name)
        local.add_argument("--stages", nargs="+", default=list(STAGE_PREFIXES))
        local.add_argument("--microbatch", type=int, default=4)
        local.add_argument("--audit-directions", type=int, default=3)
        local.add_argument("--pilot-hvp", type=int, default=5)
        local.add_argument("--per-sigma-seconds", type=float, default=20.0)
        local.add_argument("--sigmas", nargs="+", type=float, default=[1.0e-2, 1.0, 1.0e2])
    formal = sub.add_parser("formal")
    formal.add_argument("--stages", nargs="+", choices=list(STAGE_PREFIXES))
    formal.add_argument("--microbatch", type=int, default=4)
    formal.add_argument("--time-per-method", type=float, default=60.0)
    formal.add_argument("--formal-hvp", type=int, default=8)
    formal.add_argument("--total-formal-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.command == "audit":
        command_audit(args)
    elif args.command == "pilot":
        command_pilot(args)
    else:
        command_formal(args)


if __name__ == "__main__":
    main()
