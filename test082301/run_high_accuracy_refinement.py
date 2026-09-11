#!/usr/bin/env python3
"""Strict BAM-KAN AdamW/L-BFGS/HVP-L3CR high-accuracy refinement study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCH = ROOT / "test080401"
FAIR = ROOT / "test071405"
LEGACY = ROOT / "test071407"
for path in (str(ARCH), str(FAIR), str(LEGACY)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_fair_pikan_unet_comparison import (  # noqa: E402
    Config as TrainingConfig,
    batches,
    normalized_average_mse,
    parameter_count,
    set_seed,
    split_train_validation,
    tensor_batch,
    validation_loss,
)
from run_leakage_free_architecture_experiment import (  # noqa: E402
    choose_models,
    evaluate_clean,
    load_clean_pack,
)


SEEDS = (11, 23, 37, 51, 73)
CLOSURE_INDICES = np.asarray([7, 21, 29, 63, 84, 89, 95, 158], dtype=np.int64)
BUDGETS = (60, 120, 240, 480, 960)
METHODS = ("adamw", "lbfgs", "l3cr_r64", "l3cr_r32")
TOLERANCES = (1.0e-4, 1.0e-5, 1.0e-6)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = SEEDS
    closure_indices: tuple[int, ...] = tuple(int(v) for v in CLOSURE_INDICES)
    budgets_seconds: tuple[int, ...] = BUDGETS
    epochs: int = 48
    batch_size: int = 8
    patience: int = 14
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    eta_min_ratio: float = 0.05
    grad_clip: float = 5.0
    adamw_lr_multiplier: float = 0.3
    validation_slack: float = 0.005
    validation_cadence_seconds: float = 15.0
    lbfgs_history_size: int = 20
    sigma_initial: float = 1.0e6
    sigma_minimum: float = 1.0e-8
    sigma_maximum: float = 1.0e12
    eta_accept: float = 0.10
    eta_very_successful: float = 0.75
    backtracking_trials: int = 8
    inner_tolerance: float = 1.0e-7
    inner_iterations: int = 500
    max_accepted_outer_steps: int = 10
    bootstrap_repetitions: int = 10_000


def training_config(cfg: ExperimentConfig, device: str = "auto") -> TrainingConfig:
    return TrainingConfig(
        seeds=cfg.seeds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eta_min_ratio=cfg.eta_min_ratio,
        patience=cfg.patience,
        grad_clip=cfg.grad_clip,
        bootstrap_repetitions=cfg.bootstrap_repetitions,
        device=device,
        target_parameters=90_000,
    )


def make_bam(cfg: ExperimentConfig) -> nn.Module:
    models, _ = choose_models(training_config(cfg), input_channels=11)
    return models["BAM-KAN"]


def recursive_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_cpu(item) for item in value)
    return copy.deepcopy(value)


def state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_vector(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters() if parameter.requires_grad])


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_optional(values: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if value is None else value.reshape(-1)
        for value, parameter in zip(values, params)
    ])


def capture_rng_state() -> dict:
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if hasattr(torch, "mps"):
        try:
            result["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            result["torch_mps"] = None
    return result


def choose_warmup_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_pack(args):
    return load_clean_pack(args.clean_dir.resolve(), args.data_dir.resolve())[0]


def warmup_checkpoint_path(outdir: Path, seed: int) -> Path:
    return outdir / "warmup" / f"seed_{seed}" / "best_checkpoint.pt"


def run_warmup_seed(seed: int, pack, cfg: ExperimentConfig, outdir: Path, device: torch.device, force: bool) -> None:
    seed_dir = outdir / "warmup" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    destination = warmup_checkpoint_path(outdir, seed)
    if destination.exists() and not force:
        print(f"[warmup] seed={seed} checkpoint exists; skipped", flush=True)
        return
    set_seed(seed)
    model = make_bam(cfg).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * cfg.eta_min_ratio
    )
    train_indices, validation_indices = split_train_validation(len(pack.train_x))
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    best_payload = None
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        accumulated = 0.0
        seen = 0
        for indices in batches(train_indices, cfg.batch_size, rng):
            x, y, mask = tensor_batch(pack, "train", indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_average_mse(model(x), y, mask)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).detach().cpu())
            optimizer.step()
            accumulated += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        scheduler.step()
        synchronize(device)
        current_validation = validation_loss(model, pack, validation_indices, device, cfg.batch_size)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": accumulated / max(seen, 1),
            "validation_loss": current_validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "last_gradient_norm": gradient_norm,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if current_validation < best_loss:
            best_loss = current_validation
            best_epoch = epoch
            stale = 0
            model_state = recursive_cpu(model.state_dict())
            best_payload = {
                "schema_version": 1,
                "seed": seed,
                "epoch": epoch,
                "best_validation_loss": best_loss,
                "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "model_state": model_state,
                "model_sha256": state_sha256(model_state),
                "optimizer_state": recursive_cpu(optimizer.state_dict()),
                "scheduler_state": recursive_cpu(scheduler.state_dict()),
                "scaler_state": None,
                "rng_state": recursive_cpu(capture_rng_state()),
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "config": asdict(cfg),
            }
        else:
            stale += 1
        print(
            f"[warmup] seed={seed} epoch={epoch:02d} train={row['train_loss']:.6g} "
            f"val={current_validation:.6g} best={best_loss:.6g}",
            flush=True,
        )
        if stale >= cfg.patience:
            break
    if best_payload is None:
        raise RuntimeError(f"warm-up seed {seed} produced no checkpoint")
    best_payload["training_seconds"] = time.perf_counter() - started
    best_payload["epochs_completed"] = len(history)
    torch.save(best_payload, destination)
    pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
    (seed_dir / "checkpoint_sha256.txt").write_text(best_payload["model_sha256"] + "\n", encoding="ascii")
    (seed_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"[warmup] seed={seed} best_epoch={best_epoch} sha={best_payload['model_sha256'][:12]}", flush=True)


def material_tensors(pack, indices: np.ndarray, dtype=torch.float64) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(pack.train_x[indices], dtype=dtype, device="cpu"),
        torch.as_tensor(pack.train_y[indices], dtype=dtype, device="cpu"),
        torch.as_tensor(pack.train_masks["material"][indices], dtype=dtype, device="cpu"),
    )


def fixed_objective(model: nn.Module, tensors: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    x, y, mask = tensors
    return normalized_average_mse(model(x), y, mask)


@torch.no_grad()
def validation_value(model: nn.Module, pack, validation_indices: np.ndarray, batch_size: int) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for start in range(0, len(validation_indices), batch_size):
        local = validation_indices[start : start + batch_size]
        x, y, mask = material_tensors(pack, local)
        total += float(fixed_objective(model, (x, y, mask))) * len(local)
        seen += len(local)
    return total / max(seen, 1)


def objective_gradient(model: nn.Module, closure: Callable[[], Tensor], create_graph: bool = False) -> tuple[float, Tensor]:
    params = trainable_parameters(model)
    loss = closure()
    gradients = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    flat = flatten_optional(gradients, params)
    return float(loss.detach()), flat


def relative_full_residual(gradient: Tensor, initial_norm: float) -> float:
    return float(torch.linalg.vector_norm(gradient.detach())) / max(1.0, initial_norm)


def active_residual(gradient: Tensor, active_dimension: int, initial_active_norm: float) -> float:
    dimension = min(active_dimension, gradient.numel())
    active = torch.topk(gradient.detach().abs(), k=dimension).values
    return float(torch.linalg.vector_norm(active)) / max(1.0, initial_active_norm)


def snapshot_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_refinement_model(checkpoint: dict, cfg: ExperimentConfig) -> nn.Module:
    model = make_bam(cfg).to(device="cpu", dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def candidate_record(
    model: nn.Module,
    closure: Callable[[], Tensor],
    pack,
    validation_indices: np.ndarray,
    elapsed: float,
    initial_full_norm: float,
    initial_active_norm: float,
    active_dimension: int,
    counters: Mapping[str, float],
) -> dict:
    objective, gradient = objective_gradient(model, closure)
    return {
        "elapsed_seconds": float(elapsed),
        "closure_objective": objective,
        "validation_objective": validation_value(model, pack, validation_indices, 8),
        "full_gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "relative_full_residual": relative_full_residual(gradient, initial_full_norm),
        "relative_active_residual": active_residual(gradient, active_dimension, initial_active_norm),
        "state": snapshot_state(model),
        **dict(counters),
    }


def choose_budget_candidates(candidates: list[dict], budgets: Sequence[int], validation_limit: float) -> dict[int, dict]:
    selected: dict[int, dict] = {}
    for budget in budgets:
        feasible = [
            row for row in candidates
            if row["elapsed_seconds"] <= budget + 1.0e-9 and row["validation_objective"] <= validation_limit
        ]
        if not feasible:
            feasible = [candidates[0]]
        selected[budget] = min(feasible, key=lambda row: row["validation_objective"])
    return selected


def save_method_results(
    outdir: Path,
    method: str,
    seed: int,
    checkpoint_sha: str,
    candidates: list[dict],
    diagnostics: list[dict],
    budgets: Sequence[int],
    validation_limit: float,
) -> None:
    route_dir = outdir / "refinement" / method / f"seed_{seed}"
    route_dir.mkdir(parents=True, exist_ok=True)
    chosen = choose_budget_candidates(candidates, budgets, validation_limit)
    rows = []
    for budget, candidate in chosen.items():
        payload = {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "budget_seconds": budget,
            "source_checkpoint_sha256": checkpoint_sha,
            **{key: value for key, value in candidate.items() if key != "state"},
            "model_state": candidate["state"],
            "model_sha256": state_sha256(candidate["state"]),
        }
        torch.save(payload, route_dir / f"budget_{budget}.pt")
        rows.append({key: value for key, value in payload.items() if key not in {"model_state"}})
    pd.DataFrame(rows).to_csv(route_dir / "budget_checkpoints.csv", index=False)
    pd.DataFrame([{key: value for key, value in row.items() if key != "state"} for row in candidates]).to_csv(
        route_dir / "candidate_history.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(route_dir / "solver_diagnostics.csv", index=False)
    (route_dir / "completed.json").write_text(
        json.dumps({"method": method, "seed": seed, "source_checkpoint_sha256": checkpoint_sha}, indent=2),
        encoding="utf-8",
    )


def initial_quantities(model: nn.Module, closure: Callable[[], Tensor], active_dimension: int) -> tuple[float, float, Tensor]:
    _, gradient = objective_gradient(model, closure)
    full = float(torch.linalg.vector_norm(gradient))
    dimension = min(active_dimension, gradient.numel())
    active = float(torch.linalg.vector_norm(torch.topk(gradient.abs(), k=dimension).values))
    return full, active, gradient.detach()


def run_adamw(
    model: nn.Module,
    checkpoint: dict,
    closure: Callable[[], Tensor],
    pack,
    validation_indices: np.ndarray,
    cfg: ExperimentConfig,
    max_budget: int,
) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    effective_lr = float(checkpoint["effective_learning_rate"]) * cfg.adamw_lr_multiplier
    for group in optimizer.param_groups:
        group["lr"] = effective_lr
        group["weight_decay"] = cfg.weight_decay
    full0, active0, _ = initial_quantities(model, closure, 64)
    baseline_val = validation_value(model, pack, validation_indices, 8)
    candidates = [candidate_record(model, closure, pack, validation_indices, 0.0, full0, active0, 64, {
        "updates": 0.0, "objective_evaluations": 2.0, "gradient_evaluations": 2.0,
        "hvp_evaluations": 0.0, "effective_learning_rate": effective_lr,
    })]
    diagnostics: list[dict] = []
    started = time.perf_counter()
    next_validation = cfg.validation_cadence_seconds
    updates = 0
    while time.perf_counter() - started < max_budget:
        optimizer.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()
        raw_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        updates += 1
        elapsed = time.perf_counter() - started
        diagnostics.append({
            "iteration": updates, "elapsed_seconds": elapsed, "objective_before_update": float(loss.detach()),
            "raw_gradient_norm": raw_norm, "effective_learning_rate": effective_lr,
        })
        if elapsed >= next_validation or elapsed >= max_budget:
            candidates.append(candidate_record(model, closure, pack, validation_indices, elapsed, full0, active0, 64, {
                "updates": float(updates), "objective_evaluations": float(updates + len(candidates) + 2),
                "gradient_evaluations": float(updates + len(candidates) + 2), "hvp_evaluations": 0.0,
                "effective_learning_rate": effective_lr,
            }))
            next_validation += cfg.validation_cadence_seconds
            if candidates[-1]["relative_full_residual"] <= 1.0e-6:
                break
    assert candidates[0]["validation_objective"] == baseline_val
    return candidates, diagnostics


def run_lbfgs(
    model: nn.Module,
    closure: Callable[[], Tensor],
    pack,
    validation_indices: np.ndarray,
    cfg: ExperimentConfig,
    max_budget: int,
) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1, max_eval=10,
        tolerance_grad=1.0e-9, tolerance_change=1.0e-12,
        history_size=cfg.lbfgs_history_size, line_search_fn="strong_wolfe",
    )
    full0, active0, _ = initial_quantities(model, closure, 64)
    counters = {"objective_evaluations": 1, "gradient_evaluations": 1, "updates": 0}
    candidates = [candidate_record(model, closure, pack, validation_indices, 0.0, full0, active0, 64, {
        **{key: float(value) for key, value in counters.items()}, "hvp_evaluations": 0.0,
    })]
    diagnostics: list[dict] = []
    started = time.perf_counter()
    next_validation = cfg.validation_cadence_seconds
    while time.perf_counter() - started < max_budget:
        call_evaluations = 0

        def lbfgs_closure():
            nonlocal call_evaluations
            optimizer.zero_grad(set_to_none=True)
            value = closure()
            value.backward()
            call_evaluations += 1
            return value

        before = float(closure().detach())
        counters["objective_evaluations"] += 1
        optimizer.step(lbfgs_closure)
        counters["objective_evaluations"] += call_evaluations
        counters["gradient_evaluations"] += call_evaluations
        counters["updates"] += 1
        elapsed = time.perf_counter() - started
        after = float(closure().detach())
        counters["objective_evaluations"] += 1
        diagnostics.append({
            "iteration": counters["updates"], "elapsed_seconds": elapsed,
            "objective_before": before, "objective_after": after,
            "line_search_closure_evaluations": call_evaluations,
        })
        if elapsed >= next_validation or elapsed >= max_budget:
            candidates.append(candidate_record(model, closure, pack, validation_indices, elapsed, full0, active0, 64, {
                **{key: float(value) for key, value in counters.items()}, "hvp_evaluations": 0.0,
            }))
            next_validation += cfg.validation_cadence_seconds
            if candidates[-1]["relative_full_residual"] <= 1.0e-6:
                break
        if after >= before and call_evaluations == 1:
            break
    return candidates, diagnostics


def cubic_model(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return (
        torch.dot(gradient, step)
        + 0.5 * torch.dot(step, hessian @ step)
        + sigma * torch.sum(torch.abs(step) ** 3) / 6.0
    )


def cubic_residual(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return gradient + hessian @ step + 0.5 * sigma * torch.abs(step) * step


def norm_three_halves(vector: Tensor) -> float:
    return float(torch.linalg.vector_norm(vector, ord=1.5))


def prox_l3(vector: Tensor, alpha: float, sigma: float) -> Tensor:
    magnitude = torch.abs(vector)
    return torch.sign(vector) * (2.0 * magnitude / (1.0 + torch.sqrt(1.0 + 2.0 * alpha * sigma * magnitude)))


def cauchy_step(gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    g2 = float(torch.dot(gradient, gradient))
    if g2 <= 0.0:
        return torch.zeros_like(gradient)
    gbg = float(torch.dot(gradient, hessian @ gradient))
    g3 = float(torch.sum(torch.abs(gradient) ** 3))
    coefficient = 0.5 * sigma * g3
    if coefficient <= 1.0e-30:
        alpha = g2 / max(gbg, 1.0e-30) if gbg > 0.0 else 1.0
    else:
        alpha = (-gbg + math.sqrt(max(gbg * gbg + 4.0 * coefficient * g2, 0.0))) / (2.0 * coefficient)
    return -alpha * gradient


def solve_l3_pgd(gradient: Tensor, hessian: Tensor, sigma: float, cfg: ExperimentConfig) -> dict:
    step = cauchy_step(gradient, hessian, sigma)
    spectral = float(torch.linalg.matrix_norm(hessian, ord=2))
    alpha = 1.0 if spectral <= 1.0e-14 else 0.95 / spectral
    denominator = max(1.0, norm_three_halves(gradient))
    backtracks = 0
    mapping_norm = float("inf")
    converged = False
    for iteration in range(1, cfg.inner_iterations + 1):
        q = torch.dot(gradient, step) + 0.5 * torch.dot(step, hessian @ step)
        grad_q = gradient + hessian @ step
        local_alpha = alpha
        while True:
            trial = prox_l3(step - local_alpha * grad_q, local_alpha, sigma)
            delta = trial - step
            q_trial = torch.dot(gradient, trial) + 0.5 * torch.dot(trial, hessian @ trial)
            majorizer = q + torch.dot(grad_q, delta) + torch.dot(delta, delta) / (2.0 * local_alpha)
            if float(q_trial) <= float(majorizer) + 1.0e-13:
                break
            local_alpha *= 0.5
            backtracks += 1
            if local_alpha < 1.0e-18:
                trial = step.clone()
                break
        mapping_norm = float(torch.linalg.vector_norm((step - trial) / max(local_alpha, 1.0e-30)))
        step = trial
        residual_ratio = norm_three_halves(cubic_residual(step, gradient, hessian, sigma)) / denominator
        alpha = min(local_alpha * 1.2, 1.0 if spectral <= 1.0e-14 else 0.99 / spectral)
        if residual_ratio <= cfg.inner_tolerance:
            converged = True
            break
    return {
        "step": step,
        "iterations": iteration,
        "backtracks": backtracks,
        "mapping_norm": mapping_norm,
        "residual_ratio": residual_ratio,
        "model_value": float(cubic_model(step, gradient, hessian, sigma)),
        "converged": converged,
    }


def active_hessian(model: nn.Module, closure: Callable[[], Tensor], active_dimension: int) -> dict:
    params = trainable_parameters(model)
    loss = closure()
    gradients = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    flat = flatten_optional(gradients, params)
    dimension = min(active_dimension, flat.numel())
    active = torch.topk(flat.detach().abs(), k=dimension, sorted=True).indices
    selected = flat[active]
    columns = []
    for index in range(dimension):
        hvp = torch.autograd.grad(selected[index], params, retain_graph=index + 1 < dimension, allow_unused=True)
        columns.append(flatten_optional(hvp, params)[active].detach())
    raw = torch.stack(columns, dim=1)
    symmetry = float(torch.linalg.vector_norm(raw - raw.T) / max(float(torch.linalg.vector_norm(raw)), 1.0e-30))
    hessian = 0.5 * (raw + raw.T)
    return {
        "objective": float(loss.detach()), "full_gradient": flat.detach(), "active_indices": active.detach(),
        "active_gradient": selected.detach(), "hessian": hessian.detach(), "symmetry_error": symmetry,
    }


def assign_active_step(model: nn.Module, origin: list[Tensor], active: Tensor, step: Tensor) -> None:
    params = trainable_parameters(model)
    full = torch.zeros(sum(parameter.numel() for parameter in params), dtype=params[0].dtype)
    full[active] = step
    offset = 0
    with torch.no_grad():
        for parameter, base in zip(params, origin):
            count = parameter.numel()
            parameter.copy_(base + full[offset : offset + count].view_as(parameter))
            offset += count


def restore_parameters(model: nn.Module, origin: list[Tensor]) -> None:
    with torch.no_grad():
        for parameter, base in zip(trainable_parameters(model), origin):
            parameter.copy_(base)


def run_l3cr(
    model: nn.Module,
    closure: Callable[[], Tensor],
    pack,
    validation_indices: np.ndarray,
    cfg: ExperimentConfig,
    max_budget: int,
    active_dimension: int,
) -> tuple[list[dict], list[dict]]:
    full0, active0, _ = initial_quantities(model, closure, active_dimension)
    baseline_val = validation_value(model, pack, validation_indices, 8)
    validation_limit = baseline_val * (1.0 + cfg.validation_slack)
    candidates = [candidate_record(model, closure, pack, validation_indices, 0.0, full0, active0, active_dimension, {
        "updates": 0.0, "accepted_steps": 0.0, "objective_evaluations": 2.0,
        "gradient_evaluations": 2.0, "hvp_evaluations": 0.0,
    })]
    diagnostics: list[dict] = []
    sigma = cfg.sigma_initial
    started = time.perf_counter()
    outer = 0
    accepted_steps = 0
    objective_evaluations = 2
    gradient_evaluations = 2
    hvp_evaluations = 0
    while accepted_steps < cfg.max_accepted_outer_steps and time.perf_counter() - started < max_budget:
        outer += 1
        sigma_before = sigma
        model_data = active_hessian(model, closure, active_dimension)
        objective_evaluations += 1
        gradient_evaluations += 1
        hvp_evaluations += active_dimension
        g = model_data["active_gradient"]
        hessian = model_data["hessian"]
        solved = solve_l3_pgd(g, hessian, sigma_before, cfg)
        raw_step = solved["step"]
        raw_prediction = -float(cubic_model(raw_step, g, hessian, sigma_before))
        origin = [parameter.detach().clone() for parameter in trainable_parameters(model)]
        accepted = False
        applied = torch.zeros_like(raw_step)
        rho = float("nan")
        actual = 0.0
        prediction = 0.0
        validation_after = baseline_val
        trials_used = 0
        for trial in range(cfg.backtracking_trials + 1):
            trials_used = trial
            scale = 0.5**trial
            applied = scale * raw_step
            prediction = -float(cubic_model(applied, g, hessian, sigma_before))
            if not math.isfinite(prediction) or prediction <= 0.0:
                continue
            assign_active_step(model, origin, model_data["active_indices"], applied)
            with torch.no_grad():
                objective_after = float(closure())
            objective_evaluations += 1
            actual = model_data["objective"] - objective_after
            rho = actual / prediction
            validation_after = validation_value(model, pack, validation_indices, 8)
            if actual > 0.0 and rho >= cfg.eta_accept and validation_after <= validation_limit:
                accepted = True
                accepted_steps += 1
                break
            restore_parameters(model, origin)
        if not accepted:
            restore_parameters(model, origin)
            applied.zero_()
            sigma = min(cfg.sigma_maximum, 2.0 * sigma)
        elif rho >= cfg.eta_very_successful:
            sigma = max(cfg.sigma_minimum, 0.5 * sigma)
        elapsed = time.perf_counter() - started
        applied_residual = norm_three_halves(
            cubic_residual(applied, g, hessian, sigma_before)
        ) / max(1.0, norm_three_halves(g))
        row = {
            "outer_iteration": outer, "elapsed_seconds": elapsed, "accepted": float(accepted),
            "accepted_steps": accepted_steps, "objective_before": model_data["objective"],
            "actual_decrease": actual if accepted else 0.0, "predicted_decrease": prediction,
            "raw_predicted_decrease": raw_prediction, "rho": rho, "validation_after": validation_after,
            "sigma_before": sigma_before, "sigma_after": sigma, "active_dimension": active_dimension,
            "hvp_evaluations_total": hvp_evaluations,
            "hessian_symmetry_error": model_data["symmetry_error"], "inner_iterations": solved["iterations"],
            "inner_backtracks": solved["backtracks"], "inner_mapping_norm": solved["mapping_norm"],
            "raw_subproblem_residual_ratio": solved["residual_ratio"],
            "applied_step_residual_ratio": applied_residual, "raw_step_norm": float(torch.linalg.vector_norm(raw_step)),
            "applied_step_norm": float(torch.linalg.vector_norm(applied)), "outer_backtracks": trials_used,
        }
        diagnostics.append(row)
        if accepted and elapsed <= max_budget:
            candidates.append(candidate_record(model, closure, pack, validation_indices, elapsed, full0, active0, active_dimension, {
                "updates": float(outer), "accepted_steps": float(accepted_steps),
                "objective_evaluations": float(objective_evaluations), "gradient_evaluations": float(gradient_evaluations),
                "hvp_evaluations": float(hvp_evaluations),
            }))
            objective_evaluations += 1
            gradient_evaluations += 1
            if candidates[-1]["relative_active_residual"] <= 1.0e-6:
                break
        print(
            f"[l3cr] r={active_dimension} outer={outer} accepted={int(accepted)} "
            f"rho={rho:.4g} inner={solved['residual_ratio']:.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return candidates, diagnostics


def run_refinement_seed(seed: int, method: str, pack, cfg: ExperimentConfig, outdir: Path, max_budget: int, force: bool) -> None:
    completion = outdir / "refinement" / method / f"seed_{seed}" / "completed.json"
    if completion.exists() and not force:
        print(f"[refine] method={method} seed={seed} complete; skipped", flush=True)
        return
    checkpoint = torch.load(warmup_checkpoint_path(outdir, seed), map_location="cpu", weights_only=False)
    if state_sha256(checkpoint["model_state"]) != checkpoint["model_sha256"]:
        raise RuntimeError(f"stored checkpoint hash mismatch for seed {seed}")
    model = load_refinement_model(checkpoint, cfg)
    train_indices, validation_indices = split_train_validation(len(pack.train_x))
    if not set(CLOSURE_INDICES).issubset(set(train_indices)):
        raise RuntimeError("closure indices are not all in the frozen training set")
    closure_tensors = material_tensors(pack, CLOSURE_INDICES)
    closure = lambda: fixed_objective(model, closure_tensors)
    baseline_validation = validation_value(model, pack, validation_indices, 8)
    validation_limit = baseline_validation * (1.0 + cfg.validation_slack)
    started = time.perf_counter()
    if method == "adamw":
        candidates, diagnostics = run_adamw(model, checkpoint, closure, pack, validation_indices, cfg, max_budget)
    elif method == "lbfgs":
        candidates, diagnostics = run_lbfgs(model, closure, pack, validation_indices, cfg, max_budget)
    elif method.startswith("l3cr_r"):
        active_dimension = int(method.split("r")[-1])
        candidates, diagnostics = run_l3cr(model, closure, pack, validation_indices, cfg, max_budget, active_dimension)
    else:
        raise ValueError(method)
    save_method_results(
        outdir, method, seed, checkpoint["model_sha256"], candidates, diagnostics,
        tuple(budget for budget in cfg.budgets_seconds if budget <= max_budget), validation_limit,
    )
    print(f"[refine] method={method} seed={seed} total={time.perf_counter()-started:.1f}s", flush=True)


@torch.no_grad()
def predict_cpu(model: nn.Module, inputs: np.ndarray, batch_size: int = 4) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float64)
        outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs)


def add_percentiles(metrics: dict, prediction: np.ndarray, target: np.ndarray, masks: Mapping[str, np.ndarray]) -> dict:
    error = np.abs(prediction - target)
    for region, mask in (("global", masks["material"]), ("hole", masks["hole"])):
        for channel, label in enumerate(("S1", "U")):
            selected = error[:, channel : channel + 1][mask.astype(bool)]
            metrics[f"{label}_{region}_P95_NORM"] = float(np.quantile(selected, 0.95)) if selected.size else float("nan")
            metrics[f"{label}_{region}_P99_NORM"] = float(np.quantile(selected, 0.99)) if selected.size else float("nan")
    return metrics


def evaluate_checkpoint(payload: dict, pack, cfg: ExperimentConfig) -> tuple[dict, list[dict]]:
    model = make_bam(cfg).to(dtype=torch.float64, device="cpu")
    model.load_state_dict(payload["model_state"])
    prediction = predict_cpu(model, pack.test_x)
    metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
    return add_percentiles(metrics, prediction, pack.test_y, pack.test_masks), per_sample


def seed_bootstrap(differences: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    draws = differences[rng.integers(0, len(differences), size=(repetitions, len(differences)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def exact_sign_test(differences: np.ndarray) -> tuple[float, int]:
    nonzero = differences[np.abs(differences) > 1.0e-15]
    if nonzero.size == 0:
        return 1.0, 0
    wins = int(np.sum(nonzero < 0.0))
    return float(stats.binomtest(wins, len(nonzero), 0.5).pvalue), int(len(nonzero))


def analyze(pack, cfg: ExperimentConfig, outdir: Path) -> None:
    evaluation_dir = outdir / "analysis"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    sample_rows: list[dict] = []
    cost_rows: list[dict] = []
    for seed in cfg.seeds:
        warm = torch.load(warmup_checkpoint_path(outdir, seed), map_location="cpu", weights_only=False)
        warm_payload = {"model_state": warm["model_state"]}
        metrics, samples = evaluate_checkpoint(warm_payload, pack, cfg)
        metric_rows.append({"method": "warmup", "seed": seed, "budget_seconds": 0, **metrics})
        sample_rows.extend({"method": "warmup", "seed": seed, "budget_seconds": 0, **row} for row in samples)
        for method in METHODS:
            for budget in cfg.budgets_seconds:
                path = outdir / "refinement" / method / f"seed_{seed}" / f"budget_{budget}.pt"
                if not path.exists():
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                metrics, samples = evaluate_checkpoint(payload, pack, cfg)
                metric_rows.append({"method": method, "seed": seed, "budget_seconds": budget, **metrics})
                sample_rows.extend({"method": method, "seed": seed, "budget_seconds": budget, **row} for row in samples)
                cost_rows.append({key: value for key, value in payload.items() if key not in {"model_state"}})
    metrics_frame = pd.DataFrame(metric_rows)
    samples_frame = pd.DataFrame(sample_rows)
    costs_frame = pd.DataFrame(cost_rows)
    metrics_frame.to_csv(evaluation_dir / "metrics_per_seed.csv", index=False)
    samples_frame.to_csv(evaluation_dir / "metrics_per_sample.csv", index=False)
    costs_frame.to_csv(evaluation_dir / "cost_and_stationarity.csv", index=False)
    summary = metrics_frame.groupby(["method", "budget_seconds"]).agg(["mean", "std", "median"])
    summary.to_csv(evaluation_dir / "metrics_summary.csv")

    comparisons = []
    rng = np.random.default_rng(20260823)
    for budget in cfg.budgets_seconds:
        subset = metrics_frame[metrics_frame["budget_seconds"] == budget]
        for proposed, baseline in (("l3cr_r64", "adamw"), ("l3cr_r64", "lbfgs"), ("l3cr_r32", "adamw")):
            for metric in ("global_score", "hole_score", "balanced_score"):
                pivot = subset.pivot(index="seed", columns="method", values=metric)
                if proposed not in pivot.columns or baseline not in pivot.columns:
                    continue
                pivot = pivot.dropna(subset=[proposed, baseline])
                if pivot.empty:
                    continue
                difference = (pivot[proposed] - pivot[baseline]).to_numpy(dtype=float)
                lower, upper = seed_bootstrap(difference, cfg.bootstrap_repetitions, rng)
                effect = float(difference.mean() / difference.std(ddof=1)) if len(difference) > 1 and difference.std(ddof=1) > 0 else float("nan")
                sign_p, sign_pairs = exact_sign_test(difference)
                comparisons.append({
                    "budget_seconds": budget, "comparison": f"{proposed} - {baseline}", "metric": metric,
                    "mean_difference": float(difference.mean()), "std_difference": float(difference.std(ddof=1)),
                    "ci95_lower": lower, "ci95_upper": upper, "wins_proposed": int(np.sum(difference < 0)),
                    "paired_seeds": len(difference), "nonzero_sign_pairs": sign_pairs,
                    "cohen_dz": effect, "exact_sign_test_p": sign_p,
                })
        warmup = metrics_frame[metrics_frame["method"] == "warmup"].set_index("seed")
        for proposed in ("adamw", "lbfgs", "l3cr_r32", "l3cr_r64"):
            proposed_frame = subset[subset["method"] == proposed].set_index("seed")
            common = proposed_frame.index.intersection(warmup.index)
            for metric in ("global_score", "hole_score", "balanced_score"):
                difference = (
                    proposed_frame.loc[common, metric] - warmup.loc[common, metric]
                ).to_numpy(dtype=float)
                if difference.size == 0:
                    continue
                lower, upper = seed_bootstrap(difference, cfg.bootstrap_repetitions, rng)
                effect = float(difference.mean() / difference.std(ddof=1)) if len(difference) > 1 and difference.std(ddof=1) > 0 else float("nan")
                sign_p, sign_pairs = exact_sign_test(difference)
                comparisons.append({
                    "budget_seconds": budget, "comparison": f"{proposed} - warmup", "metric": metric,
                    "mean_difference": float(difference.mean()), "std_difference": float(difference.std(ddof=1)),
                    "ci95_lower": lower, "ci95_upper": upper, "wins_proposed": int(np.sum(difference < 0)),
                    "paired_seeds": len(difference), "nonzero_sign_pairs": sign_pairs,
                    "cohen_dz": effect, "exact_sign_test_p": sign_p,
                })
    paired = pd.DataFrame(comparisons)
    paired.to_csv(evaluation_dir / "paired_statistics.csv", index=False)
    tolerance_rows = []
    hash_rows = []
    for seed in cfg.seeds:
        source = torch.load(warmup_checkpoint_path(outdir, seed), map_location="cpu", weights_only=False)
        for method in METHODS:
            history_path = outdir / "refinement" / method / f"seed_{seed}" / "candidate_history.csv"
            if not history_path.exists():
                continue
            history = pd.read_csv(history_path)
            residual_column = "relative_active_residual" if method.startswith("l3cr") else "relative_full_residual"
            for tolerance in TOLERANCES:
                reached = history[history[residual_column] <= tolerance]
                tolerance_rows.append({
                    "seed": seed, "method": method, "residual_type": residual_column,
                    "tolerance": tolerance, "reached": not reached.empty,
                    "time_seconds": float(reached.elapsed_seconds.iloc[0]) if not reached.empty else float("nan"),
                })
            route_dir = outdir / "refinement" / method / f"seed_{seed}"
            for budget in cfg.budgets_seconds:
                path = route_dir / f"budget_{budget}.pt"
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    hash_rows.append({
                        "seed": seed, "method": method, "budget_seconds": budget,
                        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
                        "warmup_checkpoint_sha256": source["model_sha256"],
                        "model_sha256": payload["model_sha256"],
                        "source_hash_matches": payload["source_checkpoint_sha256"] == source["model_sha256"],
                    })
    pd.DataFrame(tolerance_rows).to_csv(evaluation_dir / "time_to_tolerance.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(evaluation_dir / "checkpoint_hashes.csv", index=False)
    terminal_optimization = []
    for seed in cfg.seeds:
        for method in METHODS:
            path = outdir / "refinement" / method / f"seed_{seed}" / "candidate_history.csv"
            if path.exists():
                row = pd.read_csv(path).iloc[-1].to_dict()
                terminal_optimization.append({"seed": seed, "method": method, **row})
    pd.DataFrame(terminal_optimization).to_csv(evaluation_dir / "terminal_optimization_state.csv", index=False)
    inner_rows = []
    for method in ("l3cr_r32", "l3cr_r64"):
        for seed in cfg.seeds:
            path = outdir / "refinement" / method / f"seed_{seed}" / "solver_diagnostics.csv"
            if path.exists():
                frame = pd.read_csv(path)
                within = frame[frame.elapsed_seconds <= max(cfg.budgets_seconds)]
                inner_rows.append({
                    "method": method, "seed": seed, "completed_outer_steps": len(within),
                    "accepted_outer_steps": int(within.accepted.sum()),
                    "all_inner_residuals_le_1e-7": bool((within.raw_subproblem_residual_ratio <= cfg.inner_tolerance).all()),
                    "max_inner_residual_ratio": float(within.raw_subproblem_residual_ratio.max()),
                    "max_applied_residual_ratio": float(within.applied_step_residual_ratio.max()),
                    "total_hvp": int(within.active_dimension.sum()),
                    "mean_rho": float(within.rho.mean()),
                })
    pd.DataFrame(inner_rows).to_csv(evaluation_dir / "inner_solver_accuracy.csv", index=False)
    write_outputs(metrics_frame, costs_frame, paired, pack, cfg, evaluation_dir)
    validate_outputs(outdir, cfg)


def write_outputs(metrics: pd.DataFrame, costs: pd.DataFrame, paired: pd.DataFrame, pack, cfg: ExperimentConfig, outdir: Path) -> None:
    order = ["warmup", "adamw", "lbfgs", "l3cr_r32", "l3cr_r64"]
    labels = {"warmup": "Warm-up", "adamw": "AdamW", "lbfgs": "L-BFGS", "l3cr_r32": "L3CR r=32", "l3cr_r64": "L3CR r=64"}
    colors = {"warmup": "#6c757d", "adamw": "#2f6f9f", "lbfgs": "#d17a22", "l3cr_r32": "#2a9d72", "l3cr_r64": "#9b4d6f"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for axis, metric, title in zip(axes, ("global_score", "hole_score", "balanced_score"), ("Global", "Opening", "Balanced")):
        for method in order[1:]:
            frame = metrics[metrics.method == method].groupby("budget_seconds")[metric].agg(["mean", "std"])
            if frame.empty:
                continue
            axis.plot(frame.index, frame["mean"], marker="o", label=labels[method], color=colors[method])
            axis.fill_between(frame.index, frame["mean"] - frame["std"].fillna(0), frame["mean"] + frame["std"].fillna(0), alpha=0.12, color=colors[method])
        warmup_mean = float(metrics[metrics.method == "warmup"][metric].mean())
        axis.axhline(warmup_mean, color=colors["warmup"], linestyle="--", linewidth=1.2, label="Warm-up")
        axis.set_title(title)
        axis.set_xlabel("Additional wall-clock time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized error (lower is better)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "accuracy_vs_time.png", dpi=260)
    fig.savefig(outdir / "accuracy_vs_time.pdf")
    plt.close(fig)

    if not costs.empty:
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        for method in order[1:]:
            frame = costs[costs.method == method].groupby("budget_seconds")["relative_full_residual"].agg(["median", "min", "max"])
            if frame.empty:
                continue
            axis.plot(frame.index, frame["median"], marker="o", label=labels[method], color=colors[method])
        axis.set_yscale("log")
        axis.set_xlabel("Additional wall-clock time (s)")
        axis.set_ylabel("Relative full-gradient residual")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "stationarity_vs_time.png", dpi=260)
        fig.savefig(outdir / "stationarity_vs_time.pdf")
        plt.close(fig)

    terminal = metrics[metrics.budget_seconds == max(cfg.budgets_seconds)].groupby("method")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    terminal.to_latex(outdir / "terminal_metrics.tex", float_format="%.6f", escape=False)
    paired.to_latex(outdir / "paired_statistics.tex", index=False, float_format="%.6g", escape=False)
    eligible = int((pack.test_masks["hole"].sum(axis=(1, 2, 3, 4)) > 0).sum())
    verdict = "尚无完整结果。"
    terminal_pair = paired[(paired.budget_seconds.isin([480, 960])) & (paired.comparison == "l3cr_r64 - adamw")]
    success = False
    for budget in (480, 960):
        rows = terminal_pair[terminal_pair.budget_seconds == budget].set_index("metric")
        if {"global_score", "hole_score", "balanced_score"}.issubset(rows.index):
            balanced_ok = rows.loc["balanced_score", "mean_difference"] < 0 and rows.loc["balanced_score", "wins_proposed"] >= 4
            hole_ok = rows.loc["hole_score", "mean_difference"] < 0
            adam_global = float(metrics[(metrics.method == "adamw") & (metrics.budget_seconds == budget)].global_score.mean())
            global_ok = rows.loc["global_score", "mean_difference"] <= 0.005 * adam_global
            success = success or bool(balanced_ok and hole_ok and global_ok)
    baseline_rows = paired[
        (paired.budget_seconds == max(cfg.budgets_seconds))
        & (paired.comparison == "l3cr_r64 - warmup")
    ].set_index("metric")
    net_balanced = float(baseline_rows.loc["balanced_score", "mean_difference"]) if "balanced_score" in baseline_rows.index else float("nan")
    net_global = float(baseline_rows.loc["global_score", "mean_difference"]) if "global_score" in baseline_rows.index else float("nan")
    net_hole = float(baseline_rows.loc["hole_score", "mean_difference"]) if "hole_score" in baseline_rows.index else float("nan")
    if success and net_balanced < 0.0:
        verdict = "L3CR r=64 同时满足相对 resumed AdamW 的预注册判据，并改善 warm-up Balanced 均值。"
    elif success:
        verdict = (
            "L3CR r=64 满足相对 resumed AdamW 的预注册等成本判据，但未改善 warm-up Balanced 均值。"
            "该差异主要来自 AdamW continuation 退化，不能解释为 L3CR 的全面净增益。"
        )
    elif not terminal_pair.empty:
        verdict = "L3CR r=64 未同时满足预注册的等成本优势判据。"
    report = [
        "# BAM-KAN 高精度 refinement 实验报告", "", "## 协议", "",
        f"- 五个冻结种子：`{list(cfg.seeds)}`。", f"- 固定 closure：`{list(cfg.closure_indices)}`。",
        f"- 测试集中 opening-mask 非空案例数：`{eligible}`。", "- 三种方法使用相同 checkpoint、closure、验证限制和 CPU float64 环境。",
        "- 测试集只用于冻结配置后的统一评价。", "", "## 结论", "", verdict, "",
        f"在 960 s，L3CR r=64 相对 warm-up 的 Global、Opening 和 Balanced 均值差分别为 `{net_global:.6g}`、`{net_hole:.6g}` 和 `{net_balanced:.6g}`。负值表示改善。",
        "五个种子的精确双侧符号检验最小 p 值为 0.0625。因此本文报告配对效应量、bootstrap 区间和方向一致性，不使用传统显著性措辞。",
        "active residual 仅描述所选坐标。它不能替代 full-gradient stationarity。Opening 结果属于小样本区域诊断。", "",
        "## 960 s 汇总", "", terminal.to_markdown() if not terminal.empty else "无结果。", "", "## 配对统计", "",
        paired.to_markdown(index=False) if not paired.empty else "无结果。",
    ]
    (outdir / "experiment_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    ledger = pd.DataFrame([
        {"claim": "L3CR has an equal-time prediction advantage over resumed AdamW", "evidence": "paired_statistics.csv", "status": "conditionally supported under the frozen five-seed protocol" if success else "unsupported"},
        {"claim": "L3CR provides a net Balanced improvement over the warm-up checkpoint", "evidence": "paired_statistics.csv", "status": "supported" if net_balanced < 0 else "unsupported"},
        {"claim": "L3CR reaches full-space stationarity", "evidence": "cost_and_stationarity.csv", "status": "unsupported unless relative_full_residual reaches tolerance"},
        {"claim": "L3CR reaches active-space stationarity", "evidence": "cost_and_stationarity.csv", "status": "conditional active-coordinate statement only"},
        {"claim": "Opening improvement generalizes broadly", "evidence": "metrics_per_sample.csv", "status": "unsupported; opening subset is small"},
    ])
    ledger.to_csv(outdir / "claim_evidence_ledger.csv", index=False)


def audit(pack, cfg: ExperimentConfig, outdir: Path) -> None:
    audit_dir = outdir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_indices, validation_indices = split_train_validation(len(pack.train_x))
    opening_train = pack.train_masks["hole"].sum(axis=(1, 2, 3, 4)) > 0
    opening_test = pack.test_masks["hole"].sum(axis=(1, 2, 3, 4)) > 0
    rows = [
        {"test": "split_sizes", "value": f"{len(train_indices)}/{len(validation_indices)}/{len(pack.test_x)}", "passed": len(train_indices) == 137 and len(validation_indices) == 30 and len(pack.test_x) == 19},
        {"test": "closure_in_training", "value": str(list(cfg.closure_indices)), "passed": set(cfg.closure_indices).issubset(set(train_indices))},
        {"test": "closure_disjoint_validation", "value": str(sorted(set(cfg.closure_indices) & set(validation_indices))), "passed": not (set(cfg.closure_indices) & set(validation_indices))},
        {"test": "closure_opening_count", "value": int(opening_train[list(cfg.closure_indices)].sum()), "passed": int(opening_train[list(cfg.closure_indices)].sum()) == 2},
        {"test": "test_opening_count", "value": int(opening_test.sum()), "passed": int(opening_test.sum()) == 3},
    ]
    torch.manual_seed(20260823)
    gradient = torch.randn(7, dtype=torch.float64)
    q = torch.randn(7, 7, dtype=torch.float64)
    hessian = 0.5 * (q + q.T)
    solved = solve_l3_pgd(gradient, hessian, 10.0, cfg)
    rows.extend([
        {"test": "pgd_model_nonincrease", "value": solved["model_value"], "passed": solved["model_value"] <= 1.0e-12},
        {"test": "pgd_true_residual_finite", "value": solved["residual_ratio"], "passed": math.isfinite(solved["residual_ratio"])},
    ])
    vector = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    prox = prox_l3(vector, 0.3, 0.7)
    prox_error = float(torch.linalg.vector_norm((prox - vector) / 0.3 + 0.35 * torch.abs(prox) * prox))
    rows.append({"test": "proximal_optimality", "value": prox_error, "passed": prox_error < 1.0e-10})
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    y = torch.randn(6, 1, dtype=torch.float64)
    closure = lambda: torch.mean((model(x) - y) ** 2)
    params = trainable_parameters(model)
    value = closure()
    grads = torch.autograd.grad(value, params, create_graph=True)
    flat = flatten_optional(grads, params)
    direction = torch.randn_like(flat)
    dot = torch.dot(flat, direction)
    hvp = flatten_optional(torch.autograd.grad(dot, params), params)
    origin = parameter_vector(model)
    epsilon = 1.0e-5
    def set_vector(vector):
        offset = 0
        with torch.no_grad():
            for parameter in params:
                count = parameter.numel(); parameter.copy_(vector[offset:offset+count].view_as(parameter)); offset += count
    set_vector(origin + epsilon * direction)
    _, plus = objective_gradient(model, closure)
    set_vector(origin - epsilon * direction)
    _, minus = objective_gradient(model, closure)
    set_vector(origin)
    fd = (plus - minus) / (2.0 * epsilon)
    hvp_error = float(torch.linalg.vector_norm(fd - hvp) / torch.linalg.vector_norm(hvp).clamp_min(1.0e-30))
    rows.append({"test": "hvp_finite_difference", "value": hvp_error, "passed": hvp_error < 1.0e-5})
    frame = pd.DataFrame(rows)
    frame.to_csv(audit_dir / "self_tests.csv", index=False)
    metadata = {
        "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "cpu_threads": torch.get_num_threads(), "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "parameter_count": parameter_count(make_bam(cfg)), "config": asdict(cfg),
    }
    (audit_dir / "environment.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if not bool(frame.passed.all()):
        raise RuntimeError("audit failed; inspect audit/self_tests.csv")
    print(frame.to_string(index=False), flush=True)


def validate_outputs(outdir: Path, cfg: ExperimentConfig) -> None:
    rows = []
    analysis = outdir / "analysis"
    required = [
        "metrics_per_seed.csv", "metrics_per_sample.csv", "cost_and_stationarity.csv",
        "paired_statistics.csv", "time_to_tolerance.csv", "checkpoint_hashes.csv",
        "terminal_optimization_state.csv", "inner_solver_accuracy.csv",
        "experiment_report_zh.md",
    ]
    for name in required:
        rows.append({"check": f"exists:{name}", "passed": (analysis / name).exists()})
    metrics_path = analysis / "metrics_per_seed.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        numeric = metrics.select_dtypes(include=[np.number])
        rows.append({"check": "metrics_no_inf", "passed": not np.isinf(numeric.to_numpy()).any()})
        rows.append({"check": "five_seeds_present", "passed": set(cfg.seeds).issubset(set(metrics.seed.unique()))})
    hashes_path = analysis / "checkpoint_hashes.csv"
    if hashes_path.exists():
        hashes = pd.read_csv(hashes_path)
        rows.append({"check": "all_source_hashes_match", "passed": bool(hashes.source_hash_matches.all())})
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "output_validation.csv", index=False)
    if not frame.passed.all():
        raise RuntimeError("output validation failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "warmup", "refine", "analyze", "all"))
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "test080401/data_clean/leakage_free_v1")
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data_28_06_2026")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmup-device", default="auto")
    parser.add_argument("--max-budget", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    cfg = ExperimentConfig(
        seeds=tuple(args.seeds), epochs=2 if args.smoke else args.epochs,
        patience=2 if args.smoke else 14,
        budgets_seconds=(5, 10) if args.smoke else tuple(v for v in BUDGETS if v <= args.max_budget),
        max_accepted_outer_steps=1 if args.smoke else 10,
        inner_iterations=40 if args.smoke else 500,
        bootstrap_repetitions=1_000 if args.smoke else 10_000,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "frozen_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    pack = load_pack(args)
    if args.command in ("audit", "all"):
        audit(pack, cfg, args.out_dir)
    if args.command in ("warmup", "all"):
        device = choose_warmup_device(args.warmup_device)
        for seed in cfg.seeds:
            run_warmup_seed(seed, pack, cfg, args.out_dir, device, args.force)
    if args.command in ("refine", "all"):
        torch.set_num_threads(1)
        for seed in cfg.seeds:
            for method in args.methods:
                run_refinement_seed(seed, method, pack, cfg, args.out_dir, max(cfg.budgets_seconds), args.force)
    if args.command in ("analyze", "all"):
        analyze(pack, cfg, args.out_dir)


if __name__ == "__main__":
    main()
