#!/usr/bin/env python3
"""Restartable P5/P6 long-time matrix-free L3CR experiment."""

from __future__ import annotations

import os

# These must be set before importing NumPy, PyTorch, or linked BLAS libraries.
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_info = lambda: []
    threadpool_limits = nullcontext


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_progressive_full_parameter_l3cr import (  # noqa: E402
    CLEAN_DIR,
    DATA_DIR,
    EXPECTED_COUNTS,
    ExperimentConfig,
    SEEDS,
    STAGE_PREFIXES,
    assign_vector,
    batches,
    configure_stage,
    evaluate_clean,
    load_clean_pack,
    load_model,
    make_bam,
    model_residual,
    model_value,
    parameter_vector,
    prox_l3,
    split_train_validation,
    vector_hash,
)


STAGES = ("P6", "P5")
METHODS = ("MF-L3CR", "Resumed-AdamW", "AdamW-0", "L-BFGS")
FORMAL_BASELINES = ("Resumed-AdamW", "L-BFGS")
PER_SEED_SAFETY_CAP = 3600.0
AUDIT_BUDGET = 1800.0
SIGMA0 = 1.0e-2
RHO_MIN = 0.10
INNER_HVP_LIMIT = 3
MAX_RSS_GIB = 27.0


def rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0**3 if sys.platform == "darwin" else 1024.0**2)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tensor_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Counters:
    objective: int = 0
    gradient: int = 0
    hvp: int = 0
    validation: int = 0


class OperationTimer:
    def __init__(self, event_path: Path, stage: str, seed: int, method: str) -> None:
        self.event_path = event_path
        self.stage = stage
        self.seed = seed
        self.method = method
        self.estimates: dict[str, float] = {}

    def record(self, operation: str, seconds: float, **extra) -> None:
        previous = self.estimates.get(operation, seconds)
        self.estimates[operation] = max(seconds, 0.65 * previous + 0.35 * seconds)
        append_jsonl(self.event_path, {
            "record_type": "operation", "stage": self.stage, "seed": self.seed,
            "method": self.method, "operation": operation, "seconds": seconds,
            "rss_gib": rss_gib(), "timestamp": time.time(), **extra,
        })

    def can_start(self, operation: str, remaining: float, fallback: float) -> bool:
        estimate = self.estimates.get(operation, fallback)
        return remaining >= 1.20 * estimate + 2.0


class CachedObjective:
    """Deterministic full-data objective with one-time CPU float64 conversion."""

    def __init__(self, model: nn.Module, pack, indices: np.ndarray, microbatch: int = 4) -> None:
        self.model = model
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        self.x = torch.as_tensor(np.asarray(pack.train_x), dtype=torch.float64).contiguous()
        self.y = torch.as_tensor(np.asarray(pack.train_y), dtype=torch.float64).contiguous()
        self.mask = torch.as_tensor(np.asarray(pack.train_masks["material"]), dtype=torch.float64).contiguous()
        self.microbatch = int(microbatch)
        self.total_weight = int(self.mask[self.indices].sum())
        if self.total_weight <= 0:
            raise RuntimeError("empty material mask")

    def local_batches(self) -> Iterable[Tensor]:
        for start in range(0, self.indices.numel(), self.microbatch):
            yield self.indices[start : start + self.microbatch]

    def batch_loss_sum(self, local: Tensor) -> Tensor:
        prediction = self.model(self.x[local])
        return (((prediction - self.y[local]).square().mean(dim=1, keepdim=True)) * self.mask[local]).sum()

    def value(self, counters: Counters | None = None) -> float:
        total = 0.0
        with torch.no_grad():
            for local in self.local_batches():
                total += float(self.batch_loss_sum(local))
        if counters:
            counters.objective += 1
        return total / self.total_weight

    def value_gradient(self, named: Sequence[tuple[str, nn.Parameter]], counters: Counters | None = None) -> tuple[float, Tensor]:
        params = [parameter for _, parameter in named]
        sums = [torch.zeros_like(parameter) for parameter in params]
        total = 0.0
        for local in self.local_batches():
            loss = self.batch_loss_sum(local)
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
            for target, value in zip(sums, gradients):
                if value is not None:
                    target.add_(value.detach())
            total += float(loss.detach())
        if counters:
            counters.objective += 1
            counters.gradient += 1
        return total / self.total_weight, torch.cat([(value / self.total_weight).reshape(-1) for value in sums])

    def hvp(self, named: Sequence[tuple[str, nn.Parameter]], direction: Tensor, counters: Counters | None = None) -> Tensor:
        params = [parameter for _, parameter in named]
        pieces, offset = [], 0
        for parameter in params:
            pieces.append(direction[offset : offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
        sums = [torch.zeros_like(parameter) for parameter in params]
        for local in self.local_batches():
            loss = self.batch_loss_sum(local)
            gradients = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            dot = sum((gradient * piece).sum() for gradient, piece in zip(gradients, pieces) if gradient is not None)
            products = torch.autograd.grad(dot, params, allow_unused=True)
            for target, value in zip(sums, products):
                if value is not None:
                    target.add_(value.detach())
        if counters:
            counters.hvp += 1
        return torch.cat([(value / self.total_weight).reshape(-1) for value in sums])

    def backward(self, named: Sequence[tuple[str, nn.Parameter]], counters: Counters | None = None) -> float:
        for _, parameter in named:
            parameter.grad = None
        total = 0.0
        for local in self.local_batches():
            loss = self.batch_loss_sum(local) / self.total_weight
            loss.backward()
            total += float(loss.detach())
        if counters:
            counters.objective += 1
            counters.gradient += 1
        return total


class CachedValidation:
    def __init__(self, model: nn.Module, pack, indices: np.ndarray, microbatch: int = 4) -> None:
        self.model = model
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        self.x = torch.as_tensor(np.asarray(pack.train_x), dtype=torch.float64).contiguous()
        self.y = torch.as_tensor(np.asarray(pack.train_y), dtype=torch.float64).contiguous()
        self.mask = torch.as_tensor(np.asarray(pack.train_masks["material"]), dtype=torch.float64).contiguous()
        self.microbatch = microbatch

    def value(self, counters: Counters | None = None) -> float:
        total, weight = 0.0, 0
        with torch.no_grad():
            for start in range(0, self.indices.numel(), self.microbatch):
                local = self.indices[start : start + self.microbatch]
                total += float((((self.model(self.x[local]) - self.y[local]).square().mean(dim=1, keepdim=True)) * self.mask[local]).sum())
                weight += int(self.mask[local].sum())
        if counters:
            counters.validation += 1
        return total / weight


def load_data():
    pack, _, _ = load_clean_pack(CLEAN_DIR, DATA_DIR)
    train, validation = split_train_validation(len(pack.train_x))
    return pack, np.asarray(train), np.asarray(validation)


def timed(timer: OperationTimer, operation: str, callback, **extra):
    started = time.perf_counter()
    result = callback()
    timer.record(operation, time.perf_counter() - started, **extra)
    return result


def save_torch_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_audit(stage: str, budget: float = AUDIT_BUDGET) -> dict:
    output = HERE / "audit" / stage
    summary_path = output / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["status"] == "PASS":
        return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "operations.jsonl"
    timer = OperationTimer(event_path, stage, 11, "AUDIT")
    started = time.perf_counter()
    model, payload = load_model(11)
    named = configure_stage(model, stage)
    pack, train_indices, _ = load_data()
    objective = CachedObjective(model, pack, train_indices, microbatch=4)
    origin = parameter_vector(named)
    rows: list[dict] = []
    status = "PASS"

    # Micro-batch invariance is checked on the objective before derivative work.
    values = {}
    for microbatch in (1, 2, 4):
        local = CachedObjective(model, pack, train_indices, microbatch=microbatch)
        values[microbatch] = timed(timer, f"objective_mb{microbatch}", local.value)
    objective_spread = max(values.values()) - min(values.values())

    generator = torch.Generator().manual_seed(20260826 + EXPECTED_COUNTS[stage])
    v = torch.randn(origin.numel(), generator=generator, dtype=torch.float64)
    v /= torch.linalg.vector_norm(v)
    u = torch.randn(origin.numel(), generator=generator, dtype=torch.float64)
    u /= torch.linalg.vector_norm(u)
    remaining = budget - (time.perf_counter() - started)
    if not timer.can_start("hvp", remaining, 180.0):
        status = "INTERRUPTED_TIME_BUDGET"
        hv, hu = None, None
    else:
        hv = timed(timer, "hvp", lambda: objective.hvp(named, v))
        hu = timed(timer, "hvp", lambda: objective.hvp(named, u))

    if hv is not None:
        left, right = float(u @ hv), float(v @ hu)
        symmetry_abs = abs(left - right)
        symmetry_rel = symmetry_abs / max(abs(left), abs(right), 1.0e-12)
        for base_h in (1.0e-3, 1.0e-4, 1.0e-5):
            h = base_h * (1.0 + float(torch.linalg.vector_norm(origin)))
            remaining = budget - (time.perf_counter() - started)
            if not timer.can_start("gradient", remaining, 90.0) or remaining < 2.4 * timer.estimates.get("gradient", 90.0):
                status = "INTERRUPTED_TIME_BUDGET"
                break
            assign_vector(named, origin + h * v)
            _, plus = timed(timer, "gradient", lambda: objective.value_gradient(named))
            assign_vector(named, origin - h * v)
            _, minus = timed(timer, "gradient", lambda: objective.value_gradient(named))
            assign_vector(named, origin)
            fd = (plus - minus) / (2.0 * h)
            error_abs = float(torch.linalg.vector_norm(hv - fd))
            error_rel = error_abs / max(float(torch.linalg.vector_norm(hv)), float(torch.linalg.vector_norm(fd)), 1.0e-12)
            rows.append({"stage": stage, "base_h": base_h, "actual_h": h, "fd_abs_error": error_abs,
                         "fd_relative_error": error_rel, "symmetry_abs_error": symmetry_abs,
                         "symmetry_relative_error": symmetry_rel})
        if len(rows) != 3:
            status = "INTERRUPTED_TIME_BUDGET"
        elif min(row["fd_relative_error"] for row in rows) > 1.0e-5:
            status = "STOP_HVP_AUDIT"
        if symmetry_rel > 1.0e-7:
            status = "STOP_HVP_AUDIT"

    assign_vector(named, origin)
    elapsed = time.perf_counter() - started
    pd.DataFrame(rows).to_csv(output / "hvp_audit.csv", index=False)
    summary = {
        "stage": stage, "status": status, "parameter_count": sum(p.numel() for _, p in named),
        "checkpoint_sha256": payload["model_sha256"], "initial_parameter_sha256": vector_hash(origin),
        "microbatch_values": values, "microbatch_objective_spread": objective_spread,
        "elapsed_seconds": elapsed, "rss_gib": rss_gib(), "operation_estimates": timer.estimates,
        "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "threadpool_info": threadpool_info(),
    }
    if summary["parameter_count"] != EXPECTED_COUNTS[stage] or objective_spread > 1.0e-10:
        summary["status"] = "STOP_NUMERICAL"
    atomic_json(summary_path, summary)
    return summary


def subproblem(objective: CachedObjective, named, gradient: Tensor, sigma: float, counters: Counters,
               timer: OperationTimer, deadline: float, fallback_hvp: float) -> dict | None:
    step = torch.zeros_like(gradient)
    hstep = torch.zeros_like(gradient)
    gnorm2 = float(gradient @ gradient)
    if gnorm2 == 0.0:
        return {"step": step, "hstep": hstep, "residual": 0.0, "converged": True, "inner_hvp": 0, "backtracks": 0}
    if not timer.can_start("hvp", deadline - time.perf_counter(), fallback_hvp):
        return None
    hg = timed(timer, "hvp", lambda: objective.hvp(named, gradient, counters))
    rayleigh = abs(float(gradient @ hg)) / max(gnorm2, 1.0e-30)
    alpha = 0.8 / max(rayleigh, 1.0e-6)
    inner_hvp, backtracks = 1, 0
    residual, converged = float("inf"), False
    while inner_hvp < INNER_HVP_LIMIT:
        trial = prox_l3(step - alpha * (gradient + hstep), alpha, sigma)
        if not timer.can_start("hvp", deadline - time.perf_counter(), fallback_hvp):
            return None
        htrial = timed(timer, "hvp", lambda: objective.hvp(named, trial, counters))
        inner_hvp += 1
        delta = trial - step
        q_current = float(gradient @ step + 0.5 * step @ hstep)
        q_trial = float(gradient @ trial + 0.5 * trial @ htrial)
        majorizer = q_current + float((gradient + hstep) @ delta) + float(delta @ delta) / (2.0 * alpha)
        if q_trial <= majorizer + 1.0e-12:
            step, hstep = trial, htrial
            residual = model_residual(step, gradient, hstep, sigma)
            converged = residual <= 1.0e-2
            if converged:
                break
            alpha *= 1.15
        else:
            alpha *= 0.5
            backtracks += 1
    return {"step": step, "hstep": hstep, "residual": residual, "converged": converged,
            "inner_hvp": inner_hvp, "backtracks": backtracks}


def method_dir(stage: str, seed: int, method: str) -> Path:
    return HERE / "runs" / stage / f"seed_{seed}" / method.replace("-", "_")


def initialize_run(stage: str, seed: int, method: str) -> tuple[nn.Module, dict, list, Path, dict | None]:
    directory = method_dir(stage, seed, method)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.pt"
    model, payload = load_model(seed)
    named = configure_stage(model, stage)
    origin_hash = vector_hash(parameter_vector(named))
    state = None
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["origin_hash"] != origin_hash:
            raise RuntimeError(f"resume origin mismatch: {stage} seed={seed} {method}")
        model.load_state_dict(state["model_state"])
        named = configure_stage(model, stage)
    return model, payload, named, directory, state


def save_run_state(directory: Path, model: nn.Module, origin_hash: str, payload: dict) -> None:
    save_torch_atomic(directory / "state.pt", {"model_state": model.state_dict(), "origin_hash": origin_hash, **payload})


def run_l3cr(stage: str, seed: int, allocation: float, max_accept: int | None = None) -> dict:
    model, payload, named, directory, state = initialize_run(stage, seed, "MF-L3CR")
    event_path = directory / "events.jsonl"
    timer = OperationTimer(event_path, stage, seed, "MF-L3CR")
    audit = json.loads((HERE / "audit" / stage / "summary.json").read_text())
    fallback_gradient = audit["operation_estimates"].get("gradient", 120.0)
    fallback_hvp = audit["operation_estimates"].get("hvp", 240.0)
    pack, train_indices, val_indices = load_data()
    objective = CachedObjective(model, pack, train_indices, 4)
    validation = CachedValidation(model, pack, val_indices, 4)
    counters = Counters(**(state.get("counters", {}) if state else {}))
    consumed = float(state.get("consumed_seconds", 0.0) if state else 0.0)
    sigma = float(state.get("sigma", SIGMA0) if state else SIGMA0)
    iteration = int(state.get("iteration", 0) if state else 0)
    accepted_count = int(state.get("accepted_count", 0) if state else 0)
    best_validation = float(state.get("best_validation", math.inf) if state else math.inf)
    origin_hash = state.get("origin_hash") if state else vector_hash(parameter_vector(named))
    current_value = state.get("current_value") if state else None
    current_gradient = state.get("current_gradient") if state else None
    run_started = time.perf_counter()
    deadline = run_started + max(0.0, allocation - consumed)
    status = "PASS"

    if current_value is None or current_gradient is None:
        if not timer.can_start("gradient", deadline - time.perf_counter(), fallback_gradient):
            status = "NOT_RUN_TIME_BUDGET"
        else:
            current_value, current_gradient = timed(timer, "gradient", lambda: objective.value_gradient(named, counters))
            append_jsonl(event_path, {"record_type": "iterate", "iteration": iteration, "objective": current_value,
                         "gradient_norm": float(torch.linalg.vector_norm(current_gradient)), "accepted": False,
                         "stage": stage, "seed": seed, "method": "MF-L3CR", "timestamp": time.time()})

    consecutive_rejections = 0
    while status == "PASS" and (max_accept is None or accepted_count < max_accept):
        remaining = deadline - time.perf_counter()
        minimum = 2.4 * timer.estimates.get("hvp", fallback_hvp) + 1.2 * timer.estimates.get("objective", audit["operation_estimates"].get("objective_mb4", 60.0))
        if remaining < minimum:
            status = "INTERRUPTED_TIME_BUDGET"
            break
        solved = subproblem(objective, named, current_gradient, sigma, counters, timer, deadline, fallback_hvp)
        if solved is None:
            status = "INTERRUPTED_TIME_BUDGET"
            break
        prediction = -model_value(solved["step"], current_gradient, solved["hstep"], sigma)
        origin = parameter_vector(named)
        accepted, rho, used_scale, trial_value = False, float("nan"), 0.0, current_value
        if math.isfinite(prediction) and prediction > 0.0:
            for trial_index in range(5):
                if not timer.can_start("objective", deadline - time.perf_counter(), audit["operation_estimates"].get("objective_mb4", 60.0)):
                    status = "INTERRUPTED_TIME_BUDGET"
                    break
                scale = 0.5**trial_index
                assign_vector(named, origin + scale * solved["step"])
                trial_value = timed(timer, "objective", lambda: objective.value(counters))
                scaled_prediction = -model_value(scale * solved["step"], current_gradient, scale * solved["hstep"], sigma)
                actual = current_value - trial_value
                rho = actual / scaled_prediction if scaled_prediction > 0.0 else float("nan")
                if actual > 0.0 and rho >= RHO_MIN:
                    accepted, used_scale = True, scale
                    break
                assign_vector(named, origin)
        if status != "PASS":
            assign_vector(named, origin)
            break
        iteration += 1
        if accepted:
            accepted_count += 1
            consecutive_rejections = 0
            if rho >= 0.75:
                sigma = max(1.0e-12, 0.5 * sigma)
            if not timer.can_start("gradient", deadline - time.perf_counter(), fallback_gradient):
                # The accepted state is retained; gradient is deferred to resume.
                current_value, current_gradient = trial_value, None
                status = "INTERRUPTED_TIME_BUDGET"
            else:
                current_value, current_gradient = timed(timer, "gradient", lambda: objective.value_gradient(named, counters))
            if timer.can_start("validation", deadline - time.perf_counter(), audit["operation_estimates"].get("objective_mb4", 60.0) * 0.25):
                val = timed(timer, "validation", lambda: validation.value(counters))
                if val < best_validation:
                    best_validation = val
                    save_torch_atomic(directory / "validation_best.pt", {"model_state": model.state_dict(), "validation": val,
                                                                           "iteration": iteration, "accepted_count": accepted_count})
            step_status = "CERTIFIED_DESCENT" if solved["converged"] else "INEXACT_DESCENT"
        else:
            assign_vector(named, origin)
            sigma = min(1.0e12, 2.0 * sigma)
            consecutive_rejections += 1
            step_status = "REJECTED"
        append_jsonl(event_path, {
            "record_type": "iterate", "stage": stage, "seed": seed, "method": "MF-L3CR",
            "iteration": iteration, "objective": current_value, "gradient_norm": None if current_gradient is None else float(torch.linalg.vector_norm(current_gradient)),
            "accepted": accepted, "accepted_count": accepted_count, "step_status": step_status,
            "sigma": sigma, "rho": rho, "predicted_decrease": prediction,
            "inner_residual": solved["residual"], "inner_converged": solved["converged"],
            "inner_hvp": solved["inner_hvp"], "inner_backtracks": solved["backtracks"],
            "step_scale": used_scale, "counters": asdict(counters), "timestamp": time.time(), "rss_gib": rss_gib(),
        })
        elapsed_now = consumed + (time.perf_counter() - run_started)
        save_run_state(directory, model, origin_hash, {
            "counters": asdict(counters), "consumed_seconds": elapsed_now, "sigma": sigma,
            "iteration": iteration, "accepted_count": accepted_count, "best_validation": best_validation,
            "current_value": current_value, "current_gradient": current_gradient, "status": status,
            "checkpoint_sha256": payload["model_sha256"],
        })
        if consecutive_rejections >= 5:
            status = "STOP_NO_DESCENT"
        if rss_gib() > MAX_RSS_GIB:
            status = "STOP_MEMORY"
        if current_gradient is None:
            break

    consumed += time.perf_counter() - run_started
    save_run_state(directory, model, origin_hash, {
        "counters": asdict(counters), "consumed_seconds": consumed, "sigma": sigma,
        "iteration": iteration, "accepted_count": accepted_count, "best_validation": best_validation,
        "current_value": current_value, "current_gradient": current_gradient, "status": status,
        "checkpoint_sha256": payload["model_sha256"],
    })
    summary = {"stage": stage, "seed": seed, "method": "MF-L3CR", "status": status,
               "allocated_seconds": allocation, "consumed_seconds": consumed, "accepted_steps": accepted_count,
               "objective": current_value, "gradient_norm": None if current_gradient is None else float(torch.linalg.vector_norm(current_gradient)),
               "best_validation": best_validation, "initial_sha256": origin_hash, "counters": asdict(counters)}
    atomic_json(directory / "summary.json", summary)
    return summary


def restore_moments(optimizer, named, payload, lr: float) -> None:
    full_model = make_bam(ExperimentConfig()).to(device="cpu", dtype=torch.float64)
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=lr, weight_decay=1.0e-5)
    full_optimizer.load_state_dict(payload["optimizer_state"])
    source = {name: full_optimizer.state[p] for (name, _), p in zip(full_model.named_parameters(), full_model.parameters())}
    for name, parameter in named:
        if name in source:
            optimizer.state[parameter] = {key: value.detach().clone().to(dtype=parameter.dtype) if torch.is_tensor(value) else copy.deepcopy(value)
                                          for key, value in source[name].items()}


def run_baseline(stage: str, seed: int, method: str, allocation: float) -> dict:
    model, payload, named, directory, state = initialize_run(stage, seed, method)
    event_path = directory / "events.jsonl"
    timer = OperationTimer(event_path, stage, seed, method)
    audit = json.loads((HERE / "audit" / stage / "summary.json").read_text())
    fallback_gradient = audit["operation_estimates"].get("gradient", 120.0)
    pack, train_indices, val_indices = load_data()
    objective = CachedObjective(model, pack, train_indices, 4)
    validation = CachedValidation(model, pack, val_indices, 4)
    counters = Counters(**(state.get("counters", {}) if state else {}))
    consumed = float(state.get("consumed_seconds", 0.0) if state else 0.0)
    iteration = int(state.get("iteration", 0) if state else 0)
    best_validation = float(state.get("best_validation", math.inf) if state else math.inf)
    current_value = state.get("current_value") if state else None
    current_gradient = state.get("current_gradient") if state else None
    origin_hash = state.get("origin_hash") if state else vector_hash(parameter_vector(named))
    lr = float(payload["effective_learning_rate"]) * 0.3
    if method in ("Resumed-AdamW", "AdamW-0"):
        optimizer = torch.optim.AdamW([p for _, p in named], lr=lr, weight_decay=1.0e-5 if method == "Resumed-AdamW" else 0.0)
        if method == "Resumed-AdamW":
            restore_moments(optimizer, named, payload, lr)
    else:
        optimizer = torch.optim.LBFGS([p for _, p in named], lr=1.0, max_iter=1, max_eval=20,
                                      tolerance_grad=1.0e-14, tolerance_change=1.0e-16,
                                      history_size=20, line_search_fn="strong_wolfe")
    if state and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    run_started = time.perf_counter()
    deadline = run_started + max(0.0, allocation - consumed)
    status = "PASS"
    if current_value is None:
        if not timer.can_start("gradient", deadline - time.perf_counter(), fallback_gradient):
            status = "NOT_RUN_TIME_BUDGET"
        else:
            current_value, current_gradient = timed(timer, "gradient", lambda: objective.value_gradient(named, counters))
    while status == "PASS":
        remaining = deadline - time.perf_counter()
        predicted = timer.estimates.get("update", 2.5 * fallback_gradient)
        if remaining < 1.2 * predicted + fallback_gradient:
            status = "INTERRUPTED_TIME_BUDGET"
            break
        before = current_value
        update_started = time.perf_counter()
        if method in ("Resumed-AdamW", "AdamW-0"):
            optimizer.zero_grad(set_to_none=True)
            objective.backward(named, counters)
            optimizer.step()
        else:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                return torch.tensor(objective.backward(named, counters), dtype=torch.float64, requires_grad=True)
            optimizer.step(closure)
        timer.record("update", time.perf_counter() - update_started)
        if not timer.can_start("gradient", deadline - time.perf_counter(), fallback_gradient):
            current_value, current_gradient = None, None
            status = "INTERRUPTED_TIME_BUDGET"
            break
        current_value, current_gradient = timed(timer, "gradient", lambda: objective.value_gradient(named, counters))
        iteration += 1
        accepted = current_value < before
        if timer.can_start("validation", deadline - time.perf_counter(), audit["operation_estimates"].get("objective_mb4", 60.0) * 0.25):
            val = timed(timer, "validation", lambda: validation.value(counters))
            if val < best_validation:
                best_validation = val
                save_torch_atomic(directory / "validation_best.pt", {"model_state": model.state_dict(), "validation": val, "iteration": iteration})
        append_jsonl(event_path, {"record_type": "iterate", "stage": stage, "seed": seed, "method": method,
                                  "iteration": iteration, "objective": current_value,
                                  "gradient_norm": float(torch.linalg.vector_norm(current_gradient)), "accepted": accepted,
                                  "counters": asdict(counters), "timestamp": time.time(), "rss_gib": rss_gib()})
        elapsed_now = consumed + time.perf_counter() - run_started
        save_run_state(directory, model, origin_hash, {"optimizer_state": optimizer.state_dict(), "counters": asdict(counters),
                         "consumed_seconds": elapsed_now, "iteration": iteration, "best_validation": best_validation,
                         "current_value": current_value, "current_gradient": current_gradient, "status": status,
                         "checkpoint_sha256": payload["model_sha256"]})
        if not accepted and method == "L-BFGS":
            status = "STOP_NO_DESCENT"
        if rss_gib() > MAX_RSS_GIB:
            status = "STOP_MEMORY"
    consumed += time.perf_counter() - run_started
    save_run_state(directory, model, origin_hash, {"optimizer_state": optimizer.state_dict(), "counters": asdict(counters),
                     "consumed_seconds": consumed, "iteration": iteration, "best_validation": best_validation,
                     "current_value": current_value, "current_gradient": current_gradient, "status": status,
                     "checkpoint_sha256": payload["model_sha256"]})
    summary = {"stage": stage, "seed": seed, "method": method, "status": status,
               "allocated_seconds": allocation, "consumed_seconds": consumed, "iterations": iteration,
               "objective": current_value, "gradient_norm": None if current_gradient is None else float(torch.linalg.vector_norm(current_gradient)),
               "best_validation": best_validation, "initial_sha256": origin_hash, "counters": asdict(counters)}
    atomic_json(directory / "summary.json", summary)
    return summary


def l3cr_allocations(stage: str) -> dict[int, float]:
    path = HERE / "allocations" / f"{stage}_l3cr.json"
    if path.exists():
        return {int(key): float(value) for key, value in json.loads(path.read_text()).items()}
    allocations = {seed: PER_SEED_SAFETY_CAP for seed in SEEDS}
    atomic_json(path, {str(key): value for key, value in allocations.items()})
    return allocations


def run_stage_l3cr(stage: str) -> list[dict]:
    audit = run_audit(stage)
    if audit["status"] != "PASS":
        raise RuntimeError(f"{stage} audit did not pass: {audit['status']}")
    summaries = [run_l3cr(stage, seed, PER_SEED_SAFETY_CAP, max_accept=3) for seed in SEEDS]
    actual = {
        seed: float(json.loads((method_dir(stage, seed, "MF-L3CR") / "summary.json").read_text())["consumed_seconds"])
        for seed in SEEDS
    }
    atomic_json(HERE / "allocations" / f"{stage}_l3cr.json", {str(k): v for k, v in actual.items()})
    return summaries


def run_stage_baselines(stage: str) -> list[dict]:
    allocation_path = HERE / "allocations" / f"{stage}_l3cr.json"
    if not allocation_path.exists():
        raise RuntimeError(f"run L3CR first for {stage}")
    allocations = {int(k): float(v) for k, v in json.loads(allocation_path.read_text()).items()}
    summaries = []
    for method in FORMAL_BASELINES:
        for seed in SEEDS:
            if allocations.get(seed, 0.0) > 0.0:
                summaries.append(run_baseline(stage, seed, method, allocations[seed]))
    return summaries


def evaluate_completed() -> None:
    pack, _, _ = load_data()
    metric_rows, sample_rows, summary_rows = [], [], []
    for stage in STAGES:
        for seed in SEEDS:
            hashes = []
            initial_model, _ = load_model(seed)
            initial_validation = CachedValidation(initial_model, pack, split_train_validation(len(pack.train_x))[1], 4).value()
            initial_predictions = []
            initial_x = torch.as_tensor(np.asarray(pack.test_x), dtype=torch.float64)
            with torch.no_grad():
                for start in range(0, len(initial_x), 4):
                    initial_predictions.append(initial_model(initial_x[start : start + 4]).numpy())
            initial_metrics, initial_samples = evaluate_clean(
                np.concatenate(initial_predictions), pack.test_y, pack.test_masks, pack.y_std
            )
            metric_rows.append({"stage": stage, "seed": seed, "method": "Frozen", "selection": "initial",
                                "validation_objective": initial_validation, **initial_metrics})
            sample_rows.extend({"stage": stage, "seed": seed, "method": "Frozen", "selection": "initial", **row}
                               for row in initial_samples)
            for method in METHODS:
                directory = method_dir(stage, seed, method)
                summary_path = directory / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                summary_rows.append(summary)
                hashes.append(summary["initial_sha256"])
                best_path = directory / "validation_best.pt"
                if not best_path.exists():
                    continue
                best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
                if initial_validation <= float(best_payload["validation"]):
                    model, _ = load_model(seed)
                    selection = "initial"
                    selected_validation = initial_validation
                else:
                    model, _ = load_model(seed)
                    model.load_state_dict(best_payload["model_state"])
                    selection = "refined"
                    selected_validation = float(best_payload["validation"])
                model.eval()
                predictions = []
                x = torch.as_tensor(np.asarray(pack.test_x), dtype=torch.float64)
                with torch.no_grad():
                    for start in range(0, len(x), 4):
                        predictions.append(model(x[start : start + 4]).numpy())
                metrics, per_sample = evaluate_clean(np.concatenate(predictions), pack.test_y, pack.test_masks, pack.y_std)
                metric_rows.append({"stage": stage, "seed": seed, "method": method, "selection": selection,
                                    "validation_objective": selected_validation, **metrics})
                sample_rows.extend({"stage": stage, "seed": seed, "method": method, "selection": selection, **row}
                                   for row in per_sample)
            if len(set(hashes)) > 1:
                raise RuntimeError(f"initial hash mismatch for {stage} seed {seed}")
    pd.DataFrame(summary_rows).to_csv(HERE / "run_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(HERE / "prediction_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(HERE / "prediction_metrics_per_sample.csv", index=False)
    operation_rows, iterate_rows = [], []
    for event_file in (HERE / "runs").glob("**/events.jsonl"):
        for row in read_jsonl(event_file):
            (operation_rows if row.get("record_type") == "operation" else iterate_rows).append(row)
    pd.DataFrame(operation_rows).to_csv(HERE / "operation_log.csv", index=False)
    pd.DataFrame(iterate_rows).to_csv(HERE / "optimization_trace.csv", index=False)


def write_config() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / "experiment_config.json", {
        "stages": STAGES, "seeds": SEEDS, "methods": METHODS,
        "formal_methods": ("MF-L3CR",) + FORMAL_BASELINES,
        "adamw0_status": "legacy snapshot retained but not rerun for the corrected table",
        "accepted_steps_per_seed": 3, "per_seed_safety_cap_seconds": PER_SEED_SAFETY_CAP,
        "audit_budget_seconds": AUDIT_BUDGET, "sigma0": SIGMA0, "rho_min": RHO_MIN,
        "inner_hvp_limit": INNER_HVP_LIMIT, "dtype": "float64", "device": "cpu",
        "microbatch": 4, "torch": torch.__version__, "python": platform.python_version(),
        "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
    })


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    write_config()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("stage", choices=STAGES)
    audit.add_argument("--budget", type=float, default=AUDIT_BUDGET)
    l3cr = sub.add_parser("l3cr")
    l3cr.add_argument("stage", choices=STAGES)
    baselines = sub.add_parser("baselines")
    baselines.add_argument("stage", choices=STAGES)
    sub.add_parser("evaluate")
    sub.add_parser("all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "audit":
            print(json.dumps(run_audit(args.stage, args.budget), ensure_ascii=False, indent=2), flush=True)
        elif args.command == "l3cr":
            print(json.dumps(run_stage_l3cr(args.stage), ensure_ascii=False, indent=2), flush=True)
        elif args.command == "baselines":
            print(json.dumps(run_stage_baselines(args.stage), ensure_ascii=False, indent=2), flush=True)
        elif args.command == "evaluate":
            evaluate_completed()
        else:
            for stage in STAGES:
                print(json.dumps(run_audit(stage), ensure_ascii=False, indent=2), flush=True)
                print(json.dumps(run_stage_l3cr(stage), ensure_ascii=False, indent=2), flush=True)
                print(json.dumps(run_stage_baselines(stage), ensure_ascii=False, indent=2), flush=True)
            evaluate_completed()


if __name__ == "__main__":
    main()
