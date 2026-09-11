#!/usr/bin/env python3
"""Small deterministic full-space optimizers for high-accuracy experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor


Objective = Callable[[Tensor], Tensor]
THRESHOLDS = (1.0e-4, 1.0e-6, 1.0e-8)


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float
    max_updates: int = 200
    max_gradient_equivalents: int = 500
    adam_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    sigma_initial: float = 1.0e-2
    sigma_minimum: float = 1.0e-14
    sigma_maximum: float = 1.0e14
    eta_accept: float = 0.10
    eta_very_successful: float = 0.75
    # High-accuracy L3CR controls.  The full-space second-order path uses
    # float64 by default and an adaptive inner forcing tolerance rather than
    # requiring a fixed 1e-12 residual at every outer iteration.
    l3cr_force_float64: bool = True
    # Backward-compatible name: this is now the minimum/floor of the adaptive
    # forcing tolerance, not a fixed tolerance imposed at every outer update.
    inner_tolerance: float = 1.0e-12
    inner_tolerance_max: float = 1.0e-4
    inner_forcing_factor: float = 1.0e-1
    inner_forcing_power: float = 1.5
    inner_iterations: int = 5000
    outer_backtracks: int = 12
    lbfgs_history_size: int = 20
    # A dense Hessian costs O(n) reverse-mode Hessian-vector products per
    # outer update.  For high-accuracy refinement, max_updates/time_limit are
    # therefore the default L3CR stopping budgets.  Set this flag to True only
    # for experiments that explicitly require a gradient-equivalent budget.
    l3cr_enforce_gradient_budget: bool = False


def vector_sha256(vector: Tensor) -> str:
    value = vector.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def value_gradient(objective: Objective, vector: Tensor, create_graph: bool = False) -> tuple[float, Tensor]:
    current = vector.detach().clone().requires_grad_(True)
    value = objective(current)
    gradient = torch.autograd.grad(value, current, create_graph=create_graph)[0]
    return float(value.detach()), gradient


def dense_hessian(objective: Objective, vector: Tensor) -> tuple[float, Tensor, Tensor, float]:
    current = vector.detach().clone().requires_grad_(True)
    value = objective(current)
    gradient = torch.autograd.grad(value, current, create_graph=True)[0]
    columns = []
    for index in range(current.numel()):
        column = torch.autograd.grad(gradient[index], current, retain_graph=index + 1 < current.numel())[0]
        columns.append(column.detach())
    raw = torch.stack(columns, dim=1)
    symmetry = float(torch.linalg.vector_norm(raw - raw.T) / torch.linalg.vector_norm(raw).clamp_min(1.0e-30))
    return float(value.detach()), gradient.detach(), 0.5 * (raw + raw.T), symmetry


def prox_l3(value: Tensor, step_size: float, sigma: float) -> Tensor:
    magnitude = torch.abs(value)
    return torch.sign(value) * (2.0 * magnitude) / (
        1.0 + torch.sqrt(1.0 + 2.0 * step_size * sigma * magnitude)
    )


def model_value(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return gradient @ step + 0.5 * step @ hessian @ step + sigma * torch.sum(torch.abs(step) ** 3) / 6.0


def model_residual(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return gradient + hessian @ step + 0.5 * sigma * torch.abs(step) * step


def norm_three_halves(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value, ord=1.5))


def cauchy_step(gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    g2 = float(gradient @ gradient)
    if g2 <= 1.0e-30:
        return torch.zeros_like(gradient)
    g_h_g = float(gradient @ hessian @ gradient)
    g3 = float(torch.sum(torch.abs(gradient) ** 3))
    coefficient = 0.5 * sigma * g3
    if coefficient <= 1.0e-30:
        alpha = g2 / g_h_g if g_h_g > 0.0 else 1.0
    else:
        discriminant = max(g_h_g * g_h_g + 4.0 * coefficient * g2, 0.0)
        alpha = (-g_h_g + math.sqrt(discriminant)) / (2.0 * coefficient)
    return -alpha * gradient


def adaptive_inner_tolerance(gradient: Tensor, config: SolverConfig) -> float:
    """Return a gradient-dependent forcing target for the cubic solve.

    The target is loose away from stationarity and tightens automatically as
    the outer gradient decreases.  With the defaults it is capped at 1e-4 and
    approaches 1e-12 in the high-accuracy regime.
    """
    gradient_scale = max(float(torch.linalg.vector_norm(gradient)), 0.0)
    raw = config.inner_forcing_factor * gradient_scale ** config.inner_forcing_power
    return min(config.inner_tolerance_max, max(config.inner_tolerance, raw))


def solve_l3_subproblem(gradient: Tensor, hessian: Tensor, sigma: float, config: SolverConfig) -> dict:
    step = cauchy_step(gradient, hessian, sigma)
    spectral = max(float(torch.linalg.matrix_norm(hessian, ord=2)), 1.0e-14)
    alpha = 0.95 / spectral
    denominator = max(1.0, norm_three_halves(gradient))
    target_tolerance = adaptive_inner_tolerance(gradient, config)
    residual_ratio = float("inf")
    backtracks = 0
    converged = False
    pg_limit = min(config.inner_iterations, 500)
    for iteration in range(1, pg_limit + 1):
        quadratic = gradient @ step + 0.5 * step @ hessian @ step
        quadratic_gradient = gradient + hessian @ step
        local_alpha = alpha
        for _ in range(80):
            trial = prox_l3(step - local_alpha * quadratic_gradient, local_alpha, sigma)
            delta = trial - step
            trial_quadratic = gradient @ trial + 0.5 * trial @ hessian @ trial
            majorizer = quadratic + quadratic_gradient @ delta + (delta @ delta) / (2.0 * local_alpha)
            if float(trial_quadratic) <= float(majorizer) + 1.0e-14:
                break
            local_alpha *= 0.5
            backtracks += 1
        step = trial
        residual_ratio = norm_three_halves(model_residual(step, gradient, hessian, sigma)) / denominator
        alpha = min(local_alpha * 1.25, 0.99 / spectral)
        if residual_ratio <= target_tolerance:
            converged = True
            break
    pg_iterations = iteration
    newton_iterations = 0
    # PG is globally safeguarded but can be very slow near a high-accuracy
    # solution.  Semismooth Newton polishes the same stationarity equation.
    if not converged:
        for newton_iterations in range(1, 81):
            residual = model_residual(step, gradient, hessian, sigma)
            residual_norm = float(torch.linalg.vector_norm(residual))
            jacobian = hessian + sigma * torch.diag(torch.abs(step))
            try:
                direction = torch.linalg.solve(jacobian, -residual)
            except RuntimeError:
                direction = torch.linalg.lstsq(jacobian, -residual.unsqueeze(1), rcond=1.0e-14).solution[:, 0]
            accepted_polish = False
            for trial in range(30):
                scale = 0.5**trial
                candidate = step + scale * direction
                candidate_norm = float(torch.linalg.vector_norm(model_residual(candidate, gradient, hessian, sigma)))
                if candidate_norm <= (1.0 - 1.0e-4 * scale) * residual_norm:
                    step = candidate
                    accepted_polish = True
                    break
            residual_ratio = norm_three_halves(model_residual(step, gradient, hessian, sigma)) / denominator
            if residual_ratio <= target_tolerance:
                converged = True
                break
            if not accepted_polish:
                break
    return {
        "step": step,
        "iterations": pg_iterations + newton_iterations,
        "pg_iterations": pg_iterations,
        "newton_iterations": newton_iterations,
        "backtracks": backtracks,
        "residual_ratio": residual_ratio,
        "target_tolerance": target_tolerance,
        "converged": converged,
        "model_value": float(model_value(step, gradient, hessian, sigma)),
    }


def _record(
    records: list[dict], method: str, iteration: int, started: float, objective: Objective,
    vector: Tensor, gradient_equivalents: int, hvp_count: int, **extra,
) -> tuple[float, float]:
    value, gradient = value_gradient(objective, vector)
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    records.append({
        "method": method,
        "iteration": iteration,
        "elapsed_seconds": time.perf_counter() - started,
        "objective": value,
        "gradient_norm": gradient_norm,
        "gradient_evaluations": int(gradient_equivalents - hvp_count),
        "hvp_evaluations": int(hvp_count),
        "gradient_equivalents": int(gradient_equivalents),
        **extra,
    })
    return value, gradient_norm


def run_adamw(initial: Tensor, objective: Objective, config: SolverConfig, method: str) -> tuple[Tensor, list[dict]]:
    vector = initial.detach().clone()
    first = torch.zeros_like(vector)
    second = torch.zeros_like(vector)
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
    records: list[dict] = []
    started = time.perf_counter()
    _, gradient_norm = _record(records, method, 0, started, objective, vector, 1, 0)
    for iteration in range(1, config.max_updates + 1):
        if time.perf_counter() - started >= config.time_limit or iteration >= config.max_gradient_equivalents:
            break
        current = vector.detach().clone().requires_grad_(True)
        value = objective(current)
        gradient = torch.autograd.grad(value, current)[0].detach()
        first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        first_hat = first / (1.0 - beta1**iteration)
        second_hat = second / (1.0 - beta2**iteration)
        vector.mul_(1.0 - config.adam_learning_rate * config.weight_decay)
        vector.addcdiv_(first_hat, torch.sqrt(second_hat) + epsilon, value=-config.adam_learning_rate)
        _, gradient_norm = _record(records, method, iteration, started, objective, vector, iteration + 1, 0)
        if gradient_norm <= min(THRESHOLDS):
            break
    return vector, records


def run_lbfgs(initial: Tensor, objective: Objective, config: SolverConfig) -> tuple[Tensor, list[dict]]:
    parameter = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.LBFGS(
        [parameter], lr=1.0, max_iter=1, max_eval=20,
        tolerance_grad=1.0e-14, tolerance_change=1.0e-16,
        history_size=config.lbfgs_history_size, line_search_fn="strong_wolfe",
    )
    records: list[dict] = []
    started = time.perf_counter()
    closure_calls = 0
    _, gradient_norm = _record(records, "L-BFGS", 0, started, objective, parameter.detach(), 1, 0)
    for iteration in range(1, config.max_updates + 1):
        if time.perf_counter() - started >= config.time_limit or closure_calls >= config.max_gradient_equivalents:
            break

        def closure() -> Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            loss = objective(parameter)
            loss.backward()
            closure_calls += 1
            return loss

        optimizer.step(closure)
        equivalents = closure_calls + 1
        _, gradient_norm = _record(records, "L-BFGS", iteration, started, objective, parameter.detach(), equivalents, 0)
        if gradient_norm <= min(THRESHOLDS):
            break
    return parameter.detach().clone(), records


def run_l3cr(initial: Tensor, objective: Objective, config: SolverConfig) -> tuple[Tensor, list[dict]]:
    # High-accuracy second-order refinement should not inherit a float32
    # training vector.  Casting here also ensures that all subsequent model
    # residuals, Hessians, predicted decreases, and ratio tests use float64.
    vector = initial.detach().clone()
    if config.l3cr_force_float64 and vector.dtype != torch.float64:
        vector = vector.to(dtype=torch.float64)

    sigma = config.sigma_initial
    records: list[dict] = []
    started = time.perf_counter()
    gradient_equivalents = 1
    hvp_count = 0
    _, gradient_norm = _record(
        records, "Full-space L3CR", 0, started, objective, vector,
        gradient_equivalents, hvp_count, working_dtype=str(vector.dtype),
    )
    dimension = vector.numel()

    for iteration in range(1, config.max_updates + 1):
        if time.perf_counter() - started >= config.time_limit:
            break

        projected_cost = gradient_equivalents + dimension + 1
        if config.l3cr_enforce_gradient_budget and projected_cost > config.max_gradient_equivalents:
            break

        before, gradient, hessian, symmetry = dense_hessian(objective, vector)
        gradient_equivalents += dimension + 1
        hvp_count += dimension
        sigma_before = sigma
        solved = solve_l3_subproblem(gradient, hessian, sigma_before, config)
        raw_step = solved["step"]
        raw_prediction = -float(model_value(raw_step, gradient, hessian, sigma_before))
        accepted = False
        rho = float("nan")
        accepted_scale = 0.0
        after = before

        # Do not hard-gate the outer iteration on an arbitrary fixed inner
        # tolerance.  Any finite descent model step is allowed to face the
        # outer actual/predicted decrease test.  The adaptive residual target
        # remains recorded as a quality certificate.
        if torch.isfinite(raw_step).all() and math.isfinite(raw_prediction) and raw_prediction > 0.0:
            for trial in range(config.outer_backtracks + 1):
                scale = 0.5**trial
                step = raw_step * scale
                prediction = -float(model_value(step, gradient, hessian, sigma_before))
                if not math.isfinite(prediction) or prediction <= 0.0:
                    continue
                candidate = vector + step
                after = float(objective(candidate).detach())
                actual = before - after
                rho = actual / prediction
                if actual > 0.0 and rho >= config.eta_accept:
                    vector = candidate.detach()
                    accepted = True
                    accepted_scale = scale
                    break

        if not accepted or not math.isfinite(rho) or rho < config.eta_accept:
            sigma = min(config.sigma_maximum, 2.0 * sigma)
        elif rho >= config.eta_very_successful:
            sigma = max(config.sigma_minimum, 0.5 * sigma)

        _, gradient_norm = _record(
            records, "Full-space L3CR", iteration, started, objective, vector,
            gradient_equivalents + 1, hvp_count,
            accepted=float(accepted), rho=rho, sigma_before=sigma_before, sigma_after=sigma,
            hessian_symmetry_error=symmetry, inner_iterations=solved["iterations"],
            inner_pg_iterations=solved["pg_iterations"], inner_newton_iterations=solved["newton_iterations"],
            inner_residual_ratio=solved["residual_ratio"],
            inner_target_tolerance=solved["target_tolerance"],
            inner_converged=float(solved["converged"]),
            raw_predicted_decrease=raw_prediction, accepted_scale=accepted_scale,
            working_dtype=str(vector.dtype),
        )
        gradient_equivalents += 1
        if gradient_norm <= min(THRESHOLDS):
            break

    return vector, records


def threshold_summary(records: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for threshold in THRESHOLDS:
        reached = next((row for row in records if row["gradient_norm"] <= threshold), None)
        suffix = f"{threshold:.0e}"
        result[f"time_to_grad_{suffix}"] = float(reached["elapsed_seconds"]) if reached else -1.0
        result[f"equivalents_to_grad_{suffix}"] = float(reached["gradient_equivalents"]) if reached else -1.0
        result[f"reached_grad_{suffix}"] = bool(reached)
    return result


def self_tests() -> list[dict]:
    rows: list[dict] = []
    matrix = torch.tensor([[3.0, 0.4], [0.4, 1.5]], dtype=torch.float64)
    linear = torch.tensor([-1.0, 0.2], dtype=torch.float64)
    objective = lambda x: 0.5 * x @ matrix @ x + linear @ x
    point = torch.tensor([0.3, -0.2], dtype=torch.float64)
    _, gradient, hessian, symmetry = dense_hessian(objective, point)
    rows.append({"test": "hessian_symmetry", "value": symmetry, "passed": symmetry < 1.0e-12})
    direction = torch.tensor([0.7, -0.4], dtype=torch.float64)
    epsilon = 1.0e-5
    _, plus = value_gradient(objective, point + epsilon * direction)
    _, minus = value_gradient(objective, point - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    hvp_error = float(torch.linalg.vector_norm(finite_difference - hessian @ direction) / torch.linalg.vector_norm(hessian @ direction))
    rows.append({"test": "hvp_finite_difference", "value": hvp_error, "passed": hvp_error < 1.0e-8})
    config = SolverConfig(time_limit=1.0, sigma_initial=0.1)
    solved = solve_l3_subproblem(gradient, hessian, 0.1, config)
    rows.append({
        "test": "inner_optimality",
        "value": solved["residual_ratio"],
        "target": solved["target_tolerance"],
        "passed": solved["residual_ratio"] <= solved["target_tolerance"],
    })
    return rows
