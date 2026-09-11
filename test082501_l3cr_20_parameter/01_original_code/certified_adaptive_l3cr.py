#!/usr/bin/env python3
"""Certified adaptive full-space L3CR for deterministic small problems."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch
from torch import Tensor


Objective = Callable[[Tensor], Tensor]
THRESHOLDS = (1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float = 2.0
    max_updates: int = 10000
    sigma_initial: float = 1.0e-2
    sigma_minimum: float = 1.0e-14
    sigma_maximum: float = 1.0e14
    eta_accept: float = 0.10
    eta_very_successful: float = 0.75
    inner_tolerance_floor: float = 1.0e-12
    inner_tolerance_maximum: float = 1.0e-4
    inner_forcing_factor: float = 1.0e-1
    inner_forcing_power: float = 1.5
    inner_iterations: int = 5000
    outer_backtracks: int = 12


def value_gradient(objective: Objective, vector: Tensor) -> tuple[float, Tensor]:
    current = vector.detach().clone().requires_grad_(True)
    value = objective(current)
    gradient = torch.autograd.grad(value, current)[0]
    return float(value.detach()), gradient.detach()


def dense_hessian(objective: Objective, vector: Tensor) -> tuple[float, Tensor, Tensor, float]:
    current = vector.detach().clone().requires_grad_(True)
    value = objective(current)
    gradient = torch.autograd.grad(value, current, create_graph=True)[0]
    columns = []
    for index in range(current.numel()):
        column = torch.autograd.grad(
            gradient[index], current, retain_graph=index + 1 < current.numel()
        )[0]
        columns.append(column.detach())
    raw = torch.stack(columns, dim=1)
    denominator = torch.linalg.vector_norm(raw).clamp_min(1.0e-30)
    symmetry = float(torch.linalg.vector_norm(raw - raw.T) / denominator)
    return float(value.detach()), gradient.detach(), 0.5 * (raw + raw.T), symmetry


def prox_l3(value: Tensor, step_size: float, sigma: float) -> Tensor:
    magnitude = torch.abs(value)
    return torch.sign(value) * (2.0 * magnitude) / (
        1.0 + torch.sqrt(1.0 + 2.0 * step_size * sigma * magnitude)
    )


def model_value(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return (
        gradient @ step
        + 0.5 * step @ hessian @ step
        + sigma * torch.sum(torch.abs(step) ** 3) / 6.0
    )


def model_residual(step: Tensor, gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    return gradient + hessian @ step + 0.5 * sigma * torch.abs(step) * step


def norm_three_halves(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value, ord=1.5))


def adaptive_inner_tolerance(gradient: Tensor, config: SolverConfig) -> float:
    scale = float(torch.linalg.vector_norm(gradient))
    raw = config.inner_forcing_factor * scale ** config.inner_forcing_power
    return min(
        config.inner_tolerance_maximum,
        max(config.inner_tolerance_floor, raw),
    )


def cauchy_step(gradient: Tensor, hessian: Tensor, sigma: float) -> Tensor:
    g2 = float(gradient @ gradient)
    if g2 <= 1.0e-30:
        return torch.zeros_like(gradient)
    ghg = float(gradient @ hessian @ gradient)
    g3 = float(torch.sum(torch.abs(gradient) ** 3))
    coefficient = 0.5 * sigma * g3
    if coefficient <= 1.0e-30:
        alpha = g2 / ghg if ghg > 0.0 else 1.0
    else:
        discriminant = max(ghg * ghg + 4.0 * coefficient * g2, 0.0)
        alpha = (-ghg + math.sqrt(discriminant)) / (2.0 * coefficient)
    return -alpha * gradient


def solve_l3_subproblem(
    gradient: Tensor, hessian: Tensor, sigma: float, config: SolverConfig
) -> dict:
    denominator = max(1.0, norm_three_halves(gradient))
    target = adaptive_inner_tolerance(gradient, config)
    step = cauchy_step(gradient, hessian, sigma)
    best_step = step.detach().clone()
    best_model = float(model_value(best_step, gradient, hessian, sigma))
    spectral = max(float(torch.linalg.matrix_norm(hessian, ord=2)), 1.0e-14)
    alpha = 0.95 / spectral
    backtracks = 0
    converged = False
    pg_iterations = 0
    newton_iterations = 0

    for pg_iterations in range(1, min(config.inner_iterations, 500) + 1):
        quadratic = gradient @ step + 0.5 * step @ hessian @ step
        quadratic_gradient = gradient + hessian @ step
        local_alpha = alpha
        trial = step
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
        current_model = float(model_value(step, gradient, hessian, sigma))
        if current_model < best_model:
            best_step = step.detach().clone()
            best_model = current_model
        residual_ratio = norm_three_halves(
            model_residual(step, gradient, hessian, sigma)
        ) / denominator
        alpha = min(local_alpha * 1.25, 0.99 / spectral)
        if residual_ratio <= target and current_model < 0.0:
            converged = True
            break

    if not converged:
        step = best_step.detach().clone()
        for newton_iterations in range(1, 81):
            residual = model_residual(step, gradient, hessian, sigma)
            residual_norm = float(torch.linalg.vector_norm(residual))
            current_model = float(model_value(step, gradient, hessian, sigma))
            jacobian = hessian + sigma * torch.diag(torch.abs(step))
            try:
                direction = torch.linalg.solve(jacobian, -residual)
            except RuntimeError:
                direction = torch.linalg.lstsq(
                    jacobian, -residual.unsqueeze(1), rcond=1.0e-14
                ).solution[:, 0]
            accepted_polish = False
            for trial_index in range(30):
                scale = 0.5**trial_index
                candidate = step + scale * direction
                candidate_residual = float(
                    torch.linalg.vector_norm(
                        model_residual(candidate, gradient, hessian, sigma)
                    )
                )
                candidate_model = float(model_value(candidate, gradient, hessian, sigma))
                residual_decrease = candidate_residual <= (
                    1.0 - 1.0e-4 * scale
                ) * residual_norm
                model_decrease = candidate_model <= current_model + 1.0e-15
                if residual_decrease and model_decrease:
                    step = candidate
                    accepted_polish = True
                    if candidate_model < best_model:
                        best_step = candidate.detach().clone()
                        best_model = candidate_model
                    break
            residual_ratio = norm_three_halves(
                model_residual(step, gradient, hessian, sigma)
            ) / denominator
            if residual_ratio <= target and float(model_value(step, gradient, hessian, sigma)) < 0.0:
                converged = True
                best_step = step.detach().clone()
                best_model = float(model_value(step, gradient, hessian, sigma))
                break
            if not accepted_polish:
                break

    final_step = best_step if converged else step
    residual_ratio = norm_three_halves(
        model_residual(final_step, gradient, hessian, sigma)
    ) / denominator
    return {
        "step": final_step,
        "converged": converged and residual_ratio <= target and best_model < 0.0,
        "target_tolerance": target,
        "residual_ratio": residual_ratio,
        "model_value": float(model_value(final_step, gradient, hessian, sigma)),
        "pg_iterations": pg_iterations,
        "newton_iterations": newton_iterations,
        "backtracks": backtracks,
    }


def _record(
    records: list[dict], iteration: int, started: float, objective: Objective,
    vector: Tensor, gradient_equivalents: int, hvp_count: int, **extra,
) -> float:
    value, gradient = value_gradient(objective, vector)
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    records.append({
        "method": "Certified adaptive L3CR",
        "iteration": iteration,
        "elapsed_seconds": time.perf_counter() - started,
        "objective": value,
        "gradient_norm": gradient_norm,
        "gradient_evaluations": int(gradient_equivalents - hvp_count),
        "hvp_evaluations": int(hvp_count),
        "gradient_equivalents": int(gradient_equivalents),
        "_state": vector.detach().clone(),
        **extra,
    })
    return gradient_norm


def run_l3cr(
    initial: Tensor, objective: Objective, config: SolverConfig
) -> tuple[Tensor, list[dict]]:
    vector = initial.detach().clone().to(device="cpu", dtype=torch.float64)
    sigma = config.sigma_initial
    records: list[dict] = []
    started = time.perf_counter()
    gradient_equivalents = 1
    hvp_count = 0
    gradient_norm = _record(
        records, 0, started, objective, vector, gradient_equivalents, hvp_count,
        accepted=0.0, inner_converged=1.0,
    )
    dimension = vector.numel()

    for iteration in range(1, config.max_updates + 1):
        if time.perf_counter() - started >= config.time_limit:
            break
        before, gradient, hessian, symmetry = dense_hessian(objective, vector)
        gradient_equivalents += dimension + 1
        hvp_count += dimension
        sigma_before = sigma
        solved = solve_l3_subproblem(gradient, hessian, sigma_before, config)
        raw_step = solved["step"]
        raw_prediction = -float(model_value(raw_step, gradient, hessian, sigma_before))
        accepted = False
        accepted_scale = 0.0
        rho = float("nan")

        if solved["converged"] and raw_prediction > 0.0 and torch.isfinite(raw_step).all():
            for trial in range(config.outer_backtracks + 1):
                scale = 0.5**trial
                step = raw_step * scale
                residual_ratio = norm_three_halves(
                    model_residual(step, gradient, hessian, sigma_before)
                ) / max(1.0, norm_three_halves(gradient))
                if residual_ratio > solved["target_tolerance"]:
                    continue
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

        gradient_norm = _record(
            records, iteration, started, objective, vector,
            gradient_equivalents + 1, hvp_count,
            accepted=float(accepted), rho=rho, sigma_before=sigma_before,
            sigma_after=sigma, hessian_symmetry_error=symmetry,
            inner_residual_ratio=solved["residual_ratio"],
            inner_target_tolerance=solved["target_tolerance"],
            inner_converged=float(solved["converged"]),
            raw_predicted_decrease=raw_prediction,
            accepted_scale=accepted_scale,
        )
        gradient_equivalents += 1
        if gradient_norm <= min(THRESHOLDS):
            break
    return vector, records


def self_tests() -> list[dict]:
    rows: list[dict] = []
    matrix = torch.tensor([[3.0, 0.4], [0.4, 1.5]], dtype=torch.float64)
    linear = torch.tensor([-1.0, 0.2], dtype=torch.float64)
    objective = lambda x: 0.5 * x @ matrix @ x + linear @ x
    point = torch.tensor([0.3, -0.2], dtype=torch.float64)
    _, gradient, hessian, symmetry = dense_hessian(objective, point)
    rows.append({"test": "hessian_symmetry", "value": symmetry, "passed": symmetry < 1.0e-10})
    direction = torch.tensor([0.7, -0.4], dtype=torch.float64)
    epsilon = 1.0e-5
    _, plus = value_gradient(objective, point + epsilon * direction)
    _, minus = value_gradient(objective, point - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    hvp_error = float(
        torch.linalg.vector_norm(finite_difference - hessian @ direction)
        / torch.linalg.vector_norm(hessian @ direction)
    )
    rows.append({"test": "hvp_finite_difference", "value": hvp_error, "passed": hvp_error < 1.0e-8})
    config = SolverConfig(time_limit=1.0, sigma_initial=0.1)
    tolerances = [
        adaptive_inner_tolerance(torch.tensor([scale], dtype=torch.float64), config)
        for scale in (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
    ]
    rows.append({
        "test": "adaptive_tolerance_monotone",
        "value": max(tolerances[index + 1] - tolerances[index] for index in range(4)),
        "passed": all(tolerances[index + 1] <= tolerances[index] for index in range(4)),
    })
    rows.append({"test": "adaptive_tolerance_floor", "value": tolerances[-1], "passed": tolerances[-1] == 1.0e-12})
    solved = solve_l3_subproblem(gradient * 1.0e-8, hessian, 0.1, config)
    rows.append({
        "test": "high_accuracy_inner_certificate",
        "value": solved["residual_ratio"],
        "target": solved["target_tolerance"],
        "passed": solved["converged"] and solved["residual_ratio"] <= solved["target_tolerance"],
    })
    return rows
