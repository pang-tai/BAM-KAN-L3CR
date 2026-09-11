#!/usr/bin/env python3
"""Run short full-space L3CR high-accuracy mechanism and BAM-head studies."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from full_space_l3cr_core import (
    SolverConfig,
    run_adamw,
    run_l3cr,
    run_lbfgs,
    self_tests,
    threshold_summary,
    value_gradient,
    vector_sha256,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCH = ROOT / "test080401"
FAIR = ROOT / "test071405"
LEGACY = ROOT / "test071407"
PRIOR = ROOT / "test082301"
for path in (ARCH, FAIR, LEGACY, PRIOR):
    sys.path.insert(0, str(path))

from run_fair_pikan_unet_comparison import split_train_validation  # noqa: E402
from run_high_accuracy_refinement import ExperimentConfig, make_bam  # noqa: E402
from run_leakage_free_architecture_experiment import evaluate_clean, load_clean_pack  # noqa: E402


FORMAL_SEEDS = (101, 211, 307, 401, 503, 601, 709, 809, 907, 1009)
BAM_SEEDS = (11, 23, 37, 51, 73)
METHODS = ("AdamW", "AdamW-0", "L-BFGS", "Full-space L3CR")
DATA_DIR = Path(__file__).resolve().parent.parent / "data_28_06_2026"
CLEAN_DIR = ARCH / "data_clean/leakage_free_v1"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_parameters() -> Tensor:
    generator = torch.Generator().manual_seed(20260824)
    return 0.45 * torch.randn(17, generator=generator, dtype=torch.float64)


def synthetic_forward(vector: Tensor, inputs: Tensor) -> Tensor:
    first_weight = vector[:8].reshape(4, 2)
    first_bias = vector[8:12]
    second_weight = vector[12:16]
    second_bias = vector[16]
    hidden = torch.tanh(inputs @ first_weight.T + first_bias)
    return hidden @ second_weight + second_bias


def synthetic_splits() -> dict[str, tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(20260825)
    teacher = synthetic_parameters()
    result = {}
    for name, count in (("train", 256), ("validation", 128), ("test", 128)):
        inputs = 2.0 * torch.rand((count, 2), generator=generator, dtype=torch.float64) - 1.0
        result[name] = (inputs, synthetic_forward(teacher, inputs).detach())
    return result


def synthetic_objective(split: tuple[Tensor, Tensor]):
    inputs, targets = split
    return lambda vector: 0.5 * torch.mean((synthetic_forward(vector, inputs) - targets) ** 2)


def run_route(initial: Tensor, objective, method: str, config: SolverConfig) -> tuple[Tensor, list[dict]]:
    if method == "AdamW":
        return run_adamw(initial, objective, config, method)
    if method == "AdamW-0":
        return run_adamw(initial, objective, config, method)
    if method == "L-BFGS":
        return run_lbfgs(initial, objective, config)
    if method == "Full-space L3CR":
        return run_l3cr(initial, objective, config)
    raise ValueError(method)


def method_config(base: SolverConfig, method: str, learning_rate: float, sigma: float) -> SolverConfig:
    values = asdict(base)
    values["adam_learning_rate"] = learning_rate
    values["sigma_initial"] = sigma
    values["weight_decay"] = 1.0e-5 if method == "AdamW" else 0.0
    return SolverConfig(**values)


def calibrate(outdir: Path) -> tuple[float, float]:
    splits = synthetic_splits()
    objective = synthetic_objective(splits["train"])
    generator = torch.Generator().manual_seed(11)
    initial = synthetic_parameters() + 1.0e-2 * torch.randn(17, generator=generator, dtype=torch.float64)
    rows = []
    for learning_rate in (1.0e-2, 3.0e-3, 1.0e-3):
        config = SolverConfig(time_limit=1.0, max_updates=200, max_gradient_equivalents=250, adam_learning_rate=learning_rate)
        final, records = run_adamw(initial, objective, config, "AdamW-0")
        value, gradient = value_gradient(objective, final)
        rows.append({"parameter": "adam_learning_rate", "candidate": learning_rate, "objective": value,
                     "gradient_norm": float(torch.linalg.vector_norm(gradient)), "elapsed_seconds": records[-1]["elapsed_seconds"]})
    for sigma in (1.0e-4, 1.0e-2, 1.0):
        config = SolverConfig(time_limit=2.0, max_updates=200, max_gradient_equivalents=500, sigma_initial=sigma)
        final, records = run_l3cr(initial, objective, config)
        value, gradient = value_gradient(objective, final)
        rows.append({"parameter": "sigma_initial", "candidate": sigma, "objective": value,
                     "gradient_norm": float(torch.linalg.vector_norm(gradient)), "elapsed_seconds": records[-1]["elapsed_seconds"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "calibration.csv", index=False)
    best_lr = float(frame[frame.parameter == "adam_learning_rate"].sort_values(["gradient_norm", "objective", "elapsed_seconds"]).iloc[0].candidate)
    best_sigma = float(frame[frame.parameter == "sigma_initial"].sort_values(["gradient_norm", "objective", "elapsed_seconds"]).iloc[0].candidate)
    return best_lr, best_sigma


def run_synthetic(outdir: Path, learning_rate: float, sigma: float, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = synthetic_splits()
    objective = synthetic_objective(splits["train"])
    validation_objective = synthetic_objective(splits["validation"])
    test_objective = synthetic_objective(splits["test"])
    seeds = FORMAL_SEEDS[:1] if smoke else FORMAL_SEEDS
    scales = (1.0e-2,) if smoke else (1.0e-2, 0.2)
    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    base = SolverConfig(time_limit=2.0 if smoke else 10.0, max_updates=200, max_gradient_equivalents=500)
    teacher = synthetic_parameters()
    for scale in scales:
        condition = "near_local_solution" if scale == 1.0e-2 else "far_negative_control"
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            initial = teacher + scale * torch.randn(17, generator=generator, dtype=torch.float64)
            initial_hash = vector_sha256(initial)
            for method in METHODS:
                config = method_config(base, method, learning_rate, sigma)
                final, records = run_route(initial, objective, method, config)
                final_value, final_gradient = value_gradient(objective, final)
                metric_rows.append({
                    "experiment": "synthetic", "condition": condition, "seed": seed, "method": method,
                    "parameter_count": 17, "initial_sha256": initial_hash,
                    "objective": final_value, "objective_gap": final_value,
                    "gradient_norm": float(torch.linalg.vector_norm(final_gradient)),
                    "validation_mse": 2.0 * float(validation_objective(final).detach()),
                    "test_mse": 2.0 * float(test_objective(final).detach()),
                    "elapsed_seconds": records[-1]["elapsed_seconds"],
                    "gradient_evaluations": records[-1]["gradient_evaluations"],
                    "hvp_evaluations": records[-1]["hvp_evaluations"],
                    "gradient_equivalents": records[-1]["gradient_equivalents"],
                    **threshold_summary(records),
                })
                diagnostic_rows.extend({"experiment": "synthetic", "condition": condition, "seed": seed, **row} for row in records)
            pd.DataFrame(metric_rows).to_csv(outdir / "synthetic_metrics_progress.csv", index=False)
            pd.DataFrame(diagnostic_rows).fillna(-1.0).to_csv(outdir / "synthetic_diagnostics_progress.csv", index=False)
            print(f"[synthetic] condition={condition} seed={seed} complete", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows).fillna(-1.0)


def checkpoint_path(seed: int) -> Path:
    return PRIOR / "warmup" / f"seed_{seed}" / "best_checkpoint.pt"


def load_bam(seed: int):
    payload = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    model = make_bam(ExperimentConfig()).to(device="cpu", dtype=torch.float64)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def head_vector(model) -> Tensor:
    return torch.cat([
        model.s1_head[1].weight.detach().reshape(-1), model.s1_head[1].bias.detach().reshape(-1),
        model.u_head[1].weight.detach().reshape(-1), model.u_head[1].bias.detach().reshape(-1),
    ]).to(torch.float64)


def assign_head_vector(model, vector: Tensor) -> None:
    with torch.no_grad():
        model.s1_head[1].weight.copy_(vector[:9].reshape_as(model.s1_head[1].weight))
        model.s1_head[1].bias.copy_(vector[9:10])
        model.u_head[1].weight.copy_(vector[10:19].reshape_as(model.u_head[1].weight))
        model.u_head[1].bias.copy_(vector[19:20])


def collect_head_features(model, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    captured: dict[str, Tensor] = {}
    handles = [
        model.s1_head[0].register_forward_hook(lambda _m, _i, output: captured.__setitem__("s1", output.detach())),
        model.u_head[0].register_forward_hook(lambda _m, _i, output: captured.__setitem__("u", output.detach())),
    ]
    with torch.no_grad():
        prediction = model(inputs)
    for handle in handles:
        handle.remove()
    return captured["s1"], captured["u"], prediction


def accumulate_design(features: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    selected = mask[:, 0].bool()
    design = features.permute(0, 2, 3, 4, 1)[selected]
    design = torch.cat([design, torch.ones((len(design), 1), dtype=design.dtype)], dim=1)
    response = target[:, 0][selected]
    return design.T @ design, design.T @ response, response @ response, int(selected.sum())


def build_bam_problem(model, pack, train_indices: np.ndarray, batch_size: int = 4) -> tuple[dict, dict]:
    gram_s = torch.zeros((10, 10), dtype=torch.float64)
    gram_u = torch.zeros((10, 10), dtype=torch.float64)
    rhs_s = torch.zeros(10, dtype=torch.float64)
    rhs_u = torch.zeros(10, dtype=torch.float64)
    constant_s = torch.zeros((), dtype=torch.float64)
    constant_u = torch.zeros((), dtype=torch.float64)
    count = 0
    direct_numerator = 0.0
    for start in range(0, len(train_indices), batch_size):
        local = train_indices[start : start + batch_size]
        inputs = torch.as_tensor(pack.train_x[local], dtype=torch.float64)
        targets = torch.as_tensor(pack.train_y[local], dtype=torch.float64)
        mask = torch.as_tensor(pack.train_masks["material"][local], dtype=torch.float64)
        s_features, u_features, prediction = collect_head_features(model, inputs)
        g, b, c, n = accumulate_design(s_features, targets[:, 0:1], mask)
        gram_s += g; rhs_s += b; constant_s += c
        g, b, c, n_u = accumulate_design(u_features, targets[:, 1:2], mask)
        gram_u += g; rhs_u += b; constant_u += c
        if n != n_u:
            raise RuntimeError("S1 and U masks differ")
        count += n
        direct_numerator += float((((prediction - targets).square().mean(dim=1, keepdim=True)) * mask).sum())
    hessian = torch.zeros((20, 20), dtype=torch.float64)
    hessian[:10, :10] = gram_s / count
    hessian[10:, 10:] = gram_u / count
    rhs = torch.cat([rhs_s, rhs_u]) / count
    constant = float((constant_s + constant_u) / (2.0 * count))
    objective = lambda vector: 0.5 * vector @ hessian @ vector - rhs @ vector + constant
    initial = head_vector(model)
    quadratic_initial = float(objective(initial).detach())
    direct_initial = direct_numerator / count
    optimum = torch.linalg.lstsq(hessian, rhs.unsqueeze(1), rcond=1.0e-12).solution[:, 0]
    optimum_value = float(objective(optimum).detach())
    rank = int(torch.linalg.matrix_rank(hessian, rtol=1.0e-12, atol=1.0e-14))
    eigenvalues = torch.linalg.eigvalsh(hessian)
    positive = eigenvalues[eigenvalues > 1.0e-14]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    problem = {"objective": objective, "initial": initial, "optimum": optimum, "optimum_value": optimum_value,
               "hessian": hessian, "rhs": rhs}
    audit = {"material_voxels": count, "quadratic_initial": quadratic_initial, "direct_initial": direct_initial,
             "gram_direct_abs_error": abs(quadratic_initial - direct_initial), "hessian_rank": rank,
             "hessian_condition_positive": condition, "optimum_value": optimum_value}
    return problem, audit


def cache_test_features(model, pack, batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    s_rows, u_rows = [], []
    for start in range(0, len(pack.test_x), batch_size):
        local = np.arange(start, min(start + batch_size, len(pack.test_x)))
        inputs = torch.as_tensor(pack.test_x[local], dtype=torch.float64)
        s_features, u_features, _ = collect_head_features(model, inputs)
        s_rows.append(s_features.cpu().numpy())
        u_rows.append(u_features.cpu().numpy())
    return np.concatenate(s_rows), np.concatenate(u_rows)


def prediction_from_features(features: tuple[np.ndarray, np.ndarray], vector: Tensor) -> np.ndarray:
    s_features, u_features = features
    weights = vector.detach().cpu().numpy()
    s1 = np.einsum("ncdhw,c->ndhw", s_features, weights[:9]) + weights[9]
    displacement = np.einsum("ncdhw,c->ndhw", u_features, weights[10:19]) + weights[19]
    return np.stack([s1, displacement], axis=1)


def run_bam(outdir: Path, learning_rate: float, sigma: float, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pack, _, _ = load_clean_pack(CLEAN_DIR, DATA_DIR)
    train_indices, _ = split_train_validation(len(pack.train_x))
    seeds = BAM_SEEDS[:1] if smoke else BAM_SEEDS
    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    audit_rows: list[dict] = []
    test_rows: list[dict] = []
    base = SolverConfig(time_limit=2.0 if smoke else 5.0, max_updates=200, max_gradient_equivalents=500)
    for seed in seeds:
        model, payload = load_bam(seed)
        problem, audit = build_bam_problem(model, pack, train_indices)
        audit_rows.append({"seed": seed, "checkpoint_sha256": payload["model_sha256"], **audit})
        features = cache_test_features(model, pack)
        initial = problem["initial"]
        initial_hash = vector_sha256(initial)
        for method in METHODS:
            config = method_config(base, method, learning_rate, sigma)
            final, records = run_route(initial, problem["objective"], method, config)
            final_value, final_gradient = value_gradient(problem["objective"], final)
            metric_rows.append({
                "experiment": "bam_head", "condition": "full_137_training_cases", "seed": seed, "method": method,
                "parameter_count": 20, "initial_sha256": initial_hash, "source_checkpoint_sha256": payload["model_sha256"],
                "objective": final_value, "objective_gap": max(final_value - problem["optimum_value"], 0.0),
                "gradient_norm": float(torch.linalg.vector_norm(final_gradient)),
                "elapsed_seconds": records[-1]["elapsed_seconds"],
                "gradient_evaluations": records[-1]["gradient_evaluations"],
                "hvp_evaluations": records[-1]["hvp_evaluations"],
                "gradient_equivalents": records[-1]["gradient_equivalents"],
                **threshold_summary(records),
            })
            prediction = prediction_from_features(features, final)
            metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
            test_rows.extend({"seed": seed, "method": method, **row} for row in per_sample)
            metric_rows[-1].update({f"test_{key}": value for key, value in metrics.items()})
            diagnostic_rows.extend({"experiment": "bam_head", "condition": "full_137_training_cases", "seed": seed, **row} for row in records)
        pd.DataFrame(metric_rows).to_csv(outdir / "bam_head_metrics_progress.csv", index=False)
        pd.DataFrame(diagnostic_rows).fillna(-1.0).to_csv(outdir / "bam_head_diagnostics_progress.csv", index=False)
        print(f"[bam-head] seed={seed} complete", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows).fillna(-1.0), pd.DataFrame(audit_rows), pd.DataFrame(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-bam", action="store_true")
    args = parser.parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    tests = pd.DataFrame(self_tests())
    tests.to_csv(outdir / "self_tests.csv", index=False)
    if not bool(tests.passed.all()):
        raise RuntimeError("full-space solver self-tests failed")
    learning_rate, sigma = calibrate(outdir)
    synthetic_metrics, synthetic_diagnostics = run_synthetic(outdir, learning_rate, sigma, args.smoke)
    synthetic_metrics.to_csv(outdir / "synthetic_metrics.csv", index=False)
    synthetic_diagnostics.to_csv(outdir / "synthetic_diagnostics.csv", index=False)
    if not args.skip_bam:
        bam_metrics, bam_diagnostics, bam_audit, bam_test = run_bam(outdir, learning_rate, sigma, args.smoke)
        bam_metrics.to_csv(outdir / "bam_head_metrics.csv", index=False)
        bam_diagnostics.to_csv(outdir / "bam_head_diagnostics.csv", index=False)
        bam_audit.to_csv(outdir / "bam_gram_audit.csv", index=False)
        bam_test.to_csv(outdir / "bam_test_metrics_per_sample.csv", index=False)
    runtime = time.perf_counter() - started
    (outdir / "runtime.json").write_text(json.dumps({
        "total_seconds": runtime, "hard_limit_seconds": 1200, "within_limit": runtime <= 1200,
        "selected_adam_learning_rate": learning_rate, "selected_sigma_initial": sigma,
        "formal_seeds": FORMAL_SEEDS, "bam_seeds": BAM_SEEDS, "smoke": args.smoke,
        "torch_version": torch.__version__, "threads": torch.get_num_threads(), "dtype": "float64",
    }, indent=2), encoding="utf-8")
    if runtime > 1200:
        raise RuntimeError(f"experiment exceeded the 20-minute hard limit: {runtime:.1f}s")
    print(f"[complete] runtime={runtime:.2f}s lr={learning_rate:g} sigma={sigma:g}", flush=True)


if __name__ == "__main__":
    main()
