#!/usr/bin/env python3
"""Compare original, user-optimized and certified L3CR on two small problems."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = ROOT / "test082401"
sys.path.insert(0, str(PRIOR))
import run_short_high_accuracy_experiment as prior  # noqa: E402
import certified_adaptive_l3cr as certified  # noqa: E402


FORMAL_SEEDS = (101, 211, 307, 401, 503, 601, 709, 809, 907, 1009)
BAM_SEEDS = (11, 23, 37, 51, 73)
METHODS = (
    "AdamW",
    "AdamW-0",
    "L-BFGS",
    "Original fixed L3CR",
    "User optimized L3CR",
    "Certified adaptive L3CR",
)
THRESHOLDS = (1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
WORK_BUDGETS = (100, 250, 500)
TIME_BUDGETS = (0.1, 0.5, 1.0, 2.0)
LEARNING_RATE = 1.0e-2
SIGMA_INITIAL = 1.0e-2


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


original = load_module("l3cr_original_snapshot", HERE / "original_core_snapshot.py")
optimized = load_module("l3cr_user_optimized_snapshot", HERE / "user_optimized_core_snapshot.py")
original.THRESHOLDS = THRESHOLDS
optimized.THRESHOLDS = THRESHOLDS


def vector_hash(vector: Tensor) -> str:
    value = vector.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def capture_states(module):
    base_record = module._record

    def wrapped(records, method, iteration, started, objective, vector, gradient_equivalents, hvp_count, **extra):
        result = base_record(
            records, method, iteration, started, objective, vector,
            gradient_equivalents, hvp_count, **extra,
        )
        records[-1]["_state"] = vector.detach().clone()
        return result

    module._record = wrapped
    try:
        yield
    finally:
        module._record = base_record


def run_method(method: str, initial: Tensor, objective) -> tuple[Tensor, list[dict]]:
    initial = initial.detach().clone().to(device="cpu", dtype=torch.float64)
    if method in ("AdamW", "AdamW-0", "L-BFGS", "Original fixed L3CR"):
        config = original.SolverConfig(
            time_limit=2.0,
            max_updates=100000,
            max_gradient_equivalents=100000,
            adam_learning_rate=LEARNING_RATE,
            weight_decay=1.0e-5 if method == "AdamW" else 0.0,
            sigma_initial=SIGMA_INITIAL,
        )
        with capture_states(original):
            if method in ("AdamW", "AdamW-0"):
                final, records = original.run_adamw(initial, objective, config, method)
            elif method == "L-BFGS":
                final, records = original.run_lbfgs(initial, objective, config)
            else:
                final, records = original.run_l3cr(initial, objective, config)
    elif method == "User optimized L3CR":
        config = optimized.SolverConfig(
            time_limit=2.0,
            max_updates=100000,
            max_gradient_equivalents=100000,
            sigma_initial=SIGMA_INITIAL,
            l3cr_force_float64=True,
            l3cr_enforce_gradient_budget=False,
        )
        with capture_states(optimized):
            final, records = optimized.run_l3cr(initial, objective, config)
    elif method == "Certified adaptive L3CR":
        config = certified.SolverConfig(
            time_limit=2.0,
            max_updates=100000,
            sigma_initial=SIGMA_INITIAL,
        )
        final, records = certified.run_l3cr(initial, objective, config)
    else:
        raise ValueError(method)

    for row in records:
        row["method"] = method
        if "_state" not in row:
            row["_state"] = final.detach().clone()
    return final, records


def choose_snapshot(records: list[dict], kind: str, budget: float) -> dict:
    field = "gradient_equivalents" if kind == "work" else "elapsed_seconds"
    candidates = [row for row in records if float(row[field]) <= budget]
    return candidates[-1] if candidates else records[0]


def threshold_fields(records: list[dict]) -> dict:
    fields: dict[str, float | bool] = {}
    for threshold in THRESHOLDS:
        reached = next((row for row in records if row["gradient_norm"] <= threshold), None)
        suffix = f"{threshold:.0e}"
        fields[f"reached_grad_{suffix}"] = bool(reached)
        fields[f"time_to_grad_{suffix}"] = float(reached["elapsed_seconds"]) if reached else -1.0
        fields[f"equivalents_to_grad_{suffix}"] = float(reached["gradient_equivalents"]) if reached else -1.0
    return fields


def diagnostic_rows(experiment: str, condition: str, seed: int, records: list[dict]) -> list[dict]:
    rows = []
    for row in records:
        clean = {key: value for key, value in row.items() if key != "_state"}
        rows.append({"experiment": experiment, "condition": condition, "seed": seed, **clean})
    return rows


def synthetic_snapshot_metrics(
    condition: str, seed: int, method: str, records: list[dict],
    train_objective, validation_objective, test_objective, initial_hash: str,
) -> list[dict]:
    rows = []
    threshold_data = threshold_fields(records)
    for kind, budgets in (("work", WORK_BUDGETS), ("time", TIME_BUDGETS)):
        for budget in budgets:
            snapshot = choose_snapshot(records, kind, budget)
            state = snapshot["_state"]
            train_value, train_gradient = original.value_gradient(train_objective, state)
            rows.append({
                "experiment": "synthetic",
                "condition": condition,
                "seed": seed,
                "method": method,
                "parameter_count": 17,
                "initial_sha256": initial_hash,
                "snapshot_kind": kind,
                "snapshot_budget": budget,
                "actual_elapsed_seconds": snapshot["elapsed_seconds"],
                "gradient_equivalents": snapshot["gradient_equivalents"],
                "hvp_evaluations": snapshot["hvp_evaluations"],
                "train_objective": train_value,
                "train_mse": 2.0 * train_value,
                "gradient_norm": float(torch.linalg.vector_norm(train_gradient)),
                "validation_mse": 2.0 * float(validation_objective(state).detach()),
                "test_mse": 2.0 * float(test_objective(state).detach()),
                **threshold_data,
            })
    return rows


def run_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = prior.synthetic_splits()
    train_objective = prior.synthetic_objective(splits["train"])
    validation_objective = prior.synthetic_objective(splits["validation"])
    test_objective = prior.synthetic_objective(splits["test"])
    teacher = prior.synthetic_parameters()
    metric_rows: list[dict] = []
    diagnostics: list[dict] = []
    for scale, condition in ((1.0e-2, "near_local_solution"), (0.2, "far_negative_control")):
        for seed in FORMAL_SEEDS:
            generator = torch.Generator().manual_seed(seed)
            initial = (
                teacher + scale * torch.randn(17, generator=generator, dtype=torch.float64)
            ).to(device="cpu", dtype=torch.float64)
            initial_hash = vector_hash(initial)
            for method in METHODS:
                _, records = run_method(method, initial, train_objective)
                metric_rows.extend(synthetic_snapshot_metrics(
                    condition, seed, method, records, train_objective,
                    validation_objective, test_objective, initial_hash,
                ))
                diagnostics.extend(diagnostic_rows("synthetic", condition, seed, records))
            pd.DataFrame(metric_rows).to_csv(HERE / "synthetic_snapshot_metrics_progress.csv", index=False)
            pd.DataFrame(diagnostics).fillna(-1.0).to_csv(HERE / "synthetic_diagnostics_progress.csv", index=False)
            print(f"[synthetic] {condition} seed={seed}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostics).fillna(-1.0)


def bam_snapshot_metrics(
    seed: int, method: str, records: list[dict], problem: dict,
    features, pack, initial_hash: str,
) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    sample_rows = []
    threshold_data = threshold_fields(records)
    for kind, budgets in (("work", WORK_BUDGETS), ("time", TIME_BUDGETS)):
        for budget in budgets:
            snapshot = choose_snapshot(records, kind, budget)
            state = snapshot["_state"]
            train_value, train_gradient = original.value_gradient(problem["objective"], state)
            prediction = prior.prediction_from_features(features, state)
            metrics, per_sample = prior.evaluate_clean(
                prediction, pack.test_y, pack.test_masks, pack.y_std
            )
            row = {
                "experiment": "bam_head",
                "condition": "full_137_training_cases",
                "seed": seed,
                "method": method,
                "parameter_count": 20,
                "initial_sha256": initial_hash,
                "snapshot_kind": kind,
                "snapshot_budget": budget,
                "actual_elapsed_seconds": snapshot["elapsed_seconds"],
                "gradient_equivalents": snapshot["gradient_equivalents"],
                "hvp_evaluations": snapshot["hvp_evaluations"],
                "train_objective": train_value,
                "objective_gap": max(train_value - problem["optimum_value"], 0.0),
                "gradient_norm": float(torch.linalg.vector_norm(train_gradient)),
                **metrics,
                **threshold_data,
            }
            metric_rows.append(row)
            sample_rows.extend({
                "seed": seed,
                "method": method,
                "snapshot_kind": kind,
                "snapshot_budget": budget,
                **sample,
            } for sample in per_sample)
    return metric_rows, sample_rows


def run_bam() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pack, _, _ = prior.load_clean_pack(prior.CLEAN_DIR, prior.DATA_DIR)
    train_indices, _ = prior.split_train_validation(len(pack.train_x))
    metric_rows: list[dict] = []
    diagnostic_rows_all: list[dict] = []
    audit_rows: list[dict] = []
    sample_rows: list[dict] = []
    for seed in BAM_SEEDS:
        model, payload = prior.load_bam(seed)
        problem, audit = prior.build_bam_problem(model, pack, train_indices)
        audit_rows.append({"seed": seed, "checkpoint_sha256": payload["model_sha256"], **audit})
        features = prior.cache_test_features(model, pack)
        initial = problem["initial"].to(device="cpu", dtype=torch.float64)
        initial_hash = vector_hash(initial)
        for method in METHODS:
            _, records = run_method(method, initial, problem["objective"])
            rows, samples = bam_snapshot_metrics(
                seed, method, records, problem, features, pack, initial_hash
            )
            metric_rows.extend(rows)
            sample_rows.extend(samples)
            diagnostic_rows_all.extend(diagnostic_rows(
                "bam_head", "full_137_training_cases", seed, records
            ))
        pd.DataFrame(metric_rows).to_csv(HERE / "bam_snapshot_metrics_progress.csv", index=False)
        pd.DataFrame(diagnostic_rows_all).fillna(-1.0).to_csv(HERE / "bam_diagnostics_progress.csv", index=False)
        print(f"[bam-head] seed={seed}", flush=True)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(diagnostic_rows_all).fillna(-1.0),
        pd.DataFrame(audit_rows),
        pd.DataFrame(sample_rows),
    )


def run_self_tests() -> pd.DataFrame:
    rows = certified.self_tests()
    rows.extend({"test": f"user_{row['test']}", **{k: v for k, v in row.items() if k != "test"}}
                for row in optimized.self_tests())
    return pd.DataFrame(rows)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    tests = run_self_tests()
    tests.to_csv(HERE / "self_tests.csv", index=False)
    if not bool(tests.passed.all()):
        raise RuntimeError("solver self-tests failed")

    synthetic_metrics, synthetic_diagnostics = run_synthetic()
    synthetic_metrics.to_csv(HERE / "synthetic_snapshot_metrics.csv", index=False)
    synthetic_diagnostics.to_csv(HERE / "synthetic_diagnostics.csv", index=False)
    bam_metrics, bam_diagnostics, bam_audit, bam_samples = run_bam()
    bam_metrics.to_csv(HERE / "bam_snapshot_metrics.csv", index=False)
    bam_diagnostics.to_csv(HERE / "bam_diagnostics.csv", index=False)
    bam_audit.to_csv(HERE / "bam_gram_audit.csv", index=False)
    bam_samples.to_csv(HERE / "bam_test_metrics_per_sample.csv", index=False)

    runtime = time.perf_counter() - started
    (HERE / "runtime.json").write_text(json.dumps({
        "total_seconds": runtime,
        "formal_seeds": FORMAL_SEEDS,
        "bam_seeds": BAM_SEEDS,
        "methods": METHODS,
        "work_budgets": WORK_BUDGETS,
        "time_budgets": TIME_BUDGETS,
        "learning_rate": LEARNING_RATE,
        "sigma_initial": SIGMA_INITIAL,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "dtype": "float64",
        "torch_version": torch.__version__,
    }, indent=2), encoding="utf-8")
    print(f"[complete] runtime={runtime:.2f}s", flush=True)


if __name__ == "__main__":
    main()
