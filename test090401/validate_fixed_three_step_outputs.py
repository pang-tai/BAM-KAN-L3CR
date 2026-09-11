#!/usr/bin/env python3
"""Validate the fixed-three-step experiment before tables are released."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STAGES = ("P5", "P6")
SEEDS = (11, 23, 37, 51, 73)
FORMAL_METHODS = ("MF-L3CR", "Resumed-AdamW", "L-BFGS")


def add(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    rows: list[dict] = []
    run = pd.read_csv(HERE / "run_summary.csv")
    l3 = run[run.method.eq("MF-L3CR")]
    add(rows, "ten_l3cr_runs", len(l3) == 10, f"rows={len(l3)}")
    add(rows, "all_exactly_three_accepted", bool(l3.accepted_steps.eq(3).all()),
        ", ".join(f"{r.stage}-{r.seed}:{int(r.accepted_steps)}" for r in l3.itertuples()))
    add(rows, "all_l3cr_pass", bool(l3.status.eq("PASS").all()), ", ".join(sorted(l3.status.unique())))

    formal = run[run.method.isin(FORMAL_METHODS)]
    expected = len(STAGES) * len(SEEDS) * len(FORMAL_METHODS)
    add(rows, "formal_method_coverage", len(formal) == expected, f"rows={len(formal)}, expected={expected}")
    finite_columns = ["consumed_seconds", "objective", "gradient_norm"]
    finite = np.isfinite(formal[finite_columns].to_numpy(dtype=float)).all()
    add(rows, "finite_terminal_metrics", bool(finite), ",".join(finite_columns))

    hash_counts = formal.groupby(["stage", "seed"]).initial_sha256.nunique()
    add(rows, "common_initial_hash", bool(hash_counts.eq(1).all()), f"max_unique={hash_counts.max()}")

    stepwise = pd.read_csv(HERE / "l3cr_stepwise_descent.csv")
    add(rows, "thirty_accepted_iterates", len(stepwise) == 30, f"rows={len(stepwise)}")
    add(rows, "positive_prediction", bool(stepwise.predicted_decrease.gt(0).all()),
        f"min={stepwise.predicted_decrease.min():.6e}")
    add(rows, "rho_acceptance", bool(stepwise.rho.ge(0.10).all()), f"min={stepwise.rho.min():.6f}")
    monotone = True
    worst_change = -np.inf
    for (_, _), group in stepwise.groupby(["stage", "seed"]):
        objectives = group.sort_values("accepted_count").objective.to_numpy()
        initial = l3[(l3.stage == group.stage.iloc[0]) & (l3.seed == group.seed.iloc[0])]
        # Initial values are available in the dedicated formal CSV.
        formal_l3 = pd.read_csv(HERE / "fixed_three_step_l3cr_per_seed.csv")
        f0 = formal_l3[(formal_l3.stage == group.stage.iloc[0]) & (formal_l3.seed == group.seed.iloc[0])].initial_objective.iloc[0]
        changes = np.diff(np.r_[f0, objectives])
        monotone &= bool((changes < 0).all())
        worst_change = max(worst_change, float(changes.max()))
    add(rows, "strict_objective_descent_each_step", monotone, f"largest_delta={worst_change:.6e}")

    audits = pd.read_csv(HERE / "hvp_audit_summary.csv")
    add(rows, "fd_audit", bool(audits.fd_relative_error.le(1e-8).all()),
        f"max={audits.fd_relative_error.max():.6e}")
    add(rows, "symmetry_audit", bool(audits.symmetry_relative_error.le(1e-12).all()),
        f"max={audits.symmetry_relative_error.max():.6e}")

    allocations_ok = True
    for stage in STAGES:
        allocation = json.loads((HERE / "allocations" / f"{stage}_l3cr.json").read_text())
        for seed in SEEDS:
            target = float(allocation[str(seed)])
            observed = float(l3[(l3.stage == stage) & (l3.seed == seed)].consumed_seconds.iloc[0])
            allocations_ok &= abs(target - observed) < 1e-6
    add(rows, "baseline_budget_source_is_l3cr_time", allocations_ok, "allocation files match L3CR consumed time")

    output = pd.DataFrame(rows)
    output.to_csv(HERE / "validation_results.csv", index=False)
    if not output.passed.all():
        raise SystemExit(output[~output.passed].to_string(index=False))
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
