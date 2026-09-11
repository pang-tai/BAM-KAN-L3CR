#!/usr/bin/env python3
"""Compile paired and solver diagnostics for active-dimension experiments."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    bootstrap_rows = []
    diagnostic_rows = []
    sources = [
        (12, "leakage_free_v1", "formal_five_seed", HERE / "formal_clean_l3cr_pgd"),
        (16, "leakage_free_v1", "formal_five_seed", HERE / "formal_clean_l3cr_k16"),
        (32, "leakage_free_v1", "formal_five_seed", HERE / "formal_clean_l3cr_k32"),
        (64, "leakage_free_v1", "formal_five_seed", HERE / "formal_clean_l3cr_k64"),
        (128, "leakage_free_v1", "smoke_seed11_one_outer_step", HERE / "pilot_clean_l3cr_k128"),
        (32, "legacy_test071601", "legacy_formal_five_seed", HERE.parent / "test071601/formal_k32"),
        (64, "legacy_test071601", "legacy_formal_five_seed", HERE.parent / "test071601/formal_k64"),
    ]
    for active, protocol, status, directory in sources:
        bootstrap = directory / "paired_bootstrap.csv"
        diagnostics = directory / "solver_diagnostics.csv"
        if bootstrap.exists():
            frame = pd.read_csv(bootstrap)
            frame["active_dimension"] = active
            frame["data_protocol"] = protocol
            frame["result_status"] = status
            bootstrap_rows.append(frame)
        if diagnostics.exists():
            frame = pd.read_csv(diagnostics)
            frame["active_dimension"] = active
            frame["data_protocol"] = protocol
            frame["result_status"] = status
            diagnostic_rows.append(frame)
    bootstrap_result = pd.concat(bootstrap_rows, ignore_index=True)
    bootstrap_result.to_csv(HERE / "12_active_dimension_paired_bootstrap.csv", index=False)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    summary = diagnostics.groupby(["active_dimension", "data_protocol", "result_status"], as_index=False).agg(
        n_diagnostic_rows=("seed", "count"), n_seeds=("seed", "nunique"), accepted_steps=("accepted", "sum"),
        hvp_evaluations=("hvp_count", "sum"), mean_active_gradient_mass=("active_gradient_mass", "mean"),
        max_outer_backtracks=("outer_backtracks", "max"), total_wall_seconds=("wall_time", "sum"),
        validation_evaluations=("validation_after", "count"),
    )
    upper = pd.read_csv(HERE / "12_full_gradient_upper_bound.csv")
    upper_summary = upper.groupby(["active_dimension", "data_protocol", "result_status"], as_index=False).agg(
        full_gradient_l2_initial_mean=("full_gradient_l2", "mean"),
        active_gradient_l2_initial_mean=("active_gradient_l2", "mean"),
        active_gradient_mass_initial_mean=("active_gradient_mass", "mean"),
        full_first_order_upper_bound_mean=("full_first_order_upper_bound", "mean"),
        active_first_order_upper_bound_mean=("active_first_order_upper_bound", "mean"),
    )
    summary = summary.merge(upper_summary, on=["active_dimension", "data_protocol", "result_status"], how="left")
    summary["full_gradient_upper_bound_definition"] = "max_step_norm * initial full-gradient L2 norm; first-order closure bound"
    summary.to_csv(HERE / "12_active_dimension_diagnostics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
