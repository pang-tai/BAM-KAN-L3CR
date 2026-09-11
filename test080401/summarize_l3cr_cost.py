#!/usr/bin/env python3
"""Summarize L3CR-PGD computational diagnostics for the cost audit."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    diagnostics = pd.read_csv(HERE / "formal_clean_l3cr_pgd/solver_diagnostics.csv")
    rows = diagnostics.groupby(["solver", "seed", "method"], as_index=False).agg(
        outer_iterations=("outer_iteration", "count"), accepted_steps=("accepted", "sum"),
        hvp_evaluations=("hvp_count", "sum"), total_solver_iterations=("solver_iterations", "sum"),
        validation_evaluations=("validation_after", "count"), wall_seconds=("wall_time", "sum"),
        max_outer_backtracks=("outer_backtracks", "max"), max_inner_backtracks=("inner_line_search_backtracks", "max"),
        active_dimension=("active_dimension", "first"),
    )
    backtracks = diagnostics.groupby(["solver", "seed", "method"], as_index=False)["outer_backtracks"].sum().rename(columns={"outer_backtracks": "outer_backtrack_trials"})
    rows = rows.merge(backtracks, on=["solver", "seed", "method"], how="left")
    rows["closure_samples"] = 8
    rows["data_protocol"] = "leakage_free_v1 formal BAM-KAN checkpoints"
    rows.to_csv(HERE / "17_l3cr_cost_diagnostics.csv", index=False)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
