#!/usr/bin/env python3
"""Prepare source-grounded tables and figures for the L3CR Word report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "test082402"
METHODS = ["Original fixed L3CR", "AdamW-0", "AdamW", "L-BFGS"]
LABELS = {
    "Original fixed L3CR": "L3CR",
    "AdamW-0": "AdamW-0",
    "AdamW": "AdamW",
    "L-BFGS": "L-BFGS",
    "User optimized L3CR": "Adaptive L3CR",
    "Certified adaptive L3CR": "Certified L3CR",
}
COLORS = {
    "Original fixed L3CR": "#156c59",
    "AdamW-0": "#365f91",
    "AdamW": "#7d8791",
    "L-BFGS": "#b97735",
    "User optimized L3CR": "#7057a3",
    "Certified adaptive L3CR": "#218778",
}


def bootstrap_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    differences = a - b
    generator = np.random.default_rng(20260825)
    indices = generator.integers(0, len(differences), size=(20000, len(differences)))
    draws = differences[indices].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int((differences < 0).sum()),
    )


def prepare_tables(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> dict:
    primary = metrics[(metrics.snapshot_kind == "work") & (metrics.snapshot_budget == 500)]
    threshold10 = thresholds[
        np.isclose(thresholds.threshold, 1.0e-10, rtol=1.0e-12, atol=0.0)
    ].set_index("method")
    rows = []
    for method in METHODS:
        local = primary[primary.method == method]
        target = threshold10.loc[method]
        rows.append({
            "method": LABELS[method],
            "source_method": method,
            "seed_count": len(local),
            "gradient_norm_mean": local.gradient_norm.mean(),
            "gradient_norm_median": local.gradient_norm.median(),
            "objective_gap_mean": local.objective_gap.mean(),
            "reached_1e10_within_500": int(target.reached_within_500_count),
            "reached_1e10_full_trace": int(target.reached_count),
            "median_equivalents_full_trace": target.median_equivalents_reached,
            "median_equivalents_within_500": target.median_equivalents_within_500,
            "median_time_seconds": target.median_time_reached,
            "global_mean": local.global_score.mean(),
            "hole_mean": local.hole_score.mean(),
            "balanced_mean": local.balanced_score.mean(),
        })
    training = pd.DataFrame(rows)
    training.to_csv(HERE / "table_training_precision.csv", index=False)

    threshold_rows = []
    for threshold in (1.0e-6, 1.0e-8, 1.0e-10):
        local = thresholds[
            np.isclose(thresholds.threshold, threshold, rtol=1.0e-12, atol=0.0)
        ].set_index("method")
        row = {"threshold": threshold}
        for method in METHODS:
            record = local.loc[method]
            prefix = LABELS[method].replace("-", "_")
            row[f"{prefix}_within_500"] = int(record.reached_within_500_count)
            row[f"{prefix}_median_equivalents"] = float(record.median_equivalents_reached)
        threshold_rows.append(row)
    threshold_matrix = pd.DataFrame(threshold_rows)
    threshold_matrix.to_csv(HERE / "table_threshold_matrix.csv", index=False)

    seed_rows = []
    for seed in sorted(primary.seed.unique()):
        local = primary[primary.seed == seed].set_index("method")
        row = {"seed": seed}
        for method in METHODS:
            record = local.loc[method]
            prefix = LABELS[method].replace("-", "_")
            row[f"{prefix}_gradient_norm"] = float(record.gradient_norm)
            row[f"{prefix}_objective_gap"] = float(record.objective_gap)
            row[f"{prefix}_equivalents_to_1e10"] = float(record["equivalents_to_grad_1e-10"])
            row[f"{prefix}_reached_within_500"] = bool(
                record["reached_grad_1e-10"] and 0 <= record["equivalents_to_grad_1e-10"] <= 500
            )
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(HERE / "table_seed_pairing.csv", index=False)

    ablation_rows = []
    for method in ("Original fixed L3CR", "User optimized L3CR", "Certified adaptive L3CR"):
        local = primary[primary.method == method]
        target = threshold10.loc[method]
        ablation_rows.append({
            "method": LABELS[method],
            "gradient_norm_mean": local.gradient_norm.mean(),
            "reached_1e10_within_500": int(target.reached_within_500_count),
            "median_equivalents": float(target.median_equivalents_reached),
            "median_time_seconds": float(target.median_time_reached),
        })
    pd.DataFrame(ablation_rows).to_csv(HERE / "table_l3cr_ablation.csv", index=False)

    l3 = primary[primary.method == "Original fixed L3CR"].set_index("seed")
    comparisons = []
    for method in METHODS[1:]:
        baseline = primary[primary.method == method].set_index("seed")
        shared = l3.index.intersection(baseline.index)
        a = np.log10(l3.loc[shared, "gradient_norm"].to_numpy().clip(1.0e-30))
        b = np.log10(baseline.loc[shared, "gradient_norm"].to_numpy().clip(1.0e-30))
        mean, low, high, wins = bootstrap_paired(a, b)
        comparisons.append({
            "comparison": f"L3CR versus {LABELS[method]}",
            "metric": "log10_gradient_norm_difference",
            "mean_difference": mean,
            "ci95_low": low,
            "ci95_high": high,
            "l3cr_lower_count": wins,
            "seed_count": len(shared),
        })
    pd.DataFrame(comparisons).to_csv(HERE / "paired_gradient_statistics.csv", index=False)

    l3_cost = float(threshold10.loc["Original fixed L3CR", "median_equivalents_reached"])
    adam_cost = float(threshold10.loc["AdamW-0", "median_equivalents_reached"])
    facts = {
        "work_budget": 500,
        "training_cases": 137,
        "material_voxels": 2159924,
        "head_parameters": 20,
        "formal_seeds": [11, 23, 37, 51, 73],
        "l3cr_median_equivalents_1e10": l3_cost,
        "adamw0_median_equivalents_1e10": adam_cost,
        "equivalent_work_reduction_percent": 100.0 * (1.0 - l3_cost / adam_cost),
        "equivalent_speedup_ratio": adam_cost / l3_cost,
        "l3cr_success_within_500": int(threshold10.loc["Original fixed L3CR", "reached_within_500_count"]),
        "adamw0_success_within_500": int(threshold10.loc["AdamW-0", "reached_within_500_count"]),
        "l3cr_median_wall_time": float(threshold10.loc["Original fixed L3CR", "median_time_reached"]),
        "adamw0_median_wall_time": float(threshold10.loc["AdamW-0", "median_time_reached"]),
    }
    (HERE / "verified_advantage_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts


def figure_convergence(diagnostics: pd.DataFrame) -> None:
    grid = np.arange(1, 501)
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for method in METHODS:
        traces = []
        for seed in sorted(diagnostics.seed.unique()):
            local = diagnostics[(diagnostics.seed == seed) & (diagnostics.method == method)].sort_values("gradient_equivalents")
            work = local.gradient_equivalents.to_numpy()
            gradients = np.minimum.accumulate(local.gradient_norm.to_numpy())
            positions = np.searchsorted(work, grid, side="right") - 1
            values = np.where(positions >= 0, gradients[np.maximum(positions, 0)], gradients[0])
            traces.append(values)
        stacked = np.vstack(traces)
        median = np.median(stacked, axis=0)
        axis.plot(grid, median, color=COLORS[method], linewidth=2.0, label=LABELS[method])
        if method in ("Original fixed L3CR", "AdamW-0"):
            axis.fill_between(grid, np.quantile(stacked, 0.25, axis=0), np.quantile(stacked, 0.75, axis=0), color=COLORS[method], alpha=0.12)
    axis.axhline(1.0e-10, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_xlim(1, 500)
    axis.set_ylim(1.0e-17, 1.0)
    axis.set_xlabel("Reverse-mode equivalent budget")
    axis.set_ylabel(r"Best attained training gradient $\|\nabla F\|_2$")
    axis.set_title("High-accuracy refinement of the frozen BAM output head")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.2)
    fig.savefig(HERE / "figure_01_training_gradient_convergence.png", dpi=320)
    fig.savefig(HERE / "figure_01_training_gradient_convergence.pdf")
    plt.close(fig)


def figure_threshold_efficiency(thresholds: pd.DataFrame) -> None:
    target = thresholds[
        np.isclose(thresholds.threshold, 1.0e-10, rtol=1.0e-12, atol=0.0)
    ].set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    plot_methods = ["Original fixed L3CR", "AdamW-0", "AdamW", "L-BFGS"]
    costs = []
    for method in plot_methods:
        value = float(target.loc[method, "median_equivalents_reached"])
        costs.append(value if value >= 0 else 0.0)
    bars = axes[0].bar([LABELS[m] for m in plot_methods], costs, color=[COLORS[m] for m in plot_methods], width=0.65)
    for bar, method, value in zip(bars, plot_methods, costs):
        label = f"{value:.0f}" if value > 0 else "Not reached"
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, label, ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel(r"Median reverse-mode equivalents to $10^{-10}$")
    axes[0].set_ylim(0, 440)
    axes[0].grid(axis="y", alpha=0.2)

    successes = [int(target.loc[method, "reached_within_500_count"]) for method in plot_methods]
    bars = axes[1].bar([LABELS[m] for m in plot_methods], successes, color=[COLORS[m] for m in plot_methods], width=0.65)
    for bar, value in zip(bars, successes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{value}/5", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 5.8)
    axes[1].set_ylabel(r"Seeds reaching $10^{-10}$ within 500 equivalents")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(HERE / "figure_02_threshold_efficiency.png", dpi=320)
    fig.savefig(HERE / "figure_02_threshold_efficiency.pdf")
    plt.close(fig)


def figure_seed_gradients(metrics: pd.DataFrame) -> None:
    primary = metrics[(metrics.snapshot_kind == "work") & (metrics.snapshot_budget == 500)]
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    positions = np.arange(len(METHODS))
    for seed in sorted(primary.seed.unique()):
        local = primary[primary.seed == seed].set_index("method")
        values = [float(local.loc[method, "gradient_norm"]) for method in METHODS]
        axis.plot(positions, values, color="#8b949d", alpha=0.35, linewidth=0.8)
    for position, method in enumerate(METHODS):
        local = primary[primary.method == method]
        axis.scatter(np.full(len(local), position), local.gradient_norm, s=38, color=COLORS[method], zorder=3)
    axis.axhline(1.0e-10, linestyle="--", color="#666666", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_xticks(positions, [LABELS[m] for m in METHODS])
    axis.set_ylabel(r"Training gradient norm at 500 equivalents")
    axis.set_title("Paired five-checkpoint comparison")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(HERE / "figure_03_seedwise_precision.png", dpi=320)
    fig.savefig(HERE / "figure_03_seedwise_precision.pdf")
    plt.close(fig)


def main() -> None:
    metrics = pd.read_csv(SOURCE / "bam_snapshot_metrics.csv")
    diagnostics = pd.read_csv(SOURCE / "bam_diagnostics.csv")
    thresholds = pd.read_csv(SOURCE / "threshold_reach_summary.csv")
    thresholds = thresholds[thresholds.experiment == "bam_head"]
    facts = prepare_tables(metrics, thresholds)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    figure_convergence(diagnostics)
    figure_threshold_efficiency(thresholds)
    figure_seed_gradients(metrics)
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
