#!/usr/bin/env python3
"""Summarize the new leakage-free run and certify reusable legacy L3CR logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FORMAL = HERE / "formal_leakage_free"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"


def hierarchical_bootstrap(per_sample: pd.DataFrame, repetitions: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    rows = []
    metrics = ["global_score", "hole_score", "balanced_score"]
    for candidate in ["BAM-KAN"]:
        for baseline in ["U-Net", "CNN"]:
            for metric in metrics:
                pivot = per_sample.pivot_table(index=["seed", "sample"], columns="model", values=metric).dropna(subset=[candidate, baseline])
                difference = (pivot[candidate] - pivot[baseline]).unstack("sample")
                difference = difference.dropna(axis=1, how="any").to_numpy(dtype=float)
                if difference.size == 0:
                    continue
                seed_count, sample_count = difference.shape
                draws = np.empty(repetitions, dtype=float)
                for start in range(0, repetitions, 2000):
                    count = min(2000, repetitions - start)
                    seed_idx = rng.integers(0, seed_count, size=(count, seed_count))
                    sample_idx = rng.integers(0, sample_count, size=(count, sample_count))
                    selected = difference[seed_idx[:, :, None], sample_idx[:, None, :]]
                    draws[start : start + count] = selected.mean(axis=(1, 2))
                mean_difference = float(difference.mean())
                row = {
                    "comparison": f"{candidate} - {baseline}",
                    "metric": metric,
                    "mean_difference": mean_difference,
                    "ci95_lower": float(np.quantile(draws, 0.025)),
                    "ci95_upper": float(np.quantile(draws, 0.975)),
                    "win_rate": float(np.mean(difference < 0.0)),
                    "paired_observations": int(difference.size),
                }
                try:
                    from scipy.stats import wilcoxon
                    test = wilcoxon(difference.ravel(), alternative="less", method="auto")
                    row.update({"wilcoxon_statistic": float(test.statistic), "wilcoxon_p_less": float(test.pvalue)})
                except Exception as exc:
                    row.update({"wilcoxon_statistic": np.nan, "wilcoxon_p_less": np.nan, "wilcoxon_error": str(exc)})
                rows.append(row)
    return pd.DataFrame(rows)


def finite_certificate(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["solver", "seed"], as_index=False).agg(
        outer_steps=("outer_iteration", "count"),
        accepted_steps=("accepted", "sum"),
        min_actual_decrease=("actual_decrease", "min"),
        total_actual_decrease=("actual_decrease", "sum"),
        max_outer_backtracks=("outer_backtracks", "max"),
        max_inner_backtracks=("inner_line_search_backtracks", "max"),
        max_final_step_norm=("final_step_norm", "max"),
        max_wall_time=("wall_time", "max"),
    )
    grouped["source_label"] = label
    grouped["finite_descent_certificate"] = (grouped["accepted_steps"] == grouped["outer_steps"]) & (grouped["min_actual_decrease"] > 0.0)
    grouped["parameter_step_bounded"] = grouped["max_final_step_norm"].notna()
    return grouped


def figure_architecture(metrics: pd.DataFrame) -> None:
    selected = ["global_score", "hole_score", "balanced_score"]
    summary = metrics.groupby("model")[selected].agg(["mean", "std"]).reindex(["CNN", "U-Net", "BAM-KAN"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(selected))
    width = 0.24
    colors = {"CNN": "#687078", "U-Net": "#2b7bba", "BAM-KAN": "#d95f02"}
    for offset, model in zip((-width, 0, width), ["CNN", "U-Net", "BAM-KAN"]):
        means = [summary.loc[model, (metric, "mean")] for metric in selected]
        errors = [summary.loc[model, (metric, "std")] for metric in selected]
        ax.bar(x + offset, means, width=width, yerr=errors, capsize=4, color=colors[model], label=model)
    ax.set_xticks(x, ["Global", "Hole", "Balanced"])
    ax.set_ylabel("Normalized RMSE score (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "leakage_free_architecture_comparison.png", dpi=240)
    fig.savefig(FIGURES / "leakage_free_architecture_comparison.pdf")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(FORMAL / "metrics_per_seed.csv")
    samples = pd.read_csv(FORMAL / "metrics_per_sample.csv")
    metrics.to_csv(TABLES / "07_main_architecture_results.csv", index=False)
    metrics.groupby("model").agg(["mean", "std", "median"]).to_csv(TABLES / "07_main_architecture_summary.csv")
    bootstrap = hierarchical_bootstrap(samples)
    bootstrap.to_csv(TABLES / "15_paired_statistics.csv", index=False)
    figure_architecture(metrics)

    certificates = pd.concat(
        [
            finite_certificate(HERE.parent / "test071502/formal/solver_diagnostics.csv", "test071502_legacy_k12"),
            finite_certificate(HERE.parent / "test071601/formal_k32/solver_diagnostics.csv", "test071601_legacy_k32"),
            finite_certificate(HERE.parent / "test071601/formal_k64/solver_diagnostics.csv", "test071601_legacy_k64"),
        ],
        ignore_index=True,
    )
    certificates.to_csv(TABLES / "11_l3cr_finite_certificate.csv", index=False)

    cost = metrics[["model", "seed", "parameters", "training_seconds", "wall_seconds", "epochs_completed"]].copy()
    cost.to_csv(TABLES / "16_computational_cost.csv", index=False)
    clean_means = metrics.groupby("model")[["global_score", "hole_score", "balanced_score"]].mean()
    lines = [
        "# 无泄漏架构实验结果",
        "",
        "本表是基于排除 c04/c10/c12 后的 leakage_free_v1 输入、相同训练/验证划分、AdamW、相同 early stopping 规则和五个随机种子的重新训练结果。它不与旧协议结果混写。",
        "",
        "## 五种子均值与标准差",
        "",
        "| Model | Global | Hole | Balanced |",
        "|---|---:|---:|---:|",
    ]
    for model in ["CNN", "U-Net", "BAM-KAN"]:
        row = metrics[metrics["model"] == model]
        lines.append(f"| {model} | {row['global_score'].mean():.6f} +/- {row['global_score'].std():.6f} | {row['hole_score'].mean():.6f} +/- {row['hole_score'].std():.6f} | {row['balanced_score'].mean():.6f} +/- {row['balanced_score'].std():.6f} |")
    lines.extend([
        "",
        "## 统计解释",
        "",
        "负的 BAM-KAN 减基线差值有利于 BAM-KAN。配对 bootstrap 和 Wilcoxon 结果见 `tables/15_paired_statistics.csv`。由于当前只进行网络结构重训，不能把该结果解释为 L3CR 优化器优势。",
        "",
        "## L3CR 旧日志证书",
        "",
        f"旧日志中可核验的 solver-seed 组数：`{len(certificates)}`；证书通过组数：`{int(certificates['finite_descent_certificate'].sum()) if not certificates.empty else 0}`。证书仅证明旧日志的有限步下降和步长记录，不证明它们从本次 leakage_free_v1 checkpoint 出发。",
        "",
        "## 尚未完成的部分",
        "",
        "BAM-MLP/BAM-Conv 隔离、BAM 路径消融、以及从本次无泄漏 BAM-KAN checkpoint 直接继续 L3CR-PGD 尚未完成；完整 Abaqus 原始场字段也仍未提供，因此 E15 网格投影误差和真正物理残差不能补写。",
    ])
    (HERE / "formal_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "formal_result_summary.json").write_text(json.dumps({"models": clean_means.to_dict(), "bootstrap_rows": len(bootstrap), "certificate_rows": len(certificates)}, indent=2), encoding="utf-8")
    print(clean_means.to_string())
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
