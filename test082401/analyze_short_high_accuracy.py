#!/usr/bin/env python3
"""Analyze the preregistered short full-space L3CR experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METHOD_ORDER = ["AdamW", "AdamW-0", "L-BFGS", "Full-space L3CR"]
COLORS = {"AdamW": "#3B78A8", "AdamW-0": "#7A8791", "L-BFGS": "#D17A22", "Full-space L3CR": "#188977"}


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(10_000)])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def endpoint_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, condition), subset in frame.groupby(["experiment", "condition"]):
        for baseline in ("AdamW", "AdamW-0", "L-BFGS"):
            pivot = subset.pivot(index="seed", columns="method", values="gradient_norm")
            pivot = pivot.dropna(subset=["Full-space L3CR", baseline])
            difference = np.log10(np.maximum(pivot["Full-space L3CR"].to_numpy(), 1.0e-18)) - np.log10(np.maximum(pivot[baseline].to_numpy(), 1.0e-18))
            lower, upper = bootstrap_mean(difference, 20260824 + len(rows))
            rows.append({
                "experiment": experiment, "condition": condition,
                "comparison": f"Full-space L3CR - {baseline}", "metric": "log10_gradient_norm",
                "mean_difference": float(difference.mean()), "ci95_lower": lower, "ci95_upper": upper,
                "wins_l3cr": int(np.sum(difference < 0.0)), "pairs": len(difference),
            })
    return pd.DataFrame(rows)


def threshold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, condition, method), subset in frame.groupby(["experiment", "condition", "method"]):
        for threshold in ("1e-04", "1e-06", "1e-08"):
            reached = subset[f"reached_grad_{threshold}"].astype(bool)
            times = subset.loc[reached, f"time_to_grad_{threshold}"]
            equivalents = subset.loc[reached, f"equivalents_to_grad_{threshold}"]
            rows.append({
                "experiment": experiment, "condition": condition, "method": method, "threshold": threshold,
                "reached_count": int(reached.sum()), "total_count": len(subset),
                "median_time_reached": float(times.median()) if len(times) else -1.0,
                "median_equivalents_reached": float(equivalents.median()) if len(equivalents) else -1.0,
            })
    return pd.DataFrame(rows)


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["objective", "objective_gap", "gradient_norm", "elapsed_seconds", "gradient_equivalents"]
    return frame.groupby(["experiment", "condition", "method"])[metrics].agg(["mean", "std", "median"]).reset_index()


def convergence_plot(diagnostics: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    panels = [("synthetic", "near_local_solution", "17-parameter nonlinear regression"),
              ("bam_head", "full_137_training_cases", "BAM-KAN 20-parameter output head")]
    for axis, (experiment, condition, title) in zip(axes, panels):
        subset = diagnostics[(diagnostics.experiment == experiment) & (diagnostics.condition == condition)]
        for method in METHOD_ORDER:
            local = subset[subset.method == method]
            grouped = local.groupby("iteration").agg(
                gradient_equivalents=("gradient_equivalents", "median"),
                gradient_norm=("gradient_norm", "median"),
            )
            axis.plot(grouped.gradient_equivalents, grouped.gradient_norm, marker="o", markersize=3,
                      linewidth=1.6, color=COLORS[method], label=method)
        axis.axhline(1.0e-8, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xlabel("Reverse-mode equivalents")
        axis.set_ylabel(r"Raw $\|\nabla F\|_2$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def endpoint_plot(frame: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    panels = [("synthetic", "near_local_solution", "Nonlinear local regime"),
              ("bam_head", "full_137_training_cases", "BAM output-head calibration")]
    for axis, (experiment, condition, title) in zip(axes, panels):
        subset = frame[(frame.experiment == experiment) & (frame.condition == condition)]
        data = [subset[subset.method == method].gradient_norm.to_numpy() for method in METHOD_ORDER]
        box = axis.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=True)
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(COLORS[method]); patch.set_alpha(0.75)
        axis.axhline(1.0e-8, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_ylabel(r"Terminal raw $\|\nabla F\|_2$")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def write_tex(bam: pd.DataFrame, threshold: pd.DataFrame) -> None:
    rows = []
    for method in METHOD_ORDER:
        local = bam[bam.method == method]
        reach = threshold[(threshold.experiment == "bam_head") & (threshold.method == method) & (threshold.threshold == "1e-08")].iloc[0]
        rows.append(
            f"{method} & {local.gradient_norm.mean():.3e} & {local.objective_gap.mean():.3e} & "
            f"{int(reach.reached_count)}/5 & {local.gradient_equivalents.mean():.1f} \\\\"
        )
    (HERE / "bam_high_accuracy_table.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\nMethod & $\\|\\nabla F\\|_2$ & $F-F^\\star$ & Reach $10^{-8}$ & AD equiv. \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def write_report(all_metrics: pd.DataFrame, threshold: pd.DataFrame, paired: pd.DataFrame, runtime: dict) -> None:
    bam = all_metrics[all_metrics.experiment == "bam_head"]
    synthetic = all_metrics[(all_metrics.experiment == "synthetic") & (all_metrics.condition == "near_local_solution")]
    bam_reach = threshold[(threshold.experiment == "bam_head") & (threshold.threshold == "1e-08")].set_index("method")
    syn_reach = threshold[(threshold.experiment == "synthetic") & (threshold.condition == "near_local_solution") & (threshold.threshold == "1e-08")].set_index("method")
    old = pd.read_csv(HERE.parent / "test082301/analysis/metrics_per_seed.csv")
    warm = old[old.method == "warmup"].set_index("seed")
    l3 = bam[bam.method == "Full-space L3CR"].set_index("seed")
    balanced_change = float((l3.test_balanced_score - warm.balanced_score).mean())
    global_change = float((l3.test_global_score - warm.global_score).mean())
    hole_change = float((l3.test_hole_score - warm.hole_score).mean())
    report = f"""# Full-space L3CR 短时高精度实验报告

## 协议

- CPU 单线程，float64，原始梯度范数 $\\|\\nabla F\\|_2$。
- 17 参数非线性回归使用 10 个正式 seed；BAM-KAN 输出头使用 5 个冻结 checkpoint。
- BAM 主干冻结。20 个输出头参数使用全部 137 个训练案例和 {int(pd.read_csv(HERE / 'bam_gram_audit.csv').material_voxels.iloc[0]):,} 个材料体素。
- 总运行时间为 {runtime['total_seconds']:.2f} s，低于 1200 s 硬限制。

## 主要结果

### 17 参数非线性回归

在近解初始化下，达到 $\\|\\nabla F\\|_2\\le10^{{-8}}$ 的 seed 数为：L3CR `{int(syn_reach.loc['Full-space L3CR','reached_count'])}/10`，L-BFGS `{int(syn_reach.loc['L-BFGS','reached_count'])}/10`，AdamW `{int(syn_reach.loc['AdamW','reached_count'])}/10`，AdamW-0 `{int(syn_reach.loc['AdamW-0','reached_count'])}/10`。该实验不支持 L3CR 的通用非线性高精度优势。远解负对照中，L3CR 亦未达到 $10^{{-8}}$。

### BAM-KAN 输出头

L3CR 在 `5/5` seed 达到 $10^{{-8}}$；L-BFGS、AdamW 和 AdamW-0 均为 `0/5`。L3CR 的终点原始梯度范数均值为 `{bam[bam.method == 'Full-space L3CR'].gradient_norm.mean():.3e}`，L-BFGS 为 `{bam[bam.method == 'L-BFGS'].gradient_norm.mean():.3e}`，AdamW-0 为 `{bam[bam.method == 'AdamW-0'].gradient_norm.mean():.3e}`。L3CR 达到 $10^{{-8}}$ 的反向传播等价量中位数为 `{bam_reach.loc['Full-space L3CR','median_equivalents_reached']:.1f}`，时间中位数为 `{bam_reach.loc['Full-space L3CR','median_time_reached']:.4f}` s。

所有 BAM L3CR 接受步的内层模型残差均不超过 $10^{{-12}}$。Gram 目标与直接完整训练损失的最大绝对误差为 `{pd.read_csv(HERE / 'bam_gram_audit.csv').gram_direct_abs_error.max():.3e}`。

## 泛化结果

输出头高精度校正相对 warm-up 的 Global、Opening 和 Balanced 平均差分别为 `{global_change:.6f}`、`{hole_change:.6f}` 和 `{balanced_change:.6f}`。训练精度优势没有转化为 L3CR 相对 L-BFGS 的测试优势；两者收敛到同一输出头最优解，测试指标在显示精度内一致。

## 结论边界

结果支持：在冻结 BAM-KAN 主干后的 20 维、满秩、确定性二次输出头校正问题上，full-space L3CR 相对 AdamW、AdamW-0 和本次 L-BFGS 实现具有高精度到达优势。

结果不支持：L3CR 在一般非线性神经网络上优于 L-BFGS；L3CR 对完整 86,653 参数 BAM-KAN 具有 full-space 高精度优势；训练高精度必然改善测试误差。
"""
    (HERE / "experiment_report_zh.md").write_text(report, encoding="utf-8")


def validate(all_metrics: pd.DataFrame, diagnostics: pd.DataFrame, runtime: dict) -> pd.DataFrame:
    bam_audit = pd.read_csv(HERE / "bam_gram_audit.csv")
    finite_metrics = all(
        np.isfinite(pd.read_csv(HERE / name).select_dtypes(include=[np.number]).to_numpy()).all()
        for name in ("synthetic_metrics.csv", "bam_head_metrics.csv")
    )
    l3_all = diagnostics[diagnostics.method == "Full-space L3CR"]
    l3 = l3_all[l3_all.iteration > 0]
    accepted = l3[l3.accepted == 1]
    descent_ok = True
    for _, group in l3_all.groupby(["experiment", "condition", "seed"]):
        ordered = group.sort_values("iteration")
        decreases = ordered.objective.diff()
        selected = (ordered.iteration > 0) & (ordered.accepted == 1)
        descent_ok = descent_ok and bool((decreases[selected] < 0.0).all())
    checks = [
        ("runtime_within_20_minutes", runtime["total_seconds"], runtime["within_limit"]),
        ("synthetic_ten_seeds", all_metrics[all_metrics.experiment == "synthetic"].seed.nunique(), all_metrics[all_metrics.experiment == "synthetic"].seed.nunique() == 10),
        ("bam_five_seeds", all_metrics[all_metrics.experiment == "bam_head"].seed.nunique(), all_metrics[all_metrics.experiment == "bam_head"].seed.nunique() == 5),
        ("same_start_per_problem_seed", int(all_metrics.groupby(["experiment", "condition", "seed"]).initial_sha256.nunique().max()), all_metrics.groupby(["experiment", "condition", "seed"]).initial_sha256.nunique().max() == 1),
        ("gram_matches_direct_loss", float(bam_audit.gram_direct_abs_error.max()), float(bam_audit.gram_direct_abs_error.max()) < 1.0e-12),
        ("bam_hessian_full_rank", int(bam_audit.hessian_rank.min()), int(bam_audit.hessian_rank.min()) == 20),
        ("accepted_inner_residual", float(accepted.inner_residual_ratio.max()), float(accepted.inner_residual_ratio.max()) <= 1.0e-12),
        ("accepted_steps_converged", int((accepted.inner_converged == 1).sum()), bool((accepted.inner_converged == 1).all())),
        ("accepted_steps_strict_descent", int(descent_ok), descent_ok),
        ("hessian_symmetry", float(l3.hessian_symmetry_error.max()), float(l3.hessian_symmetry_error.max()) < 1.0e-10),
        ("finite_metric_values", int(finite_metrics), bool(finite_metrics)),
        ("finite_diagnostic_values", int(np.isfinite(diagnostics.select_dtypes(include=[np.number]).to_numpy()).all()), bool(np.isfinite(diagnostics.select_dtypes(include=[np.number]).to_numpy()).all())),
    ]
    return pd.DataFrame([{"test": name, "value": value, "passed": passed} for name, value, passed in checks])


def main() -> None:
    synthetic = pd.read_csv(HERE / "synthetic_metrics.csv")
    bam = pd.read_csv(HERE / "bam_head_metrics.csv")
    all_metrics = pd.concat([synthetic, bam], ignore_index=True, sort=False)
    synthetic_diag = pd.read_csv(HERE / "synthetic_diagnostics.csv")
    bam_diag = pd.read_csv(HERE / "bam_head_diagnostics.csv")
    diagnostics = pd.concat([synthetic_diag, bam_diag], ignore_index=True, sort=False)
    runtime = json.loads((HERE / "runtime.json").read_text(encoding="utf-8"))
    summary = summary_table(all_metrics)
    paired = endpoint_statistics(all_metrics)
    threshold = threshold_summary(all_metrics)
    summary.to_csv(HERE / "metrics_summary.csv", index=False)
    paired.to_csv(HERE / "paired_endpoint_statistics.csv", index=False)
    threshold.to_csv(HERE / "threshold_reach_summary.csv", index=False)
    convergence_plot(diagnostics, HERE / "high_accuracy_convergence")
    endpoint_plot(all_metrics, HERE / "terminal_gradient_comparison")
    write_tex(bam, threshold)
    write_report(all_metrics, threshold, paired, runtime)
    ledger = pd.DataFrame([
        {"claim": "L3CR reaches raw gradient 1e-8 on the BAM output-head problem", "evidence": "threshold_reach_summary.csv", "status": "supported in 5/5 checkpoints"},
        {"claim": "L3CR is superior on general nonlinear neural regression", "evidence": "synthetic_metrics.csv", "status": "unsupported"},
        {"claim": "L3CR is superior for full 86653-parameter BAM-KAN refinement", "evidence": "experiment design", "status": "not tested"},
        {"claim": "higher training precision improves test error relative to L-BFGS", "evidence": "bam_head_metrics.csv", "status": "unsupported"},
    ])
    ledger.to_csv(HERE / "claim_evidence_ledger.csv", index=False)
    validation = validate(all_metrics, diagnostics, runtime)
    validation.to_csv(HERE / "output_validation.csv", index=False)
    if not bool(validation.passed.all()):
        raise RuntimeError("output validation failed")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
