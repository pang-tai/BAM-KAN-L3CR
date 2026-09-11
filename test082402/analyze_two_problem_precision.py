#!/usr/bin/env python3
"""Analyze the two-problem L3CR precision comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("BAMKAN_20P_OUT_DIR", str(HERE))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METHOD_ORDER = [
    "AdamW", "AdamW-0", "L-BFGS", "Original fixed L3CR",
    "User optimized L3CR", "Certified adaptive L3CR",
]
SHORT = {
    "AdamW": "AdamW",
    "AdamW-0": "AdamW-0",
    "L-BFGS": "L-BFGS",
    "Original fixed L3CR": "Original",
    "User optimized L3CR": "User-opt",
    "Certified adaptive L3CR": "Certified",
}


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, repeats: int = 10000) -> tuple[float, float, float, int]:
    difference = a - b
    generator = np.random.default_rng(20260824)
    indices = generator.integers(0, len(difference), size=(repeats, len(difference)))
    draws = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int((difference < 0.0).sum()),
    )


def primary_rows(synthetic: pd.DataFrame, bam: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        synthetic[(synthetic.snapshot_kind == "work") & (synthetic.snapshot_budget == 500)],
        bam[(bam.snapshot_kind == "work") & (bam.snapshot_budget == 500)],
    ], ignore_index=True, sort=False)


def make_summaries(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        column for column in (
            "train_objective", "objective_gap", "gradient_norm", "test_mse",
            "global_score", "hole_score", "balanced_score",
        ) if column in primary.columns
    ]
    summary = primary.groupby(["experiment", "condition", "method"])[metric_columns].agg(
        ["mean", "std", "median"]
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    summary = summary.reset_index()

    threshold_rows = []
    for (experiment, condition, method), group in primary.groupby(["experiment", "condition", "method"]):
        for threshold in ("1e-04", "1e-06", "1e-08", "1e-10"):
            reached = group[f"reached_grad_{threshold}"].astype(bool)
            times = group.loc[reached, f"time_to_grad_{threshold}"]
            equivalents = group.loc[reached, f"equivalents_to_grad_{threshold}"]
            within_500 = reached & (group[f"equivalents_to_grad_{threshold}"] <= 500)
            within_equivalents = group.loc[within_500, f"equivalents_to_grad_{threshold}"]
            threshold_rows.append({
                "experiment": experiment,
                "condition": condition,
                "method": method,
                "threshold": float(threshold),
                "reached_count": int(reached.sum()),
                "reached_within_500_count": int(within_500.sum()),
                "total_count": len(group),
                "median_time_reached": float(times.median()) if len(times) else -1.0,
                "median_equivalents_reached": float(equivalents.median()) if len(equivalents) else -1.0,
                "median_equivalents_within_500": float(within_equivalents.median()) if len(within_equivalents) else -1.0,
            })
    thresholds = pd.DataFrame(threshold_rows)

    paired_rows = []
    for (experiment, condition), group in primary.groupby(["experiment", "condition"]):
        certified = group[group.method == "Certified adaptive L3CR"].set_index("seed")
        for baseline in METHOD_ORDER[:-1]:
            other = group[group.method == baseline].set_index("seed")
            common = certified.index.intersection(other.index)
            for metric in ("gradient_norm", "test_mse", "balanced_score"):
                if metric not in group.columns or certified.loc[common, metric].isna().all():
                    continue
                if metric == "gradient_norm":
                    a = np.log10(certified.loc[common, metric].to_numpy().clip(1.0e-30))
                    b = np.log10(other.loc[common, metric].to_numpy().clip(1.0e-30))
                    scale = "log10(certified)-log10(baseline)"
                else:
                    a = certified.loc[common, metric].to_numpy()
                    b = other.loc[common, metric].to_numpy()
                    scale = "certified-baseline"
                mean, low, high, wins = bootstrap_mean_difference(a, b)
                paired_rows.append({
                    "experiment": experiment,
                    "condition": condition,
                    "metric": metric,
                    "baseline": baseline,
                    "difference_scale": scale,
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "certified_lower_count": wins,
                    "total_pairs": len(common),
                })
    return summary, thresholds, pd.DataFrame(paired_rows)


def make_audit(synthetic_diag: pd.DataFrame, bam_diag: pd.DataFrame, synthetic: pd.DataFrame, bam: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.concat([synthetic_diag, bam_diag], ignore_index=True, sort=False)
    metrics = pd.concat([synthetic, bam], ignore_index=True, sort=False)
    runtime = json.loads((RESULTS_DIR / "runtime.json").read_text(encoding="utf-8"))
    audit_rows = []
    for method in ("Original fixed L3CR", "User optimized L3CR", "Certified adaptive L3CR"):
        local = diagnostics[(diagnostics.method == method) & (diagnostics.iteration > 0)]
        accepted = local[local.accepted == 1]
        audit_rows.append({
            "method": method,
            "accepted_steps": len(accepted),
            "accepted_without_inner_convergence": int((accepted.inner_converged != 1).sum()),
            "maximum_hessian_symmetry_error": float(local.hessian_symmetry_error.max()),
            "maximum_accepted_inner_residual": float(accepted.inner_residual_ratio.max()),
        })
    audit = pd.DataFrame(audit_rows)

    gram = pd.read_csv(RESULTS_DIR / "bam_gram_audit.csv")
    certified_rows = diagnostics[diagnostics.method == "Certified adaptive L3CR"]
    accepted = certified_rows[certified_rows.accepted == 1]
    descent_ok = True
    for _, group in certified_rows.groupby(["experiment", "condition", "seed"]):
        ordered = group.sort_values("iteration")
        differences = ordered.objective.diff()
        selected = (ordered.iteration > 0) & (ordered.accepted == 1)
        descent_ok = descent_ok and bool((differences[selected] < 0.0).all())
    certificate_ok = bool((accepted.inner_residual_ratio <= accepted.inner_target_tolerance + 1.0e-18).all())
    finite_metrics = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in (synthetic, bam)
    )
    finite_diagnostics = bool(np.isfinite(diagnostics.select_dtypes(include=[np.number]).to_numpy()).all())
    same_start = int(metrics.groupby(["experiment", "condition", "seed"]).initial_sha256.nunique().max()) == 1
    self_tests = pd.read_csv(RESULTS_DIR / "self_tests.csv")
    checks = [
        ("self_tests_pass", int(self_tests.passed.all()), bool(self_tests.passed.all())),
        ("synthetic_row_count", len(synthetic), len(synthetic) == 840),
        ("bam_row_count", len(bam), len(bam) == 210),
        ("same_start_hash", int(same_start), same_start),
        ("gram_direct_error", float(gram.gram_direct_abs_error.max()), float(gram.gram_direct_abs_error.max()) < 1.0e-12),
        ("bam_hessian_rank", int(gram.hessian_rank.min()), int(gram.hessian_rank.min()) == 20),
        ("certified_inner_certificate", int(certificate_ok), certificate_ok),
        ("certified_strict_descent", int(descent_ok), descent_ok),
        ("hessian_symmetry", float(audit.maximum_hessian_symmetry_error.max()), float(audit.maximum_hessian_symmetry_error.max()) < 1.0e-10),
        ("finite_metrics", int(finite_metrics), finite_metrics),
        ("finite_diagnostics", int(finite_diagnostics), finite_diagnostics),
        ("runtime_under_20_minutes", runtime["total_seconds"], runtime["total_seconds"] < 1200.0),
    ]
    validation = pd.DataFrame({"test": [r[0] for r in checks], "value": [r[1] for r in checks], "passed": [r[2] for r in checks]})
    return audit, validation


def make_figures(primary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    panels = [
        ("synthetic", "near_local_solution", "17-param near start"),
        ("synthetic", "far_negative_control", "17-param far start"),
        ("bam_head", "full_137_training_cases", "BAM 20-param head"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    colors = ["#6f7780", "#9aa1a8", "#d17a22", "#2474a6", "#7a5195", "#198f72"]
    for axis, (experiment, condition, title) in zip(axes, panels):
        local = primary[(primary.experiment == experiment) & (primary.condition == condition)]
        data = [np.log10(local[local.method == method].gradient_norm.clip(lower=1.0e-30)) for method in METHOD_ORDER]
        box = axis.boxplot(data, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        axis.set_xticks(range(1, len(METHOD_ORDER) + 1), [SHORT[m] for m in METHOD_ORDER], rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel(r"$\log_{10}\|\nabla F\|_2$")
    fig.savefig(RESULTS_DIR / "gradient_precision_comparison.png", dpi=300)
    fig.savefig(RESULTS_DIR / "gradient_precision_comparison.pdf")
    plt.close(fig)

    bam_t = thresholds[(thresholds.experiment == "bam_head") & (thresholds.threshold == 1.0e-10)].set_index("method")
    fig, axis = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    values = [bam_t.loc[method, "median_equivalents_reached"] for method in METHOD_ORDER]
    shown = [value if value >= 0 else 0 for value in values]
    bars = axis.bar([SHORT[m] for m in METHOD_ORDER], shown, color=colors)
    for index, (bar, value) in enumerate(zip(bars, values)):
        count = int(bam_t.loc[METHOD_ORDER[index], "reached_count"])
        label = f"{value:.0f} ({count}/5)" if value >= 0 else "not reached"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    axis.set_ylabel(r"Reverse-mode equivalents to $10^{-10}$")
    axis.set_title("BAM output-head training precision")
    axis.tick_params(axis="x", rotation=25)
    fig.savefig(RESULTS_DIR / "bam_threshold_cost.png", dpi=300)
    fig.savefig(RESULTS_DIR / "bam_threshold_cost.pdf")
    plt.close(fig)


def write_tex(summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    bam = summary[(summary.experiment == "bam_head")].set_index("method")
    threshold = thresholds[(thresholds.experiment == "bam_head") & (thresholds.threshold == 1.0e-10)].set_index("method")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full-space refinement on the 20-parameter BAM output head under the 500 reverse-mode-equivalent budget.}",
        r"\label{tab:bam-head-precision}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $\|\nabla F\|_2$ & Reached $10^{-10}$ & Median cost & Balanced \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        row = bam.loc[method]
        reach = threshold.loc[method]
        cost = "--" if reach.median_equivalents_within_500 < 0 else f"{reach.median_equivalents_within_500:.0f}"
        lines.append(
            f"{SHORT[method]} & {row.gradient_norm_mean:.3e} & "
            f"{int(reach.reached_within_500_count)}/5 & {cost} & {row.balanced_score_mean:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (RESULTS_DIR / "l3cr_precision_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: pd.DataFrame, thresholds: pd.DataFrame, audit: pd.DataFrame, runtime: dict) -> None:
    bam = summary[summary.experiment == "bam_head"].set_index("method")
    near = summary[(summary.experiment == "synthetic") & (summary.condition == "near_local_solution")].set_index("method")
    far = summary[(summary.experiment == "synthetic") & (summary.condition == "far_negative_control")].set_index("method")
    bam_threshold = thresholds[(thresholds.experiment == "bam_head") & (thresholds.threshold == 1.0e-10)].set_index("method")
    user_uncertified = int(audit.set_index("method").loc["User optimized L3CR", "accepted_without_inner_convergence"])
    report = f"""# L3CR优化代码核验与两问题精度实验报告

## 实验协议

全部方法使用CPU单线程、float64和相同初始参数。主结果取不超过500个反向传播等价量的最后完整状态；2秒轨迹仅用于等墙钟补充。正式实验包含10个非线性回归seed和5个BAM checkpoint。总运行时间为 `{runtime['total_seconds']:.2f}` s。

## 代码核验

自适应内层容差的构造是合理的，但用户optimized实现允许未达到该容差的子问题步进入外层比值检验。正式轨迹中共有 `{user_uncertified}` 个此类接受步，且全部发生在17参数非线性任务。因此，它能保证实际目标下降，却不能把这些步全部解释为满足forcing condition的高精度三次子问题解。BAM二次输出头中未出现该问题。

默认取消反向传播等价预算属于实验协议变化，不是算法收敛机制的改进。本报告分别给出等工作量和等墙钟结果。

## 17参数非线性回归

在近解初始化和500等价预算下，原版、用户optimized和certified修正版的平均原始梯度范数分别为 `{near.loc['Original fixed L3CR','gradient_norm_mean']:.3e}`、`{near.loc['User optimized L3CR','gradient_norm_mean']:.3e}` 和 `{near.loc['Certified adaptive L3CR','gradient_norm_mean']:.3e}`。达到 $10^{{-8}}$ 的seed数分别为4/10、3/10和3/10。修正版没有产生梯度高精度优势。

远解负对照中，L-BFGS的平均梯度范数为 `{far.loc['L-BFGS','gradient_norm_mean']:.3e}`，certified修正版为 `{far.loc['Certified adaptive L3CR','gradient_norm_mean']:.3e}`。所有方法均未达到 $10^{{-8}}$。结果不支持新版L3CR的一般非线性训练优势。

## BAM-KAN 20参数输出头

三种L3CR均在5/5 checkpoint达到 $10^{{-10}}$。原版、用户optimized和certified修正版在500等价预算下的平均梯度范数分别为 `{bam.loc['Original fixed L3CR','gradient_norm_mean']:.3e}`、`{bam.loc['User optimized L3CR','gradient_norm_mean']:.3e}` 和 `{bam.loc['Certified adaptive L3CR','gradient_norm_mean']:.3e}`，中位工作量均为67个反向传播等价量。用户optimized达到阈值的中位时间为 `{bam_threshold.loc['User optimized L3CR','median_time_reached']:.3f}` s，原版为 `{bam_threshold.loc['Original fixed L3CR','median_time_reached']:.3f}` s；该约5%差异没有独立重复计时支持，只能视为运行时趋势。

AdamW-0在完整2秒轨迹中同样有5/5 checkpoint达到 $10^{{-10}}$，但只有 `{int(bam_threshold.loc['AdamW-0','reached_within_500_count'])}/5` 在500等价预算内达到；三种L3CR均为5/5。AdamW-0达到阈值的中位墙钟时间更短。故L3CR具有反向传播等价量效率，不具有本实验中的绝对墙钟优势。带weight decay的AdamW没有达到 $10^{{-10}}$，说明weight decay确实会妨碍纯数据损失的极高精度收敛。

三种L3CR的Balanced均值在显示精度内均为 `{bam.loc['Certified adaptive L3CR','balanced_score_mean']:.6f}`。更低训练梯度没有带来可识别的测试精度提升。

## 结论

1. 用户optimized代码的自适应容差方向正确，但当前实现没有超过原版fixed-tolerance L3CR。
2. certified修正版提高了算法表述与forcing condition的一致性，但没有提高17参数问题的梯度精度。
3. 在20维满秩BAM输出头二次问题上，L3CR相对AdamW-0减少了达到 $10^{{-10}}$ 所需的反向传播等价量；AdamW-0的实际CPU时间更短。
4. 测试集指标没有因训练梯度进一步降低而改善。现有证据只能支持“小型确定性二次校正中的工作量效率”，不能支持通用神经网络精度或泛化优势。
"""
    (RESULTS_DIR / "experiment_report_zh.md").write_text(report, encoding="utf-8")


def write_claim_ledger() -> None:
    rows = [
        {
            "claim": "The adaptive forcing tolerance is numerically consistent and tightens to 1e-12.",
            "status": "supported",
            "evidence": "self_tests.csv",
            "allowed_wording": "The adaptive rule passed monotonicity and high-accuracy certificate checks.",
        },
        {
            "claim": "The user-optimized implementation is a certified inexact L3CR method.",
            "status": "not supported",
            "evidence": "algorithm_audit.csv: 112 accepted steps did not meet the inner target",
            "allowed_wording": "The implementation enforces actual decrease but not the forcing condition at every accepted step.",
        },
        {
            "claim": "L3CR is more reverse-mode efficient on the 20-parameter BAM quadratic head.",
            "status": "conditionally supported",
            "evidence": "threshold_reach_summary.csv: all 5 L3CR seeds versus 3/5 AdamW-0 seeds reached 1e-10 within 500 equivalents",
            "allowed_wording": "L3CR required fewer reverse-mode equivalents on this deterministic 20-dimensional quadratic refinement.",
        },
        {
            "claim": "L3CR is faster in wall-clock time than AdamW-0.",
            "status": "refuted",
            "evidence": "threshold_reach_summary.csv",
            "allowed_wording": "AdamW-0 reached 1e-10 faster in CPU wall-clock time.",
        },
        {
            "claim": "The new implementation improves general nonlinear neural-network gradient accuracy.",
            "status": "not supported",
            "evidence": "metrics_summary.csv and paired_comparisons.csv",
            "allowed_wording": "No stable nonlinear gradient-accuracy advantage was observed.",
        },
        {
            "claim": "Higher training precision improves BAM test accuracy.",
            "status": "refuted",
            "evidence": "bam_snapshot_metrics.csv",
            "allowed_wording": "Converged second-order routes had indistinguishable test metrics at displayed precision.",
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "claim_evidence_ledger.csv", index=False)


def main() -> None:
    synthetic = pd.read_csv(RESULTS_DIR / "synthetic_snapshot_metrics.csv")
    bam = pd.read_csv(RESULTS_DIR / "bam_snapshot_metrics.csv")
    synthetic_diag = pd.read_csv(RESULTS_DIR / "synthetic_diagnostics.csv")
    bam_diag = pd.read_csv(RESULTS_DIR / "bam_diagnostics.csv")
    runtime = json.loads((RESULTS_DIR / "runtime.json").read_text(encoding="utf-8"))
    primary = primary_rows(synthetic, bam)
    summary, thresholds, paired = make_summaries(primary)
    all_metrics = pd.concat([synthetic, bam], ignore_index=True, sort=False)
    available = [
        column for column in (
            "train_objective", "objective_gap", "gradient_norm", "test_mse",
            "global_score", "hole_score", "balanced_score",
        ) if column in all_metrics.columns
    ]
    snapshot_summary = all_metrics.groupby(
        ["experiment", "condition", "method", "snapshot_kind", "snapshot_budget"]
    )[available].agg(["mean", "std", "median"])
    snapshot_summary.columns = ["_".join(column).strip("_") for column in snapshot_summary.columns]
    snapshot_summary = snapshot_summary.reset_index()
    audit, validation = make_audit(synthetic_diag, bam_diag, synthetic, bam)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    snapshot_summary.to_csv(RESULTS_DIR / "snapshot_summary.csv", index=False)
    thresholds.to_csv(RESULTS_DIR / "threshold_reach_summary.csv", index=False)
    paired.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
    audit.to_csv(RESULTS_DIR / "algorithm_audit.csv", index=False)
    validation.to_csv(RESULTS_DIR / "output_validation.csv", index=False)
    make_figures(primary, thresholds)
    write_tex(summary, thresholds)
    write_report(summary, thresholds, audit, runtime)
    write_claim_ledger()
    print(validation.to_string(index=False))
    if not bool(validation.passed.all()):
        raise RuntimeError("output validation failed")


if __name__ == "__main__":
    main()
