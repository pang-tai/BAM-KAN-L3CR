#!/usr/bin/env python3
"""Build manuscript-ready tables and a Chinese TeX appendix from frozen CSV results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
SEEDS = (11, 23, 37, 51, 73)


def f(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    metrics = pd.read_csv(ANALYSIS / "metrics_per_seed.csv")
    paired = pd.read_csv(ANALYSIS / "paired_statistics.csv")
    costs = pd.read_csv(ANALYSIS / "cost_and_stationarity.csv")
    terminal = pd.read_csv(ANALYSIS / "terminal_optimization_state.csv")
    inner = pd.read_csv(ANALYSIS / "inner_solver_accuracy.csv")

    warmup = metrics[metrics.method == "warmup"].copy()
    warmup["budget_seconds"] = 960
    terminal_metrics = pd.concat([
        warmup,
        metrics[metrics.budget_seconds == 960],
    ], ignore_index=True)
    order = ["warmup", "adamw", "lbfgs", "l3cr_r32", "l3cr_r64"]
    labels = {
        "warmup": "AdamW warm-up",
        "adamw": "Resumed AdamW",
        "lbfgs": "L-BFGS",
        "l3cr_r32": r"HVP L3CR-PGD ($r=32$)",
        "l3cr_r64": r"HVP L3CR-PGD ($r=64$)",
    }
    summary = terminal_metrics.groupby("method")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    result_rows = []
    for method in order:
        result_rows.append({
            "method": method,
            "global_mean": summary.loc[method, ("global_score", "mean")],
            "global_std": summary.loc[method, ("global_score", "std")],
            "opening_mean": summary.loc[method, ("hole_score", "mean")],
            "opening_std": summary.loc[method, ("hole_score", "std")],
            "balanced_mean": summary.loc[method, ("balanced_score", "mean")],
            "balanced_std": summary.loc[method, ("balanced_score", "std")],
        })
    pd.DataFrame(result_rows).to_csv(ANALYSIS / "formal_result_summary.csv", index=False)

    audit_rows = []
    for seed in SEEDS:
        checkpoint = torch.load(HERE / "warmup" / f"seed_{seed}" / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        audit_rows.append({
            "seed": seed, "best_epoch": checkpoint["epoch"],
            "best_validation_loss": checkpoint["best_validation_loss"],
            "effective_learning_rate": checkpoint["effective_learning_rate"],
            "optimizer_state_count": len(checkpoint["optimizer_state"]["state"]),
            "model_sha256": checkpoint["model_sha256"],
            "training_seconds": checkpoint["training_seconds"],
        })
    pd.DataFrame(audit_rows).to_csv(ANALYSIS / "warmup_checkpoint_audit.csv", index=False)

    comparisons = paired[
        (paired.budget_seconds == 960)
        & paired.comparison.isin(["l3cr_r64 - adamw", "l3cr_r64 - warmup", "l3cr_r64 - lbfgs"])
    ].copy()
    selected_cost = costs[costs.budget_seconds == 960].groupby("method")[[
        "relative_full_residual", "relative_active_residual", "closure_objective",
        "validation_objective", "hvp_evaluations", "updates",
    ]].mean()
    terminal_cost = terminal.groupby("method")[["relative_full_residual", "closure_objective"]].mean()

    table1 = []
    for row in result_rows:
        table1.append(
            f"{labels[row['method']]} & {f(row['global_mean'])} $\\pm$ {f(row['global_std'])} & "
            f"{f(row['opening_mean'])} $\\pm$ {f(row['opening_std'])} & "
            f"{f(row['balanced_mean'])} $\\pm$ {f(row['balanced_std'])} \\\\"
        )
    table2 = []
    comparison_labels = {
        "l3cr_r64 - adamw": "L3CR--resumed AdamW",
        "l3cr_r64 - warmup": "L3CR--warm-up",
        "l3cr_r64 - lbfgs": "L3CR--L-BFGS",
    }
    metric_labels = {"global_score": "Global", "hole_score": "Opening", "balanced_score": "Balanced"}
    for _, row in comparisons.iterrows():
        table2.append(
            f"{comparison_labels[row.comparison]} & {metric_labels[row.metric]} & {row.mean_difference:.6f} & "
            f"[{row.ci95_lower:.6f}, {row.ci95_upper:.6f}] & "
            f"{int(row.wins_proposed)}/5 \\\\"
        )
    table3 = []
    for method in ("adamw", "lbfgs", "l3cr_r32", "l3cr_r64"):
        table3.append(
            f"{labels[method]} & {selected_cost.loc[method, 'closure_objective']:.6f} & "
            f"{selected_cost.loc[method, 'relative_full_residual']:.3e} & "
            f"{terminal_cost.loc[method, 'closure_objective']:.6f} & "
            f"{terminal_cost.loc[method, 'relative_full_residual']:.3e} & "
            f"{selected_cost.loc[method, 'hvp_evaluations']:.1f} \\\\"
        )

    tex = rf"""\documentclass[UTF8,11pt,fontset=none]{{ctexart}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{Heiti SC}}
\setCJKmonofont{{Songti SC}}
\usepackage[a4paper,margin=2.2cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float}}
\usepackage{{siunitx}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\title{{BAM--KAN 高精度二阶段优化补充实验}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{固定问题与目标函数}}
训练、验证和测试案例数分别为 $137$、$30$ 和 $19$。五个随机种子固定为
\[
\mathcal S=\{{11,23,37,51,73\}}.
\]
确定性闭包集合固定为
\[
\mathcal I_{{\rm clo}}=\{{7,21,29,63,84,89,95,158\}}.
\]
三种 refinement 方法使用同一目标
\[
F(\theta)=
\frac{{\sum_{{q\in\mathcal I_{{\rm clo}}}}
\sum_{{v\in\Omega_q}}\frac12
\lVert \widehat y_{{qv}}(\theta)-y_{{qv}}\rVert_2^2}}
{{\sum_{{q\in\mathcal I_{{\rm clo}}}}|\Omega_q|}}.
\]

\section{{HVP L3CR--PGD}}
令 $P_j\in\mathbb R^{{n\times r}}$ 选取 $|\nabla F(\theta_j)|$ 最大的 $r$ 个坐标。受限模型为
\[
m_j(s)=g_j^\top s+\frac12s^\top B_js+
\frac{{\sigma_j}}{{6}}\lVert s\rVert_3^3,
\quad
g_j=P_j^\top\nabla F(\theta_j),
\quad
B_j=P_j^\top\nabla^2F(\theta_j)P_j.
\]
$B_j$ 由 $r$ 次自动微分 HVP 构造并对称化，不形成完整网络 Hessian。内层迭代为
\[
y^t=s^t-\alpha_t(g_j+B_js^t),
\]
\[
s_i^{{t+1}}=\operatorname{{sign}}(y_i^t)
\frac{{2|y_i^t|}}
{{1+\sqrt{{1+2\alpha_t\sigma_j|y_i^t|}}}}.
\]
停止量采用真实模型残差
\[
r_{{m,j}}(s)=
\frac{{\lVert g_j+B_js+\frac12\sigma_j|s|\odot s\rVert_{{3/2}}}}
{{\max\{{1,\lVert g_j\rVert_{{3/2}}\}}}}
\le 10^{{-7}}.
\]
外层量分别为
\[
r_{{\rm full}}(\theta)=
\frac{{\lVert\nabla F(\theta)\rVert_2}}
{{\max\{{1,\lVert\nabla F(\theta^0)\rVert_2\}}}},
\qquad
r_{{\rm act}}(\theta)=
\frac{{\lVert P(\theta)^\top\nabla F(\theta)\rVert_2}}
{{\max\{{1,\lVert P_0^\top\nabla F(\theta^0)\rVert_2\}}}}.
\]
故 $r_{{m,j}}\le10^{{-7}}$ 不推出 $r_{{\rm act}}\le10^{{-6}}$，后者亦不推出 full-space stationarity。

\section{{数值结果}}
\begin{{table}}[H]
\centering
\caption{{960 s 预算下的五种子测试误差。数值为均值 $\pm$ 标准差。}}
\begin{{tabular}}{{lccc}}
\toprule
方法 & Global & Opening & Balanced\\
\midrule
{chr(10).join(table1)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{$r=64$ 在 960 s 的配对差。负值有利于 L3CR。}}
\begin{{tabular}}{{llccc}}
\toprule
比较 & 指标 & 均值差 & bootstrap 95\% 区间 & 胜出 seed\\
\midrule
{chr(10).join(table2)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{验证选择状态与运行终点的优化量。}}
\begin{{tabular}}{{lccccc}}
\toprule
方法 & $F_{{\rm selected}}$ & $r_{{\rm full,selected}}$ & $F_{{\rm terminal}}$ & $r_{{\rm full,terminal}}$ & HVP\\
\midrule
{chr(10).join(table3)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{analysis/accuracy_vs_time.pdf}}
\caption{{测试误差与额外墙钟时间。虚线为 AdamW warm-up 起点。}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{analysis/stationarity_vs_time.pdf}}
\caption{{验证选择 checkpoint 的 full-gradient 相对残差。}}
\end{{figure}}

\section{{结论边界}}
在 $T=960$ s，$r=64$ 相对 resumed AdamW 的 Balanced 均值差为
\[
-0.012675,
\]
五个 seed 全部同方向；Opening 均值差为 $-0.007348$，四个 seed 同方向。
然而，相对 warm-up 起点，Global、Opening 和 Balanced 均值差为
\[
(0.001545,\,-0.000282,\,0.000632).
\]
因此结果支持“L3CR 相对 resumed AdamW 更稳健”，不支持“L3CR 产生全面净精度增益”。
Opening 的变化仅涉及三个有效测试案例。五个 seed 的精确双侧符号检验最小值为 $0.0625$，不作传统显著性声明。

所有方法在 960 s 内均未达到外层 $10^{{-4}}$、$10^{{-5}}$ 或 $10^{{-6}}$ stationarity。
L3CR 内层模型残差达到 $10^{{-7}}$，但该事实只验证子问题求解精度。
\end{{document}}
"""
    (HERE / "high_accuracy_refinement_appendix_zh.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
