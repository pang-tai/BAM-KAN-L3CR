#!/usr/bin/env python3
"""Build a formula-led Chinese appendix from frozen experiment CSV files."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
ORDER = ["AdamW", "AdamW-0", "L-BFGS", "Full-space L3CR"]


def table_rows(frame: pd.DataFrame, total: int) -> str:
    rows = []
    for method in ORDER:
        local = frame[frame.method == method]
        rows.append(
            f"{method} & {local.gradient_norm.mean():.3e} & {local.objective.mean():.3e} & "
            f"{int(local['reached_grad_1e-08'].sum())}/{total} & {local.gradient_equivalents.mean():.1f} \\\\"
        )
    return "\n".join(rows)


def main() -> None:
    synthetic = pd.read_csv(HERE / "synthetic_metrics.csv")
    near = synthetic[synthetic.condition == "near_local_solution"]
    far = synthetic[synthetic.condition == "far_negative_control"]
    bam = pd.read_csv(HERE / "bam_head_metrics.csv")
    audit = pd.read_csv(HERE / "bam_gram_audit.csv")
    runtime = __import__("json").loads((HERE / "runtime.json").read_text(encoding="utf-8"))
    tex = rf"""\documentclass[UTF8,11pt,fontset=none]{{ctexart}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{Heiti SC}}
\setCJKmonofont{{Songti SC}}
\usepackage[a4paper,margin=2.1cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float}}
\title{{Full-Space L3CR 短时高精度验证}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{确定性目标与精度准则}}
对参数向量 $\theta\in\mathbb R^n$，训练目标记为
\[
F(\theta)=\frac1{{2N}}\sum_{{i=1}}^N
\lVert f_\theta(x_i)-y_i\rVert_2^2.
\]
高精度判据直接采用
\[
r(\theta)=\lVert\nabla F(\theta)\rVert_2,
\qquad
r(\theta)\le10^{{-4}},\ 10^{{-6}},\ 10^{{-8}}.
\]
不作归一化。不使用验证集回退。所有路线由同一 $\theta_0$ 出发，使用 CPU、单线程和 float64。

\section{{Full-space L3CR}}
本实验令 active dimension 等于 $n$。第 $k$ 个三次模型为
\begin{{equation}}
m_k(s)=g_k^\top s+\frac12s^\top B_ks+
\frac{{\sigma_k}}6\lVert s\rVert_3^3,
\quad
g_k=\nabla F(\theta_k),
\quad
B_k=\nabla^2F(\theta_k).
\end{{equation}}
$B_k$ 由 $n$ 次 HVP 构造并对称化。PG 主迭代为
\begin{{align}}
y^t&=s^t-\alpha_t(g_k+B_ks^t),\\
s_i^{{t+1}}&=\operatorname{{sign}}(y_i^t)
\frac{{2|y_i^t|}}
{{1+\sqrt{{1+2\alpha_t\sigma_k|y_i^t|}}}}.
\end{{align}}
高精度精修求解同一驻点方程
\[
R_k(s)=g_k+B_ks+\frac{{\sigma_k}}2|s|\odot s=0,
\]
其半光滑 Newton 矩阵为
\[
J_k(s)=B_k+\sigma_k\operatorname{{Diag}}(|s|).
\]
只有满足
\[
\frac{{\lVert R_k(s)\rVert_{{3/2}}}}
{{\max\{{1,\lVert g_k\rVert_{{3/2}}\}}}}
\le10^{{-12}}
\]
的子问题解才允许进入外层接受检验。正则参数更新为
\[
\sigma_{{k+1}}=
\begin{{cases}}
0.5\sigma_k,&\rho_k\ge0.75,\\
\sigma_k,&0.10\le\rho_k<0.75,\\
2\sigma_k,&\rho_k<0.10\ \text{{或子问题未收敛}}.
\end{{cases}}
\]

\section{{BAM 输出头的完整训练目标}}
冻结主干后，每个输出通道均写成
\[
\widehat y_c=A_cw_c,
\qquad c\in\{{S_1,U\}},
\qquad w_c\in\mathbb R^{{10}}.
\]
令 $w=(w_{{S_1}}^\top,w_U^\top)^\top\in\mathbb R^{{20}}$。对全部 137 个训练案例累积
\[
G_c=A_c^\top A_c,
\qquad b_c=A_c^\top y_c,
\qquad q_c=y_c^\top y_c.
\]
于是
\begin{{align}}
F(w)
&=\frac1{{2N}}\sum_c
\left(w_c^\top G_cw_c-2b_c^\top w_c+q_c\right),\\
\nabla F(w)
&=\frac1N
\begin{{bmatrix}}G_{{S_1}}w_{{S_1}}-b_{{S_1}}\\G_Uw_U-b_U\end{{bmatrix}},\\
\nabla^2F(w)
&=\frac1N\operatorname{{blkdiag}}(G_{{S_1}},G_U).
\end{{align}}
材料体素数为 ${int(audit.material_voxels.iloc[0]):,}$。五个 Hessian 均为满秩；直接损失与 Gram 目标的最大误差为 ${audit.gram_direct_abs_error.max():.3e}$。

\section{{数值结果}}
\begin{{table}}[H]
\centering
\caption{{17 参数非线性回归：近解初始化。}}
\begin{{tabular}}{{lrrrr}}
\toprule
方法 & $\lVert\nabla F\rVert_2$ & $F$ & 达到 $10^{{-8}}$ & AD 等价量\\
\midrule
{table_rows(near, 10)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{17 参数非线性回归：远解负对照。}}
\begin{{tabular}}{{lrrrr}}
\toprule
方法 & $\lVert\nabla F\rVert_2$ & $F$ & 达到 $10^{{-8}}$ & AD 等价量\\
\midrule
{table_rows(far, 10)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{BAM-KAN 20 参数输出头。}}
\begin{{tabular}}{{lrrrr}}
\toprule
方法 & $\lVert\nabla F\rVert_2$ & $F$ & 达到 $10^{{-8}}$ & AD 等价量\\
\midrule
{table_rows(bam, 5)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\linewidth]{{high_accuracy_convergence.pdf}}
\caption{{原始梯度范数与反向传播等价量。虚线为 $10^{{-8}}$。}}
\end{{figure}}

\section{{结论边界}}
总实验时间为 ${runtime['total_seconds']:.2f}\,\mathrm{{s}}$。BAM 输出头上，L3CR 在 $5/5$ 个 checkpoint 达到 $10^{{-8}}$；其余三条路线均为 $0/5$。该结果支持20维满秩二次输出头校正中的条件性高精度优势。

17参数非线性近解实验中，L3CR仅在 $1/10$ 个 seed 达到 $10^{{-8}}$；远解负对照为 $0/10$。故不能推出一般非线性神经网络优势，也不能外推至完整 86,653 参数 BAM-KAN。
\end{{document}}
"""
    (HERE / "full_space_l3cr_high_accuracy_appendix_zh.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
