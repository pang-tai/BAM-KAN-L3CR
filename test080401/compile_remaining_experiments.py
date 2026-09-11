#!/usr/bin/env python3
"""Compile E7, E8 and E10 into the named deliverables and update status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"


def bootstrap_pair(samples: pd.DataFrame, candidate: str, baseline: str, label: str, repetitions: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    rows = []
    for metric in ["global_score", "hole_score", "balanced_score"]:
        pivot = samples.pivot_table(index=["seed", "sample"], columns="model", values=metric).dropna(subset=[candidate, baseline])
        matrix = (pivot[candidate] - pivot[baseline]).unstack("sample").dropna(axis=1, how="any").to_numpy(dtype=float)
        if matrix.size == 0:
            continue
        n_seed, n_sample = matrix.shape
        draws = np.empty(repetitions)
        for start in range(0, repetitions, 2000):
            count = min(2000, repetitions - start)
            seed_idx = rng.integers(0, n_seed, size=(count, n_seed))
            sample_idx = rng.integers(0, n_sample, size=(count, n_sample))
            draws[start : start + count] = matrix[seed_idx[:, :, None], sample_idx[:, None, :]].mean(axis=(1, 2))
        row = {
            "comparison": f"{candidate} - {baseline}",
            "metric": metric,
            "mean_difference": float(matrix.mean()),
            "ci95_lower": float(np.quantile(draws, 0.025)),
            "ci95_upper": float(np.quantile(draws, 0.975)),
            "win_rate": float(np.mean(matrix < 0.0)),
            "paired_observations": int(matrix.size),
            "experiment": label,
        }
        try:
            from scipy.stats import wilcoxon
            test = wilcoxon(matrix.ravel(), alternative="less", method="auto")
            row["wilcoxon_p_less"] = float(test.pvalue)
        except Exception as exc:
            row["wilcoxon_p_less"] = np.nan
            row["wilcoxon_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_group(summary: pd.DataFrame, order: List[str], filename: str) -> None:
    metrics = ["global_score", "hole_score", "balanced_score"]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(metrics))
    width = min(0.78 / len(order), 0.20)
    for i, name in enumerate(order):
        row = summary.loc[name]
        mean = [row[(metric, "mean")] for metric in metrics]
        std = [row[(metric, "std")] for metric in metrics]
        ax.bar(x + (i - (len(order) - 1) / 2) * width, mean, width, yerr=std, capsize=2, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x, ["Global", "Hole", "Balanced"])
    ax.set_ylabel("Normalized RMSE score (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{filename}.png", dpi=240)
    fig.savefig(FIGURES / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    main_metrics = pd.read_csv(HERE / "formal_leakage_free/metrics_per_seed.csv")
    main_samples = pd.read_csv(HERE / "formal_leakage_free/metrics_per_sample.csv")
    isolation_metrics = pd.read_csv(HERE / "formal_kan_isolation/metrics_per_seed.csv")
    isolation_samples = pd.read_csv(HERE / "formal_kan_isolation/metrics_per_sample.csv")
    ablation_metrics = pd.read_csv(HERE / "formal_path_ablation/metrics_per_seed.csv")
    ablation_samples = pd.read_csv(HERE / "formal_path_ablation/metrics_per_sample.csv")
    refinement_metrics = pd.read_csv(HERE / "formal_clean_l3cr_pgd/metrics_per_seed.csv")
    refinement_samples = pd.read_csv(HERE / "formal_clean_l3cr_pgd/metrics_per_sample.csv")

    isolation_metrics.to_csv(TABLES / "08_bam_kan_isolation_results.csv", index=False)
    ablation_metrics.to_csv(TABLES / "09_pathway_ablation_results.csv", index=False)
    refinement_metrics.to_csv(TABLES / "10_refinement_results.csv", index=False)
    isolation_samples.to_csv(TABLES / "08_bam_kan_isolation_per_sample.csv", index=False)
    ablation_samples.to_csv(TABLES / "09_pathway_ablation_per_sample.csv", index=False)
    refinement_samples.to_csv(TABLES / "10_refinement_per_sample.csv", index=False)

    legacy_certificate_path = TABLES / "11_l3cr_finite_certificate.csv"
    new_certificate = pd.read_csv(HERE / "formal_clean_l3cr_pgd/l3cr_finite_certificate.csv")
    new_certificate["source_label"] = "E10_clean_leakage_free_checkpoint"
    if legacy_certificate_path.exists():
        legacy_certificate = pd.read_csv(legacy_certificate_path)
        legacy_certificate["source_label"] = legacy_certificate.get("source_label", "legacy_L3CR_log")
        certificate_columns = sorted(set(legacy_certificate.columns) | set(new_certificate.columns))
        pd.concat([legacy_certificate.reindex(columns=certificate_columns), new_certificate.reindex(columns=certificate_columns)], ignore_index=True).to_csv(legacy_certificate_path, index=False)
    else:
        new_certificate.to_csv(legacy_certificate_path, index=False)

    stat_frames = [
        bootstrap_pair(isolation_samples, "BAM-MLP", "BAM-KAN", "E7_KAN_isolation"),
        bootstrap_pair(isolation_samples, "BAM-Conv", "BAM-KAN", "E7_Conv_isolation"),
        bootstrap_pair(ablation_samples, "BAM-KAN-no-local", "BAM-KAN-full", "E8_no_local"),
        bootstrap_pair(ablation_samples, "BAM-KAN-no-fine", "BAM-KAN-full", "E8_no_fine"),
        bootstrap_pair(ablation_samples, "BAM-KAN-no-coarse", "BAM-KAN-full", "E8_no_coarse"),
    ]
    pd.concat(stat_frames, ignore_index=True).to_csv(TABLES / "15_paired_statistics_extended.csv", index=False)

    isolation_summary = isolation_metrics.groupby("model")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    ablation_summary = ablation_metrics.groupby("model")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"])
    plot_group(isolation_summary, ["BAM-KAN", "BAM-MLP", "BAM-Conv"], "kan_isolation_comparison")
    plot_group(ablation_summary, ["BAM-KAN-full", "BAM-KAN-no-local", "BAM-KAN-no-fine", "BAM-KAN-no-coarse"], "bam_path_ablation_comparison")

    all_rows = pd.concat([
        main_metrics.assign(experiment="E6_main_architecture"),
        isolation_metrics.assign(experiment="E7_kan_isolation"),
        ablation_metrics.assign(experiment="E8_path_ablation"),
        refinement_metrics.assign(experiment="E10_clean_l3cr_pgd"),
    ], ignore_index=True)
    all_rows.to_csv(TABLES / "all_new_protocol_metrics_per_seed.csv", index=False)
    pd.DataFrame({
        "experiment": ["E6_main_architecture", "E7_kan_isolation", "E8_path_ablation", "E10_clean_l3cr_pgd"],
        "formal_seed_count": [5, 5, 5, 5],
        "model_or_route_count": [3, 3, 4, 2],
        "data_protocol": ["leakage_free_v1"] * 4,
    }).to_csv(TABLES / "experiment_coverage.csv", index=False)

    # Keep the named top-level deliverables synchronized with the detailed tables.
    for source, target in [
        (TABLES / "08_bam_kan_isolation_results.csv", HERE / "08_bam_kan_isolation_results.csv"),
        (TABLES / "09_pathway_ablation_results.csv", HERE / "09_pathway_ablation_results.csv"),
        (TABLES / "10_refinement_results.csv", HERE / "10_refinement_results.csv"),
        (TABLES / "11_l3cr_finite_certificate.csv", HERE / "11_l3cr_finite_certificate.csv"),
        (TABLES / "15_paired_statistics_extended.csv", HERE / "15_paired_statistics_extended.csv"),
    ]:
        target.write_bytes(source.read_bytes())

    status = pd.DataFrame([
        {"item": "08_bam_kan_isolation", "status": "completed", "reason": "BAM-KAN, BAM-MLP and BAM-Conv formal five-seed leakage-free retraining."},
        {"item": "09_pathway_ablation", "status": "completed", "reason": "No-local, no-fine and no-coarse formal five-seed ablations with full reference reuse."},
        {"item": "10_refinement_from_clean_checkpoint", "status": "completed", "reason": "Five new BAM-KAN AdamW checkpoints refined directly by L3CR-PGD; all 15 outer steps accepted."},
        {"item": "14_projection_audit", "status": "unavailable", "reason": "No raw Abaqus nodal/element field export in the supplied archive."},
        {"item": "19_displacement_nonnegative_audit", "status": "completed", "reason": "Reverse-normalized U audit for all 15 new architecture checkpoints."},
    ])
    status.to_csv(HERE / "experiment_status.csv", index=False)

    lines = [
        "# E7-E8-E10 补充实验汇总",
        "",
        "## E7 KAN 隔离",
        "",
        isolation_summary.to_string(),
        "",
        "BAM-MLP/BAM-Conv 的均值优于 BAM-KAN，说明当前数据和训练协议下，KAN 点映射不是性能优势的主要来源；BAM 主干、输入处理和多尺度结构更可能是主要贡献。",
        "",
        "## E8 BAM 路径消融",
        "",
        ablation_summary.to_string(),
        "",
        "no-local 和 no-fine 的结果不应被简单解释为 full 结构必然最优；它们提示当前 BAM-KAN 存在冗余或优化耦合，后续论文应报告消融而不是只展示 full 模型。no-coarse 则整体变差，表明粗尺度路径更重要。",
        "",
        "## E10 新 checkpoint 的 L3CR-PGD",
        "",
        refinement_metrics.groupby("model")[["global_score", "hole_score", "balanced_score"]].agg(["mean", "std"]).to_string(),
        "",
        "L3CR-PGD 的平均提升很小，配对置信区间跨零，因此当前证据支持‘稳定下降和轻微改进趋势’，不支持‘相对于 AdamW 有显著优势’。",
    ]
    (HERE / "remaining_experiments_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "remaining_experiments_summary.json").write_text(json.dumps({"isolation_models": isolation_metrics["model"].unique().tolist(), "ablation_models": ablation_metrics["model"].unique().tolist(), "refinement_rows": len(refinement_metrics)}, indent=2), encoding="utf-8")
    print(isolation_summary.to_string())
    print(ablation_summary.to_string())


if __name__ == "__main__":
    main()
