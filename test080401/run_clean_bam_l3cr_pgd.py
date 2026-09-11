#!/usr/bin/env python3
"""Run L3CR-PGD directly from the leakage-free BAM-KAN checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
L3CR = HERE.parent / "test071502"
LEGACY = HERE.parent / "test071407"
FAIR = HERE.parent / "test071405"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(L3CR))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(FAIR))

import run_bam_l3cr_three_solver_experiment as exp  # noqa: E402
from run_leakage_free_architecture_experiment import evaluate_clean, load_clean_pack  # noqa: E402
from run_bampikan_unet_cnn_comparison import BAMPIKAN, parameter_count  # noqa: E402


def make_clean_bam() -> BAMPIKAN:
    return BAMPIKAN(
        physical_channels=8,
        coordinate_channels=3,
        spatial_width=19,
        fine_kan_width=12,
        coarse_kan_width=12,
        grid_size=5,
    )


def paired_bootstrap(per_sample: pd.DataFrame, method: str, repetitions: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260804)
    for metric in ["global_score", "hole_score", "balanced_score"]:
        pivot = per_sample.pivot_table(index=["seed", "sample"], columns="model", values=metric).dropna(subset=[method, "BAM-KAN-AdamW"])
        matrix = (pivot[method] - pivot["BAM-KAN-AdamW"]).unstack("sample").dropna(axis=1, how="any").to_numpy(dtype=float)
        if matrix.size == 0:
            continue
        n_seed, n_sample = matrix.shape
        draws = np.empty(repetitions, dtype=float)
        for start in range(0, repetitions, 2000):
            count = min(2000, repetitions - start)
            seed_idx = rng.integers(0, n_seed, size=(count, n_seed))
            sample_idx = rng.integers(0, n_sample, size=(count, n_sample))
            draws[start : start + count] = matrix[seed_idx[:, :, None], sample_idx[:, None, :]].mean(axis=(1, 2))
        rows.append({
            "comparison": f"{method} - BAM-KAN-AdamW",
            "metric": metric,
            "mean_difference": float(matrix.mean()),
            "ci95_lower": float(np.quantile(draws, 0.025)),
            "ci95_upper": float(np.quantile(draws, 0.975)),
            "win_rate": float(np.mean(matrix < 0.0)),
            "paired_observations": int(matrix.size),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data_28_06_2026")
    parser.add_argument("--clean-dir", type=Path, default=HERE / "data_clean/leakage_free_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "formal_leakage_free")
    parser.add_argument("--out-dir", type=Path, default=HERE / "formal_clean_l3cr_pgd")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 73])
    parser.add_argument("--active-dimension", type=int, default=12)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--closure-samples", type=int, default=8)
    parser.add_argument("--inner-iterations", type=int, default=100)
    parser.add_argument("--max-abs-step", type=float, default=1.0e-3)
    parser.add_argument("--max-step-norm", type=float, default=2.0e-3)
    parser.add_argument("--sigma-init", type=float, default=1.0e6)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds[:1]) if args.smoke else tuple(args.seeds)
    outer_steps = 1 if args.smoke else args.outer_steps
    inner_iterations = min(8, args.inner_iterations) if args.smoke else args.inner_iterations
    repetitions = 200 if args.smoke else 20000
    pack, _, _ = load_clean_pack(args.clean_dir.resolve(), args.data_dir.resolve())
    device = exp.choose_device(args.device)
    baseline = pd.read_csv(args.checkpoint_dir.resolve() / "metrics_per_seed.csv")
    baseline_samples = pd.read_csv(args.checkpoint_dir.resolve() / "metrics_per_sample.csv")
    baseline = baseline[(baseline["model"] == "BAM-KAN") & baseline["seed"].isin(seeds)].copy()
    baseline_samples = baseline_samples[(baseline_samples["model"] == "BAM-KAN") & baseline_samples["seed"].isin(seeds)].copy()
    baseline["model"] = "BAM-KAN-AdamW"
    baseline_samples["model"] = "BAM-KAN-AdamW"
    method = "BAM-KAN-AdamW-L3CR-PGD"
    metric_rows = baseline.to_dict("records")
    sample_rows = baseline_samples.to_dict("records")
    diagnostics: List[Dict[str, object]] = []
    metadata: Dict[str, object] = {}
    config = exp.Config(
        seeds=seeds,
        outer_steps=outer_steps,
        active_dimension=args.active_dimension,
        closure_samples=args.closure_samples,
        inner_iterations=inner_iterations,
        max_abs_step=args.max_abs_step,
        max_step_norm=args.max_step_norm,
        sigma_init=args.sigma_init,
        bootstrap_repetitions=repetitions,
        device=args.device,
    )
    (outdir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    tests = exp.self_tests(config)
    tests.to_csv(outdir / "solver_self_tests.csv", index=False)
    if not bool(tests["passed"].all()):
        raise RuntimeError("L3CR self-tests failed")

    started = time.perf_counter()
    for seed in seeds:
        checkpoint = args.checkpoint_dir.resolve() / f"model_bam_kan_seed{seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        exp.set_seed(seed)
        model = make_clean_bam()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        print(f"[start] seed={seed} k={args.active_dimension} device={device}", flush=True)
        refined, rows, meta = exp.refine(model, "PGD", seed, pack, config, device)
        prediction = exp.predict(refined, pack.test_x, device, config.validation_batch_size)
        metrics, per_sample = evaluate_clean(prediction, pack.test_y, pack.test_masks, pack.y_std)
        metric_rows.append({"model": method, "seed": seed, "parameters": parameter_count(refined), "accepted_steps": sum(int(row["accepted"]) for row in rows), "refinement_seconds": rows[-1]["wall_time"] if rows else 0.0, **metrics})
        sample_rows.extend({"model": method, "seed": seed, **row} for row in per_sample)
        for row in rows:
            row["method"] = method
            diagnostics.append(row)
        metadata[f"seed{seed}"] = meta
        torch.save(refined.state_dict(), outdir / f"model_bam_kan_l3cr_pgd_seed{seed}.pt")
        pd.DataFrame(metric_rows).to_csv(outdir / "progress_metrics_per_seed.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(outdir / "progress_metrics_per_sample.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(outdir / "progress_solver_diagnostics.csv", index=False)
        print(f"[result] seed={seed} global={metrics['global_score']:.6f} hole={metrics['hole_score']:.6f} balanced={metrics['balanced_score']:.6f}", flush=True)

    metrics_frame = pd.DataFrame(metric_rows)
    sample_frame = pd.DataFrame(sample_rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    bootstrap = paired_bootstrap(sample_frame, method, repetitions)
    metrics_frame.to_csv(outdir / "metrics_per_seed.csv", index=False)
    sample_frame.to_csv(outdir / "metrics_per_sample.csv", index=False)
    diagnostics_frame.to_csv(outdir / "solver_diagnostics.csv", index=False)
    bootstrap.to_csv(outdir / "paired_bootstrap.csv", index=False)
    metrics_frame.groupby("model").agg(["mean", "std", "median"]).to_csv(outdir / "metrics_summary.csv")
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_means = metrics_frame.groupby("model")[["global_score", "hole_score", "balanced_score"]].mean()
    certificate = diagnostics_frame.groupby("seed").agg(
        outer_steps=("outer_iteration", "count"),
        accepted_steps=("accepted", "sum"),
        min_actual_decrease=("actual_decrease", "min"),
        total_actual_decrease=("actual_decrease", "sum"),
        max_outer_backtracks=("outer_backtracks", "max"),
        max_final_step_norm=("final_step_norm", "max"),
    ).reset_index()
    certificate["finite_descent_certificate"] = (certificate["accepted_steps"] == certificate["outer_steps"]) & (certificate["min_actual_decrease"] > 0.0)
    certificate.to_csv(outdir / "l3cr_finite_certificate.csv", index=False)
    report = [
        "# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD",
        "",
        "本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。",
        "",
        all_means.to_string(),
        "",
        "## 配对 bootstrap",
        "",
        bootstrap.to_string(index=False),
        "",
        "## 有限下降证书",
        "",
        certificate.to_string(index=False),
        "",
        f"总运行时间：{time.perf_counter() - started:.1f} s。",
    ]
    (outdir / "experiment_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (outdir / "runtime.json").write_text(json.dumps({"seconds": time.perf_counter() - started, "device": str(device)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
