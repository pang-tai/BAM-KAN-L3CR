#!/usr/bin/env python3
"""Verify integrity and manuscript-facing numerical consistency of this package."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"FAIL: {message}")


def passed(message: str) -> None:
    print(f"PASS: {message}")


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        fail(f"missing {relative}")
    return path


def rows(relative: str) -> list[dict[str, str]]:
    path = require(relative)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def close(label: str, value: float, expected: float, tolerance: float = 5e-5) -> None:
    if abs(value - expected) > tolerance:
        fail(f"{label}: {value} != {expected} within {tolerance}")
    else:
        passed(f"{label} = {value:.8g}")


def group_mean(records: list[dict[str, str]], key: str, value: str, target: str) -> float:
    values = [float(row[value]) for row in records if row[key] == target]
    if not values:
        raise ValueError(f"no rows for {key}={target} in {value}")
    return float(np.mean(values))


def check_required() -> None:
    required = [
        "manuscript/ver7.3-Tai_2nd_paper.docx",
        "manuscript/ver7.3-Tai_2nd_paper.pdf",
        "data_28_06_2026/train.npy",
        "data_28_06_2026/test.npy",
        "test080401/07_main_architecture_results.csv",
        "test080401/formal_clean_l3cr_k64/metrics_per_seed.csv",
        "test082402/metrics_summary.csv",
        "test082402/threshold_reach_summary.csv",
        "test090401/hvp_audit_summary.csv",
        "test090401/fixed_three_step_l3cr_per_seed.csv",
        "test090401/matched_budget_optimizer_summary.csv",
    ]
    for item in required:
        require(item)
    if not FAILURES:
        passed("all required source and result files are present")


def check_python_sources() -> None:
    count = 0
    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            count += 1
        except Exception as error:
            fail(f"Python parse error in {path.relative_to(ROOT)}: {error}")
    if count:
        passed(f"parsed {count} Python source files")


def check_tensors() -> None:
    train = np.load(require("data_28_06_2026/train.npy"), mmap_mode="r", allow_pickle=False)
    test = np.load(require("data_28_06_2026/test.npy"), mmap_mode="r", allow_pickle=False)
    if train.shape[0] != 167 or test.shape[0] != 19:
        fail(f"unexpected tensor case counts: train-development={train.shape[0]}, test={test.shape[0]}")
    else:
        passed("processed archive contains 167 development and 19 held-out cases")
    if tuple(train.shape[1:4]) != (64, 64, 4) or tuple(test.shape[1:4]) != (64, 64, 4):
        fail(f"unexpected spatial tensor shapes: train={train.shape}, test={test.shape}")
    else:
        passed("tensor grid is 64 x 64 x 4")


def check_architecture() -> None:
    data = rows("test080401/07_main_architecture_results.csv")
    expected = {
        "CNN": (0.7486075461, 0.5528558016, 0.6507316738),
        "U-Net": (0.8392489970, 0.5531007260, 0.6961748615),
        "BAM-KAN": (0.8041690767, 0.4706331030, 0.6374010898),
    }
    for model, values in expected.items():
        close(f"{model} Global mean", group_mean(data, "model", "global_score", model), values[0])
        close(f"{model} Opening-band mean", group_mean(data, "model", "hole_score", model), values[1])
        close(f"{model} Balanced mean", group_mean(data, "model", "balanced_score", model), values[2])


def check_active_refinement() -> None:
    data = rows("test080401/formal_clean_l3cr_k64/metrics_per_seed.csv")
    method = "BAM-KAN-AdamW-L3CR-PGD"
    close("AS-L3CR r64 Global mean", group_mean(data, "model", "global_score", method), 0.8040890396)
    close("AS-L3CR r64 Opening-band mean", group_mean(data, "model", "hole_score", method), 0.4684387147)
    close("AS-L3CR r64 Balanced mean", group_mean(data, "model", "balanced_score", method), 0.6362638772)


def check_twenty_parameter() -> None:
    data = rows("test082402/metrics_summary.csv")
    row = next((r for r in data if r.get("experiment") == "bam_head" and r.get("method") == "Original fixed L3CR"), None)
    if row is None:
        fail("missing Original fixed L3CR row in 20-parameter summary")
        return
    close("20-parameter mean terminal gradient", float(row["gradient_norm_mean"]), 4.5736134207e-16, 1e-20)
    thresholds = rows("test082402/threshold_reach_summary.csv")
    ten = next((r for r in thresholds if r.get("experiment") == "bam_head" and r.get("method") == "Original fixed L3CR" and float(r.get("threshold", "nan")) == 1e-10), None)
    if ten is None or int(ten["reached_within_500_count"]) != 5:
        fail("20-parameter 1e-10 success is not 5 of 5")
    else:
        passed("20-parameter L3CR reaches 1e-10 within budget for 5 of 5 checkpoints")


def check_full_parameter() -> None:
    hvp = rows("test090401/hvp_audit_summary.csv")
    for stage, parameters, fd, symmetry in [
        ("P5", 44149, 5.4408208773e-9, 7.1222472129e-15),
        ("P6", 86653, 2.3215967286e-9, 1.4607198482e-14),
    ]:
        row = next((r for r in hvp if r["stage"] == stage), None)
        if row is None:
            fail(f"missing {stage} HVP audit")
            continue
        if int(row["parameters"]) != parameters:
            fail(f"{stage} parameter count mismatch")
        else:
            passed(f"{stage} parameter count = {parameters}")
        close(f"{stage} FD error", float(row["fd_relative_error"]), fd, 1e-12)
        close(f"{stage} symmetry error", float(row["symmetry_relative_error"]), symmetry, 1e-17)

    fixed = rows("test090401/fixed_three_step_l3cr_per_seed.csv")
    formal = [r for r in fixed if r["method"] == "MF-L3CR" and r["stage"] in {"P5", "P6"}]
    if len(formal) != 10 or any(r["status"] != "PASS" or int(float(r["accepted_steps"])) != 3 for r in formal):
        fail("P5/P6 fixed-three-step completion is not 10 PASS rows with 3 accepted steps")
    else:
        passed("all ten P5/P6 runs have PASS status and three accepted steps")

    summary = rows("test090401/matched_budget_optimizer_summary.csv")
    targets = {
        ("P5", "MF-L3CR"): (0.1503594166, 0.1562940873),
        ("P6", "MF-L3CR"): (0.1503953269, 0.1523654457),
        ("P5", "Resumed-AdamW"): (0.1534790127, 0.2030097349),
        ("P6", "Resumed-AdamW"): (0.1535163853, 0.2108883497),
    }
    for (stage, method), expected in targets.items():
        row = next((r for r in summary if r["stage"] == stage and r["method"] == method), None)
        if row is None:
            fail(f"missing matched-budget row {stage} {method}")
            continue
        close(f"{stage} {method} terminal F", float(row["terminal_objective_mean"]), expected[0])
        close(f"{stage} {method} terminal gradient", float(row["terminal_gradient_norm_mean"]), expected[1])


def check_hashes() -> None:
    manifest = require("MANIFEST_SHA256.txt")
    if not manifest.exists():
        return
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            fail(f"manifest file missing: {relative}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            fail(f"hash mismatch: {relative}")
        checked += 1
    if checked and not any("hash mismatch" in item or "manifest file missing" in item for item in FAILURES):
        passed(f"verified SHA-256 for {checked} files")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", action="store_true", help="verify every file against MANIFEST_SHA256.txt")
    args = parser.parse_args()
    check_required()
    check_python_sources()
    check_tensors()
    check_architecture()
    check_active_refinement()
    check_twenty_parameter()
    check_full_parameter()
    if args.hashes:
        check_hashes()
    if FAILURES:
        print(f"\nVerification failed with {len(FAILURES)} issue(s).")
        return 1
    print("\nVerification passed. The archived numerical tables are internally consistent with the manuscript-facing values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
