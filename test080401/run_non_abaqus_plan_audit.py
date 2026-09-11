#!/usr/bin/env python3
"""Execute the data-audit and result-reconciliation part of the non-Abaqus plan.

This script deliberately keeps response-derived or provenance-uncertain input
channels out of the leakage-free data product.  Existing experiment tables are
copied into a clearly labelled legacy bundle; they are not silently relabelled
as leakage-free retraining results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data_28_06_2026"
KEEP_INPUTS = [2, 3, 5, 6, 7, 8, 9, 11]
TARGETS = [0, 1]
UNCERTAIN_INPUTS = [4, 10, 12]
SPLIT_SEED = 20260629


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value).strip()


def first_value(frame: Any, column: str) -> Any:
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns or frame.empty:
        return ""
    return frame.iloc[0][column]


def case_family(name: str) -> str:
    lowered = name.lower()
    if "no-rebar" in lowered or "ratio=0%" in lowered:
        return "no-rebar"
    if "rebar" in lowered or "ratio=" in lowered:
        return "reinforced"
    return "unknown"


def case_geometry_group(name: str) -> str:
    # Geometry must not absorb reinforcement ratio, temperature or job index.
    # Include the fields that can change the physical geometry or discretization
    # in both naming conventions found in the archive.
    patterns = [
        ("L", r"\bL=([^\s\)]+)"),
        ("W", r"\bW=([^\s\)]+)"),
        ("H", r"\bH=([^\s\)]+)"),
        ("TH", r"\bTH=([^\s\)]+)"),
        ("grid", r"Grid of slab=([^\s\)]+)"),
        ("gs", r"G-S=([^\s\)]+)"),
        ("hole", r"hole=([^\s\)]+)"),
        ("r", r"\br=([^\s\)]+)"),
    ]
    values = []
    for label, pattern in patterns:
        match = re.search(pattern, name)
        if match:
            values.append(f"{label}={match.group(1)}")
    return "|".join(values) if values else "unknown_geometry"


def flatten_parameters(parameters: np.ndarray, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(parameters.tolist()):
        if not isinstance(item, Mapping):
            rows.append({"split": split, "sample": index, "name": "", "parse_status": "invalid"})
            continue
        name = short_text(item.get("name"))
        global_frame = item.get("global")
        concrete_frame = item.get("concrete")
        rebar_frame = item.get("rebar")
        row: Dict[str, Any] = {
            "split": split,
            "sample": index,
            "name": name,
            "case_family": case_family(name),
            "geometry_group": case_geometry_group(name),
            "parse_status": "ok",
        }
        columns = {
            "length_m": (global_frame, "Length/(m)"),
            "width_m": (global_frame, "Width/(m)"),
            "thickness_m": (global_frame, "Thikness/(m)"),
            "grid_size_m": (global_frame, "Grid size"),
            "temperature_c": (global_frame, "Temperature/(℃)"),
            "concrete_E_MPa": (concrete_frame, "Young's modulus(MPa)"),
            "concrete_nu": (concrete_frame, "Poisson's ratio"),
            "shrinkage_strain": (concrete_frame, "Shrinkage Strain ε"),
            "concrete_strength_MPa": (concrete_frame, "Strength (MPa)"),
            "rebar_E_Pa": (rebar_frame, "Young's modulus/(Pa)"),
            "rebar_diameter_m": (rebar_frame, "Diameter/(m)"),
            "reinforcement_ratio_pct": (rebar_frame, "Reinforement Ratio(%)"),
        }
        for output, (frame, column) in columns.items():
            row[output] = first_value(frame, column)
        rows.append(row)
    return pd.DataFrame(rows)


def file_inventory(data_dir: Path, output_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir)
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(relative),
                "top_level": relative.parts[0] if relative.parts else "",
                "suffix": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def duplicate_groups(inventory: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if inventory.empty:
        result = pd.DataFrame(columns=["sha256", "size_bytes", "file_count", "paths"])
    else:
        result = (
            inventory.groupby(["sha256", "size_bytes"], as_index=False)
            .agg(file_count=("relative_path", "size"), paths=("relative_path", lambda x: " || ".join(x)))
        )
        result = result[result["file_count"] > 1].copy()
    result.to_csv(output_path, index=False)
    return result


def array_fingerprints(data_dir: Path, parameters: Dict[str, np.ndarray], output_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        tensor = np.load(data_dir / f"{split}.npy", mmap_mode="r")
        for sample in range(tensor.shape[0]):
            digest = hashlib.sha256(np.asarray(tensor[sample], dtype=np.float32).tobytes()).hexdigest()
            item = parameters[split][sample]
            name = short_text(item.get("name")) if isinstance(item, Mapping) else ""
            rows.append(
                {
                    "split": split,
                    "sample": sample,
                    "name": name,
                    "case_family": case_family(name),
                    "geometry_group": case_geometry_group(name),
                    "tensor_sha256": digest,
                    "tensor_shape": "x".join(map(str, tensor.shape[1:])),
                }
            )
    frame = pd.DataFrame(rows)
    frame["exact_tensor_duplicate_count"] = frame.groupby("tensor_sha256")["tensor_sha256"].transform("size")
    frame.to_csv(output_path, index=False)
    return frame


def create_clean_data(data_dir: Path, output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    train = np.load(data_dir / "train.npy").astype(np.float32)
    test = np.load(data_dir / "test.npy").astype(np.float32)
    if train.ndim != 5 or test.ndim != 5 or train.shape[-1] != 13 or test.shape[-1] != 13:
        raise ValueError(f"Expected five-dimensional 13-channel arrays, got {train.shape}, {test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("NaN or Inf found in raw FEM tensors")

    material_train = train[..., 3] > 0.0
    material_test = test[..., 3] > 0.0
    raw_train_x = train[..., KEEP_INPUTS]
    raw_test_x = test[..., KEEP_INPUTS]
    raw_train_y = train[..., TARGETS]
    raw_test_y = test[..., TARGETS]
    x_mean = np.asarray([raw_train_x[..., i][material_train].mean() for i in range(len(KEEP_INPUTS))], dtype=np.float32)
    x_std = np.asarray([raw_train_x[..., i][material_train].std() for i in range(len(KEEP_INPUTS))], dtype=np.float32)
    y_mean = np.asarray([raw_train_y[..., i][material_train].mean() for i in range(len(TARGETS))], dtype=np.float32)
    y_std = np.asarray([raw_train_y[..., i][material_train].std() for i in range(len(TARGETS))], dtype=np.float32)
    x_std = np.where(x_std > 1.0e-12, x_std, 1.0).astype(np.float32)
    y_std = np.where(y_std > 1.0e-12, y_std, 1.0).astype(np.float32)

    clean_dir = output_dir / "leakage_free_v1"
    clean_dir.mkdir(parents=True, exist_ok=True)
    np.save(clean_dir / "train_targets.npy", raw_train_y)
    np.save(clean_dir / "test_targets.npy", raw_test_y)
    np.save(clean_dir / "train_inputs.npy", raw_train_x)
    np.save(clean_dir / "test_inputs.npy", raw_test_x)
    metadata = {
        "raw_channel_order": list(range(13)),
        "target_channels": TARGETS,
        "retained_input_channels": KEEP_INPUTS,
        "excluded_input_channels": UNCERTAIN_INPUTS,
        "coordinate_channels": ["x", "y", "z"],
        "normalization_fit": "training material voxels only",
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "material_voxels_train": int(material_train.sum()),
        "material_voxels_test": int(material_test.sum()),
        "warning": "c04, c10 and c12 remain excluded until their provenance is confirmed from FEM export metadata.",
    }
    (clean_dir / "normalization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    for index, channel in enumerate(range(13)):
        if channel in TARGETS:
            role, decision, reason = "target", "target", "reported FEM output S1 or U"
        elif channel in KEEP_INPUTS:
            role, decision, reason = "input", "retain", "geometry/material/control field according to channel map"
        else:
            role, decision, reason = "input", "exclude", "response-derived or provenance uncertain; exclusion is conservative"
        rows.append({"channel": channel, "role": role, "decision": decision, "reason": reason})
    provenance = pd.DataFrame(rows)
    provenance.to_csv(output_dir.parent / "channel_provenance.csv", index=False)
    return provenance, metadata


def metric_self_tests(output_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20260804)
    target = rng.normal(size=(2, 2, 4, 4, 2)).astype(np.float32)
    prediction = target.copy()
    material = np.ones_like(target[:, :1], dtype=bool)
    material[..., 0, 0, 0] = False
    hole = np.zeros_like(material)
    hole[..., 1, 1, 0] = True
    hole &= material
    material_full = np.broadcast_to(material, target.shape)
    hole_full = np.broadcast_to(hole, target.shape)
    cases = [("zero_error_global", prediction, target, material_full, 0.0), ("zero_error_hole", prediction, target, hole_full, 0.0)]
    for name, pred, truth, mask, expected in cases:
        value = float(np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2))) if mask.any() else float("nan")
        rows.append({"test": name, "value": value, "expected": expected, "passed": bool(np.isclose(value, expected))})
    prediction = target.copy()
    prediction[:, 0] += 2.0
    channel0_mask = material_full.copy()
    channel0_mask[:, 1] = False
    value = float(np.sqrt(np.mean((prediction[channel0_mask] - target[channel0_mask]) ** 2)))
    rows.append({"test": "positive_masked_error", "value": value, "expected": 2.0, "passed": bool(np.isclose(value, 2.0))})
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def read_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "source_label", label)
    return frame


def merge_legacy_results(root: Path, output_dir: Path) -> Dict[str, int]:
    architecture = read_metrics(root / "test071407/formal/metrics_per_seed.csv", "legacy_test071407")
    if not architecture.empty:
        architecture.to_csv(output_dir / "legacy_architecture_results.csv", index=False)
    l3cr_paths = [
        (root / "test071502/formal/metrics_per_seed.csv", "legacy_adamw_l3cr"),
        (root / "test071703/tuned_pgd_formal/metrics_per_seed.csv", "legacy_adam_l3cr_pgd"),
    ]
    l3cr = pd.concat([read_metrics(path, label) for path, label in l3cr_paths if path.exists()], ignore_index=True)
    if not l3cr.empty:
        l3cr.to_csv(output_dir / "legacy_l3cr_results.csv", index=False)

    physical = read_metrics(root / "test071704/analysis/physical_unit_metrics.csv", "legacy_physical_units")
    if not physical.empty:
        physical.to_csv(output_dir / "legacy_physical_metrics.csv", index=False)
    hole = read_metrics(root / "test071704/analysis/hole_band_sensitivity_summary.csv", "legacy_hole_band")
    if not hole.empty:
        hole.to_csv(output_dir / "legacy_hole_radius_results.csv", index=False)
    subgroup = read_metrics(root / "test071704/analysis/subgroup_metrics.csv", "legacy_subgroup")
    if not subgroup.empty:
        subgroup.to_csv(output_dir / "legacy_subgroup_results.csv", index=False)

    statistical_rows = []
    if not architecture.empty and "model" in architecture.columns:
        for metric in ["global_score", "hole_score", "balanced_score"]:
            pivot = architecture.pivot(index="seed", columns="model", values=metric)
            if {"BAM-PIKAN", "U-Net", "CNN"}.issubset(pivot.columns):
                for baseline in ["U-Net", "CNN"]:
                    difference = pivot["BAM-PIKAN"] - pivot[baseline]
                    row = {"comparison": f"BAM-PIKAN - {baseline}", "metric": metric, "mean_difference": float(difference.mean()), "n": int(difference.notna().sum())}
                    try:
                        from scipy.stats import wilcoxon
                        test = wilcoxon(difference.dropna().to_numpy(), alternative="less", method="auto")
                        row.update({"wilcoxon_statistic": float(test.statistic), "wilcoxon_p_less": float(test.pvalue)})
                    except Exception as exc:
                        row.update({"wilcoxon_statistic": float("nan"), "wilcoxon_p_less": float("nan"), "wilcoxon_error": str(exc)})
                    statistical_rows.append(row)
    pd.DataFrame(statistical_rows).to_csv(output_dir / "legacy_statistical_tests.csv", index=False)
    return {"architecture_rows": len(architecture), "l3cr_rows": len(l3cr), "physical_rows": len(physical), "hole_rows": len(hole), "subgroup_rows": len(subgroup)}


def write_reports(
    output_dir: Path,
    data_dir: Path,
    inventory: pd.DataFrame,
    duplicates: pd.DataFrame,
    fingerprints: pd.DataFrame,
    provenance: pd.DataFrame,
    metadata: Dict[str, Any],
    metric_tests: pd.DataFrame,
    legacy_counts: Dict[str, int],
) -> None:
    train_names = fingerprints[fingerprints["split"] == "train"]
    test_names = fingerprints[fingerprints["split"] == "test"]
    group_overlap = set(train_names["geometry_group"]) & set(test_names["geometry_group"])
    report = [
        "# 非 Abaqus 实验补充包：数据与结果审计",
        "",
        "## 已确认的数据边界",
        "",
        f"- 数据目录：`{data_dir}`。",
        f"- 原始张量：train `{metadata['train_shape']}`，test `{metadata['test_shape']}`。",
        f"- 文件总数：`{len(inventory)}`；按 SHA-256 和文件大小识别的完全重复文件组：`{len(duplicates)}`。",
        f"- 张量样本：train `{len(train_names)}`，test `{len(test_names)}`；完全重复张量数量：`{int((fingerprints['exact_tensor_duplicate_count'] > 1).sum())}`。",
        f"- 几何指纹在 train/test 之间重合组数：`{len(group_overlap)}`。这意味着只凭现有数组不能声称已经完成严格的几何组独立测试。",
        "",
        "## 无泄漏输入产品",
        "",
        "保留目标通道 c00/c01 和输入 c02、c03、c05、c06、c07、c08、c09、c11；排除 c04、c10、c12。后 3 个通道可能包含响应量或接触响应，当前压缩数据没有足够的导出元数据证明它们是纯先验输入，因此采用保守排除。",
        "",
        f"训练材料体素归一化参数已写入 `data_clean/leakage_free_v1/normalization.json`，训练材料体素数为 `{metadata['material_voxels_train']}`，测试材料体素数为 `{metadata['material_voxels_test']}`。",
        "",
        "## 指标自检",
        "",
        f"- 指标自检通过：`{bool(metric_tests['passed'].all())}`。详细结果见 `metric_self_tests.csv`。",
        "- Global 使用材料掩码；Hole 使用孔洞边界邻域与材料掩码的交集；Balanced 定义为 Global 与 Hole 两者的平均。",
        "- 当前数据没有完整 Abaqus 节点位移向量、六分量应力、本构参数场、边界法向量与接触状态，因此不能据此合法构造强形式 PIKAN 的平衡、本构和接触残差。",
        "",
        "## 已有结果的使用规则",
        "",
        f"- 已统一索引的旧架构结果行数：`{legacy_counts['architecture_rows']}`；旧 L3CR 结果行数：`{legacy_counts['l3cr_rows']}`。",
        "- 这些结果来自旧实验协议，保留为 legacy reference；它们不能替代基于上述无泄漏输入的重新训练。",
        "- 物理单位、Hole 半径、子组和统计表如果存在，已复制到 `tables/`，并带有来源标签。",
        "",
        "## 当前仍需重新完成的实验",
        "",
        "1. 用 leakage_free_v1 输入重新训练 CNN、U-Net、BAM-KAN，并锁定同一 checkpoint 选择协议。",
        "2. 进行 BAM-MLP/BAM-Conv 的 KAN 隔离实验和 BAM 路径消融。",
        "3. 从同一 AdamW checkpoint 启动 L3CR-PGD，生成有限终止证书、接受步、回溯和实际闭包下降。",
        "4. 由于缺少原始 Abaqus 场字段，E15 网格投影误差和真正的强形式物理残差暂不能执行，不能用代理量填充。",
    ]
    (output_dir / "audit_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    dataset_card = [
        "# Leakage-free v1 dataset card",
        "",
        "This product retains only channels whose role is supported by the local channel map and excludes response-derived or provenance-uncertain channels.",
        "",
        f"- Targets: `{TARGETS}`.",
        f"- Retained inputs: `{KEEP_INPUTS}`.",
        f"- Excluded inputs: `{UNCERTAIN_INPUTS}`.",
        "- Normalization: training material voxels only; test data never enters the statistics.",
        "- Coordinates: generated normalized x/y/z channels are permitted as geometric coordinates, not target-derived features.",
        "- Limitation: this is leakage-free with respect to the documented channel map, but strict group-independent generalization requires the original case grouping/provenance table.",
    ]
    (output_dir / "leakage_free_v1_dataset_card.md").write_text("\n".join(dataset_card) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--root", type=Path, default=HERE.parent)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["data_audit", "data_clean", "tables", "legacy_reuse"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    parameters = {
        "train": np.load(args.data_dir / "train_parameters.npy", allow_pickle=True),
        "test": np.load(args.data_dir / "test_parameters.npy", allow_pickle=True),
    }
    parameter_frames = pd.concat(
        [flatten_parameters(parameters["train"], "train"), flatten_parameters(parameters["test"], "test")],
        ignore_index=True,
    )
    parameter_frames.to_csv(args.out_dir / "data_audit/case_metadata.csv", index=False)
    inventory = file_inventory(args.data_dir, args.out_dir / "data_audit/file_inventory.csv")
    duplicates = duplicate_groups(inventory, args.out_dir / "data_audit/duplicate_groups.csv")
    fingerprints = array_fingerprints(args.data_dir, parameters, args.out_dir / "data_audit/case_fingerprints.csv")
    provenance, metadata = create_clean_data(args.data_dir, args.out_dir / "data_clean")
    metric_tests = metric_self_tests(args.out_dir / "data_audit/metric_self_tests.csv")
    shutil.copy2(args.out_dir / "data_audit/case_metadata.csv", args.out_dir / "tables/case_metadata.csv")
    legacy_counts = merge_legacy_results(args.root, args.out_dir / "tables")
    (args.out_dir / "run_summary.json").write_text(
        json.dumps({"data_dir": str(args.data_dir), "metadata": metadata, "legacy_counts": legacy_counts}, indent=2),
        encoding="utf-8",
    )
    write_reports(args.out_dir, args.data_dir, inventory, duplicates, fingerprints, provenance, metadata, metric_tests, legacy_counts)
    print(json.dumps({"metadata": metadata, "legacy_counts": legacy_counts, "metric_tests_passed": bool(metric_tests["passed"].all())}, indent=2))


if __name__ == "__main__":
    main()
