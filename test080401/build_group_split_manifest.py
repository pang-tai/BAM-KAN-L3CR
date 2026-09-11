#!/usr/bin/env python3
"""Build a deterministic geometry-group holdout manifest without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_28_06_2026"


def internal_void(mask: np.ndarray) -> np.ndarray:
    void = ~mask.astype(bool)
    labels, _ = label(void)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    result = void.copy()
    for value in border:
        result &= labels != value
    return result


def has_opening(raw: np.ndarray) -> np.ndarray:
    material = raw[..., 3] > 0.0
    result = np.zeros(raw.shape[0], dtype=bool)
    for sample in range(raw.shape[0]):
        for z in range(raw.shape[3]):
            if internal_void(material[sample, :, :, z]).any():
                result[sample] = True
                break
    return result


def main() -> None:
    fingerprints = pd.read_csv(HERE / "data_audit/case_fingerprints.csv")
    train_raw = np.load(DATA / "train.npy", mmap_mode="r")
    test_raw = np.load(DATA / "test.npy", mmap_mode="r")
    opening = np.concatenate([has_opening(np.asarray(train_raw)), has_opening(np.asarray(test_raw))])
    fingerprints["has_opening"] = opening
    fingerprints["case_id"] = fingerprints["split"] + "_" + fingerprints["sample"].astype(str)
    groups = fingerprints["geometry_group"].fillna("unknown").astype(str).unique().tolist()
    rng = np.random.default_rng(20260804)
    rng.shuffle(groups)
    group_sizes = fingerprints.groupby("geometry_group").size().to_dict()
    total = len(fingerprints)
    targets = {"test": round(0.18 * total), "validation": round(0.18 * total)}
    assigned = {"test": [], "validation": [], "train": []}
    counts = {key: 0 for key in assigned}
    group_opening = fingerprints.groupby("geometry_group")["has_opening"].sum().to_dict()
    opening_groups = sorted(
        [group for group in groups if group_opening.get(group, 0) > 0],
        key=lambda value: (-group_opening.get(value, 0), value),
    )
    # The test set must contain an opening. If a second independent opening
    # geometry exists, keep it in validation so the training/test assignment
    # is explicit rather than accidentally produced by the greedy allocator.
    if opening_groups:
        group = opening_groups.pop(0)
        assigned["test"].append(group)
        counts["test"] += group_sizes.get(group, 0)
    if opening_groups:
        group = opening_groups.pop(0)
        assigned["validation"].append(group)
        counts["validation"] += group_sizes.get(group, 0)
    # Greedy assignment keeps each group intact and aims at the target number
    # of cases; group identity is more important than exact split proportions.
    already_assigned = set(assigned["test"] + assigned["validation"] + assigned["train"])
    for group in sorted([value for value in groups if value not in already_assigned], key=lambda value: (-group_sizes.get(value, 0), value)):
        candidates = ["test", "validation", "train"]
        candidates.sort(key=lambda split: (counts[split] / max(1, targets.get(split, total)), counts[split]))
        chosen = candidates[0]
        if chosen == "train" and counts["test"] < targets["test"]:
            chosen = "test"
        elif chosen == "train" and counts["validation"] < targets["validation"]:
            chosen = "validation"
        assigned[chosen].append(group)
        counts[chosen] += group_sizes.get(group, 0)
    mapping = {group: split for split, values in assigned.items() for group in values}
    fingerprints["group_split"] = fingerprints["geometry_group"].map(mapping)
    manifest = fingerprints[["case_id", "split", "sample", "group_split", "geometry_group", "case_family", "has_opening", "name", "tensor_sha256"]].copy()
    manifest = manifest.rename(columns={"group_split": "protocol_split"})
    out = HERE / "data_clean/leakage_free_v1/split_manifest_geometry_group.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    manifest.to_csv(HERE / "05_split_manifest_geometry_group.csv", index=False)

    random_manifest = pd.read_csv(HERE / "05_split_manifest.csv")
    random_manifest.to_csv(HERE / "data_clean/leakage_free_v1/split_manifest_random.csv", index=False)
    random_manifest.to_csv(HERE / "05_split_manifest_random.csv", index=False)

    summary = manifest.groupby("protocol_split").agg(cases=("case_id", "size"), geometry_groups=("geometry_group", "nunique"), opening_cases=("has_opening", "sum"), no_rebar=("case_family", lambda x: int((x == "no-rebar").sum()))).reset_index()
    summary.to_csv(HERE / "data_audit/geometry_group_split_summary.csv", index=False)
    pd.DataFrame([
        {"channel": 4, "decision": "exclude", "reason": "case-level displacement descriptor is response-derived or provenance uncertain"},
        {"channel": 10, "decision": "exclude", "reason": "contact shear stress is response-derived or provenance uncertain"},
        {"channel": 12, "decision": "exclude", "reason": "contact displacement is response-derived or provenance uncertain"},
    ]).to_csv(HERE / "data_audit/exclusion_log.csv", index=False)
    group_sets = {key: set(value) for key, value in assigned.items()}
    report = {
        "total_cases": total,
        "geometry_group_count": len(groups),
        "split_counts": counts,
        "group_overlap_pairs": {f"{left}_{right}": len(group_sets[left] & group_sets[right]) for left in group_sets for right in group_sets if left < right},
        "opening_cases_by_split": summary.set_index("protocol_split")["opening_cases"].to_dict(),
        "warning": "This manifest is a strict group assignment under the available geometry_group proxy; it is not used as the main formal retraining split because the source archive does not expose complete geometry genealogy.",
    }
    (HERE / "data_audit/geometry_group_split_report.json").write_text(json.dumps(report, indent=2, default=int), encoding="utf-8")
    print(json.dumps(report, indent=2, default=int))


if __name__ == "__main__":
    main()
