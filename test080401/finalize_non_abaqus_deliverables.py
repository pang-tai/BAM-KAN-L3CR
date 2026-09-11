#!/usr/bin/env python3
"""Create the named deliverables and explicit status records for incomplete items."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def main() -> None:
    audit = HERE / "data_audit"
    TABLES = HERE / "tables"
    fingerprints = pd.read_csv(audit / "case_fingerprints.csv")
    train = fingerprints[fingerprints["split"] == "train"].copy()
    rng = np.random.default_rng(20260629)
    permutation = rng.permutation(len(train))
    validation_count = max(1, int(round(0.18 * len(train))))
    validation_indices = set(train.iloc[permutation[:validation_count]]["sample"].astype(int))
    train["protocol_split"] = train["sample"].map(lambda x: "validation" if int(x) in validation_indices else "train")
    test = fingerprints[fingerprints["split"] == "test"].copy()
    test["protocol_split"] = "test"
    manifest = pd.concat([train, test], ignore_index=True)
    train_groups = set(train["geometry_group"])
    manifest["geometry_group_overlaps_train_test"] = manifest["geometry_group"].isin(train_groups) & (manifest["split"] == "test")
    manifest.to_csv(audit / "split_manifest.csv", index=False)

    aliases = {
        HERE / "channel_provenance.csv": HERE / "01_channel_provenance.csv",
        audit / "file_inventory.csv": HERE / "02_file_inventory.csv",
        audit / "duplicate_groups.csv": HERE / "03_duplicate_groups.csv",
        audit / "case_fingerprints.csv": HERE / "04_case_fingerprints.csv",
        audit / "split_manifest.csv": HERE / "05_split_manifest.csv",
        HERE / "data_clean/leakage_free_v1/normalization.json": HERE / "06_normalization.json",
        TABLES / "07_main_architecture_results.csv": HERE / "07_main_architecture_results.csv",
        TABLES / "11_l3cr_finite_certificate.csv": HERE / "11_l3cr_finite_certificate.csv",
        HERE / "tables/15_paired_statistics.csv": HERE / "15_paired_statistics.csv",
        HERE / "tables/16_computational_cost.csv": HERE / "16_computational_cost.csv",
        HERE / "tables/19_displacement_nonnegative_audit.csv": HERE / "19_displacement_nonnegative_audit.csv",
    }
    for source, target in aliases.items():
        copy_if_exists(source, target)

    status_rows = [
        {"item": "08_bam_kan_isolation", "status": "not_run", "reason": "Requires new parameter-matched BAM-MLP and BAM-Conv retraining."},
        {"item": "09_pathway_ablation", "status": "not_run", "reason": "Requires formal removal-path retraining under the leakage-free input protocol."},
        {"item": "10_refinement_from_clean_checkpoint", "status": "not_run", "reason": "Legacy L3CR logs are certified separately; they do not start from the new leakage-free BAM-KAN checkpoints."},
        {"item": "14_projection_audit", "status": "unavailable", "reason": "The supplied archive contains S1/U and metadata, not raw Abaqus nodal/element fields for independent-grid projection."},
        {"item": "19_displacement_nonnegative_audit", "status": "completed", "reason": "Reverse-normalized U predictions were checked for negative values for all 15 new checkpoints."},
    ]
    status = pd.DataFrame(status_rows)
    status.to_csv(HERE / "experiment_status.csv", index=False)
    status[status["item"] == "08_bam_kan_isolation"].to_csv(HERE / "08_bam_kan_isolation_results.csv", index=False)
    status[status["item"] == "09_pathway_ablation"].to_csv(HERE / "09_pathway_ablation_results.csv", index=False)
    status[status["item"] == "10_refinement_from_clean_checkpoint"].to_csv(HERE / "10_refinement_results.csv", index=False)
    status[status["item"] == "14_projection_audit"].to_csv(HERE / "14_projection_audit.csv", index=False)
    copy_if_exists(TABLES / "legacy_subgroup_results.csv", HERE / "12_subgroup_results.csv")
    copy_if_exists(TABLES / "legacy_hole_radius_results.csv", HERE / "13_hole_radius_results.csv")
    copy_if_exists(HERE / "figures/leakage_free_architecture_comparison.pdf", HERE / "17_architecture_comparison.pdf")

    reproducibility = [
        "# Reproducibility report",
        "",
        "## Completed in this package",
        "",
        "- Raw file inventory, SHA-256 hashes, duplicate-file groups, case metadata and tensor fingerprints.",
        "- Conservative leakage-free v1 product excluding c04/c10/c12, with training-only normalization.",
        "- Deterministic metric self-tests and a five-seed CNN/U-Net/BAM-KAN retraining on the cleaned input.",
        "- New checkpoints, per-seed metrics, per-sample metrics, history and paired statistics.",
        "- Static BAM thickness-direction audit and finite-descent certificates from reusable legacy L3CR logs.",
        "",
        "## Not silently claimed",
        "",
        "- Existing legacy L3CR tables are not presented as continuation from the new leakage-free BAM-KAN checkpoints.",
        "- Missing Abaqus tensor fields are not reconstructed from S1/U; strong-form PIKAN residuals and independent-grid projection error remain unavailable.",
        "- The current tensor split has geometry-group overlap indicators; strict group-independent generalization requires the original case genealogy.",
        "",
        "## Re-run commands",
        "",
        "```bash",
        '"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_non_abaqus_plan_audit.py',
        '"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_leakage_free_architecture_experiment.py --epochs 48 --seeds 11 23 37 51 73 --out-dir test080401/formal_leakage_free --device mps',
        '"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/summarize_non_abaqus_results.py',
        "```",
    ]
    (HERE / "18_reproducibility_report.md").write_text("\n".join(reproducibility) + "\n", encoding="utf-8")
    mapping = [
        "# Manuscript table and figure mapping",
        "",
        "| Manuscript item | File | Interpretation |",
        "|---|---|---|",
        "| Data provenance | `01_channel_provenance.csv`, `02_file_inventory.csv`, `03_duplicate_groups.csv`, `04_case_fingerprints.csv` | Audit evidence; not a performance claim. |",
        "| Split and normalization | `05_split_manifest.csv`, `06_normalization.json` | Defines leakage-free v1 protocol. |",
        "| Main architecture table | `07_main_architecture_results.csv`, `tables/07_main_architecture_summary.csv` | Five-seed retraining; lower scores are better. |",
        "| Main architecture figure | `17_architecture_comparison.pdf` | Global/Hole/Balanced with seed standard deviations. |",
        "| Paired inference | `15_paired_statistics.csv` | Bootstrap CI and Wilcoxon; do not report as universal superiority when CI crosses zero. |",
        "| L3CR certificate | `tables/11_l3cr_finite_certificate.csv` | Legacy log certificate only; not new-clean-checkpoint refinement. |",
        "| Missing experiments | `08_bam_kan_isolation_results.csv`, `09_pathway_ablation_results.csv`, `10_refinement_results.csv`, `14_projection_audit.csv` | Explicit status records; do not turn into numerical claims. |",
    ]
    (HERE / "19_manuscript_table_mapping.md").write_text("\n".join(mapping) + "\n", encoding="utf-8")
    summary = {"manifest_rows": len(manifest), "group_overlap_test_rows": int(manifest["geometry_group_overlaps_train_test"].sum()), "status_rows": len(status)}
    (HERE / "deliverable_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
