#!/usr/bin/env python3
"""Index reusable pre-audit result files without copying or modifying them."""

from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    rows = [
        {"source_file": str(ROOT / "test071407/formal/metrics_per_seed.csv"), "model": "legacy CNN/U-Net/BAM comparison", "seed": "multiple", "protocol": "legacy", "paper_table": "pre-audit architecture comparison"},
        {"source_file": str(ROOT / "test071601/formal_k32/metrics_per_seed.csv"), "model": "BAM-AdamW-L3CR-PGD-k32", "seed": "11,23,37,51,73", "protocol": "legacy", "paper_table": "active-dimension sensitivity"},
        {"source_file": str(ROOT / "test071601/formal_k64/metrics_per_seed.csv"), "model": "BAM-AdamW-L3CR-PGD-k64", "seed": "11,23,37,51,73", "protocol": "legacy", "paper_table": "active-dimension sensitivity"},
        {"source_file": str(ROOT / "test071704/analysis/hole_band_sensitivity_per_seed.csv"), "model": "legacy hole-band sensitivity", "seed": "multiple", "protocol": "legacy", "paper_table": "pre-audit hole-band sensitivity"},
        {"source_file": str(ROOT / "test071505/formal/metrics_per_seed.csv"), "model": "BAM-Adam-L3CR three-solvers", "seed": "11,23,37,51,73", "protocol": "legacy", "paper_table": "solver comparison"},
    ]
    frame = pd.DataFrame(rows)
    frame["exists"] = frame["source_file"].map(lambda value: Path(value).exists())
    frame.to_csv(HERE / "legacy_result_index.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
