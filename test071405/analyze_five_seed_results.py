#!/usr/bin/env python3
"""Combine the locked three-seed run and two confirmatory seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_fair_pikan_unet_comparison import (
    Config,
    paired_bootstrap,
    save_plots,
    write_report,
)


ROOT = Path(__file__).resolve().parent
SOURCES = [ROOT / "formal", ROOT / "formal_extra"]
OUTPUT = ROOT / "formal_five_seed"


def combine(filename: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(source / filename) for source in SOURCES], ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = combine("metrics_per_seed.csv").sort_values(["seed", "model"])
    per_sample = combine("metrics_per_sample.csv").sort_values(["seed", "sample", "model"])
    history = combine("training_history.csv").sort_values(["seed", "model", "epoch"])
    expected = {11, 23, 37, 51, 73}
    if set(metrics["seed"]) != expected or len(metrics) != 10:
        raise RuntimeError("The five-seed result set is incomplete.")

    specifications = json.loads((SOURCES[0] / "model_specifications.json").read_text(encoding="utf-8"))
    config = Config(seeds=(11, 23, 37, 51, 73))
    bootstrap = paired_bootstrap(per_sample, config.bootstrap_repetitions)
    metrics.to_csv(OUTPUT / "metrics_per_seed.csv", index=False)
    per_sample.to_csv(OUTPUT / "metrics_per_sample.csv", index=False)
    history.to_csv(OUTPUT / "training_history.csv", index=False)
    bootstrap.to_csv(OUTPUT / "paired_bootstrap.csv", index=False)
    metrics.groupby("model").agg(["mean", "std", "median"]).to_csv(OUTPUT / "metrics_summary.csv")
    save_plots(metrics, history, OUTPUT)
    write_report(config, specifications, metrics, bootstrap, OUTPUT)

    model_manifest = {
        f"{row.model}_seed{int(row.seed)}": str(
            (SOURCES[0] if int(row.seed) in {11, 23, 37} else SOURCES[1])
            / f"model_{str(row.model).replace('-', '').lower()}_seed{int(row.seed)}.pt"
        )
        for row in metrics.itertuples()
    }
    (OUTPUT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
