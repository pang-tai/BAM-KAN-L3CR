#!/usr/bin/env python3
"""Benchmark saved-model inference and record available device memory fields."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "test071407"))
sys.path.insert(0, str(HERE.parent / "test071405"))

from run_leakage_free_architecture_experiment import (  # noqa: E402
    Config,
    choose_models,
    load_clean_pack,
)


def synchronize(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> tuple[float | None, float | None]:
    if device.type == "cuda":
        return float(torch.cuda.memory_allocated(device)) / 2**20, float(torch.cuda.max_memory_allocated(device)) / 2**20
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        peak = getattr(torch.mps, "driver_allocated_memory", None)
        current_value = float(current()) / 2**20 if callable(current) else None
        peak_value = float(peak()) / 2**20 if callable(peak) else None
        return current_value, peak_value
    return None, None


def main() -> None:
    config = Config(seeds=(11,), epochs=48, batch_size=8, patience=14, bootstrap_repetitions=100, device="auto", target_parameters=90000)
    pack, _, _ = load_clean_pack(HERE / "data_clean/leakage_free_v1", HERE.parent / "data_28_06_2026")
    device_name = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    models, specs = choose_models(config, input_channels=11)
    rows = []
    repeats = 10
    warmup = 3
    for name, model in models.items():
        checkpoint = HERE / "formal_leakage_free" / f"model_{name.lower().replace('-', '_')}_seed11.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.to(device).eval()
        for batch_name, batch in (("single_case", pack.test_x[:1]), ("test_batch_19", pack.test_x)):
            batch = torch.as_tensor(batch, dtype=torch.float32, device=device)
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(batch)
                synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(batch)
                synchronize(device)
                samples.append(time.perf_counter() - start)
            current_mb, peak_mb = memory_snapshot(device)
            rows.append({
                "model": name,
                "parameters": specs[name]["parameters"],
                "device": device_name,
                "batch_name": batch_name,
                "batch_size": int(batch.shape[0]),
                "repeats": repeats,
                "warmup": warmup,
                "mean_seconds": float(np.mean(samples)),
                "std_seconds": float(np.std(samples, ddof=1)),
                "median_seconds": float(np.median(samples)),
                "mean_ms_per_case": float(np.mean(samples) / batch.shape[0] * 1000.0),
                "current_memory_mb": current_mb,
                "peak_or_driver_memory_mb": peak_mb,
            })
    output = HERE / "17_inference_cost.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    metadata = {
        "device": device_name,
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "note": "MPS does not expose a CUDA-equivalent peak allocator on all versions; unavailable fields remain null.",
    }
    (HERE / "17_inference_cost_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
