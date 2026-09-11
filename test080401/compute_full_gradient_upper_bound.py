#!/usr/bin/env python3
"""Compute a declared first-order full-gradient upper bound for E12.

For a closure objective L and a parameter step ||s||_2 <= delta,
|<grad L, s>| <= delta ||grad L||_2. The table reports this bound at the
initial checkpoint, using the same fixed closure as the corresponding PGD
run. It is a first-order bound, not a claim about the nonlinear objective.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "test071502"))
sys.path.insert(0, str(HERE.parent / "test071407"))
sys.path.insert(0, str(HERE.parent / "test071405"))

import run_bam_l3cr_three_solver_experiment as exp  # noqa: E402
from run_clean_bam_l3cr_pgd import make_clean_bam  # noqa: E402
from run_leakage_free_architecture_experiment import load_clean_pack  # noqa: E402


def flatten(grads, params):
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for g, p in zip(grads, params)])


def main() -> None:
    data_dir = HERE.parent / "data_28_06_2026"
    pack, _, _ = load_clean_pack(HERE / "data_clean/leakage_free_v1", data_dir)
    device = torch.device("cpu")
    rows = []
    settings = [
        (12, (11, 23, 37, 51, 73), "formal_five_seed"),
        (16, (11, 23, 37, 51, 73), "formal_five_seed"),
        (32, (11, 23, 37, 51, 73), "formal_five_seed"),
        (64, (11, 23, 37, 51, 73), "formal_five_seed"),
        (128, (11,), "smoke_seed11_one_outer_step"),
    ]
    for active_dimension, seeds, result_status in settings:
        for seed in seeds:
            started = time.perf_counter()
            exp.set_seed(seed)
            cfg = exp.Config(seeds=seeds, active_dimension=active_dimension, closure_samples=8, max_step_norm=2.0e-3, device="cpu")
            train_indices, _ = exp.split_train_validation(len(pack.train_x))
            closure_indices = exp.fixed_closure_indices(train_indices, pack, cfg.closure_samples, seed)
            x, y, masks = exp.material_batch(pack, closure_indices, device)
            model = make_clean_bam().to(device).eval()
            checkpoint = HERE / "formal_leakage_free" / f"model_bam_kan_seed{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            loss = exp.component_mse(model(x), y, masks)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            full = flatten(grads, params).detach()
            dimension = min(active_dimension, full.numel())
            active = torch.topk(full.abs(), k=dimension, largest=True, sorted=True).indices
            full_norm = float(torch.linalg.vector_norm(full))
            active_norm = float(torch.linalg.vector_norm(full[active]))
            delta = cfg.max_step_norm
            rows.append({
                "active_dimension": active_dimension, "seed": seed, "data_protocol": "leakage_free_v1",
                "result_status": result_status, "closure_samples": cfg.closure_samples,
                "closure_objective": float(loss.detach()), "full_gradient_l2": full_norm,
                "active_gradient_l2": active_norm, "active_gradient_mass": active_norm / max(full_norm, 1.0e-30),
                "full_first_order_upper_bound": delta * full_norm,
                "active_first_order_upper_bound": delta * active_norm,
                "max_step_norm": delta, "gradient_evaluations": 1,
                "closure_indices": ",".join(str(int(i)) for i in closure_indices),
                "seconds": time.perf_counter() - started,
            })
    # The legacy k=32/k=64 runs retained their original closure indices and
    # baseline checkpoints, so the same first-order diagnostic can be computed
    # without rerunning the refinement itself.
    legacy_data = exp.load_data(exp.DATA_DIR)
    for active_dimension in (32, 64):
        metadata_path = HERE.parent / "test071601" / f"formal_k{active_dimension}/run_metadata.json"
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        for seed in (11, 23, 37, 51, 73):
            started = time.perf_counter()
            exp.set_seed(seed)
            closure_indices = np.asarray(metadata_dict[f"BAM-AdamW-L3CR-PGD-k{active_dimension}:seed{seed}"]["closure_indices"], dtype=int)
            x, y, masks = exp.material_batch(legacy_data, closure_indices, device)
            model = exp.make_bam().to(device).eval()
            checkpoint = HERE.parent / "test071407/formal" / f"model_bampikan_seed{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            loss = exp.component_mse(model(x), y, masks)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            full = flatten(grads, params).detach()
            active = torch.topk(full.abs(), k=min(active_dimension, full.numel()), largest=True, sorted=True).indices
            full_norm = float(torch.linalg.vector_norm(full))
            active_norm = float(torch.linalg.vector_norm(full[active]))
            rows.append({
                "active_dimension": active_dimension, "seed": seed, "data_protocol": "legacy_test071601",
                "result_status": "legacy_formal_five_seed", "closure_samples": len(closure_indices),
                "closure_objective": float(loss.detach()), "full_gradient_l2": full_norm,
                "active_gradient_l2": active_norm, "active_gradient_mass": active_norm / max(full_norm, 1.0e-30),
                "full_first_order_upper_bound": 2.0e-3 * full_norm,
                "active_first_order_upper_bound": 2.0e-3 * active_norm,
                "max_step_norm": 2.0e-3, "gradient_evaluations": 1,
                "closure_indices": ",".join(str(int(i)) for i in closure_indices),
                "seconds": time.perf_counter() - started,
            })
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "12_full_gradient_upper_bound.csv", index=False)
    print(result[["active_dimension", "seed", "result_status", "full_gradient_l2", "active_gradient_l2", "active_gradient_mass", "full_first_order_upper_bound", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
