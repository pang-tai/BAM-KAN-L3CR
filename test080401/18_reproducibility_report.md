# Reproducibility report

## Completed in this package

- Raw file inventory, SHA-256 hashes, duplicate-file groups, case metadata and tensor fingerprints.
- Conservative leakage-free v1 product excluding c04/c10/c12, with training-only normalization.
- Deterministic metric self-tests and a five-seed CNN/U-Net/BAM-KAN retraining on the cleaned input.
- New checkpoints, per-seed metrics, per-sample metrics, history and paired statistics.
- Formal five-seed BAM-MLP/BAM-Conv KAN isolation and three BAM path ablations.
- Direct five-seed AdamW-to-L3CR-PGD continuation from the new leakage-free BAM-KAN checkpoints, including a finite-descent certificate.
- Static BAM thickness-direction audit and finite-descent certificates from reusable legacy L3CR logs.
- Public-grid projection audit for all 19 selected test Excel cases with raw `X,Y,Z,S1,U` nodal fields; the audit projects to `64x64x4` and interpolates back to the original nodes.
- Saved-checkpoint hole-radius sensitivity for radii 1-4, physical-unit Figure 8, inference timing, L3CR HVP cost, clean subgroup diagnostics and Holm-adjusted paired statistics.

## Not silently claimed

- Existing legacy L3CR tables are not presented as continuation from the new leakage-free BAM-KAN checkpoints.
- Missing Abaqus tensor fields are not reconstructed from S1/U; strong-form PIKAN residuals remain unavailable. The public-grid projection audit is available, but it is a discretization audit rather than model error or mesh-convergence evidence. Only 2/19 cases have a topology-defined hole band, so hole-band projection metrics are reported only for those cases.
- The current tensor split has geometry-group overlap indicators; strict group-independent generalization requires the original case genealogy.
- L3CR-PGD improved the mean score only slightly and its paired intervals cross zero; the certificate proves accepted descent, not a significant test-set advantage.
- The active-dimension sensitivity table contains clean five-seed results for `r=12/16/32/64` and a clean seed-11 smoke result for `r=128`. The `r=128` row is not treated as a formal five-seed result. Reusable legacy `r=32/64` refinements remain separately labeled for protocol context. The full-gradient first-order bound is recomputed for all clean and legacy rows without imputing refinement metrics.

## Re-run commands

```bash
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_non_abaqus_plan_audit.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_leakage_free_architecture_experiment.py --epochs 48 --seeds 11 23 37 51 73 --out-dir test080401/formal_leakage_free --device mps
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/summarize_non_abaqus_results.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_bam_kan_isolation_experiment.py --epochs 48 --seeds 11 23 37 51 73 --out-dir test080401/formal_kan_isolation --device mps
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_bam_path_ablation_experiment.py --epochs 48 --seeds 11 23 37 51 73 --out-dir test080401/formal_path_ablation --device mps
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_clean_bam_l3cr_pgd.py --seeds 11 23 37 51 73 --out-dir test080401/formal_clean_l3cr_pgd --device cpu
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_clean_bam_l3cr_pgd.py --seeds 11 23 37 51 73 --active-dimension 16 --out-dir test080401/formal_clean_l3cr_k16 --device cpu
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/run_clean_bam_l3cr_pgd.py --smoke --seeds 11 --active-dimension 128 --out-dir test080401/pilot_clean_l3cr_k128 --device cpu
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/compile_active_dimension_all.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/compile_active_dimension_diagnostics.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/compute_full_gradient_upper_bound.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/compute_hole_radius_sensitivity.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/generate_figure8_physical_units.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/benchmark_inference_cost.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/summarize_l3cr_cost.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/create_refinement_statistics.py
"/Users/pangtai/Desktop/Experiment/plates with holes/test071003/.venv/bin/python" test080401/apply_holm_correction.py
```
