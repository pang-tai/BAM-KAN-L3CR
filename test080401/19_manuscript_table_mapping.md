# Manuscript table and figure mapping

| Manuscript item | File | Interpretation |
|---|---|---|
| Data provenance | `01_channel_provenance.csv`, `02_file_inventory.csv`, `03_duplicate_groups.csv`, `04_case_fingerprints.csv` | Audit evidence; not a performance claim. |
| Leakage diagnostic | `legacy_vs_leakage_free_bam.csv`, `legacy_vs_leakage_free_bam_summary.csv` | Internal diagnostic only; legacy and clean checkpoints are not a perfectly matched causal pair. |
| Split and normalization | `05_split_manifest.csv`, `06_normalization.json` | Defines leakage-free v1 protocol. |
| Main architecture table | `07_main_architecture_results.csv`, `tables/07_main_architecture_summary.csv` | Five-seed retraining; lower scores are better. |
| Main architecture figure | `17_architecture_comparison.pdf` | Global/Hole/Balanced with seed standard deviations. |
| Checkpoint consistency | `09_checkpoint_consistency_audit.csv` | Verifies repeated full-BAM rows and the 15 main checkpoints. |
| Paired inference | `15_paired_statistics.csv` | Bootstrap CI and Wilcoxon; do not report as universal superiority when CI crosses zero. |
| L3CR certificate | `11_l3cr_finite_certificate.csv`, `formal_clean_l3cr_pgd/l3cr_finite_certificate.csv` | Contains legacy certificates and the new-clean-checkpoint PGD certificate; distinguish the source labels. |
| KAN isolation | `08_bam_kan_isolation_results.csv`, `figures/kan_isolation_comparison.pdf` | BAM-KAN versus parameter-matched BAM-MLP/BAM-Conv. |
| BAM path ablation | `09_pathway_ablation_results.csv`, `figures/bam_path_ablation_comparison.pdf` | No-local, no-fine and no-coarse variants. |
| L3CR continuation | `10_refinement_results.csv`, `10_refinement_paired_statistics.csv`, `15_paired_statistics_holm.csv` | Direct AdamW-to-PGD continuation; report the small, non-significant mean improvement accurately with corrected paired inference. |
| Active-dimension sensitivity | `12_active_dimension_sensitivity.csv`, `12_active_dimension_all_summary.csv`, `12_active_dimension_all_per_seed.csv`, `12_active_dimension_diagnostics.csv`, `12_full_gradient_upper_bound.csv`, `12_active_dimension_accuracy_cost.pdf` | Primary result: clean `r=12/16/32/64` formal five-seed runs; `r=128` is a clean seed-11 smoke point for the appendix. Legacy `r=32/64` remains separately labeled for protocol context. The full-gradient field is a declared first-order bound, not a nonlinear descent guarantee. |
| Subgroup audit | `12_subgroup_results_clean.csv`, `12_subgroup_results.csv` | New leakage-free diagnostic subgroup results plus legacy audit; groups with fewer than five test cases are diagnostic only. |
| Hole-radius sensitivity | `13_hole_radius_results.csv`, `13_hole_radius_per_seed.csv` | Re-evaluation of all new leakage-free checkpoints at radii 1-4; no retraining. |
| Projection audit | `14_projection_audit.csv`, `figures/projection_audit_summary.pdf` | 19/19 raw Excel nodal S1/U fields; report as discretization/projection error, not model error or mesh convergence. |
| Physical-unit Figure 8 | `17_figure8_physical_units.pdf`, `figures/figure8_physical_units.png` | Saved-checkpoint comparison in MPa/mm with geometry-derived hole mask and predeclared median-hole-case selection. |
| Inference cost | `17_inference_cost.csv`, `17_inference_cost_metadata.json` | Ten timed repetitions after three warmups on MPS; report hardware and memory-field limitations. |
| L3CR cost | `17_l3cr_cost_diagnostics.csv` | HVP count, solver iterations, validation evaluations, wall time and outer backtrack trials for the new formal PGD run. |
| Remaining unavailable item | Strong-form physics residuals | Requires full displacement/stress/BC/contact fields; do not infer them from S1/U. |
