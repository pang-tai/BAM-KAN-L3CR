# BAM-KAN ver7.3 reproducibility package

This folder consolidates the experiments and data used by `ver7.3-Tai_2nd_paper.docx`. It brings together the reported results, saved checkpoints, processed tensors, finite-element source materials, and the scripts used in the numerical studies.

## Reproducibility overview

The machine-learning and optimization results reported in the manuscript can be reproduced from the processed tensor archive and saved checkpoints included in this package. The package contains the data products, experiment configurations, code, checkpoints, evaluation outputs, and verification utilities needed for the reported numerical studies.

The finite-element source materials include ten `.cae` models and fourteen raw Excel nodal exports. These files provide additional provenance for the processed dataset and document representative stages of the finite-element workflow used in the study.

## Package map

| Location | Purpose |
|---|---|
| `manuscript/` | Ver7.3 Word manuscript and its rendered PDF |
| `data_28_06_2026/` | Processed tensor archive used by the numerical work |
| `fem_source/` | Abaqus `.cae` models and raw Excel nodal exports |
| `test071405/`, `test071407/` | Shared model and training modules used by later experiments |
| `test080401/` | Five-seed CNN, U-Net and BAM-KAN comparison; cleaned data product; radius sensitivity; active-subspace refinement |
| `test082301/`, `test082401/` | Checkpoint preparation and shared optimization code |
| `test082402/` | Deterministic 20-parameter stationarity audit and source results |
| `test082501_l3cr_20_parameter/` | Curated code/data/report snapshot for the 20-parameter study |
| `test090401/` | P5/P6 full-data HVP audit, fixed-three-step MF-L3CR runs and matched-budget baselines |
| `manuscript_support/` | Manuscript update source and prior definition audit |
| `EXPERIMENT_INDEX.csv` | Experiment-to-code/data/output map |
| `CLAIM_EVIDENCE_LEDGER.md` | Paper claim-to-source traceability ledger |
| `verify_package.py` | Integrity, syntax and numerical consistency checks |
| `MANIFEST_SHA256.txt` | SHA-256 manifest for all packaged files except the manifest itself |

## Environment

The timing results in the paper were obtained on an Apple M4 Mac mini with a 10-core CPU and 32 GB RAM, using Python 3.12.2, PyTorch 2.5.1, CPU float64, one intra-op thread, one inter-op thread and micro-batch size 4. Training runs may also use Apple MPS.

Create a Python 3.12 environment and install `requirements.txt`. Abaqus is the proprietary software used for the packaged `.cae` models and can be used to open or rerun those models.

## First check

Run from this folder:

```bash
python verify_package.py
```

This command performs package-level verification by checking required files, parsing the packaged Python source files, checking tensor shapes, and comparing the principal CSV results with Tables 3-10 of the manuscript.

For a full byte-level integrity check, use:

```bash
python verify_package.py --hashes
```

## Main rerun commands

The commands below write reproduced results to new `reproduced_*` directories while preserving the archived reference outputs.

### Five-seed architecture comparison

```bash
python test080401/run_leakage_free_architecture_experiment.py \
  --epochs 48 --seeds 11 23 37 51 73 --device mps \
  --out-dir reproduced_architecture
```

For CPU execution, use `--device cpu`.

### Active-subspace L3CR refinement at r = 64

```bash
python test080401/run_clean_bam_l3cr_pgd.py \
  --seeds 11 23 37 51 73 --active-dimension 64 --device cpu \
  --checkpoint-dir test080401/formal_leakage_free \
  --out-dir reproduced_as_l3cr_r64
```

The paper's Table 8 values map to `test080401/formal_clean_l3cr_k64/` and `test080401/12_active_dimension_sensitivity.csv`.

### Deterministic 20-parameter stationarity audit

```bash
BAMKAN_20P_OUT_DIR="$PWD/reproduced_20_parameter" \
  python test082402/run_two_problem_precision_experiment.py
BAMKAN_20P_OUT_DIR="$PWD/reproduced_20_parameter" \
  python test082402/analyze_two_problem_precision.py
```

### P5 and P6 full-parameter experiment

The archived driver is `test090401/run_fixed_three_step_experiment.py`. It supports stage-specific audit and continuation modes. The run configuration is recorded in `test090401/experiment_config.json`, and the corresponding experiment report is `test090401/experiment_report_zh.md`. On the reported CPU setup, each P5/P6 seed takes approximately 25-30 minutes. The archived result tables and operation logs are included and are checked by `verify_package.py`.

## Archived outputs and reruns

Files under the original `test*` directories provide the archived reference outputs. Rerun results can be written to the `reproduced_*` directories shown above, which keeps the archived outputs and `MANIFEST_SHA256.txt` unchanged.
