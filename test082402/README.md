# L3CR optimized-code audit and two-problem precision experiment

This directory is independent of `test082401`. It preserves exact snapshots of
the original core and the user-provided optimized core, and adds a certified
adaptive implementation.

## Reproduce

```bash
cd "/Users/pangtai/Desktop/Experiment/plates with holes"
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl-test082402 \
  /opt/anaconda3/bin/python3 test082402/run_two_problem_precision_experiment.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl-test082402 \
  /opt/anaconda3/bin/python3 test082402/analyze_two_problem_precision.py
```

The formal run uses CPU, one thread, float64, ten synthetic seeds and five
frozen BAM-KAN checkpoints. No BAM backbone is retrained. Work-budget results
at 500 reverse-mode equivalents are primary; wall-clock snapshots are
supplementary.

See `experiment_report_zh.md`, `algorithm_audit.csv`,
`threshold_reach_summary.csv`, and `claim_evidence_ledger.csv` before using any
result in the paper.
