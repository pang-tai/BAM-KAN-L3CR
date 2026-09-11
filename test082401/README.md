# Full-space L3CR short high-accuracy study

This directory contains the frozen code and results for the 17-parameter
nonlinear regression and 20-parameter BAM-KAN output-head experiments.

## Reproduce

```bash
cd "/Users/pangtai/Desktop/Experiment/plates with holes/test082401"
MPLCONFIGDIR=/private/tmp/mpl-test082401 \
  /opt/anaconda3/bin/python run_short_high_accuracy_experiment.py
MPLCONFIGDIR=/private/tmp/mpl-test082401 \
  /opt/anaconda3/bin/python analyze_short_high_accuracy.py
/opt/anaconda3/bin/python build_appendix.py
xelatex -interaction=nonstopmode -halt-on-error \
  full_space_l3cr_high_accuracy_appendix_zh.tex
```

The formal experiment used CPU, one thread, float64, ten synthetic seeds and
five frozen BAM-KAN checkpoints. Its measured runtime was 116.38 seconds.

## Evidence boundary

- On the full-rank 20-parameter BAM output-head problem, full-space L3CR reached
  raw gradient norm `1e-8` in 5/5 checkpoints. AdamW, AdamW-0 and L-BFGS reached
  it in 0/5 under the frozen budget.
- On the 17-parameter nonlinear problem, L3CR reached `1e-8` in only 1/10 near
  starts and 0/10 far starts. This does not support a general nonlinear neural
  optimization advantage.
- The experiment does not test full-space L3CR on all 86,653 BAM-KAN parameters.

See `experiment_report_zh.md`, `threshold_reach_summary.csv`, and
`full_space_l3cr_high_accuracy_appendix_zh.pdf` for the reportable results.
