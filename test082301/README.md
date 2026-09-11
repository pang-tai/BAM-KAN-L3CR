# BAM-KAN high-accuracy refinement experiment

This directory contains the strict same-checkpoint comparison of resumed AdamW,
strong-Wolfe L-BFGS, and HVP L3CR-PGD. The frozen training/validation/test split
contains 137/30/19 cases. All refinement routes use closure indices
`[7, 21, 29, 63, 84, 89, 95, 158]` and the same standardized material-voxel MSE.

## Commands

```bash
export MPLCONFIGDIR=/private/tmp/mpl-test082301
PY=/opt/anaconda3/bin/python

$PY run_high_accuracy_refinement.py audit
$PY run_high_accuracy_refinement.py warmup --warmup-device cpu
$PY run_high_accuracy_refinement.py refine
$PY run_high_accuracy_refinement.py analyze
```

The installed PyTorch 2.5.1 runtime is not linked with MPS support. The formal
run therefore records CPU warm-up as a protocol deviation. Refinement is CPU,
single-threaded, and float64 by design.

Test metrics are produced only by `analyze`, after every configuration and
budget checkpoint has been frozen. Existing model-only checkpoints are not used
because they do not contain the AdamW momentum and scheduler states required by
the strict continuation comparison.
