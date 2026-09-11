#!/bin/sh
set -eu

export MPLCONFIGDIR=/private/tmp/mpl-test082301
PY=/opt/anaconda3/bin/python
SCRIPT="$(dirname "$0")/run_high_accuracy_refinement.py"

"$PY" "$SCRIPT" audit
"$PY" "$SCRIPT" warmup --warmup-device cpu
"$PY" "$SCRIPT" refine
"$PY" "$SCRIPT" analyze
