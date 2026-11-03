#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"
export MPLCONFIGDIR="$ROOT/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$ROOT/results/batch" "$ROOT/results/continuous"

python3 check_environment.py
zsh verify_package_integrity.sh
python3 audit_model_structure.py
python3 run_batch_1nt1aa.py --hours 8 --points 600 --outdir "$ROOT/results/batch"
python3 run_tx1nt_tlg1_micro3_longrun.py \
  --gammas 0.5 \
  --t-end-h 72 \
  --n-points 1200 \
  --stability-window-h 24 \
  --outdir "$ROOT/results/continuous"
python3 verify_results.py
python3 plot_results.py

echo "Independent PURE simulation and verification completed."
