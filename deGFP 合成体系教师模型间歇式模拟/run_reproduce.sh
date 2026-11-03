#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

python3 check_environment.py
python3 run_batch_1nt1aa.py --hours 8 --points 600 --outdir results/batch
python3 run_tx1nt_tlg1_micro3_longrun.py --gammas 0.5 --t-end-h 72 --n-points 2400 --outdir results/continuous
python3 plot_results.py
python3 verify_results.py

echo "deGFP 单蛋白间歇式/连续式模拟复现完成。"
