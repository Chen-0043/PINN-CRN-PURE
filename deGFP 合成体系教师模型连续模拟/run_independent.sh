#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.mplconfig"
case "${1:-verify}" in
  verify) python3 check_environment.py ;;
  run)
    mkdir -p reproduced_results/continuous
    python3 run_tx1nt_tlg1_micro3_longrun.py --gammas 0.5 --t-end-h 72 --n-points 2400 --outdir reproduced_results/continuous
    ;;
  *) echo "usage: $0 {verify|run}" >&2; exit 2 ;;
esac

