#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.mplconfig"
case "${1:-verify}" in
  verify) python3 check_environment.py ;;
  run)
    mkdir -p reproduced_results/batch
    python3 run_batch_1nt1aa.py --hours 8 --points 600 --outdir reproduced_results/batch
    ;;
  *) echo "usage: $0 {verify|run}" >&2; exit 2 ;;
esac

