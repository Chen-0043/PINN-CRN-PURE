#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
case "${1:-verify}" in
  verify) python3 verify_formal_results.py ;;
  run|from-weights) python3 run_full_history_from_weights.py ;;
  evaluate) python3 evaluate_results.py "${@:2}" ;;
  figures) python3 绘图/update_figures.py ;;
  *) echo 'usage: run_independent.sh {verify|run|evaluate|figures}' >&2; exit 2 ;;
esac
