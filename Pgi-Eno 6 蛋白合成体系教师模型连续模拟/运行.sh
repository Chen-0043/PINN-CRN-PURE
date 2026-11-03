#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Activate the documented environment before calling this script.
export PYTHONDONTWRITEBYTECODE=1
case "${1:-复现}" in
  检查|verify) bash run_independent.sh verify ;;
  监督数据|supervision) python3 generate_pinn_supervision.py ;;
  评价|evaluate) python3 evaluate_results.py ;;
  复现|run)
    bash run_independent.sh run
    python3 generate_pinn_supervision.py
    python3 evaluate_results.py
    ;;
  *) echo '用法：bash 运行.sh [检查|复现|评价]' >&2; exit 2 ;;
esac
