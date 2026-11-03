#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Activate the documented environment before calling this script.
export PYTHONDONTWRITEBYTECODE=1
case "${1:-复现}" in
  检查|verify) bash run_independent.sh verify ;;
  评价|evaluate) python3 evaluate_results.py ;;
  复现|run)
    bash run_independent.sh replay
    python3 evaluate_results.py --prediction "reproduced_results/replayed_prior/prior_prediction_6h.npz"
    ;;
  训练|train) bash run_independent.sh from-initialization ;;
  绘图|figures) python3 绘图/update_figures.py ;;
  *) echo '用法：bash 运行.sh [检查|复现|评价|绘图|训练]' >&2; exit 2 ;;
esac
