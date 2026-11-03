#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Activate the documented environment before calling this script.
export PYTHONDONTWRITEBYTECODE=1
case "${1:-复现}" in
  检查|verify) bash run_independent.sh verify ;;
  评价|evaluate) python3 evaluate_results.py ;;
  复现|run)
    python3 run_correction_from_weights.py
    python3 evaluate_results.py --prediction "model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn/from_prior_stage_3_shooting_nodes.npz"
    ;;
  绘图|figures) python3 绘图/update_figures.py ;;
  *) echo '用法：bash 运行.sh [检查|复现|评价|绘图]' >&2; exit 2 ;;
esac
