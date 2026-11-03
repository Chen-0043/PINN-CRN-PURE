#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
case "${1:-复现}" in
 检查|verify) bash run_independent.sh verify ;;
 评价|evaluate) python3 evaluate_results.py; python3 verify_perturbation_conclusions.py ;;
 复现|run)
  python3 prepare_fresh_outputs.py endpoints uncertainty neighbors
  bash run_independent.sh endpoints
  bash run_independent.sh uncertainty
  bash run_independent.sh local-optima
  python3 recalculate_topology_from_network.py
  python3 evaluate_results.py
  python3 verify_perturbation_conclusions.py
  ;;
 端点|endpoints) python3 prepare_fresh_outputs.py endpoints; bash run_independent.sh endpoints; python3 evaluate_results.py ;;
 不确定性|uncertainty) python3 prepare_fresh_outputs.py uncertainty neighbors; bash run_independent.sh uncertainty ;;
 局部最优|local-optima) python3 prepare_fresh_outputs.py neighbors; bash run_independent.sh local-optima ;;
 拓扑|topology) python3 recalculate_topology_from_network.py ;;
 绘图|figures) bash run_independent.sh figures ;;
 *) echo '用法：bash 运行.sh [检查|复现|评价|端点|不确定性|局部最优|拓扑|绘图]' >&2; exit 2 ;;
esac
