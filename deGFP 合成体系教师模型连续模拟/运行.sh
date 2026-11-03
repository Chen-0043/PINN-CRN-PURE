#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Activate the documented environment before calling this script.
export PYTHONDONTWRITEBYTECODE=1
case "${1:-复现}" in
  检查|verify) bash run_independent.sh verify ;;
  监督数据|supervision) python3 export_all_species_reference.py --mode continuous --gamma-h 0.5 --t-end-h 6 --time-points-h 0,0.25,0.5,1,2,3,4,5,6 --tag continuous_existing_teacher_train0to5_validate6h ;;
  评价|evaluate) python3 evaluate_results.py ;;
  复现|run)
    bash run_independent.sh run
    python3 export_all_species_reference.py --mode continuous --gamma-h 0.5 --t-end-h 6 --time-points-h 0,0.25,0.5,1,2,3,4,5,6 --tag continuous_existing_teacher_train0to5_validate6h
    python3 evaluate_results.py
    ;;
  *) echo '用法：bash 运行.sh [检查|复现|评价]' >&2; exit 2 ;;
esac
