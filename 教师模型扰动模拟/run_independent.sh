#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-verify}" in
  verify)
    python3 verify_perturbation_conclusions.py
    ;;
  run) python3 run_latest_perturbation_simulations.py ;;
  endpoints) python3 run_missing_article_scale_degfp_endpoints.py --force ;;
  uncertainty) python3 analyze_5axis_uncertainty_20261027.py --mode all --samples 8 ;;
  local-optima)
    python3 scan_adjacent_local_optima_20261027.py
    python3 retry_adjacent_failures_20261027.py
    python3 -c 'import pandas as pd; import scan_adjacent_local_optima_20261027 as s; s.summarize(pd.read_csv(s.OUTDIR/"adjacent_local_optimum_new_simulations.csv"))'
    ;;
  recalibrate)
    python3 statistical_recalibration_20261027.py --mode all
    python3 finalize_parameter_uncertainty_audit_20261027.py
    ;;
  topology) python3 verify_perturbation_conclusions.py ;;
  figures) python3 run_article_3_3_figures.py ;;
  *) echo "usage: $0 {verify|run|endpoints|uncertainty|local-optima|recalibrate|topology|figures}" >&2; exit 2 ;;
esac

