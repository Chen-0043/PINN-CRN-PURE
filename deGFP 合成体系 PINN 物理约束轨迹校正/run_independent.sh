#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-verify}" in
  verify) python3 run_3_4_pipeline.py verify-assets ;;
  verify-deps) python3 run_3_4_pipeline.py verify-deps ;;
  smoke) python3 run_3_4_pipeline.py correct-reference --steps 1 --output-prefix reproduced_smoke_correction ;;
  run) python3 run_correction_from_weights.py ;;
  replay-saved-nodes) python3 run_3_4_pipeline.py correct-reference ;;
  from-prior) python3 run_from_prior.py ;;
  figures) python3 run_3_4_pipeline.py plot ;;
  performance) bash 性能对比/run_performance.sh "${2:-verify}" "${3:-}" ;;
  *) echo "usage: $0 {verify|verify-deps|smoke|run|from-prior|figures|performance [verify|subnetwork|full-bdf|full-radau]}" >&2; exit 2 ;;
esac

