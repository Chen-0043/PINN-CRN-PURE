#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-verify}" in
  verify) python3 run_3_4_pipeline.py verify-assets ;;
  verify-deps) python3 run_3_4_pipeline.py verify-deps ;;
  smoke) python3 run_3_4_pipeline.py retrain-prior --steps 1 --output-prefix reproduced_smoke_prior ;;
  run) python3 run_3_4_pipeline.py retrain-prior ;;
  from-initialization) python3 train_prior_from_initialization.py degfp ;;
  *) echo "usage: $0 {verify|verify-deps|smoke|run|from-initialization}" >&2; exit 2 ;;
esac

