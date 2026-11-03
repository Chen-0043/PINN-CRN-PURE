#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
case "${1:-verify}" in
  verify)
    python3 - <<'PY'
from pathlib import Path
genes=['Pgi','PfkA','TpiA','GAPDH','Pgk','Eno']
required=[Path(f'inputs/original_18p88_unified_shared_scan/tx3000_e0.24/single_{g}/panel.json') for g in genes]
required += [Path('inputs/original_18p88_unified_shared_scan/tx3000_e0.24/mux/panel.json'), Path('reference_results/coexpression_72h/sixprotein_continuous_timeseries.csv')]
missing=[str(p) for p in required if not p.exists()]
assert not missing, missing
print('OK: six single-expression and one coexpression continuous inputs are present.')
PY
    ;;
  run) python3 run_all_7_continuous.py ;;
  compare) python3 verify_reproduced_results.py ;;
  *) echo "usage: $0 {verify|run|compare}" >&2; exit 2 ;;
esac

