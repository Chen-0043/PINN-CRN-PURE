#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
FULL="$PWD/../model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn"
case "${1:-verify}" in
  verify)
    python3 - <<'PY'
import json
from pathlib import Path
same=json.loads(Path('reviewer_samecondition_full_ode_test_20261026.json').read_text())
online=json.loads(Path('reviewer_runtime_benchmark_20261025_final.json').read_text())
assert same['locked_conditions']['species_count']==5772
assert same['locked_conditions']['reaction_count']==7061
assert {r['method'] for r in same['runs']}=={'BDF','Radau'}
assert abs(online['pinn_inference']['median_s']-0.0008729789997232729)<1e-12
fresh=[]
for method in ('bdf','radau'):
    path=Path('reproduced_results')/f'samecondition_{method}_run.json'
    if path.exists():
        row=json.loads(path.read_text())
        assert row['status']=='timeout_without_complete_trajectory', row
        assert row['trajectory_available'] is False, row
        fresh.append(method)
print('OK: historical ODE/PINN performance records are present; file presence is not an operator-identity certificate.')
if fresh:
    print('OK: fresh 700 s timeout runs without complete trajectories:', ', '.join(fresh))
audit=Path('同算子核查_20261102/operator_identity_actual.json')
if audit.exists() and not json.loads(audit.read_text())['old_declared_operator_identity_is_reproduced']:
    print('WARNING: the legacy uM-rate ODE entry changes 1807 constants relative to the frozen PINN. The old same-condition declaration is not reproduced. See 同算子核查_20261102.')
PY
    ;;
  subnetwork)
    mkdir -p reproduced_results
    python3 reviewer_runtime_benchmark_20261025.py \
      --assets "$FULL" \
      --source "$FULL/reviewer_benchmark_finetune_200_prediction.npz" \
      --checkpoint "$FULL/reviewer_benchmark_finetune_200_checkpoint.pt" \
      --output reproduced_results/subnetwork_and_pinn_benchmark.json \
      --rtol 1e-8 --atol-M 1e-18 --preequilibrate-s 600 --time-points 25 --repeats 500
    ;;
  full-bdf)
    echo 'LEGACY settings: rate-unit=uM changes k relative to the frozen PINN; use same-operator-bdf for the audited comparison.' >&2
    python3 run_full_ode_with_timeout.py --method BDF --timeout-s "${2:-600}" ;;
  full-radau)
    echo 'LEGACY settings: rate-unit=uM changes k relative to the frozen PINN; use same-operator-radau for the audited comparison.' >&2
    python3 run_full_ode_with_timeout.py --method Radau --timeout-s "${2:-600}" ;;
  verify-operator) python3 同算子核查_20261102/prove_operator_identity.py ;;
  same-operator-bdf) python3 同算子核查_20261102/run_sameoperator_probe.py BDF --timeout "${2:-180}" ;;
  same-operator-radau) python3 同算子核查_20261102/run_sameoperator_probe.py Radau --timeout "${2:-180}" ;;
  *) echo "usage: $0 {verify|verify-operator|subnetwork|full-bdf [legacy timeout_s]|full-radau [legacy timeout_s]|same-operator-bdf [timeout_s]|same-operator-radau [timeout_s]}" >&2; exit 2 ;;
esac
