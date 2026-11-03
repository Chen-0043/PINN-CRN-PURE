#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
ASSETS="$PWD/model_assets/original_18p88_full_elementary"
POOL="$PWD/inputs/input_pool_override_sg_pure2_paper_mapped_20260424.json"
SUP="$ASSETS/sixprotein_teacher_supervision_all_metabolites_dt0p125.npz"
SCALE="$ASSETS/state_scale_from_degfp_probe_M.npy"
STAGE3="$PWD/inputs/stage3_initial/prior_pinn_checkpoint.pt"
case "${1:-verify}" in
  verify)
    python3 - <<'PY'
import json
from pathlib import Path
p=Path('reference_results/final_prior/prior_training_report.json')
d=json.loads(p.read_text())
assert d['full_states']==49149
assert d['elementary_reactions']==57368
assert d['direct_teacher_constraints']+d['aggregate_teacher_constraints']==9198
assert abs(d['physical_integral_metrics']['rhs_integral_gt_1p0_uM_mean_NRMSR_percent']-103.3920336381)<1e-6
assert Path('inputs/stage3_initial/prior_pinn_checkpoint.pt').exists()
print('OK: archived prior assets and historical 9,198-trajectory report are consistent; current evaluation uses energy_scale=0.24.')
PY
    ;;
  replay)
    mkdir -p reproduced_results/replayed_prior
    python3 src/train_teacher_supervised_sixprotein_pinn.py \
      --assets "$ASSETS" --pool-json "$POOL" --teacher-supervision "$SUP" --state-scale "$SCALE" \
      --output-dir reproduced_results/replayed_prior --include-all-teacher-metabolites \
      --resume-checkpoint reference_results/final_prior/prior_pinn_checkpoint.pt \
      --steps 0 --energy-scale 0.24 --teacher-train-max-s 19800 --teacher-derivative-weight 0.1 \
      --terminal-teacher-weight 100 --terminal-derivative-weight 20 --moiety-weight 1 --physics-weight 0
    ;;
  run|train)
    mkdir -p reproduced_results/retrained_prior
    python3 src/train_teacher_supervised_sixprotein_pinn.py \
      --assets "$ASSETS" --pool-json "$POOL" --teacher-supervision "$SUP" --state-scale "$SCALE" \
      --output-dir reproduced_results/retrained_prior --include-all-teacher-metabolites \
      --resume-checkpoint "$STAGE3" \
      --steps 1000 --lr 1e-4 --teacher-train-max-s 19800 --teacher-derivative-weight 0.1 \
      --terminal-teacher-weight 100 --terminal-derivative-weight 20 --moiety-weight 1 --physics-weight 0
    ;;
  from-initialization) python3 train_prior_from_initialization.py sixprotein ;;
  *) echo "usage: $0 {verify|replay|run|train|from-initialization}" >&2; exit 2 ;;
esac

