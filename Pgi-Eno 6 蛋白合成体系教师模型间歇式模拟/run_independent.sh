#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
case "${1:-verify}" in
  verify)
    python3 - <<'PY'
from pathlib import Path
required = [
 Path('src/run_unified_shared_scan.py'),
 Path('src/run_legacy_unified_shared_scan.py'),
 Path('legacy_runtime/simulate_sixgene_pool_chain_v3.cpython-314.pyc'),
 Path('inputs/output_sixgene_single_gene_txfit_then_all_bgtx2p25_muxTotal2p8_20260425/sixgene_panel_single_gene_txfit_absolutechains.json'),
 Path('inputs/output_sixgene_single_gene_proxy_paperanchored_20260424/selected_sixgene_single_gene_vs_lit_paperanchored.csv'),
 Path('model_assets/output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json'),
 Path('reference_results/original_18p88_unified_shared_scan/tx3000_e0.24/mux/panel.json'),
]
missing=[str(p) for p in required if not p.exists()]
assert not missing, missing
print('OK: seven-system batch package is complete.')
PY
    ;;
  run)
    mkdir -p reproduced_results
    python3 src/run_legacy_unified_shared_scan.py \
      --outdir reproduced_results/seven_systems \
      --panel-json inputs/output_sixgene_single_gene_txfit_then_all_bgtx2p25_muxTotal2p8_20260425/sixgene_panel_single_gene_txfit_absolutechains.json \
      --single-proxy-csv inputs/output_sixgene_single_gene_proxy_paperanchored_20260424/selected_sixgene_single_gene_vs_lit_paperanchored.csv \
      --formal-summary-json model_assets/output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json \
      --pool-json input_pool_override_sg_pure2_paper_mapped_20260424.json \
      --tx-list 3000 --energy-list 0.24 --tl-uniform 2800 \
      --dna-total-M 2.8e-9 --t-end-h 6 --dt-s 2 --record-every-s 10800 \
      --bg-tx-coupling-power 2.25
    ;;
  *) echo "usage: $0 {verify|run}" >&2; exit 2 ;;
esac

