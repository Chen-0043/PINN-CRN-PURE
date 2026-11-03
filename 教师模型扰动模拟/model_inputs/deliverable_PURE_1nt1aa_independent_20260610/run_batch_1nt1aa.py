#!/usr/bin/env python3
"""Run the validated batch PURE model with the packaged inputs only."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import tellurium as te

import fit_batch_then_continuous_full_crn_background as core
from run_tx1nt_tlg1_micro3_longrun import apply_params, load_sources, set_fine_node_env


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAM_SUMMARY = PACKAGE_DIR / "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"
DEFAULT_HF_SUMMARY = PACKAGE_DIR / "output_full_crn_background/batch_then_continuous_summary.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the packaged 1nt/1aa batch PURE model.")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=600000)
    ap.add_argument("--outdir", type=Path, default=PACKAGE_DIR / "results/batch")
    ap.add_argument("--param-summary", type=Path, default=DEFAULT_PARAM_SUMMARY)
    ap.add_argument("--hf-summary", type=Path, default=DEFAULT_HF_SUMMARY)
    args = ap.parse_args()

    os.environ["PURE_MAX_STEPS"] = str(int(args.max_steps))
    set_fine_node_env()
    args.outdir.mkdir(parents=True, exist_ok=True)

    params, hf_fit, keff_map = load_sources(args.param_summary, args.hf_summary)
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    sbml, counts, _ = core.build_background_sbml(
        mode="batch",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)

    t0 = time.time()
    df = core.simulate_degfp(
        rr,
        t_end_s=float(args.hours) * 3600.0,
        n_points=int(args.points),
        reset_to_origin=False,
    )
    runtime_s = float(time.time() - t0)
    metrics = core.batch_metrics(df)

    timeseries = args.outdir / "batch_timeseries.csv"
    summary_path = args.outdir / "batch_summary.json"
    df.to_csv(timeseries, index=False)
    summary = {
        "model": "PURE deGFP batch, 1nt TX + 1aa TL",
        "node": {
            "tx_pathway_mode": "1nt",
            "tx_step_mode": "micro3",
            "tl_pathway_mode": "step_chain",
            "tl_qssa_mode": "factorized",
            "tl_chain_group_aa": 1.0,
            "energy_regen_mode": "background",
            "legacy_species_pruned": True,
        },
        "hours": float(args.hours),
        "counts": counts,
        "param_apply": param_apply,
        "runtime_s": runtime_s,
        "metrics": metrics,
        "files": {"timeseries_csv": str(timeseries)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
