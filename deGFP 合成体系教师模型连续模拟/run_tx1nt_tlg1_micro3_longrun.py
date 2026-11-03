#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tellurium as te

import fit_batch_then_continuous_full_crn_background as core


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAM_SUMMARY = PACKAGE_DIR / (
    "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"
)
DEFAULT_HF_SUMMARY = PACKAGE_DIR / "output_full_crn_background/batch_then_continuous_summary.json"
DEFAULT_OUTDIR = PACKAGE_DIR / "results/continuous"


def parse_gammas(text: str) -> List[float]:
    vals: List[float] = []
    for chunk in str(text).split(","):
        s = chunk.strip()
        if not s:
            continue
        v = float(s)
        if v < 0.0:
            raise ValueError(f"gamma must be >=0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("No gamma values parsed from --gammas")
    return vals


def gamma_tag(gamma_h: float) -> str:
    txt = str(float(gamma_h))
    if "." in txt:
        return txt.replace(".", "p")
    return f"{txt}p0"


def set_fine_node_env() -> None:
    # Finest node fixed by handover:
    # tx=1nt, tx-step=micro3, tl=step_chain(g=1), qssa=factorized.
    os.environ["PURE_TX_PATHWAY_MODE"] = "1nt"
    os.environ["PURE_TX_STEP_MODE"] = "micro3"
    os.environ["PURE_ENERGY_REGEN_MODE"] = "background"
    os.environ["PURE_TL_PATHWAY_MODE"] = "step_chain"
    os.environ["PURE_TL_QSSA_MODE"] = "factorized"
    os.environ["PURE_TL_CHAIN_GROUP_AA"] = "1.0"
    os.environ["PURE_TL_CHAIN_SCALE"] = "30.0"
    os.environ["PURE_TL_CHAIN_KINETICS"] = "effective"
    os.environ["PURE_TL_CHAIN_MA_K"] = ""
    os.environ["PURE_TL_CHAIN_STEPS"] = ""
    os.environ["PURE_RIBO_BIND_MODE"] = "explicit"
    os.environ["PURE_RIBO_BIND_SCALE"] = "1.0"
    os.environ["PURE_TX_NET_SCALE"] = "0.35"
    os.environ["PURE_FAST_SIDE_SCALE"] = "1.0"
    os.environ["PURE_FAST_SIDE_MIN_K"] = "10.0"
    os.environ["PURE_BG_FAMILY_EXCHANGE_BOOKKEEP"] = "1"
    os.environ["PURE_BG_WATER_INIT_M"] = "55.0"
    os.environ["PURE_PLATEAU_SLOPE_THR_UMPH"] = "5e-3"
    os.environ["PURE_PLATEAU_BAND_REL"] = "5e-3"
    os.environ["PURE_PLATEAU_BAND_ABS_UM"] = "0.0"
    os.environ["PURE_PLATEAU_MIN_START_H"] = "0.5"
    os.environ["PURE_REMOVE_LEGACY_MRNA_RS"] = "1"
    os.environ["PURE_REMOVE_LEGACY_RS_INIT"] = "1"
    os.environ["PURE_REMOVE_LEGACY_RS_ELONG_TERM"] = "1"
    os.environ["PURE_REMOVE_LEGACY_RS_ENERGY_SIDE"] = "1"
    os.environ["PURE_REMOVE_LEGACY_RS_ASSEMBLY"] = "1"
    os.environ["PURE_PRUNE_LEGACY_SPECIES"] = "1"


def load_sources(
    param_summary: Path,
    hf_summary: Path,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
    base = json.loads(param_summary.read_text(encoding="utf-8"))
    src = json.loads(hf_summary.read_text(encoding="utf-8"))
    params = dict(base.get("batch_best_params", {}))
    hf_fit = dict(src.get("hf_fit_params", {}))
    keff_map = dict(src.get("batch_fit", {}).get("keff_map", {}))
    if not params:
        raise RuntimeError(f"Missing batch_best_params in {param_summary}")
    if not hf_fit:
        raise RuntimeError(f"Missing hf_fit_params in {hf_summary}")
    if not keff_map:
        raise RuntimeError(f"Missing batch_fit.keff_map in {hf_summary}")
    return params, hf_fit, keff_map


def apply_params(rr, params: Dict[str, float]) -> Dict[str, int]:
    pids = set(rr.getGlobalParameterIds())
    applied = 0
    missing = 0
    for pid, val in params.items():
        if pid in pids:
            rr[pid] = float(val)
            applied += 1
        else:
            missing += 1
    return {"applied": int(applied), "missing": int(missing)}


def compute_metrics(
    df: pd.DataFrame,
    t_end_h: float,
    stability_window_h: float,
) -> tuple[Dict[str, Optional[float]], Dict[str, float]]:
    t_h = df["time"].to_numpy(dtype=float) / 3600.0
    p_uM = df["[deGFP_m]"].to_numpy(dtype=float) * 1e6

    sample_hours = [2.0, 4.0, 8.0, 24.0, 48.0, 72.0]
    metrics: Dict[str, Optional[float]] = {}
    for h in sample_hours:
        key = f"protein_{int(h)}h_uM"
        if h <= float(t_end_h) + 1e-12:
            metrics[key] = float(np.interp(h, t_h, p_uM))
        else:
            metrics[key] = None

    win = max(1e-9, float(stability_window_h))
    window_end = float(t_end_h)
    window_start = max(0.0, window_end - win)
    mask = t_h >= window_start
    y = p_uM[mask]
    x = t_h[mask]
    mean_uM = float(np.mean(y))
    max_abs_drift_uM = float(np.max(np.abs(y - mean_uM)))
    abs_slope_uM_per_h = float(abs(np.polyfit(x, y, 1)[0])) if y.size >= 5 else 0.0
    y_start = float(np.interp(window_start, t_h, p_uM))
    y_end = float(np.interp(window_end, t_h, p_uM))
    stability = {
        "window_h": f"{window_start:g}-{window_end:g}",
        "mean_uM": mean_uM,
        "max_abs_drift_uM": max_abs_drift_uM,
        "abs_slope_uM_per_h": abs_slope_uM_per_h,
        f"delta_{window_end:g}_minus_{window_start:g}_uM": float(y_end - y_start),
    }
    return metrics, stability


def run_one_gamma(
    *,
    gamma_h: float,
    t_end_h: float,
    n_points: int,
    feed_multiplier: float,
    product_gamma_h: Optional[float],
    params: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    ic_uM: Dict[str, float],
    stability_window_h: float,
    outdir: Path,
) -> Dict[str, object]:
    sbml_cont, counts_cont, _ = core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=float(gamma_h),
        feed_multiplier=float(feed_multiplier),
        product_outflow_h=(None if product_gamma_h is None else float(product_gamma_h)),
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml_cont)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)

    t0 = time.time()
    df = core.simulate_degfp(
        rr,
        t_end_s=float(t_end_h) * 3600.0,
        n_points=int(n_points),
        reset_to_origin=False,
    )
    runtime_s = float(time.time() - t0)
    metrics, stability = compute_metrics(
        df=df,
        t_end_h=float(t_end_h),
        stability_window_h=float(stability_window_h),
    )

    tag = gamma_tag(float(gamma_h))
    ts_csv = outdir / f"gamma_{tag}_continuous_{int(t_end_h)}h_timeseries.csv"
    tbl_csv = outdir / f"gamma_{tag}_continuous_final_table.csv"
    sum_json = outdir / f"gamma_{tag}_continuous_{int(t_end_h)}h_summary.json"

    df.to_csv(ts_csv, index=False)
    pd.DataFrame(
        [
            {"gamma_h": float(gamma_h), "time_h": float(t_end_h), "deGFP_reactor_uM": metrics[f"protein_{int(t_end_h)}h_uM"]},
            {"gamma_h": float(gamma_h), "time_h": max(0.0, float(t_end_h) - float(stability_window_h)), "deGFP_reactor_uM": float(np.interp(max(0.0, float(t_end_h) - float(stability_window_h)), df["time"].to_numpy(dtype=float) / 3600.0, df["[deGFP_m]"].to_numpy(dtype=float) * 1e6))},
        ]
    ).sort_values("time_h").to_csv(tbl_csv, index=False)

    summary: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": {
            "tx_pathway_mode": "1nt",
            "tx_step_mode": "micro3",
            "tl_pathway_mode": "step_chain",
            "tl_qssa_mode": "factorized",
            "tl_chain_group_aa": 1.0,
            "tl_chain_scale": 30.0,
            "energy_regen_mode": "background",
            "ribo_bind_mode": "explicit",
        },
        "continuous_setup": {
            "gamma_h": float(gamma_h),
            "feed_multiplier": float(feed_multiplier),
            "product_outflow_h": (None if product_gamma_h is None else float(product_gamma_h)),
        },
        "model_counts_continuous": counts_cont,
        "param_apply": param_apply,
        "runtime_s": runtime_s,
        "reactor_metrics_uM": metrics,
        "stability": stability,
        "files": {
            "timeseries_csv": str(ts_csv),
            "final_table_csv": str(tbl_csv),
        },
    }
    sum_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "gamma_h": float(gamma_h),
        **{k: v for k, v in metrics.items() if v is not None},
        "mean_window_uM": float(stability["mean_uM"]),
        "max_abs_drift_window_uM": float(stability["max_abs_drift_uM"]),
        "abs_slope_window_uM_per_h": float(stability["abs_slope_uM_per_h"]),
        "runtime_s": runtime_s,
        "reactions": int(counts_cont["reactions_total"]),
        "species": int(counts_cont["species_total"]),
        "summary_json": str(sum_json),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the finest-node continuous simulation: tx1nt + micro3 + tl g1.")
    ap.add_argument("--gammas", type=str, default="0.5,0.8,1.0")
    ap.add_argument("--t-end-h", type=float, default=72.0)
    ap.add_argument("--n-points", type=int, default=2400)
    ap.add_argument("--stability-window-h", type=float, default=24.0)
    ap.add_argument("--feed-multiplier", type=float, default=1.0)
    ap.add_argument("--product-gamma-h", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=600000)
    ap.add_argument("--param-summary", type=Path, default=DEFAULT_PARAM_SUMMARY)
    ap.add_argument("--hf-summary", type=Path, default=DEFAULT_HF_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    os.environ["PURE_MAX_STEPS"] = str(int(args.max_steps))
    set_fine_node_env()

    args.outdir.mkdir(parents=True, exist_ok=True)
    params, hf_fit, keff_map = load_sources(args.param_summary, args.hf_summary)
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    gammas = parse_gammas(args.gammas)

    rows: List[Dict[str, object]] = []
    for g in gammas:
        row = run_one_gamma(
            gamma_h=float(g),
            t_end_h=float(args.t_end_h),
            n_points=int(args.n_points),
            feed_multiplier=float(args.feed_multiplier),
            product_gamma_h=(None if args.product_gamma_h is None else float(args.product_gamma_h)),
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            ic_uM=ic_uM,
            stability_window_h=float(args.stability_window_h),
            outdir=args.outdir,
        )
        rows.append(row)
        print(
            "[OK] gamma={:.3g} 48h={} 72h={} drift={} slope={:.3e} runtime={:.1f}s".format(
                float(g),
                row.get("protein_48h_uM"),
                row.get("protein_72h_uM"),
                row.get("max_abs_drift_window_uM"),
                float(row.get("abs_slope_window_uM_per_h", 0.0)),
                float(row.get("runtime_s", 0.0)),
            ),
            flush=True,
        )

    final_df = pd.DataFrame(rows).sort_values("gamma_h")
    final_csv = args.outdir / "gamma_stability_final_table.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[DONE] {final_csv}", flush=True)


if __name__ == "__main__":
    main()
