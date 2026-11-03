#!/usr/bin/env python3
"""抽样复跑技术报告中已经采用的扰动点，核对6 h deGFP终点。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rerun_synced_ntp_aars_perturbations_20260711 as ntp_base  # noqa: E402
import run_final_total_aars_pool as total_aars  # noqa: E402


OUTDIR = ROOT / "run_outputs" / "report_sample_recheck_20260902"
REPORT_RECORDS = ROOT / "technical_report_pure_20260713" / "report_audit_20260723" / "unified_perturbation_timeseries_records_20260723.csv"

SAMPLES = [
    ("ATP", 0.5),
    ("GTP", 3.0),
    ("CTP", 0.01),
    ("UTP", 0.01),
    ("total_aars_pool", 0.22),
]


def lookup_report_value(component: str, scale: float) -> float:
    df = pd.read_csv(REPORT_RECORDS)
    sub = df[
        (df["component"].astype(str) == component)
        & (pd.to_numeric(df["scale"], errors="coerce").sub(float(scale)).abs() < 1e-9)
        & (pd.to_numeric(df["time_h"], errors="coerce").sub(6.0).abs() < 1e-9)
    ]
    if sub.empty:
        raise RuntimeError(f"Cannot find report reference for {component} x{scale}")
    return float(sub.iloc[0]["deGFP"])


def simulate_sample(component: str, scale: float, params, hf_fit, keff_map, base_ic_uM) -> dict:
    if component == "total_aars_pool":
        row, _ts, _endpoint, _sparse, _reaction_table = total_aars.simulate_total_aars_condition(
            scale=float(scale),
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            base_ic_uM=base_ic_uM,
            gamma_h=0.5,
            t_end_h=6.0,
            n_points=121,
            snapshot_time_points_h=[],
        )
    else:
        row, _ts, _endpoint = ntp_base.simulate_condition(
            component=component,
            scale=float(scale),
            base_ic_uM=base_ic_uM,
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            gamma_h=0.5,
            t_end_h=6.0,
            n_points=121,
        )
    return row


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    params, hf_fit, keff_map, base_ic_uM = ntp_base.load_sources()
    rows = []
    for i, (component, scale) in enumerate(SAMPLES, start=1):
        print(f"[RUN] {i}/{len(SAMPLES)} {component} x{scale:g}", flush=True)
        ref = lookup_report_value(component, scale)
        row = simulate_sample(component, scale, params, hf_fit, keff_map, base_ic_uM)
        rerun = float(row["deGFP_end_uM"])
        abs_diff = float(rerun - ref)
        rel_diff = float(abs_diff / ref) if ref != 0 else np.nan
        rec = {
            "component": component,
            "scale": float(scale),
            "report_6h_deGFP_uM": ref,
            "rerun_6h_deGFP_uM": rerun,
            "abs_diff_uM": abs_diff,
            "rel_diff": rel_diff,
            "runtime_s": float(row["runtime_s"]),
            "perturbation_kind": row.get("perturbation_kind", ""),
            "n_scaled_species": row.get("n_scaled_species", ""),
            "scaled_params": row.get("scaled_params", ""),
            "set_params": row.get("set_params", ""),
            "status": "match" if abs(abs_diff) <= 1e-6 else "check",
        }
        rows.append(rec)
        pd.DataFrame(rows).to_csv(OUTDIR / "sample_recheck_partial.csv", index=False)
        print(
            f"[OK] {component} x{scale:g}: report={ref:.9g}, rerun={rerun:.9g}, diff={abs_diff:.3g}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "sample_recheck_results.csv", index=False)
    summary = {
        "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_csv": str(REPORT_RECORDS),
        "samples": SAMPLES,
        "max_abs_diff_uM": float(out["abs_diff_uM"].abs().max()),
        "max_rel_diff": float(out["rel_diff"].abs().max()),
        "all_match_at_1e-6_uM": bool((out["abs_diff_uM"].abs() <= 1e-6).all()),
        "results_csv": str(OUTDIR / "sample_recheck_results.csv"),
    }
    (OUTDIR / "sample_recheck_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_match_at_1e-6_uM"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
