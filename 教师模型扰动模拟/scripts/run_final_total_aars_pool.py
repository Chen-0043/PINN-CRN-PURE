#!/usr/bin/env python3
"""Final total-aaRS-pool perturbation runner.

This is the final "total aaRS supply" axis used in the report:
all aaRS-containing species are scaled together, and the aggregate AA
translation limiter/feed are scaled in synchrony.  This restores the expected
0x -> near-zero deGFP behavior in the reduced translation-chain model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import libsbml
import numpy as np
import pandas as pd

import rerun_synced_ntp_aars_perturbations_20260711 as base


ROOT = Path(__file__).resolve().parents[1]
OUTDIR_DEFAULT = ROOT / "run_outputs" / "final_total_aars_pool"


def parse_scales(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No scales provided")
    return vals


def sync_total_aars_pool(model: "libsbml.Model", scale: float) -> Dict[str, object]:
    species_scaled: List[str] = []
    params_scaled: List[str] = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        if any(tag in sid for tag in base.AARS):
            if base.scale_species_initial(model, sid, scale):
                species_scaled.append(sid)
    if base.scale_species_initial(model, "AA", scale):
        species_scaled.append("AA")
    for pid in ["bg_feed_AA", "bg_AA_feed"]:
        if base.scale_parameter_value(model, pid, scale):
            params_scaled.append(pid)
    return {
        "kind": "total_aars_pool_all_aars_plus_AA_and_feed",
        "species_scaled": species_scaled,
        "params_scaled": params_scaled,
        "params_set": {},
        "qssa_floors_zeroed": [],
    }


def build_total_aars_rr(*, scale: float, base_ic_uM, hf_fit, gamma_h: float, keff_map):
    base.set_environment()
    sbml, counts, meta = base.core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=base_ic_uM,
        gamma_h=float(gamma_h),
        feed_multiplier=1.0,
        product_outflow_h=float(gamma_h),
        keff_map=keff_map,
    )
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse generated SBML")
    perturb_meta = sync_total_aars_pool(model, scale)
    sbml_perturbed = libsbml.writeSBMLToString(doc)
    rr = base.te.loadSBMLModel(sbml_perturbed)
    base.core.configure_rr(rr)
    rr.resetToOrigin()
    return rr, counts, meta, perturb_meta, sbml_perturbed


def simulate_total_aars_condition(*, scale: float, params, hf_fit, keff_map, base_ic_uM, gamma_h: float, t_end_h: float, n_points: int, snapshot_time_points_h: List[float]):
    rr, counts, _meta, perturb_meta, sbml = build_total_aars_rr(
        scale=scale,
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    reaction_table = base.reaction_table_from_sbml(sbml) if snapshot_time_points_h else None
    row, ts, endpoint, sparse_bundle = base.run_rr_and_collect(
        rr=rr,
        counts=counts,
        params=params,
        component="total_aars_pool",
        scale=float(scale),
        perturbation_kind=str(perturb_meta["kind"]),
        qssa_floors_zeroed=list(perturb_meta.get("qssa_floors_zeroed", [])),
        n_scaled_species=int(len(perturb_meta.get("species_scaled", []))),
        scaled_species=list(perturb_meta.get("species_scaled", [])),
        scaled_params=list(perturb_meta.get("params_scaled", [])),
        set_params=dict(perturb_meta.get("params_set", {})),
        t_end_h=t_end_h,
        n_points=n_points,
        gamma_h=gamma_h,
        reaction_table=reaction_table,
        snapshot_time_points_h=snapshot_time_points_h,
    )
    return row, ts, endpoint, sparse_bundle, reaction_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Run final total-aaRS-pool perturbations.")
    ap.add_argument("--scales", default="0,0.01,0.03,0.18,0.20,0.22,1,4")
    ap.add_argument("--snapshot-time-points-h", default="0,0.2,0.5,1,3,4,6")
    ap.add_argument("--gamma-h", type=float, default=0.5)
    ap.add_argument("--t-end-h", type=float, default=6.0)
    ap.add_argument("--n-points", type=int, default=121)
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "endpoints").mkdir(exist_ok=True)
    if args.snapshot_time_points_h:
        (outdir / "sparse_snapshots" / "species").mkdir(parents=True, exist_ok=True)
        (outdir / "sparse_snapshots" / "reactions").mkdir(parents=True, exist_ok=True)
        (outdir / "sparse_snapshots" / "reaction_classes").mkdir(parents=True, exist_ok=True)

    scales = parse_scales(args.scales)
    snapshot_t = base.parse_snapshot_time_points(args.snapshot_time_points_h)
    params, hf_fit, keff_map, base_ic_uM = base.load_sources()

    rows = []
    manifest_rows = []
    for i, scale in enumerate(scales, start=1):
        label = f"total_aars_pool_x{base.tag_float(scale)}"
        print(f"[RUN] {i}/{len(scales)} {label}", flush=True)
        row, _ts, endpoint, sparse_bundle, reaction_table = simulate_total_aars_condition(
            scale=scale,
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            base_ic_uM=base_ic_uM,
            gamma_h=float(args.gamma_h),
            t_end_h=float(args.t_end_h),
            n_points=int(args.n_points),
            snapshot_time_points_h=snapshot_t,
        )
        row["status"] = "ok"
        ep_path = outdir / "endpoints" / f"endpoint_{label}.csv"
        pd.DataFrame({"species": list(endpoint.keys()), "final_uM": list(endpoint.values())}).to_csv(ep_path, index=False)
        row["endpoint_csv"] = str(ep_path)
        if snapshot_t:
            meta = base.write_sparse_snapshot_files(
                outdir=outdir,
                label=label,
                component="total_aars_pool",
                scale=float(scale),
                sparse_bundle=sparse_bundle,
                reaction_table=reaction_table,
            )
            row["species_snapshot_csv_gz"] = meta["species_snapshot_csv_gz"]
            row["reaction_snapshot_csv_gz"] = meta["reaction_snapshot_csv_gz"]
            row["reaction_class_snapshot_csv_gz"] = meta["reaction_class_snapshot_csv_gz"]
            manifest_rows.append(meta)
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "total_aars_pool_metrics_partial.csv", index=False)
        print(f"[OK] {label} deGFP={row['deGFP_end_uM']:.6g} peak={row['deGFP_peak_uM']:.6g} t_peak={row['deGFP_peak_time_h']:.3g}h", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "total_aars_pool_metrics.csv", index=False)
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(outdir / "sparse_snapshot_manifest.csv", index=False)
    summary = {
        "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "semantics": "final total aaRS pool = all aaRS species + aggregate AA + AA feed scaled synchronously",
        "metrics_csv": str(outdir / "total_aars_pool_metrics.csv"),
        "sparse_snapshot_manifest_csv": str(outdir / "sparse_snapshot_manifest.csv") if manifest_rows else None,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {outdir / 'total_aars_pool_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
