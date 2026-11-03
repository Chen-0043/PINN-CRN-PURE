#!/usr/bin/env python3
"""Export 6 h all-species snapshots from the existing deGFP teacher scripts.

This does not define a new model. It imports the existing validated deGFP
package code and only changes RoadRunner selections to all floating species so
that the PINN can be evaluated on a broad 6 h validation set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tellurium as te


ROOT = Path(__file__).resolve().parents[1]
DEGFP = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parent / "reproduced_supervision_source" / "original_export"


def load_existing_modules():
    sys.path.insert(0, str(DEGFP))
    import fit_batch_then_continuous_full_crn_background as core  # type: ignore
    from run_tx1nt_tlg1_micro3_longrun import apply_params, load_sources, set_fine_node_env  # type: ignore

    return core, apply_params, load_sources, set_fine_node_env


def build_rr(mode: str, gamma_h: float):
    core, apply_params, load_sources, set_fine_node_env = load_existing_modules()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-pure")
    os.environ.setdefault("PURE_MAX_STEPS", "800000")
    set_fine_node_env()
    params, hf_fit, keff_map = load_sources(
        DEGFP / "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json",
        DEGFP / "output_full_crn_background/batch_then_continuous_summary.json",
    )
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    sbml, counts, hf_meta = core.build_background_sbml(
        mode=mode,
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=gamma_h,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)
    return rr, counts, hf_meta, param_apply


def parse_time_points_h(text: str | None, t_end_h: float) -> list[float]:
    if not text:
        return [0.0, float(t_end_h)]
    pts = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        pts.append(float(part))
    pts = sorted(set(pts))
    if not pts or pts[0] != 0.0:
        pts = [0.0] + pts
    return pts


def export_snapshot(
    mode: str,
    gamma_h: float,
    t_end_h: float,
    tag: str,
    time_points_h: list[float] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    rr, counts, hf_meta, param_apply = build_rr(mode, gamma_h)
    build_s = time.perf_counter() - started
    species_ids = [str(x) for x in rr.getFloatingSpeciesIds()]
    selections = ["time"] + [f"[{sid}]" for sid in species_ids]
    sim_started = time.perf_counter()
    if time_points_h is None:
        time_points_h = [0.0, float(t_end_h)]
    time_points_s = np.array(time_points_h, dtype=float) * 3600.0
    pieces = []
    # This RoadRunner build does not accept an arbitrary time vector as the
    # first argument to simulate(). Keep the existing model state and march
    # through requested checkpoints, recording each segment endpoint.
    current = float(time_points_s[0])
    pieces.append(
        np.concatenate(
            [
                np.array([current], dtype=float),
                np.asarray(rr.getFloatingSpeciesConcentrations(), dtype=float),
            ]
        )
    )
    for target in time_points_s[1:]:
        seg = rr.simulate(current, float(target), 2, selections)
        pieces.append(np.asarray(seg, dtype=float)[-1])
        current = float(target)
    result = np.vstack(pieces)
    sim_s = time.perf_counter() - sim_started
    arr = np.asarray(result, dtype=float)
    x_traj = np.maximum(arr[:, 1:], 0.0)
    x = x_traj[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    out_npz = OUT / f"{tag}_all_species_6h_snapshot.npz"
    np.savez_compressed(
        out_npz,
        time_s=np.array([float(arr[-1, 0])]),
        species_id=np.array(species_ids, dtype=object),
        concentrations_M=x[None, :],
    )
    out_csv = OUT / f"{tag}_all_species_6h_snapshot.csv"
    pd.DataFrame({"species_id": species_ids, "teacher_6h_M": x, "teacher_6h_uM": x * 1e6}).to_csv(
        out_csv, index=False
    )
    out_traj_npz = OUT / f"{tag}_all_species_trajectory.npz"
    np.savez_compressed(
        out_traj_npz,
        time_s=arr[:, 0].astype(float),
        species_id=np.array(species_ids, dtype=object),
        concentrations_M=x_traj,
    )
    out_traj_csv = OUT / f"{tag}_all_species_trajectory_long.csv"
    rows = []
    for ti, t_s in enumerate(arr[:, 0].astype(float)):
        for sid, val in zip(species_ids, x_traj[ti]):
            rows.append(
                {
                    "time_s": float(t_s),
                    "time_h": float(t_s / 3600.0),
                    "species_id": sid,
                    "teacher_M": float(val),
                    "teacher_uM": float(val * 1e6),
                }
            )
    pd.DataFrame(rows).to_csv(out_traj_csv, index=False)
    summary = {
        "scope": "all-floating-species trajectory/snapshot exported from existing deGFP teacher package",
        "mode": mode,
        "gamma_h": gamma_h,
        "t_end_h": t_end_h,
        "time_points_h": [float(x) for x in time_points_h],
        "species_count": int(len(species_ids)),
        "counts": counts,
        "hf_meta_continuous_exchange_policy": (hf_meta or {}).get("continuous_exchange_policy"),
        "param_apply": param_apply,
        "build_s": float(build_s),
        "simulate_s": float(sim_s),
        "output_npz": str(out_npz),
        "output_csv": str(out_csv),
        "output_trajectory_npz": str(out_traj_npz),
        "output_trajectory_long_csv": str(out_traj_csv),
        "finite_fraction": float(np.mean(np.isfinite(x))),
        "negative_fraction_lt_minus_1e_18": float(np.mean(x < -1e-18)),
    }
    out_json = OUT / f"{tag}_all_species_6h_snapshot_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "continuous"], required=True)
    parser.add_argument("--gamma-h", type=float, default=None)
    parser.add_argument("--t-end-h", type=float, default=6.0)
    parser.add_argument(
        "--time-points-h",
        type=str,
        default=None,
        help="Comma-separated time points in hours for all-species trajectory export.",
    )
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    gamma = 0.0 if args.mode == "batch" else 0.5
    if args.gamma_h is not None:
        gamma = float(args.gamma_h)
    result = export_snapshot(
        args.mode,
        gamma,
        float(args.t_end_h),
        args.tag,
        parse_time_points_h(args.time_points_h, float(args.t_end_h)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
