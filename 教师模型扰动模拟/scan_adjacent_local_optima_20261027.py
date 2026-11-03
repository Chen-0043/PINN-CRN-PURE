#!/usr/bin/env python3
"""Check local optima under the existing +/-10% joint parameter samples.

The baseline and original candidate-optimum simulations are reused from
``five_axis_uncertainty_20261027``.  Only the two neighbouring dose levels are
newly simulated for each axis and parameter sample.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rerun_synced_ntp_aars_perturbations_20260711 as ntp  # noqa: E402
import run_final_total_aars_pool as aars  # noqa: E402


SOURCE = ROOT / "run_outputs" / "five_axis_uncertainty_20261027"
OUTDIR = ROOT / "run_outputs" / "adjacent_local_optima_20261027"

# Original article grid: 0, .01, .03, .06, .1, .22, .25, .5, 1, 2, 3, 4.
AXES = {
    "ATP": {"component": "ATP", "left": 0.25, "center": 0.5, "right": 1.0},
    "GTP": {"component": "GTP", "left": 2.0, "center": 3.0, "right": 4.0},
    "CTP": {"component": "CTP", "left": 0.0, "center": 0.01, "right": 0.03},
    "UTP": {"component": "UTP", "left": 0.0, "center": 0.01, "right": 0.03},
    "total_aars_pool": {
        "component": "total_aars_pool", "left": 0.1, "center": 0.22, "right": 0.25,
    },
}


def _scale_tag(scale: float) -> str:
    return ("%g" % scale).replace(".", "p")


def _build_rr(component: str, scale: float):
    os.environ["PURE_MAX_STEPS"] = "600000"
    params0, hf_fit, keff_map, base_ic_uM = ntp.load_sources()
    if component == "total_aars_pool":
        rr, counts, _meta, perturb_meta, _sbml = aars.build_total_aars_rr(
            scale=scale, base_ic_uM=base_ic_uM, hf_fit=hf_fit,
            gamma_h=0.5, keff_map=keff_map,
        )
    else:
        rr, counts, _meta, perturb_meta, _sbml = ntp.build_perturbed_rr(
            component=component, scale=scale, base_ic_uM=base_ic_uM,
            hf_fit=hf_fit, gamma_h=0.5, keff_map=keff_map,
        )
    integ = rr.getIntegrator()
    integ.setValue("relative_tolerance", 1e-6)
    integ.setValue("absolute_tolerance", 1e-12)
    try:
        integ.setValue("maximum_num_steps", 600000)
    except Exception:
        pass
    return rr, counts, perturb_meta, params0


def _run_condition(
    axis: str, component: str, side: str, scale: float, samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    build_start = time.time()
    rr, counts, perturb_meta, params0 = _build_rr(component, scale)
    build_s = time.time() - build_start
    partial = OUTDIR / f"worker_{axis}_{side}_x{_scale_tag(scale)}.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[int(row["sample_id"])] = row

    for sample in samples:
        sid = int(sample["sample_id"])
        if sid in existing and existing[sid].get("status") == "ok":
            continue
        multipliers = {k: float(v) for k, v in sample.items() if k != "sample_id"}
        params = {k: float(v) * multipliers.get(k, 1.0) for k, v in params0.items()}
        started = time.time()
        try:
            rr.resetToOrigin()
            row0, _ts, _endpoint, _sparse = ntp.run_rr_and_collect(
                rr=rr,
                counts=counts,
                params=params,
                component=component,
                scale=scale,
                perturbation_kind=str(perturb_meta["kind"]),
                qssa_floors_zeroed=list(perturb_meta.get("qssa_floors_zeroed", [])),
                n_scaled_species=int(len(perturb_meta.get("species_scaled", []))),
                scaled_species=list(perturb_meta.get("species_scaled", [])),
                scaled_params=list(perturb_meta.get("params_scaled", [])),
                set_params=dict(perturb_meta.get("params_set", {})),
                t_end_h=6.0,
                n_points=2,
                gamma_h=0.5,
            )
            row = {
                "axis": axis,
                "component": component,
                "side": side,
                "scale": scale,
                "sample_id": sid,
                "deGFP_6h_uM": float(row0["deGFP_end_uM"]),
                "network_build_runtime_s": build_s,
                "simulation_runtime_s": float(time.time() - started),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            row = {
                "axis": axis, "component": component, "side": side,
                "scale": scale, "sample_id": sid, "deGFP_6h_uM": np.nan,
                "network_build_runtime_s": build_s,
                "simulation_runtime_s": float(time.time() - started),
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
        existing[sid] = row
        partial.write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in existing.values()) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{axis} {side} x{scale:g}] sample={sid} {row['status']} "
            f"y6={row['deGFP_6h_uM']:.9g} sim={row['simulation_runtime_s']:.1f}s",
            flush=True,
        )
    return list(existing.values())


def summarize(new_rows: pd.DataFrame) -> None:
    old = pd.read_csv(SOURCE / "parameter_robustness_raw.csv")
    retry_path = OUTDIR / "tight_retry_results.csv"
    retry = pd.read_csv(retry_path) if retry_path.exists() else pd.DataFrame()
    if "status" not in old:
        old["status"] = "ok"
    old["status"] = old["status"].fillna("ok")
    rows = []
    for axis, cfg in AXES.items():
        for sid in sorted(old["sample_id"].unique()):
            base = old[(old.sample_id == sid) & (old.condition == "baseline") & (old.status == "ok")]
            center = old[(old.sample_id == sid) & (old.condition == axis) & (old.status == "ok")]
            left = new_rows[(new_rows.sample_id == sid) & (new_rows.axis == axis) & (new_rows.side == "left") & (new_rows.status == "ok")]
            right = new_rows[(new_rows.sample_id == sid) & (new_rows.axis == axis) & (new_rows.side == "right") & (new_rows.status == "ok")]
            # ATP's right neighbour is exactly the already available 1x
            # baseline; reuse it instead of running the same model again.
            if float(cfg["left"]) == 1.0:
                left = base.rename(columns={"deGFP_6h_uM": "deGFP_6h_uM"})
            if float(cfg["right"]) == 1.0:
                right = base.rename(columns={"deGFP_6h_uM": "deGFP_6h_uM"})
            center_solver = "production"
            left_solver = "production"
            if len(center) != 1 and not retry.empty:
                recovered = retry[
                    (retry.sample_id == sid)
                    & (retry.condition == axis)
                    & (retry.status == "ok")
                ]
                if len(recovered):
                    center = recovered.iloc[[0]]
                    center_solver = str(recovered.iloc[0].solver_label)
            if len(left) != 1 and not retry.empty:
                recovered = retry[
                    (retry.sample_id == sid)
                    & (retry.condition == f"{axis}_left")
                    & (retry.status == "ok")
                ]
                if len(recovered):
                    left = recovered.iloc[[0]]
                    left_solver = str(recovered.iloc[0].solver_label)
            complete = all(len(x) == 1 for x in (base, center, left, right))
            if not complete:
                rows.append({"axis": axis, "sample_id": sid, "complete": False})
                continue
            y_base = float(base.iloc[0].deGFP_6h_uM)
            y_left = float(left.iloc[0].deGFP_6h_uM)
            y_center = float(center.iloc[0].deGFP_6h_uM)
            y_right = float(right.iloc[0].deGFP_6h_uM)
            candidates = {
                float(cfg["left"]): y_left,
                float(cfg["center"]): y_center,
                float(cfg["right"]): y_right,
            }
            best_scale = max(candidates, key=candidates.get)
            best_y = candidates[best_scale]
            rows.append(
                {
                    "axis": axis,
                    "sample_id": sid,
                    "complete": True,
                    "baseline_scale": 1.0,
                    "baseline_deGFP_6h_uM": y_base,
                    "left_scale": cfg["left"],
                    "left_deGFP_6h_uM": y_left,
                    "center_scale": cfg["center"],
                    "center_deGFP_6h_uM": y_center,
                    "right_scale": cfg["right"],
                    "right_deGFP_6h_uM": y_right,
                    "center_solver": center_solver,
                    "left_solver": left_solver,
                    "local_best_scale": best_scale,
                    "local_best_deGFP_6h_uM": best_y,
                    "center_is_local_max": bool(best_scale == float(cfg["center"])),
                    "center_beats_baseline": bool(y_center > y_base),
                    "local_best_beats_baseline": bool(best_y > y_base),
                    "center_minus_best_neighbor_uM": y_center - max(y_left, y_right),
                    "center_gain_vs_baseline_percent": 100.0 * (y_center / y_base - 1.0),
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(OUTDIR / "adjacent_local_optimum_paired_comparison.csv", index=False)
    complete = paired[paired.complete == True].copy()  # noqa: E712
    summary = complete.groupby("axis").agg(
        n_complete=("sample_id", "size"),
        n_center_is_local_max=("center_is_local_max", "sum"),
        n_center_beats_baseline=("center_beats_baseline", "sum"),
        n_local_best_beats_baseline=("local_best_beats_baseline", "sum"),
        local_best_scales=("local_best_scale", lambda x: ",".join("%g" % v for v in x)),
        min_center_minus_best_neighbor_uM=("center_minus_best_neighbor_uM", "min"),
        max_center_minus_best_neighbor_uM=("center_minus_best_neighbor_uM", "max"),
        min_center_gain_vs_baseline_percent=("center_gain_vs_baseline_percent", "min"),
        max_center_gain_vs_baseline_percent=("center_gain_vs_baseline_percent", "max"),
    ).reset_index()
    summary["all_center_is_local_max"] = summary.n_center_is_local_max == summary.n_complete
    summary["all_center_beats_baseline"] = summary.n_center_beats_baseline == summary.n_complete
    summary["all_local_best_beats_baseline"] = summary.n_local_best_beats_baseline == summary.n_complete
    summary.to_csv(OUTDIR / "adjacent_local_optimum_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SOURCE / "parameter_multiplier_samples.csv").to_dict("records")
    jobs = []
    for axis, cfg in AXES.items():
        for side in ("left", "right"):
            if float(cfg[side]) == 1.0:
                continue
            jobs.append((axis, str(cfg["component"]), side, float(cfg[side]), samples))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_condition, *job): job[:4] for job in jobs}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                batch = fut.result()
                rows.extend(batch)
                print(f"Merged {label}: {len(batch)} samples", flush=True)
            except Exception as exc:
                print(f"FAILED batch {label}: {type(exc).__name__}: {exc}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTDIR / "adjacent_local_optimum_new_simulations.csv", index=False)
    summarize(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
