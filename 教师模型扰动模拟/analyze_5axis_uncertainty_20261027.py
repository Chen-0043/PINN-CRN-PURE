#!/usr/bin/env python3
"""Numerical-convergence and fitted-parameter robustness analysis for Fig. 3.3.1.

The script deliberately separates two questions:

1. Numerical convergence of the deterministic CVODE solution.
2. Robustness to a declared uncertainty range of the 27 calibrated kinetic
   parameters stored in ``batch_best_params``.

The second result is an uncertainty/robustness interval conditional on the
declared parameter range, not an experimental confidence interval.  Parameter
sets are shared (paired) between the 1x baseline and all five candidate optima,
so intervals for the relative gain do not confound independent parameter draws.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rerun_synced_ntp_aars_perturbations_20260711 as ntp  # noqa: E402
import run_final_total_aars_pool as aars  # noqa: E402


OUTDIR_DEFAULT = ROOT / "run_outputs" / "five_axis_uncertainty_20261027"
CONDITIONS = [
    ("baseline", "__baseline__", 1.0),
    ("ATP", "ATP", 0.5),
    ("GTP", "GTP", 3.0),
    ("CTP", "CTP", 0.01),
    ("UTP", "UTP", 0.01),
    ("total_aars_pool", "total_aars_pool", 0.22),
]

_SOURCES: tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], dict[str, float]] | None = None


def _load_sources():
    global _SOURCES
    if _SOURCES is None:
        os.environ["PURE_MAX_STEPS"] = "600000"
        _SOURCES = ntp.load_sources()
    return _SOURCES


def _set_solver(rr, rtol: float, atol: float) -> None:
    integ = rr.getIntegrator()
    integ.setValue("relative_tolerance", float(rtol))
    integ.setValue("absolute_tolerance", float(atol))
    try:
        integ.setValue("maximum_num_steps", 600000)
    except Exception:
        pass


def _simulate(task: dict[str, Any]) -> dict[str, Any]:
    params0, hf_fit, keff_map, base_ic_uM = _load_sources()
    params = dict(params0)
    multipliers = task.get("multipliers") or {}
    for name, mult in multipliers.items():
        params[name] = float(params[name]) * float(mult)

    component = str(task["component"])
    scale = float(task["scale"])
    rtol = float(task["rtol"])
    atol = float(task["atol"])
    started = time.time()

    if component == "__baseline__":
        rr, counts, _meta, _sbml = ntp.build_baseline_rr(
            base_ic_uM=base_ic_uM,
            hf_fit=hf_fit,
            gamma_h=0.5,
            keff_map=keff_map,
        )
        perturb_meta = {
            "kind": "shared_baseline_1x",
            "qssa_floors_zeroed": [],
            "species_scaled": [],
            "params_scaled": [],
            "params_set": {},
        }
    elif component == "total_aars_pool":
        rr, counts, _meta, perturb_meta, _sbml = aars.build_total_aars_rr(
            scale=scale,
            base_ic_uM=base_ic_uM,
            hf_fit=hf_fit,
            gamma_h=0.5,
            keff_map=keff_map,
        )
    else:
        rr, counts, _meta, perturb_meta, _sbml = ntp.build_perturbed_rr(
            component=component,
            scale=scale,
            base_ic_uM=base_ic_uM,
            hf_fit=hf_fit,
            gamma_h=0.5,
            keff_map=keff_map,
        )

    _set_solver(rr, rtol=rtol, atol=atol)
    row, _ts, _endpoint, _sparse = ntp.run_rr_and_collect(
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
    return {
        "analysis": task["analysis"],
        "solver_label": task.get("solver_label", ""),
        "sample_id": int(task.get("sample_id", -1)),
        "condition": task["condition"],
        "component": component,
        "scale": scale,
        "rtol": rtol,
        "atol": atol,
        "deGFP_6h_uM": float(row["deGFP_end_uM"]),
        "runtime_s": float(time.time() - started),
        "multipliers_json": json.dumps(multipliers, sort_keys=True),
    }


def _task_key(task: dict[str, Any]) -> tuple:
    multipliers = task.get("multipliers")
    if multipliers is None:
        value = task.get("multipliers_json", "{}")
        multipliers = json.loads(value) if isinstance(value, str) and value.strip() else {}
    # Include the actual parameter draw and solver settings when resuming.
    signature = tuple(sorted((name, round(float(value), 12)) for name, value in multipliers.items()))
    return (
        str(task["analysis"]),
        int(task.get("sample_id", -1)),
        str(task.get("solver_label", "")),
        str(task["condition"]),
        str(task["component"]),
        float(task["scale"]),
        float(task["rtol"]),
        float(task["atol"]),
        signature,
    )


def _run_tasks(
    tasks: list[dict[str, Any]], workers: int, partial_path: Path, resume: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if resume and partial_path.exists():
        previous = pd.read_csv(partial_path).to_dict("records")
        requested = {_task_key(task) for task in tasks}
        matching = {}
        for row in previous:
            if str(row.get("status", "ok")) != "ok":
                continue
            key = _task_key(row)
            if key in requested and np.isfinite(float(row.get("deGFP_6h_uM", np.nan))):
                matching[key] = row
        rows.extend(matching.values())
        completed = set(matching)
        tasks = [task for task in tasks if _task_key(task) not in completed]
        print(f"Resuming with {len(previous)} completed rows; {len(tasks)} tasks remain.", flush=True)
    if not tasks:
        return pd.DataFrame(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_simulate, task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            task = futures[fut]
            try:
                row = fut.result()
                row["status"] = "ok"
                row["error"] = ""
            except Exception as exc:
                row = {
                    "analysis": task["analysis"],
                    "solver_label": task.get("solver_label", ""),
                    "sample_id": int(task.get("sample_id", -1)),
                    "condition": task["condition"],
                    "component": task["component"],
                    "scale": float(task["scale"]),
                    "rtol": float(task["rtol"]),
                    "atol": float(task["atol"]),
                    "deGFP_6h_uM": np.nan,
                    "runtime_s": np.nan,
                    "multipliers_json": json.dumps(task.get("multipliers") or {}, sort_keys=True),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            if row["status"] == "ok":
                print(
                    f"[{i}/{len(tasks)}] {row['analysis']} sample={row['sample_id']} "
                    f"{row['condition']} y6={row['deGFP_6h_uM']:.9g} uM "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"[{i}/{len(tasks)}] FAILED sample={row['sample_id']} "
                    f"{row['condition']}: {row['error']}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def run_convergence(outdir: Path, workers: int) -> None:
    solver_settings = [
        ("production", 1e-6, 1e-12),
        ("tight", 1e-7, 1e-13),
        ("very_tight", 1e-8, 1e-14),
    ]
    tasks = []
    for label, rtol, atol in solver_settings:
        for condition, component, scale in CONDITIONS:
            tasks.append(
                {
                    "analysis": "convergence",
                    "solver_label": label,
                    "sample_id": -1,
                    "condition": condition,
                    "component": component,
                    "scale": scale,
                    "rtol": rtol,
                    "atol": atol,
                    "multipliers": {},
                }
            )
    raw = _run_tasks(tasks, workers, outdir / "convergence_partial.csv")
    raw.to_csv(outdir / "convergence_raw.csv", index=False)
    pivot = raw.pivot(index="condition", columns="solver_label", values="deGFP_6h_uM").reset_index()
    for col in ["production", "tight"]:
        pivot[f"rel_diff_{col}_vs_very_tight"] = (
            (pivot[col] - pivot["very_tight"]).abs() / pivot["very_tight"].abs().clip(lower=1e-30)
        )
    pivot["numerical_envelope_uM"] = (
        pivot[["production", "tight", "very_tight"]].max(axis=1)
        - pivot[["production", "tight", "very_tight"]].min(axis=1)
    )
    pivot.to_csv(outdir / "convergence_summary.csv", index=False)

    # Paired relative gains at every solver setting.
    gain_rows = []
    for label, _rtol, _atol in solver_settings:
        y0 = float(raw[(raw["solver_label"] == label) & (raw["condition"] == "baseline")]["deGFP_6h_uM"].iloc[0])
        for condition, _component, _scale in CONDITIONS[1:]:
            y = float(raw[(raw["solver_label"] == label) & (raw["condition"] == condition)]["deGFP_6h_uM"].iloc[0])
            gain_rows.append({"solver_label": label, "condition": condition, "gain_percent_vs_1x": 100.0 * (y / y0 - 1.0)})
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(outdir / "convergence_relative_gains.csv", index=False)


def make_lhs_multipliers(param_names: list[str], n_samples: int, rel_range: float, seed: int) -> list[dict[str, float]]:
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    u = sampler.random(n=n_samples)
    lo = math.log(1.0 - rel_range)
    hi = math.log(1.0 + rel_range)
    mult = np.exp(lo + (hi - lo) * u)
    return [dict(zip(param_names, row.astype(float))) for row in mult]


def run_parameter_robustness(outdir: Path, workers: int, n_samples: int, rel_range: float, seed: int) -> None:
    params, _hf, _keff, _ic = _load_sources()
    param_names = sorted(params)
    samples = make_lhs_multipliers(param_names, n_samples=n_samples, rel_range=rel_range, seed=seed)
    pd.DataFrame([{"sample_id": i, **sample} for i, sample in enumerate(samples)]).to_csv(
        outdir / "parameter_multiplier_samples.csv", index=False
    )

    tasks = []
    for sample_id, multipliers in enumerate(samples):
        for condition, component, scale in CONDITIONS:
            tasks.append(
                {
                    "analysis": "parameter_robustness",
                    "solver_label": "production",
                    "sample_id": sample_id,
                    "condition": condition,
                    "component": component,
                    "scale": scale,
                    "rtol": 1e-6,
                    "atol": 1e-12,
                    "multipliers": multipliers,
                }
            )
    raw = _run_tasks(tasks, workers, outdir / "parameter_robustness_partial.csv", resume=True)
    raw.to_csv(outdir / "parameter_robustness_raw.csv", index=False)

    if "status" not in raw.columns:
        raw["status"] = "ok"
    else:
        raw["status"] = raw["status"].fillna("ok")
    successful = raw[raw["status"] == "ok"].copy()
    wide = successful.pivot(index="sample_id", columns="condition", values="deGFP_6h_uM").sort_index()
    complete_ids = wide.dropna(subset=[condition for condition, _component, _scale in CONDITIONS]).index
    wide = wide.loc[complete_ids]
    gain = pd.DataFrame(index=wide.index)
    for condition, _component, _scale in CONDITIONS[1:]:
        gain[condition] = 100.0 * (wide[condition] / wide["baseline"] - 1.0)
    gain.to_csv(outdir / "parameter_robustness_paired_gains_percent.csv")

    summary_rows = []
    for condition in gain.columns:
        arr = gain[condition].to_numpy(dtype=float)
        summary_rows.append(
            {
                "condition": condition,
                "n_complete_paired_samples": int(arr.size),
                "parameter_range": f"independent log-uniform multipliers in [{1-rel_range:.3g}, {1+rel_range:.3g}]",
                "gain_median_percent": float(np.median(arr)),
                "gain_mean_percent": float(np.mean(arr)),
                "gain_q2p5_percent": float(np.quantile(arr, 0.025)),
                "gain_q97p5_percent": float(np.quantile(arr, 0.975)),
                "probability_gain_gt_0": float(np.mean(arr > 0.0)),
                "gain_min_percent": float(np.min(arr)),
                "gain_max_percent": float(np.max(arr)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(outdir / "parameter_robustness_summary.csv", index=False)

    # Rank which calibrated parameters drive each paired gain in this ensemble.
    mult_df = pd.DataFrame(samples).loc[gain.index]
    corr_rows = []
    for condition in gain.columns:
        for pname in param_names:
            rho, pval = spearmanr(mult_df[pname].to_numpy(), gain[condition].to_numpy())
            corr_rows.append(
                {
                    "condition": condition,
                    "parameter": pname,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                    "abs_rho": float(abs(rho)),
                }
            )
    corr = pd.DataFrame(corr_rows).sort_values(["condition", "abs_rho"], ascending=[True, False])
    corr.to_csv(outdir / "parameter_gain_spearman_ranking.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["convergence", "parameters", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--relative-range", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "workers": args.workers,
        "samples": args.samples,
        "relative_parameter_range": args.relative_range,
        "seed": args.seed,
        "interpretation": "parameter intervals are robustness intervals conditional on the declared range, not experimental confidence intervals",
        "conditions": CONDITIONS,
    }
    (outdir / "analysis_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode in {"convergence", "all"}:
        run_convergence(outdir, workers=args.workers)
    if args.mode in {"parameters", "all"}:
        run_parameter_robustness(
            outdir,
            workers=args.workers,
            n_samples=args.samples,
            rel_range=args.relative_range,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
