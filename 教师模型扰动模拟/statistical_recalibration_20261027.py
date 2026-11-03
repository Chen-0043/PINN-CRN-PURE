#!/usr/bin/env python3
"""Experimental-data extraction and local identifiability audit.

This script does not invent parameter error bars.  It rebuilds the three
available replicate experiments (transcription only, translation only and
combined expression), evaluates the active teacher model, and computes the
finite-difference Jacobian of every fitted runtime parameter with respect to
the experimentally observed trajectories.

The resulting weighted SVD is the prerequisite for deciding which parameters
can receive data-supported confidence intervals.  Parameters in the numerical
null space are explicitly reported as non-identifiable.
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


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "model_inputs" / "deliverable_PURE_1nt1aa_independent_20260610"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(SCRIPTS))

import tellurium as te  # noqa: E402

import fit_batch_then_continuous_full_crn_background as core  # noqa: E402
from run_tx1nt_tlg1_micro3_longrun import apply_params, load_sources, set_fine_node_env  # noqa: E402


OUTDIR = ROOT / "run_outputs" / "statistical_recalibration_20261027"
RAW_ROOT = PACKAGE / "raw_literature_data"
TX_RAW = RAW_ROOT / "MGapt_deGFP_Exp_PURE_Data" / "2022.08.19_OnePot_PURE_mGapt_uM.csv"
COMBINED_RAW = RAW_ROOT / "MGapt_deGFP_Exp_PURE_Data" / "2022.11.14_OnePotPURE_deGFPuM.csv"
TL_RAW = RAW_ROOT / "Calibration_MGapt" / "2023.07.28_PUREwmRNA_updated.xlsx"

PARAM_SUMMARY = PACKAGE / "output_full_crn_background" / "formal_mech_only_20260409_210539_rechecked.json"
HF_SUMMARY = PACKAGE / "output_full_crn_background" / "batch_then_continuous_summary.json"


def _time_to_h(value: Any) -> float:
    if hasattr(value, "hour"):
        return float(value.hour + value.minute / 60.0 + value.second / 3600.0)
    text = str(value)
    parts = [float(x) for x in text.split(":")]
    if len(parts) == 3:
        return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    raise ValueError(f"Unsupported time value: {value!r}")


def extract_observations() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    tx = pd.read_csv(TX_RAW)
    tx_t = tx["timepoints"].to_numpy(dtype=float) / 3600.0
    tx_reps = tx[["mRNA0", "mRNA1", "mRNA2", "mRNA3"]].to_numpy(dtype=float)
    rows.append(_summarize_replicates("transcription_only", "mRNA_t", tx_t, tx_reps))

    combined = pd.read_csv(COMBINED_RAW, skipinitialspace=True)
    combined.columns = [str(c).strip() for c in combined.columns]
    c_t = combined["Time"].to_numpy(dtype=float) / 60.0
    c_reps = combined[["deGFP_1", "deGFP_2", "deGFP_3"]].to_numpy(dtype=float)
    rows.append(_summarize_replicates("combined_expression", "deGFP_m", c_t, c_reps))

    tl = pd.read_excel(TL_RAW, sheet_name="gfp61_data")
    tl_t = np.asarray([_time_to_h(v) for v in tl["Time"]], dtype=float)
    neg = tl["B20"].to_numpy(dtype=float)
    tl_reps = tl[["B21", "B22", "B23"]].to_numpy(dtype=float)
    tl_reps = (tl_reps - neg[:, None]) / 5542.4
    tl_reps = tl_reps - tl_reps[0, :]
    rows.append(_summarize_replicates("translation_only", "deGFP_m", tl_t, tl_reps))

    out = pd.concat(rows, ignore_index=True)
    return out


def _summarize_replicates(
    condition: str, observable: str, time_h: np.ndarray, replicates: np.ndarray
) -> pd.DataFrame:
    n = replicates.shape[1]
    mean = np.mean(replicates, axis=1)
    sd = np.std(replicates, axis=1, ddof=1)
    sem = sd / math.sqrt(n)
    data: dict[str, Any] = {
        "condition": condition,
        "observable": observable,
        "time_h": time_h,
        "n_replicates": n,
        "experiment_mean_uM": mean,
        "experiment_sd_uM": sd,
        "experiment_sem_uM": sem,
    }
    for j in range(n):
        data[f"replicate_{j+1}_uM"] = replicates[:, j]
    return pd.DataFrame(data)


def _configure_environment() -> None:
    os.environ["PURE_MAX_STEPS"] = "600000"
    os.chdir(PACKAGE)
    set_fine_node_env()


def _build_rr(condition: str, params: dict[str, float], hf_fit: dict, keff_map: dict):
    _configure_environment()
    if condition == "transcription_only":
        source = PACKAGE / "results" / "platform_raw_validation" / "MGapt_90nt_T7RNAP_mRNA_t_source.xml"
        if not source.exists():
            raise FileNotFoundError(source)
        core.TX_1NT_SOURCE_SBML = source
        ic = core.load_initial_uM_table(core.INIT_CSV)
        ic.update({"DNA": 0.005, "T7RNAP": 0.1, "ATP": 3000.0, "GTP": 3000.0,
                   "CTP": 1000.0, "UTP": 1000.0, "Ribosome": 0.0, "RS70S": 0.0})
        sbml, _counts, _meta = core.build_background_sbml(
            mode="batch", fitted_hf={}, ic_uM=ic, gamma_h=0.0,
            feed_multiplier=1.0, product_outflow_h=None, keff_map=None,
        )
        observable = "mRNA_t"
        end_h = 15.0
    elif condition in {"translation_only", "combined_expression"}:
        core.TX_1NT_SOURCE_SBML = PACKAGE / (
            "external_PURE_CRN_models_sparse/Models/TXTL_PURE_MGapt-deGFP_model.xml"
        )
        ic = core.load_initial_uM_table(core.INIT_CSV)
        if condition == "translation_only":
            ic.update({"DNA": 0.0, "T7RNAP": 0.0, "mRNA_t": 0.548})
            end_h = 12.0
        else:
            ic["DNA"] = 0.001
            end_h = 6.0
        sbml, _counts, _meta = core.build_background_sbml(
            mode="batch", fitted_hf=hf_fit, ic_uM=ic, gamma_h=0.0,
            feed_multiplier=1.0, product_outflow_h=None, keff_map=keff_map,
        )
        observable = "deGFP_m"
    else:
        raise ValueError(condition)

    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    applied = apply_params(rr, params)
    return rr, observable, end_h, applied


def simulate_condition(
    condition: str,
    parameter: str | None,
    log_step: float,
    observation_times_h: list[float],
) -> dict[str, Any]:
    params, hf_fit, keff_map = load_sources(PARAM_SUMMARY, HF_SUMMARY)
    if parameter is not None:
        params = dict(params)
        params[parameter] = float(params[parameter]) * math.exp(float(log_step))
    start = time.time()
    rr, observable, end_h, applied = _build_rr(condition, params, hf_fit, keff_map)
    n_points = max(121, int(round(end_h * 20)) + 1)
    result = rr.simulate(0.0, end_h * 3600.0, n_points, ["time", f"[{observable}]"])
    t_h = np.asarray(result[:, 0], dtype=float) / 3600.0
    y_uM = np.asarray(result[:, 1], dtype=float) * 1e6
    pred = np.interp(np.asarray(observation_times_h, dtype=float), t_h, y_uM)
    return {
        "condition": condition,
        "parameter": parameter or "__baseline__",
        "log_step": float(log_step),
        "observable": observable,
        "applied": int(applied["applied"]),
        "missing": int(applied["missing"]),
        "runtime_s": float(time.time() - start),
        "predictions_uM": pred.tolist(),
    }


def simulate_condition_batch(
    condition: str,
    parameters: list[str | None],
    log_step: float,
    observation_times_h: list[float],
) -> list[dict[str, Any]]:
    """Build one condition once, then reuse it for all parameter perturbations.

    Network generation and RoadRunner compilation dominate the wall time.  A
    reset followed by re-applying the complete parameter dictionary is
    mathematically equivalent to rebuilding the same SBML for each global
    parameter, while avoiding that repeated compilation cost.
    """
    base_params, hf_fit, keff_map = load_sources(PARAM_SUMMARY, HF_SUMMARY)
    build_start = time.time()
    rr, observable, end_h, _applied = _build_rr(
        condition, dict(base_params), hf_fit, keff_map
    )
    build_runtime = float(time.time() - build_start)
    n_points = max(121, int(round(end_h * 20)) + 1)
    worker_partial = OUTDIR / f"sensitivity_worker_{condition}.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if worker_partial.exists():
        for line in worker_partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old = json.loads(line)
                existing[old["parameter"]] = old

    output: list[dict[str, Any]] = []
    for parameter in parameters:
        key = parameter or "__baseline__"
        if key in existing and existing[key].get("status") == "ok":
            output.append(existing[key])
            continue
        start = time.time()
        params = dict(base_params)
        if parameter is not None:
            params[parameter] = float(params[parameter]) * math.exp(float(log_step))
        try:
            rr.resetToOrigin()
            applied = apply_params(rr, params)
            result = rr.simulate(
                0.0, end_h * 3600.0, n_points, ["time", f"[{observable}]"],
            )
            t_h = np.asarray(result[:, 0], dtype=float) / 3600.0
            y_uM = np.asarray(result[:, 1], dtype=float) * 1e6
            pred = np.interp(np.asarray(observation_times_h, dtype=float), t_h, y_uM)
            row = {
                "condition": condition,
                "parameter": key,
                "log_step": 0.0 if parameter is None else float(log_step),
                "observable": observable,
                "applied": int(applied["applied"]),
                "missing": int(applied["missing"]),
                "network_build_runtime_s": build_runtime,
                "runtime_s": float(time.time() - start),
                "predictions_uM": pred.tolist(),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            row = {
                "condition": condition,
                "parameter": key,
                "log_step": 0.0 if parameter is None else float(log_step),
                "network_build_runtime_s": build_runtime,
                "runtime_s": float(time.time() - start),
                "predictions_uM": [],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        existing[key] = row
        output.append(row)
        worker_partial.write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in existing.values()) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{condition}] {key} {row['status']} "
            f"build={build_runtime:.1f}s simulate={row['runtime_s']:.1f}s",
            flush=True,
        )
    return output


def run_sensitivity(observations: pd.DataFrame, workers: int, rel_step: float) -> None:
    params, _hf, _keff = load_sources(PARAM_SUMMARY, HF_SUMMARY)
    param_names = sorted(params)
    conditions = ["transcription_only", "translation_only", "combined_expression"]
    obs_times = {
        c: observations.loc[observations["condition"] == c, "time_h"].astype(float).tolist()
        for c in conditions
    }
    log_step = math.log1p(rel_step)
    tasks = [(c, None, 0.0, obs_times[c]) for c in conditions]
    tasks += [(c, p, log_step, obs_times[c]) for c in conditions for p in param_names]

    partial = OUTDIR / "sensitivity_simulations_partial.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[(row["condition"], row["parameter"])] = row
    pending = [t for t in tasks if (t[0], t[1] or "__baseline__") not in completed]
    print(f"Completed={len(completed)} pending={len(pending)}", flush=True)

    # One worker per experimental condition.  Each worker compiles its network
    # once and then reuses it for every still-pending parameter perturbation.
    batches: list[tuple[str, list[str | None], float, list[float]]] = []
    for condition in conditions:
        pending_params = [
            t[1] for t in pending if t[0] == condition
        ]
        if pending_params:
            batches.append((condition, pending_params, log_step, obs_times[condition]))
    if not batches:
        _summarize_identifiability(observations, completed, param_names, log_step)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(batches))) as pool:
        futs = {pool.submit(simulate_condition_batch, *batch): batch[0] for batch in batches}
        for fut in as_completed(futs):
            condition = futs[fut]
            try:
                new_rows = fut.result()
            except Exception as exc:
                print(f"Batch failed: {condition}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for row in new_rows:
                completed[(row["condition"], row["parameter"])] = row
            partial.write_text(
                "\n".join(json.dumps(v, ensure_ascii=False) for v in completed.values()) + "\n",
                encoding="utf-8",
            )
            print(f"Merged batch {condition}: {len(new_rows)} rows", flush=True)

    _summarize_identifiability(observations, completed, param_names, log_step)


def _summarize_identifiability(
    obs: pd.DataFrame,
    rows: dict[tuple[str, str], dict[str, Any]],
    param_names: list[str],
    log_step: float,
) -> None:
    blocks: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    row_meta: list[pd.DataFrame] = []
    for condition in ["transcription_only", "translation_only", "combined_expression"]:
        d = obs[obs["condition"] == condition].copy().reset_index(drop=True)
        base = rows[(condition, "__baseline__")]
        if base.get("status") != "ok":
            continue
        y0 = np.asarray(base["predictions_uM"], dtype=float)
        J = np.full((len(d), len(param_names)), np.nan, dtype=float)
        for j, pname in enumerate(param_names):
            row = rows.get((condition, pname), {})
            if row.get("status") == "ok":
                J[:, j] = (np.asarray(row["predictions_uM"], dtype=float) - y0) / log_step
        # Conservative SD floors prevent near-zero early measurements from
        # dominating the information matrix.
        positive_sd = d.loc[d["experiment_sd_uM"] > 0, "experiment_sd_uM"].to_numpy(dtype=float)
        floor = max(1e-3, 0.1 * float(np.median(positive_sd))) if positive_sd.size else 1e-3
        sigma = np.maximum(d["experiment_sd_uM"].to_numpy(dtype=float), floor)
        blocks.append(J)
        residuals.append(d["experiment_mean_uM"].to_numpy(dtype=float) - y0)
        weights.append(1.0 / sigma)
        d["model_baseline_uM"] = y0
        d["weight_sd_floor_uM"] = floor
        row_meta.append(d)

    J = np.vstack(blocks)
    r = np.concatenate(residuals)
    w = np.concatenate(weights)
    finite_cols = np.all(np.isfinite(J), axis=0)
    Jw = J[:, finite_cols] * w[:, None]
    rw = r * w
    U, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    tol = max(Jw.shape) * np.finfo(float).eps * s[0] if s.size else 0.0
    rank_machine = int(np.sum(s > tol))
    rank_practical = int(np.sum(s / max(s[0], 1e-300) > 1e-6)) if s.size else 0
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")

    # Null-space participation: large values mean the parameter cannot be
    # separated from correlated alternatives by the available observations.
    null = Vt[rank_practical:, :] if rank_practical < Vt.shape[0] else np.zeros((0, Vt.shape[1]))
    null_part = np.sqrt(np.sum(null * null, axis=0)) if null.size else np.zeros(Vt.shape[1])
    active_names = np.asarray(param_names)[finite_cols]
    ident = pd.DataFrame({
        "parameter": active_names,
        "jacobian_l2_weighted": np.linalg.norm(Jw, axis=0),
        "null_space_participation": null_part,
        "practically_identifiable": null_part < 0.1,
    }).sort_values(["practically_identifiable", "jacobian_l2_weighted"], ascending=[False, False])

    obs_out = pd.concat(row_meta, ignore_index=True)
    obs_out.to_csv(OUTDIR / "experimental_observations_with_baseline.csv", index=False)
    ident.to_csv(OUTDIR / "parameter_local_identifiability.csv", index=False)
    np.savez_compressed(
        OUTDIR / "weighted_local_jacobian.npz",
        J=J, residual=r, weight=w, parameter_names=np.asarray(param_names),
        singular_values=s,
    )
    summary = {
        "n_observations": int(J.shape[0]),
        "n_parameters_requested": int(J.shape[1]),
        "n_parameters_with_complete_derivatives": int(np.sum(finite_cols)),
        "machine_rank": rank_machine,
        "practical_rank_relative_threshold_1e-6": rank_practical,
        "condition_number": cond,
        "weighted_residual_sum_squares_at_original_parameters": float(np.sum(rw * rw)),
        "parameters_practically_identifiable": ident.loc[ident["practically_identifiable"], "parameter"].tolist(),
        "interpretation": (
            "Confidence intervals are only valid for practically identifiable parameter combinations. "
            "Parameters with large null-space participation must be profiled, fixed from independent data, "
            "or reported as non-identifiable."
        ),
    }
    (OUTDIR / "identifiability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "sensitivity", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--relative-step", type=float, default=0.01)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    observations = extract_observations()
    observations.to_csv(OUTDIR / "experimental_replicate_summary.csv", index=False)
    if args.mode in {"sensitivity", "all"}:
        run_sensitivity(observations, workers=args.workers, rel_step=args.relative_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
