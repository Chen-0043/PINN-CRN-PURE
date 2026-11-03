#!/usr/bin/env python3
"""Evaluate integral NRMSR for every state in a saved six-protein PINN path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_resolution_pinn import DEFAULT_ASSETS, DEFAULT_POOL, HybridResolutionOperator
from train_teacher_supervised_sixprotein_pinn import aggregate, load_supervision


def detailed_nrmsr_summary(
    residual_uM: np.ndarray,
    rhs_integral_uM: np.ndarray,
    interval_h: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int | None]:
    """Report the manuscript's per-state and pooled NRMSR thresholds."""
    residual_rms = np.sqrt(np.mean(residual_uM[:, indices] ** 2, axis=0))
    rhs_rms = np.sqrt(np.mean(rhs_integral_uM[:, indices] ** 2, axis=0))
    output: dict[str, float | int | None] = {"state_count": int(len(indices))}
    for floor_uM in (1e-12, 0.001, 0.01, 0.1, 1.0):
        active = rhs_rms > floor_uM
        key = str(floor_uM).replace(".", "p").replace("-", "m")
        output[f"rhs_rms_gt_{key}_uM_count"] = int(np.sum(active))
        if not np.any(active):
            output[f"rhs_rms_gt_{key}_uM_mean_species_NRMSR_percent"] = None
            output[f"rhs_rms_gt_{key}_uM_rms_species_NRMSR_percent"] = None
            output[f"rhs_rms_gt_{key}_uM_pooled_NRMSR_percent"] = None
            continue
        ratio = residual_rms[active] / rhs_rms[active]
        output[f"rhs_rms_gt_{key}_uM_mean_species_NRMSR_percent"] = float(np.mean(ratio) * 100.0)
        output[f"rhs_rms_gt_{key}_uM_rms_species_NRMSR_percent"] = float(np.sqrt(np.mean(ratio**2)) * 100.0)
        output[f"rhs_rms_gt_{key}_uM_pooled_NRMSR_percent"] = float(
            np.sqrt(np.mean(residual_uM[:, indices][:, active] ** 2))
            / max(np.sqrt(np.mean(rhs_integral_uM[:, indices][:, active] ** 2)), 1e-30)
            * 100.0
        )
    rhs_rate_rms = np.sqrt(np.mean((rhs_integral_uM[:, indices] / interval_h[:, None]) ** 2, axis=0))
    for floor_uM_per_h in (0.004, 0.01, 0.1, 1.0):
        active = rhs_rate_rms > floor_uM_per_h
        key = str(floor_uM_per_h).replace(".", "p")
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_count"] = int(np.sum(active))
        if not np.any(active):
            output[f"rhs_rate_rms_gt_{key}_uM_per_h_mean_species_NRMSR_percent"] = None
            output[f"rhs_rate_rms_gt_{key}_uM_per_h_pooled_NRMSR_percent"] = None
            continue
        ratio = residual_rms[active] / rhs_rms[active]
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_mean_species_NRMSR_percent"] = float(np.mean(ratio) * 100.0)
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_pooled_NRMSR_percent"] = float(
            np.sqrt(np.mean(residual_uM[:, indices][:, active] ** 2))
            / max(np.sqrt(np.mean(rhs_integral_uM[:, indices][:, active] ** 2)), 1e-30)
            * 100.0
        )
    return output


def stats(mask: np.ndarray, rhs: np.ndarray, residual: np.ndarray) -> dict[str, float | int | None]:
    if not np.any(mask):
        return {"states": 0, "mean_NRMSR_percent": None, "median_NRMSR_percent": None,
                "pooled_NRMSR_percent": None}
    ratios = np.sqrt(np.mean(residual[:, mask] ** 2, axis=0)) / np.maximum(
        np.sqrt(np.mean(rhs[:, mask] ** 2, axis=0)), 1e-30
    )
    return {
        "states": int(np.sum(mask)),
        "mean_NRMSR_percent": float(np.mean(ratios) * 100.0),
        "median_NRMSR_percent": float(np.median(ratios) * 100.0),
        "pooled_NRMSR_percent": float(
            np.sqrt(np.mean(residual[:, mask] ** 2) / max(np.mean(rhs[:, mask] ** 2), 1e-300)) * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--energy-scale", type=float, default=0.8)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--threshold-uM-per-h", type=float, default=0.004)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--metric-window-s",
        type=float,
        default=None,
        help=(
            "Optionally aggregate internal integration steps into fixed metric windows "
            "before calculating NRMSR, e.g. 900 for the original 15-minute windows."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-all-teacher-metabolites", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64
    operator = HybridResolutionOperator(
        args.assets,
        args.pool_json,
        energy_scale=args.energy_scale,
        gamma_h=args.gamma_h,
        include_all_teacher_metabolites=args.include_all_teacher_metabolites,
        dtype=dtype,
    )
    supervision = load_supervision(args.supervision, operator.n_species, dtype)
    data = np.load(args.prediction, allow_pickle=True)
    time_s = data["time_s"].astype(float)
    state_np = data["prediction_M"].astype(float)
    if state_np.shape != (len(time_s), operator.n_species):
        raise ValueError("Prediction dimensions do not match the full-state operator")
    rhs_chunks = []
    with torch.no_grad():
        for start in range(0, len(time_s), args.batch_size):
            values = torch.tensor(state_np[start : start + args.batch_size], dtype=dtype)
            rhs_chunks.append(operator(values).cpu().numpy())
    rhs = np.concatenate(rhs_chunks, axis=0)
    dt_h = np.diff(time_s) / 3600.0
    if not np.allclose(dt_h, dt_h[0]):
        raise ValueError("Saved trajectory time grid must be uniform")
    step_integral = 0.5 * np.diff(time_s)[:, None] * (rhs[1:] + rhs[:-1])
    step_delta = np.diff(state_np, axis=0)
    if args.metric_window_s is None:
        rhs_integral = step_integral
        delta = step_delta
        metric_dt_h = dt_h
    else:
        ratio = float(args.metric_window_s) / float(np.diff(time_s)[0])
        n_substeps = int(round(ratio))
        if n_substeps < 1 or not np.isclose(ratio, n_substeps):
            raise ValueError("metric-window-s must be an integer multiple of the saved time step")
        if len(step_integral) % n_substeps:
            raise ValueError("Saved trajectory length does not divide evenly into metric windows")
        rhs_integral = step_integral.reshape(
            -1, n_substeps, operator.n_species
        ).sum(axis=1)
        delta = state_np[n_substeps::n_substeps] - state_np[:-1:n_substeps]
        metric_dt_h = np.full(len(rhs_integral), float(args.metric_window_s) / 3600.0)
    residual = delta - rhs_integral
    rhs_rms_uM = np.sqrt(np.mean(rhs_integral**2, axis=0)) * 1e6
    residual_rms_uM = np.sqrt(np.mean(residual**2, axis=0)) * 1e6
    activity_uM_h = rhs_rms_uM / float(metric_dt_h[0])
    nrmsr = residual_rms_uM / np.maximum(rhs_rms_uM, 1e-30) * 100.0

    ids = np.asarray(operator.species_ids, dtype=str)
    direct = np.zeros(operator.n_species, dtype=bool)
    direct[supervision["direct_indices_np"]] = True
    aggregate_mask = np.zeros(operator.n_species, dtype=bool)
    # A state covered only through an aggregate target remains a hidden microstate,
    # but is distinguished from states absent from every teacher constraint.
    sup_data = np.load(args.supervision, allow_pickle=True)
    members = sup_data["group_member_indices"].astype(np.int64)
    aggregate_mask[members] = True
    hidden = ~direct
    threshold = float(args.threshold_uM_per_h)
    active = activity_uM_h > threshold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "species_id": ids,
        "direct_teacher_state": direct,
        "covered_by_aggregate_teacher_constraint": aggregate_mask,
        "hidden_state": hidden,
        "rhs_activity_rms_uM_per_h": activity_uM_h,
        "rhs_integral_rms_uM": rhs_rms_uM,
        "residual_rms_uM": residual_rms_uM,
        "NRMSR_percent": nrmsr,
        "above_activity_threshold": active,
    })
    frame.to_csv(args.output_dir / "fullstate_nrmsr_by_species.csv", index=False)
    report = {
        "formula": "NRMSR_i = RMS_t(Delta x_i - integral f_i dt) / RMS_t(integral f_i dt)",
        "activity_threshold_uM_per_h": threshold,
        "native_window_s": float(metric_dt_h[0] * 3600.0),
        "time_points": int(len(rhs_integral) + 1),
        "internal_step_s": float(dt_h[0] * 3600.0),
        "all_states": stats(np.ones(operator.n_species, dtype=bool), rhs_integral * 1e6, residual * 1e6),
        "active_all": stats(active, rhs_integral * 1e6, residual * 1e6),
        "active_direct_teacher_states": stats(active & direct, rhs_integral * 1e6, residual * 1e6),
        "active_aggregate_covered_hidden_states": stats(active & aggregate_mask & hidden, rhs_integral * 1e6, residual * 1e6),
        "active_uncovered_hidden_states": stats(active & ~aggregate_mask & hidden, rhs_integral * 1e6, residual * 1e6),
        "output_csv": str(args.output_dir / "fullstate_nrmsr_by_species.csv"),
    }
    (args.output_dir / "fullstate_nrmsr_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Complete manuscript-facing evaluation: all 9,198 comparable states plus
    # every full and hidden elementary state.  This is deliberately separate
    # from the compact activity-threshold summary above.
    direct_indices_np = supervision["direct_indices_np"]
    direct_indices_t = torch.tensor(direct_indices_np, dtype=torch.long)
    state_t = torch.tensor(state_np, dtype=dtype)
    direct_prediction = state_t.index_select(1, direct_indices_t).cpu().numpy()
    group_prediction = aggregate(supervision["group_matrix"], state_t).cpu().numpy()
    combined_prediction = np.concatenate((direct_prediction[-1], group_prediction[-1]))
    combined_target = np.concatenate((supervision["direct_target_np"][-1], supervision["group_target_np"][-1]))
    nonzero = np.abs(combined_target) > 1e-15
    concentration_error = np.abs(combined_prediction - combined_target)
    direct_set = set(direct_indices_np.tolist())
    hidden_elementary_indices = np.asarray(
        [index for index in range(operator.base_species_count) if index not in direct_set], dtype=int
    )
    sup_data = np.load(args.supervision, allow_pickle=True)
    group_members = sup_data["group_member_indices"].astype(int)
    grouped_hidden_indices = np.asarray(sorted(set(group_members.tolist()) - direct_set), dtype=int)
    all_indices = np.arange(operator.n_species, dtype=int)
    residual_uM_all = residual * 1e6
    rhs_integral_uM_all = rhs_integral * 1e6
    complete_report = {
        "prediction": str(args.prediction),
        "model_inventory": {
            "states": int(operator.n_species),
            "full_elementary_states": int(operator.base_species_count),
            "mass_action_elementary_reactions": int(operator.n_elementary_reactions),
            "teacher_effective_reactions": int(operator.n_teacher_reactions),
            "energy_scale": float(args.energy_scale),
            "gamma_h_inv": float(args.gamma_h),
            "include_all_teacher_metabolites": bool(args.include_all_teacher_metabolites),
        },
        "validation_time_h": float(time_s[-1] / 3600.0),
        "all_comparable_concentration_error": {
            "direct_states": int(len(direct_prediction[-1])),
            "aggregate_states": int(len(group_prediction[-1])),
            "total_comparable_states": int(len(combined_target)),
            "nonzero_states_entering_MRE": int(np.sum(nonzero)),
            "MRE_percent": float(np.mean(concentration_error[nonzero] / np.abs(combined_target[nonzero])) * 100.0),
            "concentration_weighted_relative_error_percent": float(
                np.sum(concentration_error) / max(np.sum(np.abs(combined_target)), 1e-30) * 100.0
            ),
        },
        "NRMSR_definition": {
            "mean_species": "mean_i[RMS_t(residual_i)/RMS_t(integral_rhs_i)]",
            "rms_species": "sqrt(mean_i[(RMS_t(residual_i)/RMS_t(integral_rhs_i))^2])",
            "pooled": "RMS_i,t(residual)/RMS_i,t(integral_rhs)",
            "saved_grid": "25 points over 0--6 h (15 min intervals)",
        },
        "all_full_states": detailed_nrmsr_summary(residual_uM_all, rhs_integral_uM_all, metric_dt_h, all_indices),
        "hidden_full_elementary_states": detailed_nrmsr_summary(
            residual_uM_all, rhs_integral_uM_all, metric_dt_h, hidden_elementary_indices
        ),
        "teacher_aggregate_member_hidden_states": detailed_nrmsr_summary(
            residual_uM_all, rhs_integral_uM_all, metric_dt_h, grouped_hidden_indices
        ),
        "scope_counts": {
            "all_PINN_states": int(operator.n_species),
            "base_full_elementary_states": int(operator.base_species_count),
            "direct_comparable_states": int(len(direct_indices_np)),
            "hidden_full_elementary_states": int(len(hidden_elementary_indices)),
            "unique_teacher_aggregate_member_hidden_states": int(len(grouped_hidden_indices)),
        },
    }
    (args.output_dir / "fullstate_and_hidden_nrmsr.json").write_text(
        json.dumps(complete_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(args.output_dir / "fullstate_and_hidden_nrmsr.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
