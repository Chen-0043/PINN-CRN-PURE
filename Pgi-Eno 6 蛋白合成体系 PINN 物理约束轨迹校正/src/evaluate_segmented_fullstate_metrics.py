#!/usr/bin/env python3
"""Evaluate held-out concentration error and full/hidden-state NRMSR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_resolution_pinn import DEFAULT_ASSETS, DEFAULT_POOL, HybridResolutionOperator
from train_teacher_supervised_sixprotein_pinn import aggregate, load_supervision


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION = (
    ROOT
    / "outputs"
    / "sixprotein_teacher_pinn_positional_mechanistic_v9_segmented"
    / "segmented_prediction_6h.npz"
)
DEFAULT_SUPERVISION = (
    DEFAULT_ASSETS / "sixprotein_teacher_supervision_positional_6h_converged.npz"
)


def nrmsr_summary(
    residual_uM: np.ndarray,
    rhs_integral_uM: np.ndarray,
    interval_h: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int | None]:
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
        output[f"rhs_rms_gt_{key}_uM_mean_species_NRMSR_percent"] = float(
            np.mean(ratio) * 100.0
        )
        output[f"rhs_rms_gt_{key}_uM_rms_species_NRMSR_percent"] = float(
            np.sqrt(np.mean(ratio**2)) * 100.0
        )
        output[f"rhs_rms_gt_{key}_uM_pooled_NRMSR_percent"] = float(
            np.sqrt(np.mean(residual_uM[:, indices][:, active] ** 2))
            / max(np.sqrt(np.mean(rhs_integral_uM[:, indices][:, active] ** 2)), 1e-30)
            * 100.0
        )
    rhs_rate_rms = np.sqrt(
        np.mean((rhs_integral_uM[:, indices] / interval_h[:, None]) ** 2, axis=0)
    )
    for floor_uM_per_h in (0.004, 0.01, 0.1, 1.0):
        active = rhs_rate_rms > floor_uM_per_h
        key = str(floor_uM_per_h).replace(".", "p")
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_count"] = int(np.sum(active))
        if not np.any(active):
            output[f"rhs_rate_rms_gt_{key}_uM_per_h_mean_species_NRMSR_percent"] = None
            output[f"rhs_rate_rms_gt_{key}_uM_per_h_pooled_NRMSR_percent"] = None
            continue
        ratio = residual_rms[active] / rhs_rms[active]
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_mean_species_NRMSR_percent"] = float(
            np.mean(ratio) * 100.0
        )
        output[f"rhs_rate_rms_gt_{key}_uM_per_h_pooled_NRMSR_percent"] = float(
            np.sqrt(np.mean(residual_uM[:, indices][:, active] ** 2))
            / max(np.sqrt(np.mean(rhs_integral_uM[:, indices][:, active] ** 2)), 1e-30)
            * 100.0
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--energy-scale", type=float, required=True)
    parser.add_argument("--gamma-h", type=float, required=True)
    parser.add_argument("--include-all-teacher-metabolites", action="store_true")
    parser.add_argument("--output", type=Path)
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
    prediction = np.load(args.prediction, allow_pickle=True)
    time_s = prediction["time_s"].astype(float)
    state = torch.tensor(prediction["prediction_M"].astype(float), dtype=dtype)
    direct_indices_np = supervision["direct_indices_np"]
    direct_indices = torch.tensor(direct_indices_np, dtype=torch.long)
    group_matrix = supervision["group_matrix"]

    direct_prediction = state.index_select(1, direct_indices).cpu().numpy()
    group_prediction = aggregate(group_matrix, state).cpu().numpy()
    direct_target = supervision["direct_target_np"]
    group_target = supervision["group_target_np"]
    holdout = -1
    combined_prediction = np.concatenate((direct_prediction[holdout], group_prediction[holdout]))
    combined_target = np.concatenate((direct_target[holdout], group_target[holdout]))
    nonzero = np.abs(combined_target) > 1e-15
    error = np.abs(combined_prediction - combined_target)

    with torch.no_grad():
        rhs = operator(state).cpu().numpy()
    dt = np.diff(time_s)[:, None]
    interval_h = np.diff(time_s) / 3600.0
    rhs_integral_uM = 0.5 * dt * (rhs[1:] + rhs[:-1]) * 1e6
    residual_uM = np.diff(state.cpu().numpy(), axis=0) * 1e6 - rhs_integral_uM

    all_indices = np.arange(operator.n_species, dtype=int)
    direct_set = set(direct_indices_np.tolist())
    hidden_elementary_indices = np.asarray(
        [
            index
            for index in range(operator.base_species_count)
            if index not in direct_set
        ],
        dtype=int,
    )
    group_members = np.load(args.supervision, allow_pickle=True)["group_member_indices"].astype(int)
    grouped_hidden_indices = np.asarray(
        sorted(set(group_members.tolist()) - direct_set), dtype=int
    )
    hidden_elementary_set = set(hidden_elementary_indices.tolist())
    grouped_hidden_set = set(grouped_hidden_indices.tolist())

    report = {
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
        "validation_time_h": float(time_s[holdout] / 3600.0),
        "all_comparable_concentration_error": {
            "direct_states": int(direct_target.shape[1]),
            "aggregate_states": int(group_target.shape[1]),
            "total_comparable_states": int(len(combined_target)),
            "nonzero_states_entering_MRE": int(np.sum(nonzero)),
            "MRE_percent": float(np.mean(error[nonzero] / np.abs(combined_target[nonzero])) * 100.0),
            "concentration_weighted_relative_error_percent": float(
                np.sum(error) / max(np.sum(np.abs(combined_target)), 1e-30) * 100.0
            ),
        },
        "NRMSR_definition": {
            "mean_species": "mean_i[RMS_t(residual_i)/RMS_t(integral_rhs_i)]",
            "rms_species": "sqrt(mean_i[(RMS_t(residual_i)/RMS_t(integral_rhs_i))^2])",
            "pooled": "RMS_i,t(residual)/RMS_i,t(integral_rhs)",
            "saved_grid": "25 points over 0--6 h (15 min intervals)",
        },
        "all_full_states": nrmsr_summary(
            residual_uM, rhs_integral_uM, interval_h, all_indices
        ),
        "hidden_full_elementary_states": nrmsr_summary(
            residual_uM, rhs_integral_uM, interval_h, hidden_elementary_indices
        ),
        "teacher_aggregate_member_hidden_states": nrmsr_summary(
            residual_uM, rhs_integral_uM, interval_h, grouped_hidden_indices
        ),
        "scope_counts": {
            "all_PINN_states": int(operator.n_species),
            "base_full_elementary_states": int(operator.base_species_count),
            "direct_comparable_states": int(len(direct_indices_np)),
            "hidden_full_elementary_states": int(len(hidden_elementary_indices)),
            "unique_teacher_aggregate_member_hidden_states": int(len(grouped_hidden_indices)),
        },
    }
    output = args.output or args.prediction.parent / "fullstate_and_hidden_nrmsr.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    species_ids = np.asarray(operator.species_ids, dtype=str)
    residual_rms = np.sqrt(np.mean(residual_uM**2, axis=0))
    rhs_rms = np.sqrt(np.mean(rhs_integral_uM**2, axis=0))
    for index in all_indices:
        rows.append(
            {
                "species_index": int(index),
                "species_id": species_ids[index],
                "direct_comparable": bool(index in direct_set),
                "hidden_full_elementary": bool(index in hidden_elementary_set),
                "teacher_aggregate_member_hidden": bool(index in grouped_hidden_set),
                "rhs_integral_rms_uM": rhs_rms[index],
                "residual_rms_uM": residual_rms[index],
                "NRMSR_percent": residual_rms[index] / max(rhs_rms[index], 1e-12) * 100.0,
            }
        )
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
