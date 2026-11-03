#!/usr/bin/env python3
"""Evaluate six-protein PINN physics on nonredundant biological trajectories.

All teacher states remain in concentration validation.  For the physical
integral metric, position-resolved transcription and translation occupancies
are summed by gene before NRMSR is calculated.  This prevents a macroscopic
activity cutoff from discarding thousands of nonzero but tiny positional
occupancies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_resolution_pinn import HybridResolutionOperator
from train_teacher_supervised_sixprotein_pinn import aggregate, load_supervision


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ROOT / "outputs" / "original_18p88_full_elementary"
DEFAULT_POOL = ROOT / "input_pool_override_sg_pure2_paper_mapped_20260424.json"
DEFAULT_SUPERVISION = DEFAULT_ASSETS / "sixprotein_teacher_supervision_all_metabolites_dt0p125.npz"
DEFAULT_PREDICTION = (
    ROOT
    / "outputs"
    / "original_18p88_teacher_pinn_all_metabolites_stage3"
    / "prior_prediction_6h.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "original_18p88_teacher_pinn_all_metabolites_stage3"
    / "biological_panel"
)
GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--energy-scale", type=float, default=0.24)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--activity-threshold-uM-per-h", type=float, default=0.004)
    parser.add_argument("--mre-floor-M", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dtype = torch.float64
    operator = HybridResolutionOperator(
        args.assets,
        args.pool_json,
        energy_scale=args.energy_scale,
        gamma_h=args.gamma_h,
        include_all_teacher_metabolites=True,
        dtype=dtype,
    )
    supervision = load_supervision(args.supervision, operator.n_species, dtype)
    raw_supervision = np.load(args.supervision, allow_pickle=True)
    prediction_data = np.load(args.prediction, allow_pickle=True)

    time_s = prediction_data["time_s"].astype(float)
    state_np = prediction_data["prediction_M"].astype(float)
    if state_np.shape != (len(time_s), operator.n_species):
        raise ValueError(
            f"Prediction shape {state_np.shape} does not match "
            f"({len(time_s)}, {operator.n_species})"
        )
    if not np.allclose(time_s, supervision["time_s"]):
        raise ValueError("Prediction and teacher supervision time grids differ")

    def sparse_abs(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.coalesce()
        return torch.sparse_coo_tensor(
            matrix.indices(), matrix.values().abs(), matrix.shape, dtype=matrix.dtype
        ).coalesce()

    elementary_stoich_abs = sparse_abs(operator.S_elementary)
    teacher_stoich_abs = sparse_abs(operator.S_teacher)
    rhs_chunks: list[np.ndarray] = []
    turnover_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(time_s), args.batch_size):
            state = torch.tensor(state_np[start : start + args.batch_size], dtype=dtype)
            rhs_chunks.append(operator(state).cpu().numpy())
            elementary_turnover = torch.sparse.mm(
                elementary_stoich_abs, operator.elementary_flux(state).T
            ).T
            if operator.n_species > operator.base_species_count:
                elementary_turnover = torch.cat(
                    (
                        elementary_turnover,
                        state.new_zeros(
                            (len(state), operator.n_species - operator.base_species_count)
                        ),
                    ),
                    dim=1,
                )
            teacher_turnover = torch.sparse.mm(
                teacher_stoich_abs, operator.teacher_flux(state).T
            ).T
            turnover_chunks.append(
                (elementary_turnover + teacher_turnover + operator.exchange_rhs(state).abs())
                .cpu()
                .numpy()
            )
    rhs_np = np.concatenate(rhs_chunks, axis=0)
    turnover_np = np.concatenate(turnover_chunks, axis=0)

    direct_indices = supervision["direct_indices_np"]
    direct_names = list(supervision["direct_names"])
    group_names = list(supervision["group_names"])
    direct_prediction = state_np[:, direct_indices]
    group_prediction = aggregate(
        supervision["group_matrix"], torch.tensor(state_np, dtype=dtype)
    ).cpu().numpy()
    direct_target = supervision["direct_target_np"]
    group_target = supervision["group_target_np"]

    dt_s = np.diff(time_s)
    if not np.allclose(dt_s, dt_s[0]):
        raise ValueError("The saved trajectory must use a uniform time grid")
    dt_h = float(dt_s[0] / 3600.0)
    rhs_integral = 0.5 * dt_s[:, None] * (rhs_np[1:] + rhs_np[:-1])
    delta = np.diff(state_np, axis=0)
    direct_theoretical = rhs_integral[:, direct_indices]
    direct_residual = delta[:, direct_indices] - direct_theoretical
    group_theoretical = aggregate(
        supervision["group_matrix"], torch.tensor(rhs_integral, dtype=dtype)
    ).cpu().numpy()
    group_residual = aggregate(
        supervision["group_matrix"], torch.tensor(delta - rhs_integral, dtype=dtype)
    ).cpu().numpy()
    direct_turnover = turnover_np[:, direct_indices]
    group_turnover = aggregate(
        supervision["group_matrix"], torch.tensor(turnover_np, dtype=dtype)
    ).cpu().numpy()

    entries: list[dict[str, object]] = []

    def add_entry(
        name: str,
        category: str,
        source_names: list[str],
        prediction: np.ndarray,
        target: np.ndarray,
        theoretical: np.ndarray,
        residual: np.ndarray,
        turnover: np.ndarray,
    ) -> None:
        teacher_rate = np.diff(target, axis=0) / dt_h
        rhs_rate = theoretical / dt_h
        rhs_integral_rms_uM = float(_rms(theoretical[:, None])[0] * 1e6)
        residual_rms_uM = float(_rms(residual[:, None])[0] * 1e6)
        rhs_activity = float(_rms(rhs_rate[:, None])[0] * 1e6)
        teacher_activity = float(_rms(teacher_rate[:, None])[0] * 1e6)
        gross_turnover_activity = float(_rms(turnover[:, None])[0] * 1e6 * 3600.0)
        floor_integral_uM = float(args.activity_threshold_uM_per_h * dt_h)
        target_6h = float(target[-1])
        pred_6h = float(prediction[-1])
        entries.append(
            {
                "trajectory": name,
                "category": category,
                "source_teacher_states": len(source_names),
                "source_constraints": ";".join(source_names),
                "teacher_6h_uM": target_6h * 1e6,
                "PINN_6h_uM": pred_6h * 1e6,
                "6h_relative_error_percent": (
                    abs(pred_6h - target_6h) / abs(target_6h) * 100.0
                    if abs(target_6h) > args.mre_floor_M
                    else np.nan
                ),
                "teacher_activity_rms_uM_per_h": teacher_activity,
                "PINN_rhs_activity_rms_uM_per_h": rhs_activity,
                "PINN_gross_turnover_rms_uM_per_h": gross_turnover_activity,
                "rhs_integral_rms_uM": rhs_integral_rms_uM,
                "residual_rms_uM": residual_rms_uM,
                "NRMSR_percent": residual_rms_uM / max(rhs_integral_rms_uM, 1e-30) * 100.0,
                "stabilized_NRMSR_percent": (
                    residual_rms_uM / max(rhs_integral_rms_uM, floor_integral_uM) * 100.0
                ),
                "net_rhs_above_0p004_uM_per_h": rhs_activity > args.activity_threshold_uM_per_h,
                "gross_turnover_above_0p004_uM_per_h": (
                    gross_turnover_activity > args.activity_threshold_uM_per_h
                ),
            }
        )

    for column, name in enumerate(direct_names):
        add_entry(
            name,
            "direct_teacher_state",
            [name],
            direct_prediction[:, column],
            direct_target[:, column],
            direct_theoretical[:, column],
            direct_residual[:, column],
            direct_turnover[:, column],
        )

    group_lookup = {name: column for column, name in enumerate(group_names)}
    for gene in GENES:
        initiation_name = f"gene::{gene}::ribosome_mrna_assembly"
        initiation_column = group_lookup[initiation_name]
        add_entry(
            initiation_name,
            "translation_initiation_assembly",
            [initiation_name],
            group_prediction[:, initiation_column],
            group_target[:, initiation_column],
            group_theoretical[:, initiation_column],
            group_residual[:, initiation_column],
            group_turnover[:, initiation_column],
        )
        for chain, category in (
            ("tx_pos", "transcription_chain_total"),
            ("tl_pos", "translation_chain_total"),
        ):
            prefix = f"gene::{gene}::{chain}::"
            columns = [i for i, group_name in enumerate(group_names) if group_name.startswith(prefix)]
            source_names = [group_names[i] for i in columns]
            if not columns:
                raise ValueError(f"No {chain} constraints found for {gene}")
            add_entry(
                f"gene::{gene}::{category}",
                category,
                source_names,
                np.sum(group_prediction[:, columns], axis=1),
                np.sum(group_target[:, columns], axis=1),
                np.sum(group_theoretical[:, columns], axis=1),
                np.sum(group_residual[:, columns], axis=1),
                np.sum(group_turnover[:, columns], axis=1),
            )

    panel = pd.DataFrame(entries)
    if len(panel) != 84:
        raise ValueError(f"Expected 84 biological panel trajectories, found {len(panel)}")

    all_target = np.concatenate((direct_target, group_target), axis=1)
    all_prediction = np.concatenate((direct_prediction, group_prediction), axis=1)
    teacher_empirical_rate = np.diff(all_target, axis=0) / dt_h
    teacher_activity = _rms(teacher_empirical_rate) * 1e6
    all_gross_turnover = np.concatenate((direct_turnover, group_turnover), axis=1)
    all_gross_turnover_activity = _rms(all_gross_turnover) * 1e6 * 3600.0
    all_nonzero = np.any(np.diff(all_target, axis=0) != 0.0, axis=0)
    all_valid_6h = np.abs(all_target[-1]) > args.mre_floor_M
    all_mre_6h = np.abs(all_prediction[-1] - all_target[-1]) / np.maximum(
        np.abs(all_target[-1]), args.mre_floor_M
    )
    panel_valid = panel["6h_relative_error_percent"].notna().to_numpy()
    panel_net_active = panel["net_rhs_above_0p004_uM_per_h"].to_numpy(dtype=bool)
    panel_turnover_active = panel["gross_turnover_above_0p004_uM_per_h"].to_numpy(dtype=bool)

    summary = {
        "interpretation": (
            "All 9198 teacher states are retained for concentration validation. "
            "For biological NRMSR reporting, position-resolved occupancies are summed "
            "by gene into transcription- and translation-chain totals."
        ),
        "all_teacher_states": int(all_target.shape[1]),
        "all_teacher_states_with_nonzero_dynamics": int(np.sum(all_nonzero)),
        "all_teacher_states_teacher_activity_gt_0p004_uM_per_h": int(
            np.sum(teacher_activity > args.activity_threshold_uM_per_h)
        ),
        "all_teacher_states_PINN_gross_turnover_gt_0p004_uM_per_h": int(
            np.sum(all_gross_turnover_activity > args.activity_threshold_uM_per_h)
        ),
        "all_teacher_states_6h_MRE_values": int(np.sum(all_valid_6h)),
        "all_teacher_states_6h_MRE_percent": float(np.mean(all_mre_6h[all_valid_6h]) * 100.0),
        "biological_panel_trajectories": int(len(panel)),
        "biological_panel_source_teacher_states": int(panel["source_teacher_states"].sum()),
        "biological_panel_valid_6h_MRE_values": int(np.sum(panel_valid)),
        "biological_panel_6h_MRE_percent": float(
            panel.loc[panel_valid, "6h_relative_error_percent"].mean()
        ),
        "biological_panel_net_rhs_above_0p004_uM_per_h": int(np.sum(panel_net_active)),
        "biological_panel_gross_turnover_above_0p004_uM_per_h": int(
            np.sum(panel_turnover_active)
        ),
        "biological_panel_turnover_active_mean_NRMSR_percent": (
            float(panel.loc[panel_turnover_active, "NRMSR_percent"].mean())
            if np.any(panel_turnover_active)
            else None
        ),
        "biological_panel_all_stabilized_mean_NRMSR_percent": float(
            panel["stabilized_NRMSR_percent"].mean()
        ),
        "biological_panel_all_stabilized_median_NRMSR_percent": float(
            panel["stabilized_NRMSR_percent"].median()
        ),
        "activity_threshold_uM_per_h": float(args.activity_threshold_uM_per_h),
        "stabilized_NRMSR_denominator_floor_uM": float(
            args.activity_threshold_uM_per_h * dt_h
        ),
        "native_window_h": dt_h,
        "categories": {
            str(category): int(count)
            for category, count in panel.groupby("category").size().items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "sixprotein_biological_panel_metrics.csv"
    summary_path = args.output_dir / "sixprotein_biological_panel_summary.json"
    panel.to_csv(panel_path, index=False)
    summary["panel_csv"] = str(panel_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
