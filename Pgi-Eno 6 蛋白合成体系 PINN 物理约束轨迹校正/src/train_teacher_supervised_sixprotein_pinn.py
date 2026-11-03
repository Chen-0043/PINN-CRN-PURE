#!/usr/bin/env python3
"""Train the six-protein prior PINN using all mapped teacher constraints.

The detailed reduced teacher contributes direct states and position-resolved
aggregate constraints over corresponding full-elementary microstate families.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from hybrid_resolution_pinn import DEFAULT_ASSETS, DEFAULT_POOL, HybridResolutionOperator, scipy_to_torch
from train_hybrid_resolution_pinn import HardICPositivePINN, time_derivative


ROOT = Path(__file__).resolve().parent
DEFAULT_SUPERVISION = (
    DEFAULT_ASSETS / "sixprotein_teacher_supervision_positional_6h_converged.npz"
)
DEFAULT_SCALE = DEFAULT_ASSETS / "state_scale_from_degfp_probe_M.npy"
DEFAULT_OUTPUT = ROOT / "outputs" / "sixprotein_teacher_pinn_positional_6h"


def load_supervision(path: Path, n_species: int, dtype: torch.dtype) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    offsets = data["group_member_offsets"].astype(np.int64)
    members = data["group_member_indices"].astype(np.int64)
    n_groups = len(offsets) - 1
    rows: list[int] = []
    cols: list[int] = []
    for group in range(n_groups):
        group_members = members[offsets[group] : offsets[group + 1]]
        rows.extend([group] * len(group_members))
        cols.extend(group_members.tolist())
    group_matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)), shape=(n_groups, n_species)
    )
    return {
        "time_s": data["time_s"].astype(float),
        "direct_names": [str(x) for x in data["direct_names"].tolist()],
        "direct_indices_np": data["direct_indices"].astype(np.int64),
        "direct_target_np": data["direct_target_M"].astype(float),
        "group_names": [str(x) for x in data["group_names"].tolist()],
        "group_target_np": data["group_target_M"].astype(float),
        "group_matrix": scipy_to_torch(group_matrix, dtype),
    }


def target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    log_eps_M: float,
    linear_weight: float,
    row_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_residual = torch.log10(prediction + log_eps_M) - torch.log10(target + log_eps_M)
    log_error = log_residual.square()
    linear_error = ((prediction - target) / scale).square()
    if row_weight is not None:
        log_error = log_error * row_weight[:, None]
        linear_error = linear_error * row_weight[:, None]
    log_loss = torch.mean(log_error)
    linear_loss = torch.mean(linear_error)
    return log_loss + float(linear_weight) * linear_loss, log_loss, linear_loss


def aggregate(group_matrix: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(group_matrix, state.T).T


def finite_difference(values: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
    """Differentiable finite differences on an ordered time grid."""
    if values.shape[0] < 3:
        raise ValueError("At least three time points are required for derivative supervision")
    derivative = torch.empty_like(values)
    derivative[0] = (values[1] - values[0]) / (time_s[1] - time_s[0])
    derivative[-1] = (values[-1] - values[-2]) / (time_s[-1] - time_s[-2])
    denominator = time_s[2:] - time_s[:-2]
    derivative[1:-1] = (values[2:] - values[:-2]) / denominator[:, None]
    return derivative


def error_metrics(prediction: np.ndarray, target: np.ndarray, floor_M: float) -> dict[str, float | int | None]:
    valid = np.abs(target) > float(floor_M)
    error = np.abs(prediction - target)
    return {
        "values": int(target.size),
        "mre_values": int(np.sum(valid)),
        "MRE_percent": (
            float(np.mean(error[valid] / np.abs(target[valid])) * 100.0) if np.any(valid) else None
        ),
        "weighted_relative_error_percent": float(
            np.sum(error) / max(np.sum(np.abs(target)), 1e-30) * 100.0
        ),
        "RMSE_uM": float(np.sqrt(np.mean((prediction - target) ** 2)) * 1e6),
    }


def grouped_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    names: list[str],
    rows: np.ndarray,
    floor_M: float,
) -> dict[str, dict[str, float | int | None]]:
    categories = {
        "transcription_positions": np.asarray(
            ["::tx_pos::" in name or name.endswith("::tx_chain_total") for name in names]
        ),
        "translation_positions": np.asarray(
            ["::tl_pos::" in name or name.endswith("::tl_chain_total") for name in names]
        ),
        "translation_initiation_assembly": np.asarray(
            [name.endswith("::ribosome_mrna_assembly") for name in names]
        ),
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for category, mask in categories.items():
        columns = np.flatnonzero(mask)
        result[category] = error_metrics(
            prediction[np.ix_(rows, columns)], target[np.ix_(rows, columns)], floor_M
        )
    return result


def nrmsr_metrics(
    state: torch.Tensor,
    rhs: torch.Tensor,
    time_s: np.ndarray,
    direct_indices: torch.Tensor,
    group_matrix: torch.Tensor,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    dt_seconds = np.diff(time_s)[:, None]
    dt = torch.tensor(dt_seconds, dtype=state.dtype)
    rhs_integral = 0.5 * dt * (rhs[1:] + rhs[:-1])
    delta = state[1:] - state[:-1]
    direct_residual = (delta - rhs_integral).index_select(1, direct_indices)
    direct_rhs = rhs_integral.index_select(1, direct_indices)
    group_residual = aggregate(group_matrix, delta - rhs_integral)
    group_rhs = aggregate(group_matrix, rhs_integral)
    residual = torch.cat((direct_residual, group_residual), dim=1).detach().cpu().numpy()
    theoretical = torch.cat((direct_rhs, group_rhs), dim=1).detach().cpu().numpy()
    residual_rms = np.sqrt(np.mean(residual**2, axis=0)) * 1e6
    rhs_rms = np.sqrt(np.mean(theoretical**2, axis=0)) * 1e6
    rhs_rate_uM_per_h = theoretical * 1e6 / (dt_seconds / 3600.0)
    rhs_rate_rms_uM_per_h = np.sqrt(np.mean(rhs_rate_uM_per_h**2, axis=0))
    ratio = residual_rms / np.maximum(rhs_rms, 1e-12)
    detail = pd.DataFrame(
        {
            "rhs_integral_rms_uM": rhs_rms,
            "rhs_rate_rms_uM_per_h": rhs_rate_rms_uM_per_h,
            "residual_rms_uM": residual_rms,
            "NRMSR_percent": ratio * 100.0,
        }
    )
    summary: dict[str, float | int | None] = {
        "all_constraints_n": int(len(ratio)),
        "all_constraints_mean_NRMSR_percent": float(np.mean(ratio) * 100.0),
        "all_constraints_global_NRMSR_percent": float(
            np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(theoretical**2)), 1e-30) * 100.0
        ),
    }
    for floor in (0.01, 1.0):
        mask = rhs_rms > floor
        key = str(floor).replace(".", "p")
        summary[f"rhs_integral_gt_{key}_uM_n"] = int(np.sum(mask))
        summary[f"rhs_integral_gt_{key}_uM_mean_NRMSR_percent"] = (
            float(np.mean(ratio[mask]) * 100.0) if np.any(mask) else None
        )
    active_mask = rhs_rate_rms_uM_per_h > 0.004
    summary["rhs_rate_gt_0p004_uM_per_h_n"] = int(np.sum(active_mask))
    summary["rhs_rate_gt_0p004_uM_per_h_mean_NRMSR_percent"] = (
        float(np.mean(ratio[active_mask]) * 100.0) if np.any(active_mask) else None
    )
    summary["rhs_rate_gt_0p004_uM_per_h_global_NRMSR_percent"] = (
        float(
            np.sqrt(np.mean(residual[:, active_mask] ** 2))
            / max(np.sqrt(np.mean(theoretical[:, active_mask] ** 2)), 1e-30)
            * 100.0
        )
        if np.any(active_mask)
        else None
    )
    return detail, summary


def train(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float64
    operator = HybridResolutionOperator(
        args.assets,
        args.pool_json,
        energy_scale=args.energy_scale,
        gamma_h=args.gamma_h,
        include_all_teacher_metabolites=args.include_all_teacher_metabolites,
        dtype=dtype,
    )
    supervision = load_supervision(args.teacher_supervision, operator.n_species, dtype)
    time_np = supervision["time_s"]
    direct_indices_np = supervision["direct_indices_np"]
    direct_target_np = supervision["direct_target_np"]
    group_target_np = supervision["group_target_np"]
    direct_indices = torch.tensor(direct_indices_np, dtype=torch.long)
    direct_target = torch.tensor(direct_target_np, dtype=dtype)
    group_target = torch.tensor(group_target_np, dtype=dtype)
    group_matrix = supervision["group_matrix"]
    time_all = torch.tensor(time_np[:, None], dtype=dtype)
    train_rows_np = np.flatnonzero(time_np <= float(args.teacher_train_max_s) + 1e-9)
    holdout_rows_np = np.flatnonzero(np.isclose(time_np, time_np[-1], rtol=0.0, atol=1e-9))
    if len(holdout_rows_np) != 1 or time_np[holdout_rows_np[0]] <= float(args.teacher_train_max_s):
        raise ValueError("Expected one final-time teacher holdout strictly after the training window")
    unused_rows_np = np.flatnonzero(
        (time_np > float(args.teacher_train_max_s) + 1e-9)
        & (~np.isclose(time_np, time_np[-1], rtol=0.0, atol=1e-9))
    )
    train_rows = torch.tensor(train_rows_np, dtype=torch.long)
    if not np.array_equal(train_rows_np, np.arange(len(train_rows_np))):
        raise ValueError("Derivative supervision expects a contiguous training window starting at t=0")

    # Estimate trajectory slopes strictly inside the 0--5 h training window.
    # In particular, the one-sided derivative at 5 h does not read the unused
    # 5.25--5.75 h rows or the 6 h holdout.
    train_time_np = time_np[train_rows_np]
    direct_derivative_np = np.gradient(
        direct_target_np[train_rows_np], train_time_np, axis=0, edge_order=2
    )
    group_derivative_np = np.gradient(
        group_target_np[train_rows_np], train_time_np, axis=0, edge_order=2
    )

    base_scale = np.load(args.state_scale).astype(float)
    if base_scale.shape != (operator.base_species_count,):
        raise ValueError("Transferred state scale does not match full-elementary species count")
    scale_np = np.full(operator.n_species, float(args.state_scale_floor_M), dtype=float)
    scale_np[: operator.base_species_count] = np.maximum(base_scale, float(args.state_scale_floor_M))
    for column, index in enumerate(direct_indices_np):
        scale_np[index] = max(scale_np[index], float(np.max(direct_target_np[:, column])))
    scale = torch.tensor(scale_np, dtype=dtype)
    direct_scale = torch.tensor(
        np.maximum(np.max(np.abs(direct_target_np), axis=0), float(args.direct_scale_floor_M)),
        dtype=dtype,
    )
    group_scale = torch.tensor(
        np.maximum(np.max(np.abs(group_target_np), axis=0), float(args.group_scale_floor_M)),
        dtype=dtype,
    )
    direct_derivative_target = torch.tensor(direct_derivative_np, dtype=dtype)
    group_derivative_target = torch.tensor(group_derivative_np, dtype=dtype)
    direct_derivative_scale = torch.tensor(
        np.maximum(
            np.max(np.abs(direct_derivative_np), axis=0),
            float(args.direct_derivative_scale_floor_M_per_s),
        ),
        dtype=dtype,
    )
    group_derivative_scale = torch.tensor(
        np.maximum(
            np.max(np.abs(group_derivative_np), axis=0),
            float(args.group_derivative_scale_floor_M_per_s),
        ),
        dtype=dtype,
    )
    derivative_row_weight = torch.ones(len(train_rows_np), dtype=dtype)
    derivative_row_weight[-1] = float(args.terminal_derivative_weight)
    derivative_row_weight /= torch.mean(derivative_row_weight)
    concentration_row_weight = torch.ones(len(train_rows_np), dtype=dtype)
    concentration_row_weight[-1] = float(args.terminal_teacher_weight)
    concentration_row_weight /= torch.mean(concentration_row_weight)
    direct_name_to_column = {
        name: column for column, name in enumerate(supervision["direct_names"])
    }
    moiety_columns = []
    for label, member_names in (
        ("adenylate", ("pool::ATP", "pool::ADP")),
        ("guanylate", ("pool::GTP", "pool::GDP")),
    ):
        columns = [direct_name_to_column[name] for name in member_names]
        target_total = float(np.sum(direct_target_np[0, columns]))
        moiety_columns.append((label, columns, target_total))
    gene_activity_decay_h_inv = {
        "Pgi": 1.6378420809649084,
        "PfkA": 1.5902573893951593,
        "TpiA": 1.6317999584739176,
        "GAPDH": 1.6412709904046225,
        "Pgk": 1.8358385248316258,
        "Eno": 1.5834609468078493,
    }
    product_pairs = []
    for gene in ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno"):
        peptide_name = f"gene::{gene}::released_peptide"
        product_name = f"gene::{gene}::protein"
        if peptide_name not in direct_name_to_column or product_name not in direct_name_to_column:
            continue
        product_pairs.append(
            (
                gene,
                direct_name_to_column[peptide_name],
                direct_name_to_column[product_name],
                gene_activity_decay_h_inv[gene],
            )
        )
    waste_columns = {
        sid: direct_name_to_column[f"pool::{sid}"] for sid in ("PPi", "PO4")
    }
    energy_columns = {
        sid: direct_name_to_column[f"pool::{sid}"] for sid in ("ATP", "ADP", "GTP", "GDP")
    }
    energy_terminal_derivative = {
        sid: float(direct_derivative_np[-1, column])
        for sid, column in energy_columns.items()
    }
    energy_terminal_state = {
        sid: float(direct_target_np[train_rows_np[-1], column])
        for sid, column in energy_columns.items()
    }
    initial_state_np = operator.initial_state_M.detach().cpu().numpy()
    constant_regeneration_gate = 1.0
    for sid, half_saturation in (
        ("Glucose", operator.K_glucose_M),
        ("NAD", operator.K_nad_M),
        ("TPP", operator.K_tpp_M),
        ("Mg", operator.K_mg_M),
    ):
        value = float(initial_state_np[operator.species_index[sid]])
        constant_regeneration_gate *= value / (value + float(half_saturation))
    terminal_regeneration_flux = (
        operator.k_atp_reg_M_s
        * constant_regeneration_gate
        * energy_terminal_state["ADP"]
        / (energy_terminal_state["ADP"] + operator.K_adp_reg_M)
    )
    ndk_rate_M_per_s = 6e-6 * float(args.energy_scale)
    ndk_K_gdp_M = 3e-4
    ndk_K_atp_M = 5e-4
    mg_initial_M = float(initial_state_np[operator.species_index["Mg"]])
    terminal_ndk_flux = (
        ndk_rate_M_per_s
        * energy_terminal_state["GDP"]
        / (energy_terminal_state["GDP"] + ndk_K_gdp_M)
        * energy_terminal_state["ATP"]
        / (energy_terminal_state["ATP"] + ndk_K_atp_M)
        * mg_initial_M
        / (mg_initial_M + operator.K_mg_M)
    )
    terminal_atp_consumption = max(
        0.0,
        energy_terminal_derivative["ADP"]
        + terminal_regeneration_flux
        - terminal_ndk_flux
        + operator.gamma_per_s * energy_terminal_state["ADP"],
    )
    terminal_gtp_consumption = max(
        0.0,
        energy_terminal_derivative["GDP"]
        + terminal_ndk_flux
        + operator.gamma_per_s * energy_terminal_state["GDP"],
    )
    derivative_scale = torch.clamp(
        scale / max(float(time_np[-1]), 1.0), min=float(args.derivative_scale_floor_M_per_s)
    )

    model = HardICPositivePINN(
        x0=operator.initial_state_M,
        scale=scale,
        t_end_s=float(time_np[-1]),
        ic_gate_timescale_s=float(args.ic_gate_timescale_s),
        width=args.width,
        depth=args.depth,
        dtype=dtype,
    )
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_species = [str(sid) for sid in checkpoint.get("species_ids", [])]
        current_species = list(operator.species_ids)
        if checkpoint_species == current_species:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif checkpoint_species and current_species[: len(checkpoint_species)] == checkpoint_species:
            old_state = checkpoint["model_state_dict"]
            new_state = model.state_dict()
            old_count = len(checkpoint_species)
            for key, old_value in old_state.items():
                if key not in new_state:
                    continue
                if old_value.shape == new_state[key].shape:
                    new_state[key] = old_value
                elif old_value.ndim >= 1 and old_value.shape[0] == old_count:
                    expanded = new_state[key].clone()
                    expanded[:old_count] = old_value
                    if key.endswith("net.8.weight") or key.endswith("net.8.bias"):
                        expanded[old_count:] = 0.0
                    new_state[key] = expanded
                else:
                    raise ValueError(f"Cannot expand checkpoint tensor {key}: {old_value.shape}")
            model.load_state_dict(new_state)
        else:
            raise ValueError("Resume checkpoint species ordering does not match the hybrid operator")
    # At t>0 the zero-initialized output branches otherwise start near
    # softplus(0)*1 nM for every microstate.  Summing thousands of such states
    # creates a fictitious micromolar chain occupancy.  A negative output bias
    # initializes those hidden states near zero while preserving an exact x(0).
    if args.resume_checkpoint is None:
        with torch.no_grad():
            final_layer = model.net[-1]
            if isinstance(final_layer, torch.nn.Linear):
                zero_initial = operator.initial_state_M <= 0.0
                final_layer.bias[zero_initial] = float(args.zero_state_initial_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    generator = torch.Generator().manual_seed(args.seed)

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.teacher_derivative_weight > 0.0 or args.teacher_batch >= len(train_rows):
            selected_rows = train_rows
        else:
            choice = torch.randperm(len(train_rows), generator=generator)[: args.teacher_batch]
            selected_rows = train_rows.index_select(0, choice)
        teacher_time = time_all.index_select(0, selected_rows)
        if args.moiety_weight > 0.0 or args.product_balance_weight > 0.0:
            all_state_for_constraints = model(time_all)
            teacher_state = all_state_for_constraints.index_select(0, selected_rows)
        else:
            all_state_for_constraints = None
            teacher_state = model(teacher_time)
        direct_prediction = teacher_state.index_select(1, direct_indices)
        group_prediction = aggregate(group_matrix, teacher_state)
        direct_loss, direct_log, direct_linear = target_loss(
            direct_prediction,
            direct_target.index_select(0, selected_rows),
            direct_scale,
            log_eps_M=args.teacher_log_eps_M,
            linear_weight=args.linear_teacher_weight,
            row_weight=concentration_row_weight.index_select(0, selected_rows),
        )
        group_loss, group_log, group_linear = target_loss(
            group_prediction,
            group_target.index_select(0, selected_rows),
            group_scale,
            log_eps_M=args.teacher_log_eps_M,
            linear_weight=args.linear_teacher_weight,
            row_weight=concentration_row_weight.index_select(0, selected_rows),
        )
        teacher_loss = 0.5 * (direct_loss + group_loss)

        if args.teacher_derivative_weight > 0.0:
            direct_derivative_prediction = finite_difference(
                direct_prediction, teacher_time[:, 0]
            )
            group_derivative_prediction = finite_difference(
                group_prediction, teacher_time[:, 0]
            )
            direct_derivative_residual = (
                direct_derivative_prediction
                - direct_derivative_target.index_select(0, selected_rows)
            ) / direct_derivative_scale
            group_derivative_residual = (
                group_derivative_prediction
                - group_derivative_target.index_select(0, selected_rows)
            ) / group_derivative_scale
            selected_derivative_weight = derivative_row_weight.index_select(0, selected_rows)[:, None]
            direct_derivative_loss = torch.mean(
                torch.nn.functional.smooth_l1_loss(
                    torch.clamp(direct_derivative_residual, min=-100.0, max=100.0),
                    torch.zeros_like(direct_derivative_residual),
                    reduction="none",
                )
                * selected_derivative_weight
            )
            group_derivative_loss = torch.mean(
                torch.nn.functional.smooth_l1_loss(
                    torch.clamp(group_derivative_residual, min=-100.0, max=100.0),
                    torch.zeros_like(group_derivative_residual),
                    reduction="none",
                )
                * selected_derivative_weight
            )
            teacher_derivative_loss = 0.5 * (
                direct_derivative_loss + group_derivative_loss
            )
        else:
            direct_derivative_loss = teacher_loss.new_zeros(())
            group_derivative_loss = teacher_loss.new_zeros(())
            teacher_derivative_loss = teacher_loss.new_zeros(())

        if all_state_for_constraints is not None:
            all_direct_prediction = all_state_for_constraints.index_select(1, direct_indices)
        else:
            all_direct_prediction = None
        direct_derivative_all = None
        if args.moiety_weight > 0.0 and all_direct_prediction is not None:
            moiety_terms = []
            moiety_row_weight = torch.ones(len(time_all), dtype=dtype)
            moiety_row_weight[time_np > float(args.teacher_train_max_s) + 1e-9] = float(
                args.unlabeled_physics_weight
            )
            moiety_row_weight /= torch.mean(moiety_row_weight)
            for _label, columns, target_total in moiety_columns:
                predicted_total = torch.sum(all_direct_prediction[:, columns], dim=1)
                moiety_terms.append(
                    torch.mean(
                        ((predicted_total - target_total) / target_total).square()
                        * moiety_row_weight
                    )
                )
            moiety_loss = torch.mean(torch.stack(moiety_terms))
        else:
            moiety_loss = teacher_loss.new_zeros(())
        if args.product_balance_weight > 0.0 and all_direct_prediction is not None:
            if not product_pairs:
                raise ValueError("Product-balance loss requires teacher released-peptide targets")
            product_terms = []
            full_time = time_all[:, 0]
            future_weight = torch.ones(len(time_all), dtype=dtype)
            future_weight[time_np > float(args.teacher_train_max_s) + 1e-9] = float(
                args.unlabeled_physics_weight
            )
            future_weight /= torch.mean(future_weight)
            fold_rate_per_s = float(args.product_fold_rate_per_s)
            dilution_rate_per_s = float(args.gamma_h) / 3600.0
            for _gene, peptide_column, product_column, _decay_h_inv in product_pairs:
                peptide = all_direct_prediction[:, peptide_column]
                product = all_direct_prediction[:, product_column]
                product_derivative = finite_difference(product[:, None], full_time)[:, 0]
                balance_rhs = fold_rate_per_s * peptide - dilution_rate_per_s * product
                product_scale = max(
                    float(np.max(np.abs(direct_target_np[:, product_column])))
                    / max(float(time_np[-1]), 1.0),
                    float(args.product_balance_scale_floor_M_per_s),
                )
                product_terms.append(
                    torch.mean(
                        torch.nn.functional.smooth_l1_loss(
                            (product_derivative - balance_rhs) / product_scale,
                            torch.zeros_like(product_derivative),
                            reduction="none",
                        )
                        * future_weight
                    )
                )
            product_balance_loss = torch.mean(torch.stack(product_terms))
        else:
            product_balance_loss = teacher_loss.new_zeros(())
        if args.energy_continuation_weight > 0.0 and all_direct_prediction is not None:
            direct_derivative_all = finite_difference(all_direct_prediction, time_all[:, 0])
            continuation_mask = time_np >= float(args.teacher_train_max_s) - 1e-9
            continuation_rows = torch.tensor(np.flatnonzero(continuation_mask), dtype=torch.long)
            continuation_time = time_all.index_select(0, continuation_rows)[:, 0]
            elapsed_h = (continuation_time - float(args.teacher_train_max_s)) / 3600.0
            consumption_decay = torch.exp(-float(args.activity_decay_h_inv) * elapsed_h)
            atp = all_direct_prediction.index_select(0, continuation_rows)[
                :, energy_columns["ATP"]
            ]
            adp = all_direct_prediction.index_select(0, continuation_rows)[
                :, energy_columns["ADP"]
            ]
            gtp = all_direct_prediction.index_select(0, continuation_rows)[
                :, energy_columns["GTP"]
            ]
            gdp = all_direct_prediction.index_select(0, continuation_rows)[
                :, energy_columns["GDP"]
            ]
            regeneration_flux = (
                float(operator.k_atp_reg_M_s)
                * float(constant_regeneration_gate)
                * adp
                / (adp + float(operator.K_adp_reg_M) + 1e-30)
            )
            ndk_flux = (
                float(ndk_rate_M_per_s)
                * gdp
                / (gdp + float(ndk_K_gdp_M) + 1e-30)
                * atp
                / (atp + float(ndk_K_atp_M) + 1e-30)
                * float(mg_initial_M)
                / (float(mg_initial_M) + float(operator.K_mg_M))
            )
            atp_consumption = float(terminal_atp_consumption) * consumption_decay
            gtp_consumption = float(terminal_gtp_consumption) * consumption_decay
            energy_rhs = torch.stack(
                (
                    -atp_consumption
                    + regeneration_flux
                    - ndk_flux
                    + float(operator.gamma_per_s)
                    * (float(operator.feed_target_M[operator.species_index["ATP"]]) - atp),
                    atp_consumption
                    - regeneration_flux
                    + ndk_flux
                    - float(operator.gamma_per_s) * adp,
                    -gtp_consumption
                    + ndk_flux
                    + float(operator.gamma_per_s)
                    * (float(operator.feed_target_M[operator.species_index["GTP"]]) - gtp),
                    gtp_consumption - ndk_flux - float(operator.gamma_per_s) * gdp,
                ),
                dim=1,
            )
            energy_prediction_derivative = torch.stack(
                tuple(
                    direct_derivative_all.index_select(0, continuation_rows)[:, energy_columns[sid]]
                    for sid in ("ATP", "ADP", "GTP", "GDP")
                ),
                dim=1,
            )
            energy_scale = torch.tensor(
                [
                    max(abs(energy_terminal_derivative[sid]), float(args.energy_scale_floor_M_per_s))
                    for sid in ("ATP", "ADP", "GTP", "GDP")
                ],
                dtype=dtype,
            )
            energy_continuation_loss = torch.nn.functional.smooth_l1_loss(
                (energy_prediction_derivative - energy_rhs) / energy_scale,
                torch.zeros_like(energy_prediction_derivative),
            )
        else:
            energy_continuation_loss = teacher_loss.new_zeros(())
        if args.continuation_balance_weight > 0.0 and all_direct_prediction is not None:
            if not product_pairs:
                raise ValueError("Continuation-balance loss requires teacher released-peptide targets")
            if direct_derivative_all is None:
                direct_derivative_all = finite_difference(all_direct_prediction, time_all[:, 0])
            continuation_mask = time_np >= float(args.teacher_train_max_s) - 1e-9
            continuation_rows = torch.tensor(np.flatnonzero(continuation_mask), dtype=torch.long)
            continuation_time = time_all.index_select(0, continuation_rows)[:, 0]
            elapsed_h = (continuation_time - float(args.teacher_train_max_s)) / 3600.0
            continuation_terms = []
            fold_rate_per_s = float(args.product_fold_rate_per_s)
            for _gene, peptide_column, _product_column, decay_h_inv in product_pairs:
                terminal_peptide = float(
                    direct_target_np[train_rows_np[-1], peptide_column]
                )
                terminal_formation = max(
                    0.0,
                    float(direct_derivative_np[-1, peptide_column])
                    + fold_rate_per_s * terminal_peptide,
                )
                peptide = all_direct_prediction.index_select(0, continuation_rows)[
                    :, peptide_column
                ]
                peptide_rhs = (
                    terminal_formation * torch.exp(-float(decay_h_inv) * elapsed_h)
                    - fold_rate_per_s * peptide
                )
                peptide_derivative = direct_derivative_all.index_select(0, continuation_rows)[
                    :, peptide_column
                ]
                peptide_scale = max(
                    abs(float(direct_derivative_np[-1, peptide_column])),
                    float(args.continuation_scale_floor_M_per_s),
                )
                continuation_terms.append(
                    torch.nn.functional.smooth_l1_loss(
                        (peptide_derivative - peptide_rhs) / peptide_scale,
                        torch.zeros_like(peptide_derivative),
                    )
                )
            for _sid, column in waste_columns.items():
                terminal_state = float(direct_target_np[train_rows_np[-1], column])
                terminal_generation = max(
                    0.0,
                    float(direct_derivative_np[-1, column])
                    + float(operator.gamma_per_s) * terminal_state,
                )
                state_now = all_direct_prediction.index_select(0, continuation_rows)[:, column]
                rhs_now = (
                    terminal_generation
                    * torch.exp(-float(args.activity_decay_h_inv) * elapsed_h)
                    - float(operator.gamma_per_s) * state_now
                )
                derivative_now = direct_derivative_all.index_select(0, continuation_rows)[:, column]
                scale_now = max(
                    abs(float(direct_derivative_np[-1, column])),
                    float(args.continuation_scale_floor_M_per_s),
                )
                continuation_terms.append(
                    torch.nn.functional.smooth_l1_loss(
                        (derivative_now - rhs_now) / scale_now,
                        torch.zeros_like(derivative_now),
                    )
                )
            continuation_balance_loss = torch.mean(torch.stack(continuation_terms))
        else:
            continuation_balance_loss = teacher_loss.new_zeros(())

        if step < args.teacher_warmup_steps:
            physics_weight = float(args.physics_weight_start)
        else:
            ramp = min(
                1.0,
                (step - args.teacher_warmup_steps + 1) / max(1, args.physics_ramp_steps),
            )
            physics_weight = float(args.physics_weight_start) + ramp * (
                float(args.physics_weight) - float(args.physics_weight_start)
            )
        if physics_weight > 0.0:
            collocation_time = (
                torch.rand((args.physics_batch, 1), generator=generator, dtype=dtype)
                * (float(time_np[-1]) - float(args.physics_min_time_s))
                + float(args.physics_min_time_s)
            )
            physics_state, physics_derivative = time_derivative(model, collocation_time)
            physics_rhs = operator(physics_state)
            raw_physics_residual = physics_derivative - physics_rhs
            if args.physics_loss_mode == "relative_nrmsr":
                # Match the reported relative residual instead of allowing a
                # few very stiff, high-rate states to determine the loss scale.
                # The detached denominator prevents the model from reducing
                # the objective merely by inflating its own reaction rates.
                relative_scale = torch.maximum(
                    torch.abs(physics_rhs.detach()),
                    torch.abs(physics_derivative.detach()),
                )
                relative_scale = torch.clamp(
                    relative_scale,
                    min=float(args.physics_relative_floor_M_per_s),
                )
                scaled_residual = raw_physics_residual / relative_scale
                element_loss = torch.nn.functional.smooth_l1_loss(
                    scaled_residual,
                    torch.zeros_like(scaled_residual),
                    beta=1.0,
                    reduction="none",
                )
                hard_fraction = min(max(float(args.physics_hard_fraction), 0.0), 1.0)
                flat_loss = element_loss.reshape(-1)
                if hard_fraction < 0.999:
                    hard_count = max(1, int(np.ceil(hard_fraction * flat_loss.numel())))
                    physics_loss = torch.mean(
                        torch.topk(flat_loss, k=hard_count, largest=True).values
                    )
                else:
                    physics_loss = torch.mean(flat_loss)
            else:
                scaled_residual = raw_physics_residual / derivative_scale
                physics_loss = torch.nn.functional.smooth_l1_loss(
                    scaled_residual, torch.zeros_like(scaled_residual), beta=1.0
                )
        else:
            physics_loss = teacher_loss.new_zeros(())
        loss = (
            float(args.teacher_weight) * teacher_loss
            + float(args.teacher_derivative_weight) * teacher_derivative_loss
            + float(args.moiety_weight) * moiety_loss
            + float(args.product_balance_weight) * product_balance_loss
            + float(args.energy_continuation_weight) * energy_continuation_loss
            + float(args.continuation_balance_weight) * continuation_balance_loss
            + physics_weight * physics_loss
        )
        if step < args.steps:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            gradient_norm = torch.tensor(float("nan"))
        if step in {0, 1, 2, 5, 10, args.steps} or step % max(1, args.report_every) == 0:
            row = {
                "step": int(step),
                "loss": float(loss.detach()),
                "teacher_loss": float(teacher_loss.detach()),
                "direct_log_loss": float(direct_log.detach()),
                "direct_linear_loss": float(direct_linear.detach()),
                "group_log_loss": float(group_log.detach()),
                "group_linear_loss": float(group_linear.detach()),
                "teacher_derivative_loss": float(teacher_derivative_loss.detach()),
                "direct_derivative_loss": float(direct_derivative_loss.detach()),
                "group_derivative_loss": float(group_derivative_loss.detach()),
                "moiety_loss": float(moiety_loss.detach()),
                "product_balance_loss": float(product_balance_loss.detach()),
                "energy_continuation_loss": float(energy_continuation_loss.detach()),
                "continuation_balance_loss": float(continuation_balance_loss.detach()),
                "physics_loss": float(physics_loss.detach()),
                "physics_weight": float(physics_weight),
                "gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    with torch.no_grad():
        state = model(time_all)
        direct_prediction = state.index_select(1, direct_indices).cpu().numpy()
        group_prediction = aggregate(group_matrix, state).cpu().numpy()
    prediction_path = args.output_dir / "prior_prediction_6h.npz"
    np.savez_compressed(
        prediction_path,
        time_s=time_np,
        species_id=np.asarray(operator.species_ids, dtype=str),
        prediction_M=state.cpu().numpy(),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "configuration": vars(args),
            "species_ids": operator.species_ids,
        },
        args.output_dir / "prior_pinn_checkpoint.pt",
    )

    train_direct = error_metrics(
        direct_prediction[train_rows_np], direct_target_np[train_rows_np], args.mre_floor_M
    )
    train_group = error_metrics(
        group_prediction[train_rows_np], group_target_np[train_rows_np], args.mre_floor_M
    )
    holdout_direct = error_metrics(
        direct_prediction[holdout_rows_np], direct_target_np[holdout_rows_np], args.mre_floor_M
    )
    holdout_group = error_metrics(
        group_prediction[holdout_rows_np], group_target_np[holdout_rows_np], args.mre_floor_M
    )
    train_group_categories = grouped_error_metrics(
        group_prediction,
        group_target_np,
        supervision["group_names"],
        train_rows_np,
        args.mre_floor_M,
    )
    holdout_group_categories = grouped_error_metrics(
        group_prediction,
        group_target_np,
        supervision["group_names"],
        holdout_rows_np,
        args.mre_floor_M,
    )
    with torch.no_grad():
        rhs = operator(state)
    detail, nrmsr = nrmsr_metrics(state, rhs, time_np, direct_indices, group_matrix)
    detail.insert(
        0,
        "constraint",
        supervision["direct_names"] + supervision["group_names"],
    )
    detail.to_csv(args.output_dir / "prior_nrmsr_by_constraint.csv", index=False)

    report = {
        "status": "smoke_test" if args.steps <= 5 else "prior_training_completed",
        "method": "teacher-supervised full-state PINN followed by full mass-action physics evaluation",
        "teacher_train_window_h": [0.0, float(args.teacher_train_max_s / 3600.0)],
        "teacher_holdout_window_h": (
            [float(time_np[holdout_rows_np[0]] / 3600.0), float(time_np[-1] / 3600.0)]
            if len(holdout_rows_np)
            else None
        ),
        "teacher_unused_gap_times_h": (time_np[unused_rows_np] / 3600.0).tolist(),
        "full_states": operator.n_species,
        "elementary_reactions": operator.n_elementary_reactions,
        "teacher_effective_reactions": operator.n_teacher_reactions,
        "direct_teacher_constraints": len(supervision["direct_names"]),
        "aggregate_teacher_constraints": len(supervision["group_names"]),
        "hidden_elementary_state_targets_invented": False,
        "teacher_derivative_supervision": {
            "weight": float(args.teacher_derivative_weight),
            "source_window_h": [0.0, float(args.teacher_train_max_s / 3600.0)],
            "uses_unused_gap_or_holdout": False,
            "terminal_training_concentration_weight": float(args.terminal_teacher_weight),
            "terminal_training_slope_weight": float(args.terminal_derivative_weight),
        },
        "unlabeled_physics_constraints": {
            "moiety_weight": float(args.moiety_weight),
            "moiety_targets_from": "0 h initial ATP+ADP and GTP+GDP totals",
            "product_balance_weight": float(args.product_balance_weight),
            "product_balance": "dProtein/dt = k_fold*released_peptide - gamma*Protein",
            "energy_continuation_weight": float(args.energy_continuation_weight),
            "energy_continuation": (
                "5 h ATP/GTP consumption extrapolated with the calibrated activity decay; "
                "ATP regeneration and NDK recomputed from predicted pools"
            ),
            "terminal_atp_consumption_M_per_s": float(terminal_atp_consumption),
            "terminal_gtp_consumption_M_per_s": float(terminal_gtp_consumption),
            "continuation_balance_weight": float(args.continuation_balance_weight),
            "continuation_balance": (
                "5 h peptide/PPi/PO4 generation extrapolated with calibrated activity decay; "
                "folding and dilution recomputed from predicted states"
            ),
            "uses_unused_gap_or_holdout_concentrations": False,
        },
        "initial_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint),
        "runtime_s": float(time.time() - started),
        "scale_source": str(args.state_scale),
        "train_metrics": {
            "direct": train_direct,
            "aggregate": train_group,
            **train_group_categories,
        },
        "holdout_metrics": {
            "direct": holdout_direct,
            "aggregate": holdout_group,
            **holdout_group_categories,
        },
        "physical_integral_metrics": nrmsr,
        "prediction": str(prediction_path),
        "history": history,
    }
    (args.output_dir / "prior_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--teacher-supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--state-scale", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--include-all-teacher-metabolites",
        action="store_true",
        help="Append the teacher AA bookkeeping pool so every teacher metabolite is comparable.",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--teacher-train-max-s", type=float, default=19800.0)
    parser.add_argument("--teacher-batch", type=int, default=21)
    parser.add_argument("--physics-batch", type=int, default=2)
    parser.add_argument("--physics-min-time-s", type=float, default=60.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-derivative-weight", type=float, default=0.0)
    parser.add_argument("--terminal-teacher-weight", type=float, default=1.0)
    parser.add_argument("--terminal-derivative-weight", type=float, default=5.0)
    parser.add_argument("--moiety-weight", type=float, default=0.0)
    parser.add_argument("--product-balance-weight", type=float, default=0.0)
    parser.add_argument("--energy-continuation-weight", type=float, default=0.0)
    parser.add_argument("--continuation-balance-weight", type=float, default=0.0)
    parser.add_argument("--unlabeled-physics-weight", type=float, default=5.0)
    parser.add_argument("--product-fold-rate-per-s", type=float, default=0.001667)
    parser.add_argument("--product-balance-scale-floor-M-per-s", type=float, default=1e-14)
    parser.add_argument("--activity-decay-h-inv", type=float, default=1.6534115659796805)
    parser.add_argument("--energy-scale-floor-M-per-s", type=float, default=1e-12)
    parser.add_argument("--continuation-scale-floor-M-per-s", type=float, default=1e-16)
    parser.add_argument("--physics-weight", type=float, default=1e-8)
    parser.add_argument("--physics-weight-start", type=float, default=0.0)
    parser.add_argument(
        "--physics-loss-mode",
        choices=("legacy", "relative_nrmsr"),
        default="legacy",
    )
    parser.add_argument(
        "--physics-relative-floor-M-per-s",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--physics-hard-fraction",
        type=float,
        default=1.0,
        help="Fraction of largest per-state relative physics losses retained.",
    )
    parser.add_argument("--teacher-warmup-steps", type=int, default=500)
    parser.add_argument("--physics-ramp-steps", type=int, default=2500)
    parser.add_argument("--linear-teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-log-eps-M", type=float, default=1e-12)
    parser.add_argument("--direct-scale-floor-M", type=float, default=1e-12)
    parser.add_argument("--group-scale-floor-M", type=float, default=1e-15)
    parser.add_argument("--state-scale-floor-M", type=float, default=1e-9)
    parser.add_argument("--derivative-scale-floor-M-per-s", type=float, default=1e-15)
    parser.add_argument("--direct-derivative-scale-floor-M-per-s", type=float, default=1e-15)
    parser.add_argument("--group-derivative-scale-floor-M-per-s", type=float, default=1e-18)
    parser.add_argument("--ic-gate-timescale-s", type=float, default=300.0)
    parser.add_argument("--zero-state-initial-bias", type=float, default=-12.0)
    parser.add_argument("--energy-scale", type=float, default=0.8)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--mre-floor-M", type=float, default=1e-15)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
