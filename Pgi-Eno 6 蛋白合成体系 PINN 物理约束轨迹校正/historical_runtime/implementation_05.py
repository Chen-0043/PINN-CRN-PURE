#!/usr/bin/env python3
"""Second-stage full-state physical correction for the six-protein PINN.

The loss mirrors the deGFP node-shooting stage: teacher concentration loss,
trapezoidal full-operator integral consistency, an enhanced key-constraint
integral loss, and proximity to the prior PINN trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_resolution_pinn import DEFAULT_ASSETS, DEFAULT_POOL, HybridResolutionOperator
from train_teacher_supervised_sixprotein_pinn import (
    aggregate,
    error_metrics,
    grouped_error_metrics,
    load_supervision,
    nrmsr_metrics,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PRIOR = (
    ROOT / "outputs" / "sixprotein_teacher_pinn_positional_v1" / "prior_prediction_6h.npz"
)
DEFAULT_SUPERVISION = (
    DEFAULT_ASSETS / "sixprotein_teacher_supervision_positional_6h_converged.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "sixprotein_teacher_pinn_positional_corrected_v1"


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=1e-30)
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value) + 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--teacher-supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--refined-prior",
        type=Path,
        default=None,
        help=(
            "Optional full-state trajectory on the physics-substep grid used to "
            "initialize hidden fast-state paths; coarse teacher targets remain unchanged."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--teacher-train-max-s", type=float, default=18000.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--integral-weight", type=float, default=1.0)
    parser.add_argument("--key-integral-weight", type=float, default=2.0)
    parser.add_argument("--active-integral-weight", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-nrmsr-weight",
        type=float,
        default=0.0,
        help="Article-aligned mean per-trajectory NRMSR loss on source-active teacher constraints.",
    )
    parser.add_argument(
        "--current-trajectory-nrmsr-weight",
        type=float,
        default=0.0,
        help=(
            "Mean per-trajectory NRMSR evaluated with the current corrected RHS, "
            "matching the reported article metric; the active trajectory set is "
            "fixed from the input trajectory."
        ),
    )
    parser.add_argument(
        "--hidden-trajectory-nrmsr-weight",
        type=float,
        default=0.0,
        help="Article-aligned mean per-trajectory NRMSR loss on source-active hidden states.",
    )
    parser.add_argument("--comparable-rhs-activity-weight", type=float, default=0.0)
    parser.add_argument("--comparable-rhs-activity-fraction", type=float, default=0.5)
    parser.add_argument(
        "--active-rhs-rate-uM-per-h",
        type=float,
        default=0.004,
        help=(
            "Fixed source-trajectory activity threshold for the enhanced comparable-"
            "trajectory integral loss."
        ),
    )
    parser.add_argument("--prior-proximity-weight", type=float, default=0.005)
    parser.add_argument("--rhs-floor-uM", type=float, default=0.01)
    parser.add_argument("--hidden-integral-weight", type=float, default=0.0)
    parser.add_argument("--hidden-current-integral-weight", type=float, default=0.0)
    parser.add_argument("--hidden-rhs-activity-weight", type=float, default=0.0)
    parser.add_argument("--hidden-rhs-activity-fraction", type=float, default=0.5)
    parser.add_argument("--hidden-rhs-floor-uM", type=float, default=0.001)
    parser.add_argument(
        "--hidden-species-mode",
        choices=("uncovered", "all_non_direct"),
        default="uncovered",
    )
    parser.add_argument("--hidden-include-regex", default=None)
    parser.add_argument("--freeze-nonhidden-state", action="store_true")
    parser.add_argument(
        "--preserve-source-group-totals",
        action="store_true",
        help=(
            "Hard-project aggregate-covered microstates so every teacher group "
            "retains its source trajectory total while hidden states redistribute "
            "within the group. This prevents hidden-state correction from trading "
            "aggregate concentration accuracy for a lower residual."
        ),
    )
    parser.add_argument(
        "--hidden-state-parameterization",
        choices=("softplus", "scaled_clamp", "source_delta"),
        default="softplus",
        help=(
            "Use the shared softplus variables or independent nonnegative hidden-state "
            "variables expressed relative to a fixed concentration scale, or bounded "
            "source-relative corrections scaled by each hidden state's source RHS."
        ),
    )
    parser.add_argument("--hidden-state-scale-M", type=float, default=1e-6)
    parser.add_argument("--hidden-delta-scale-multiplier", type=float, default=2.0)
    parser.add_argument("--hidden-lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-active-rhs-rate-uM-per-h",
        type=float,
        default=0.004,
        help="Fixed source-trajectory activity threshold for hidden-state physics loss.",
    )
    parser.add_argument(
        "--optimization-initial-floor-M",
        type=float,
        default=1e-30,
        help=(
            "Positive numerical seed used only to initialize shooting variables. "
            "It prevents zero-valued hidden complexes from receiving a vanishing "
            "softplus gradient; it is not a concentration constraint."
        ),
    )
    parser.add_argument(
        "--optimization-seed-scope",
        choices=("all", "unconstrained"),
        default="all",
        help=(
            "Apply the numerical seed to every state or only to full-CRN states "
            "that are absent from both direct and aggregate teacher constraints."
        ),
    )
    parser.add_argument(
        "--initial-state-mode",
        choices=("fixed", "optimize_all"),
        default="optimize_all",
        help=(
            "Keep the full 0 h state fixed, or optimize every time node as in the "
            "deGFP node-shooting correction. Teacher losses still constrain the "
            "observed 0 h concentrations in optimize_all mode."
        ),
    )
    parser.add_argument(
        "--integral-loss",
        choices=("mse", "smooth_l1"),
        default="mse",
        help="Robust smooth_l1 avoids domination by a few ultra-stiff residuals.",
    )
    parser.add_argument("--log-eps-M", type=float, default=1e-12)
    parser.add_argument("--mre-floor-M", type=float, default=1e-15)
    parser.add_argument("--energy-scale", type=float, default=0.8)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument(
        "--include-all-teacher-metabolites",
        action="store_true",
        help="Use the hybrid state space that retains every teacher metabolite.",
    )
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=1,
        help="Refine every teacher interval with trainable physical shooting nodes.",
    )
    args = parser.parse_args()

    def make_optimizer(parameters, lr: float | None = None):
        optimizer_type = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
        if lr is None:
            return optimizer_type(parameters)
        return optimizer_type(parameters, lr=lr)

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
    prior_data = np.load(args.prior, allow_pickle=True)
    teacher_time_np = prior_data["time_s"].astype(float)
    teacher_prior_np = np.maximum(prior_data["prediction_M"].astype(float), 0.0)
    if teacher_prior_np.shape != (len(teacher_time_np), operator.n_species):
        raise ValueError("Prior trajectory dimensions do not match the hybrid operator")
    if not np.allclose(teacher_time_np, supervision["time_s"]):
        raise ValueError("Prior and teacher time grids differ")
    if args.physics_substeps < 1:
        raise ValueError("--physics-substeps must be at least 1")
    coarse_indices_np = np.arange(len(teacher_time_np), dtype=np.int64) * args.physics_substeps
    if args.physics_substeps == 1:
        time_np = teacher_time_np.copy()
        prior_np = teacher_prior_np.copy()
    else:
        time_np = np.linspace(
            teacher_time_np[0], teacher_time_np[-1],
            (len(teacher_time_np) - 1) * args.physics_substeps + 1,
        )
        interval = np.minimum(
            np.arange(len(time_np), dtype=np.int64) // args.physics_substeps,
            len(teacher_time_np) - 2,
        )
        fraction = (time_np - teacher_time_np[interval]) / (
            teacher_time_np[interval + 1] - teacher_time_np[interval]
        )
        prior_np = (
            (1.0 - fraction[:, None]) * teacher_prior_np[interval]
            + fraction[:, None] * teacher_prior_np[interval + 1]
        )
        prior_np[-1] = teacher_prior_np[-1]

    if args.refined_prior is not None:
        refined_data = np.load(args.refined_prior, allow_pickle=True)
        refined_time = refined_data["time_s"].astype(float)
        refined_state = np.maximum(refined_data["prediction_M"].astype(float), 0.0)
        if refined_state.shape != (len(time_np), operator.n_species):
            raise ValueError("Refined prior dimensions do not match the physics grid/operator")
        if not np.allclose(refined_time, time_np, rtol=0.0, atol=1e-8):
            raise ValueError("Refined prior time grid does not match --physics-substeps")
        prior_np = refined_state

    direct_indices = torch.tensor(supervision["direct_indices_np"], dtype=torch.long)
    group_matrix = supervision["group_matrix"]
    direct_target = torch.tensor(supervision["direct_target_np"], dtype=dtype)
    group_target = torch.tensor(supervision["group_target_np"], dtype=dtype)
    supervised_indices = np.unique(
        np.concatenate(
            (
                supervision["direct_indices_np"],
                group_matrix.coalesce().indices()[1].cpu().numpy(),
            )
        )
    )
    unconstrained_mask = np.ones(operator.n_species, dtype=bool)
    unconstrained_mask[supervised_indices] = False
    if args.hidden_species_mode == "all_non_direct":
        hidden_mask = np.ones(operator.n_species, dtype=bool)
        hidden_mask[supervision["direct_indices_np"]] = False
    else:
        hidden_mask = unconstrained_mask.copy()
    if args.hidden_include_regex:
        hidden_pattern = re.compile(args.hidden_include_regex)
        hidden_mask &= np.asarray(
            [bool(hidden_pattern.search(species_id)) for species_id in operator.species_ids],
            dtype=bool,
        )
    hidden_indices_np = np.flatnonzero(hidden_mask)
    hidden_indices = torch.tensor(hidden_indices_np, dtype=torch.long)
    group_member_indices_np = np.load(
        args.teacher_supervision, allow_pickle=True
    )["group_member_indices"].astype(np.int64)
    group_member_offsets_np = np.load(
        args.teacher_supervision, allow_pickle=True
    )["group_member_offsets"].astype(np.int64)
    group_member_group_np = np.repeat(
        np.arange(len(group_member_offsets_np) - 1, dtype=np.int64),
        np.diff(group_member_offsets_np),
    )
    if len(np.unique(group_member_indices_np)) != len(group_member_indices_np):
        raise ValueError(
            "Hard group-total projection requires non-overlapping aggregate groups"
        )
    group_member_indices = torch.tensor(group_member_indices_np, dtype=torch.long)
    group_member_group = torch.tensor(group_member_group_np, dtype=torch.long)
    teacher_mask_np = teacher_time_np <= float(args.teacher_train_max_s) + 1e-9
    teacher_mask = torch.tensor(coarse_indices_np[teacher_mask_np], dtype=torch.long)
    holdout_mask_np = np.isclose(teacher_time_np, teacher_time_np[-1], rtol=0.0, atol=1e-9)
    unused_gap_mask_np = (~teacher_mask_np) & (~holdout_mask_np)
    prior = torch.tensor(prior_np, dtype=dtype)
    source_group_totals = aggregate(group_matrix, prior)
    fixed_initial = operator.initial_state_M.unsqueeze(0)
    optimization_seed = prior_np.copy()
    if args.optimization_seed_scope == "unconstrained":
        optimization_seed[:, unconstrained_mask] = np.maximum(
            optimization_seed[:, unconstrained_mask],
            float(args.optimization_initial_floor_M),
        )
    else:
        optimization_seed = np.maximum(
            optimization_seed, float(args.optimization_initial_floor_M)
        )
    if args.initial_state_mode == "optimize_all":
        optimizable_initial = torch.tensor(optimization_seed, dtype=dtype)
    else:
        optimizable_initial = torch.tensor(optimization_seed[1:], dtype=dtype)
    y = torch.nn.Parameter(inverse_softplus(optimizable_initial))
    hidden_y: torch.nn.Parameter | None = None
    if args.hidden_state_parameterization == "scaled_clamp":
        hidden_seed_np = np.maximum(
            prior_np[:, hidden_indices_np],
            float(args.optimization_initial_floor_M),
        )
        hidden_seed = torch.tensor(hidden_seed_np, dtype=dtype)
        hidden_y = torch.nn.Parameter(
            hidden_seed / float(args.hidden_state_scale_M)
        )
        if args.freeze_nonhidden_state:
            optimizer = make_optimizer([hidden_y], lr=float(args.hidden_lr))
            optimized_parameters = [hidden_y]
        else:
            optimizer = make_optimizer(
                [
                    {"params": [y], "lr": float(args.lr)},
                    {"params": [hidden_y], "lr": float(args.hidden_lr)},
                ]
            )
            optimized_parameters = [y, hidden_y]
    elif args.hidden_state_parameterization == "source_delta":
        hidden_y = torch.nn.Parameter(
            torch.zeros((prior_np.shape[0], len(hidden_indices_np)), dtype=dtype)
        )
        if args.freeze_nonhidden_state:
            optimizer = make_optimizer([hidden_y], lr=float(args.hidden_lr))
            optimized_parameters = [hidden_y]
        else:
            optimizer = make_optimizer(
                [
                    {"params": [y], "lr": float(args.lr)},
                    {"params": [hidden_y], "lr": float(args.hidden_lr)},
                ]
            )
            optimized_parameters = [y, hidden_y]
    else:
        if args.freeze_nonhidden_state:
            raise ValueError(
                "--freeze-nonhidden-state requires a hidden-state parameterization"
            )
        optimizer = make_optimizer([y], lr=args.lr)
        optimized_parameters = [y]
    time = torch.tensor(time_np[:, None], dtype=dtype)
    dt = time[1:] - time[:-1]
    with torch.no_grad():
        source_rhs = operator(prior)
        source_rhs_integral = 0.5 * dt * (source_rhs[1:] + source_rhs[:-1])
        source_hidden_rhs_integral = source_rhs_integral.index_select(1, hidden_indices)
        source_hidden_scale_raw = torch.sqrt(
            torch.mean(source_hidden_rhs_integral.square(), dim=0)
        )
        source_hidden_scale = torch.clamp(
            source_hidden_scale_raw,
            min=float(args.hidden_rhs_floor_uM) * 1e-6,
        )
        interval_hours = (dt[:, 0] / 3600.0).detach()
        source_hidden_rate_rms = torch.sqrt(
            torch.mean(
                (
                    source_hidden_rhs_integral
                    * 1e6
                    / interval_hours[:, None]
                ).square(),
                dim=0,
            )
        )
        source_hidden_active_mask = (
            source_hidden_rate_rms
            > float(args.hidden_active_rhs_rate_uM_per_h)
        )
        source_direct_rhs_integral = source_rhs_integral.index_select(
            1, direct_indices
        )
        source_group_rhs_integral = aggregate(group_matrix, source_rhs_integral)
        source_direct_exact_scale = torch.clamp(
            torch.sqrt(torch.mean(source_direct_rhs_integral.square(), dim=0)),
            min=1e-30,
        )
        source_group_exact_scale = torch.clamp(
            torch.sqrt(torch.mean(source_group_rhs_integral.square(), dim=0)),
            min=1e-30,
        )
        source_direct_rate_rms = torch.sqrt(
            torch.mean(
                (
                    source_direct_rhs_integral
                    * 1e6
                    / interval_hours[:, None]
                ).square(),
                dim=0,
            )
        )
        source_group_rate_rms = torch.sqrt(
            torch.mean(
                (
                    source_group_rhs_integral
                    * 1e6
                    / interval_hours[:, None]
                ).square(),
                dim=0,
            )
        )
        source_direct_active_mask = (
            source_direct_rate_rms > float(args.active_rhs_rate_uM_per_h)
        )
        source_group_active_mask = (
            source_group_rate_rms > float(args.active_rhs_rate_uM_per_h)
        )

    direct_names = supervision["direct_names"]
    group_names = supervision["group_names"]
    key_pool_names = {
        "pool::ATP",
        "pool::ADP",
        "pool::GTP",
        "pool::GDP",
        "pool::PPi",
        "pool::PO4",
        "pool::Ribosome",
        "pool::IF2",
        "pool::IF2_GTP",
        "pool::EFTu",
        "pool::EFTu_GTP",
        "pool::EFG",
        "pool::EFG_GTP",
    }
    key_direct_columns = torch.tensor(
        [
            index
            for index, name in enumerate(direct_names)
            if name in key_pool_names
            or name.endswith("::mrna_product")
            or name.endswith("::protein")
        ],
        dtype=torch.long,
    )
    key_group_columns = torch.tensor(
        [
            index
            for index, name in enumerate(group_names)
            if name.endswith("::ribosome_mrna_assembly")
        ],
        dtype=torch.long,
    )
    history: list[dict[str, float | int]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def current_trajectory() -> torch.Tensor:
        if args.freeze_nonhidden_state:
            trajectory = prior.clone()
        else:
            optimized = torch.nn.functional.softplus(y)
            if args.initial_state_mode == "optimize_all":
                trajectory = optimized
            else:
                trajectory = torch.cat((fixed_initial, optimized), dim=0)
        if hidden_y is not None:
            trajectory = trajectory.clone()
            if args.hidden_state_parameterization == "source_delta":
                trajectory[:, hidden_indices] = torch.clamp(
                    prior[:, hidden_indices]
                    + float(args.hidden_delta_scale_multiplier)
                    * source_hidden_scale[None, :]
                    * torch.tanh(hidden_y),
                    min=0.0,
                )
            else:
                trajectory[:, hidden_indices] = (
                    torch.clamp(hidden_y, min=0.0)
                    * float(args.hidden_state_scale_M)
                )
            if args.preserve_source_group_totals:
                current_group_totals = aggregate(group_matrix, trajectory)
                group_scale = torch.where(
                    source_group_totals > 0.0,
                    source_group_totals
                    / torch.clamp(current_group_totals, min=1e-300),
                    torch.zeros_like(source_group_totals),
                )
                projected_members = trajectory.index_select(
                    1, group_member_indices
                ) * group_scale.index_select(1, group_member_group)
                trajectory[:, group_member_indices] = projected_members
            if args.initial_state_mode == "fixed":
                trajectory = torch.cat((fixed_initial, trajectory[1:]), dim=0)
        return trajectory

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x = current_trajectory()
        rhs = operator(x)
        rhs_integral = 0.5 * dt * (rhs[1:] + rhs[:-1])
        delta = x[1:] - x[:-1]
        residual = delta - rhs_integral

        direct_residual = residual.index_select(1, direct_indices)
        direct_rhs_integral = rhs_integral.index_select(1, direct_indices)
        group_residual = aggregate(group_matrix, residual)
        group_rhs_integral = aggregate(group_matrix, rhs_integral)
        direct_integral_scale = torch.clamp(
            torch.sqrt(torch.mean(direct_rhs_integral.detach().square(), dim=0)),
            min=float(args.rhs_floor_uM) * 1e-6,
        )
        group_integral_scale = torch.clamp(
            torch.sqrt(torch.mean(group_rhs_integral.detach().square(), dim=0)),
            min=float(args.rhs_floor_uM) * 1e-6,
        )
        direct_integral_normalized = torch.clamp(
            direct_residual / direct_integral_scale, -1e5, 1e5
        )
        group_integral_normalized = torch.clamp(
            group_residual / group_integral_scale, -1e5, 1e5
        )
        comparable_integral_normalized = torch.cat(
            (direct_integral_normalized.reshape(-1), group_integral_normalized.reshape(-1))
        )
        if args.integral_loss == "smooth_l1":
            integral_loss = torch.nn.functional.smooth_l1_loss(
                comparable_integral_normalized,
                torch.zeros_like(comparable_integral_normalized),
            )
        else:
            integral_loss = torch.mean(comparable_integral_normalized.square())

        direct_prediction = x.index_select(1, direct_indices)
        group_prediction = aggregate(group_matrix, x)
        teacher_direct_prediction = direct_prediction.index_select(0, teacher_mask)
        teacher_group_prediction = group_prediction.index_select(0, teacher_mask)
        teacher_direct_error = (
            torch.log10(teacher_direct_prediction + args.log_eps_M)
            - torch.log10(direct_target[teacher_mask_np] + args.log_eps_M)
        )
        teacher_group_error = (
            torch.log10(teacher_group_prediction + args.log_eps_M)
            - torch.log10(group_target[teacher_mask_np] + args.log_eps_M)
        )
        teacher_loss = torch.mean(
            torch.cat((teacher_direct_error.reshape(-1), teacher_group_error.reshape(-1))).square()
        )

        key_terms: list[torch.Tensor] = []
        if key_direct_columns.numel():
            key_direct_residual = direct_residual.index_select(1, key_direct_columns)
            key_direct_scale = direct_integral_scale.index_select(0, key_direct_columns)
            key_direct_normalized = torch.clamp(
                key_direct_residual / key_direct_scale, -1e5, 1e5
            )
            key_terms.append(key_direct_normalized.reshape(-1))
        if key_group_columns.numel():
            key_group_residual = group_residual.index_select(1, key_group_columns)
            key_group_scale = group_integral_scale.index_select(0, key_group_columns)
            key_group_normalized = torch.clamp(
                key_group_residual / key_group_scale, -1e5, 1e5
            )
            key_terms.append(key_group_normalized.reshape(-1))
        key_normalized = torch.cat(key_terms)
        key_integral_loss = (
            torch.nn.functional.smooth_l1_loss(
                key_normalized,
                torch.zeros_like(key_normalized),
            )
            if args.integral_loss == "smooth_l1"
            else torch.mean(key_normalized.square())
        )
        active_integral_loss = torch.tensor(0.0, dtype=dtype)
        comparable_rhs_activity_loss = torch.tensor(0.0, dtype=dtype)
        trajectory_nrmsr_loss = torch.tensor(0.0, dtype=dtype)
        current_trajectory_nrmsr_loss = torch.tensor(0.0, dtype=dtype)
        if float(args.trajectory_nrmsr_weight) > 0:
            track_ratios: list[torch.Tensor] = []
            if bool(torch.any(source_direct_active_mask)):
                direct_track_rms = torch.sqrt(
                    torch.mean(
                        direct_residual[:, source_direct_active_mask].square(), dim=0
                    )
                    + 1e-60
                )
                track_ratios.append(
                    direct_track_rms
                    / source_direct_exact_scale[source_direct_active_mask]
                )
            if bool(torch.any(source_group_active_mask)):
                group_track_rms = torch.sqrt(
                    torch.mean(
                        group_residual[:, source_group_active_mask].square(), dim=0
                    )
                    + 1e-60
                )
                track_ratios.append(
                    group_track_rms
                    / source_group_exact_scale[source_group_active_mask]
                )
            trajectory_nrmsr_loss = torch.cat(track_ratios).mean()
        if float(args.current_trajectory_nrmsr_weight) > 0:
            current_track_ratios: list[torch.Tensor] = []
            if bool(torch.any(source_direct_active_mask)):
                direct_current_scale = torch.sqrt(
                    torch.mean(
                        direct_rhs_integral[:, source_direct_active_mask].square(),
                        dim=0,
                    )
                    + 1e-60
                )
                direct_current_residual = torch.sqrt(
                    torch.mean(
                        direct_residual[:, source_direct_active_mask].square(),
                        dim=0,
                    )
                    + 1e-60
                )
                current_track_ratios.append(
                    direct_current_residual / direct_current_scale
                )
            if bool(torch.any(source_group_active_mask)):
                group_current_scale = torch.sqrt(
                    torch.mean(
                        group_rhs_integral[:, source_group_active_mask].square(),
                        dim=0,
                    )
                    + 1e-60
                )
                group_current_residual = torch.sqrt(
                    torch.mean(
                        group_residual[:, source_group_active_mask].square(),
                        dim=0,
                    )
                    + 1e-60
                )
                current_track_ratios.append(
                    group_current_residual / group_current_scale
                )
            current_trajectory_nrmsr_loss = torch.cat(
                current_track_ratios
            ).mean()
        if float(args.active_integral_weight) > 0:
            active_terms: list[torch.Tensor] = []
            if bool(torch.any(source_direct_active_mask)):
                active_terms.append(
                    direct_integral_normalized[:, source_direct_active_mask].reshape(-1)
                )
            if bool(torch.any(source_group_active_mask)):
                active_terms.append(
                    group_integral_normalized[:, source_group_active_mask].reshape(-1)
                )
            active_integral_loss = torch.mean(
                torch.cat(active_terms).square()
            )
        if float(args.comparable_rhs_activity_weight) > 0:
            current_direct_rate_rms = torch.sqrt(
                torch.mean(
                    (
                        direct_rhs_integral
                        * 1e6
                        / interval_hours[:, None]
                    ).square(),
                    dim=0,
                )
                + 1e-60
            )
            current_group_rate_rms = torch.sqrt(
                torch.mean(
                    (
                        group_rhs_integral
                        * 1e6
                        / interval_hours[:, None]
                    ).square(),
                    dim=0,
                )
                + 1e-60
            )
            activity_deficits: list[torch.Tensor] = []
            if bool(torch.any(source_direct_active_mask)):
                direct_target_rate = (
                    float(args.comparable_rhs_activity_fraction)
                    * source_direct_rate_rms[source_direct_active_mask]
                )
                activity_deficits.append(
                    torch.relu(
                        torch.log10(direct_target_rate + 1e-30)
                        - torch.log10(
                            current_direct_rate_rms[source_direct_active_mask] + 1e-30
                        )
                    ).reshape(-1)
                )
            if bool(torch.any(source_group_active_mask)):
                group_target_rate = (
                    float(args.comparable_rhs_activity_fraction)
                    * source_group_rate_rms[source_group_active_mask]
                )
                activity_deficits.append(
                    torch.relu(
                        torch.log10(group_target_rate + 1e-30)
                        - torch.log10(
                            current_group_rate_rms[source_group_active_mask] + 1e-30
                        )
                    ).reshape(-1)
                )
            comparable_rhs_activity_loss = torch.mean(
                torch.cat(activity_deficits).square()
            )
        hidden_integral_loss = torch.tensor(0.0, dtype=dtype)
        hidden_current_integral_loss = torch.tensor(0.0, dtype=dtype)
        hidden_rhs_activity_loss = torch.tensor(0.0, dtype=dtype)
        hidden_trajectory_nrmsr_loss = torch.tensor(0.0, dtype=dtype)
        if hidden_indices.numel() and (
            float(args.hidden_integral_weight) > 0
            or float(args.hidden_current_integral_weight) > 0
            or float(args.hidden_trajectory_nrmsr_weight) > 0
        ):
            hidden_residual = residual.index_select(1, hidden_indices)
            hidden_rhs_integral = rhs_integral.index_select(1, hidden_indices)
            active_hidden_residual = hidden_residual[:, source_hidden_active_mask]
            if float(args.hidden_integral_weight) > 0:
                active_source_scale = source_hidden_scale[source_hidden_active_mask]
                hidden_source_normalized = torch.clamp(
                    active_hidden_residual / active_source_scale,
                    -1e5,
                    1e5,
                )
                hidden_integral_loss = torch.mean(hidden_source_normalized.square())
            if float(args.hidden_trajectory_nrmsr_weight) > 0:
                hidden_track_rms = torch.sqrt(
                    torch.mean(active_hidden_residual.square(), dim=0) + 1e-60
                )
                hidden_trajectory_nrmsr_loss = torch.mean(
                    hidden_track_rms / source_hidden_scale_raw[source_hidden_active_mask]
                )
            if float(args.hidden_current_integral_weight) > 0:
                hidden_current_scale = torch.clamp(
                    torch.sqrt(
                        torch.mean(hidden_rhs_integral.detach().square(), dim=0)
                    ),
                    min=float(args.hidden_rhs_floor_uM) * 1e-6,
                )
                active_current_scale = hidden_current_scale[source_hidden_active_mask]
                hidden_current_normalized = torch.clamp(
                    active_hidden_residual / active_current_scale,
                    -1e5,
                    1e5,
                )
                hidden_current_integral_loss = torch.mean(
                    hidden_current_normalized.square()
                )
            if float(args.hidden_rhs_activity_weight) > 0:
                current_hidden_rate_rms = torch.sqrt(
                    torch.mean(
                        (
                            hidden_rhs_integral
                            * 1e6
                            / interval_hours[:, None]
                        ).square(),
                        dim=0,
                    )
                    + 1e-60
                )
                source_active_rate = source_hidden_rate_rms[source_hidden_active_mask]
                current_active_rate = current_hidden_rate_rms[source_hidden_active_mask]
                minimum_rate = (
                    float(args.hidden_rhs_activity_fraction) * source_active_rate
                )
                hidden_rhs_activity_loss = torch.mean(
                    torch.relu(
                        torch.log10(minimum_rate + 1e-30)
                        - torch.log10(current_active_rate + 1e-30)
                    ).square()
                )
        proximity_start = 0 if args.initial_state_mode == "optimize_all" else 1
        proximity_loss = torch.mean(
            (
                torch.log10(x[proximity_start:] + args.log_eps_M)
                - torch.log10(prior[proximity_start:] + args.log_eps_M)
            ).square()
        )
        loss = (
            float(args.teacher_weight) * teacher_loss
            + float(args.integral_weight) * integral_loss
            + float(args.key_integral_weight) * key_integral_loss
            + float(args.active_integral_weight) * active_integral_loss
            + float(args.trajectory_nrmsr_weight) * trajectory_nrmsr_loss
            + float(args.current_trajectory_nrmsr_weight)
            * current_trajectory_nrmsr_loss
            + float(args.comparable_rhs_activity_weight)
            * comparable_rhs_activity_loss
            + float(args.hidden_integral_weight) * hidden_integral_loss
            + float(args.hidden_trajectory_nrmsr_weight)
            * hidden_trajectory_nrmsr_loss
            + float(args.hidden_current_integral_weight)
            * hidden_current_integral_loss
            + float(args.hidden_rhs_activity_weight) * hidden_rhs_activity_loss
            + float(args.prior_proximity_weight) * proximity_loss
        )
        if step < args.steps:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
            optimizer.step()
        else:
            gradient_norm = torch.tensor(float("nan"))
        if step in {0, 1, 2, 5, 10, args.steps} or step % max(1, args.report_every) == 0:
            row = {
                "step": int(step),
                "loss": float(loss.detach()),
                "teacher_loss": float(teacher_loss.detach()),
                "integral_loss": float(integral_loss.detach()),
                "key_integral_loss": float(key_integral_loss.detach()),
                "active_integral_loss": float(active_integral_loss.detach()),
                "trajectory_nrmsr_loss": float(trajectory_nrmsr_loss.detach()),
                "current_trajectory_nrmsr_loss": float(
                    current_trajectory_nrmsr_loss.detach()
                ),
                "comparable_rhs_activity_loss": float(
                    comparable_rhs_activity_loss.detach()
                ),
                "hidden_integral_loss": float(hidden_integral_loss.detach()),
                "hidden_trajectory_nrmsr_loss": float(
                    hidden_trajectory_nrmsr_loss.detach()
                ),
                "hidden_current_integral_loss": float(
                    hidden_current_integral_loss.detach()
                ),
                "hidden_rhs_activity_loss": float(hidden_rhs_activity_loss.detach()),
                "proximity_loss": float(proximity_loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    with torch.no_grad():
        corrected = current_trajectory()
        corrected_rhs = operator(corrected)
        direct_prediction_np = corrected.index_select(1, direct_indices).cpu().numpy()
        group_prediction_np = aggregate(group_matrix, corrected).cpu().numpy()
    corrected_np = corrected.cpu().numpy()
    output_npz = args.output_dir / "corrected_prediction_6h.npz"
    np.savez_compressed(
        output_npz,
        time_s=time_np,
        species_id=np.asarray(operator.species_ids, dtype=str),
        optimized_M=corrected_np,
        prediction_M=corrected_np,
    )
    detail, nrmsr = nrmsr_metrics(
        corrected, corrected_rhs, time_np, direct_indices, group_matrix
    )
    detail.insert(0, "constraint", direct_names + group_names)
    detail.to_csv(args.output_dir / "corrected_nrmsr_by_constraint.csv", index=False)
    coarse_direct_prediction_np = direct_prediction_np[coarse_indices_np]
    coarse_group_prediction_np = group_prediction_np[coarse_indices_np]
    extension_weights = (
        float(args.active_integral_weight),
        float(args.trajectory_nrmsr_weight),
        float(args.hidden_trajectory_nrmsr_weight),
        float(args.comparable_rhs_activity_weight),
        float(args.current_trajectory_nrmsr_weight),
        float(args.hidden_integral_weight),
        float(args.hidden_current_integral_weight),
        float(args.hidden_rhs_activity_weight),
    )
    degfp_four_term_aligned = (
        np.allclose(extension_weights, 0.0)
        and np.isclose(float(args.teacher_weight), 1.0)
        and np.isclose(float(args.integral_weight), 1.0)
        and np.isclose(float(args.key_integral_weight), 2.0)
        and np.isclose(float(args.prior_proximity_weight), 0.005)
        and np.isclose(float(args.rhs_floor_uM), 0.01)
        and np.isclose(float(args.log_eps_M), 1e-12)
        and args.initial_state_mode == "optimize_all"
        and args.integral_loss == "mse"
        and int(args.physics_substeps) == 1
        and args.optimizer == "adam"
        and args.optimization_seed_scope == "all"
        and not args.freeze_nonhidden_state
        and not args.preserve_source_group_totals
        and args.hidden_state_parameterization == "softplus"
    )
    report = {
        "status": "physical_correction_completed",
        "method": (
            "deGFP-aligned second-stage four-term node correction adapted to direct-and-aggregate teacher mapping"
            if degfp_four_term_aligned
            else "six-protein second-stage node correction with deviations from the deGFP four-term source formulation"
        ),
        "alignment_scope": (
            "This flag evaluates the second-stage node-shooting formulation only; "
            "it does not certify that the upstream prior-PINN training is deGFP-aligned."
        ),
        "degfp_four_term_source_aligned": bool(degfp_four_term_aligned),
        "mapping_adaptation": (
            "The six-protein teacher states are represented by direct full-CRN states "
            "or fixed stoichiometric sums of sequence-specific microstates; this changes "
            "only the observation operator, not the four-term loss definition."
        ),
        "loss_formula": (
            "L_total = L_data + L_int + 2*L_key + 0.005*L_prox"
            if degfp_four_term_aligned
            else "See loss_weights and correction settings for deviations from the deGFP base formulation"
        ),
        "model_inventory": {
            "states": int(operator.n_species),
            "full_elementary_states": int(operator.base_species_count),
            "mass_action_elementary_reactions": int(operator.n_elementary_reactions),
            "teacher_effective_reactions": int(operator.n_teacher_reactions),
            "assets": str(args.assets),
            "pool_json": str(args.pool_json),
            "teacher_supervision": str(args.teacher_supervision),
            "prior": str(args.prior),
            "refined_prior": None if args.refined_prior is None else str(args.refined_prior),
        },
        "loss_weights": {
            "teacher": float(args.teacher_weight),
            "integral": float(args.integral_weight),
            "key_integral": float(args.key_integral_weight),
            "active_integral": float(args.active_integral_weight),
            "trajectory_nrmsr": float(args.trajectory_nrmsr_weight),
            "current_trajectory_nrmsr": float(
                args.current_trajectory_nrmsr_weight
            ),
            "hidden_trajectory_nrmsr": float(args.hidden_trajectory_nrmsr_weight),
            "comparable_rhs_activity": float(args.comparable_rhs_activity_weight),
            "hidden_integral": float(args.hidden_integral_weight),
            "hidden_current_integral": float(args.hidden_current_integral_weight),
            "hidden_rhs_activity": float(args.hidden_rhs_activity_weight),
            "prior_proximity": float(args.prior_proximity_weight),
        },
        "rhs_floor_uM": float(args.rhs_floor_uM),
        "optimization_initial_floor_M": float(args.optimization_initial_floor_M),
        "optimization_seed_scope": args.optimization_seed_scope,
        "hidden_intermediate_species_count": int(hidden_indices.numel()),
        "hidden_species_mode": args.hidden_species_mode,
        "hidden_include_regex": args.hidden_include_regex,
        "freeze_nonhidden_state": bool(args.freeze_nonhidden_state),
        "preserve_source_group_totals": bool(args.preserve_source_group_totals),
        "hidden_state_parameterization": args.hidden_state_parameterization,
        "hidden_state_scale_M": float(args.hidden_state_scale_M),
        "hidden_delta_scale_multiplier": float(args.hidden_delta_scale_multiplier),
        "hidden_lr": float(args.hidden_lr),
        "optimizer": args.optimizer,
        "hidden_loss_active_species_count": int(
            torch.sum(source_hidden_active_mask).item()
        ),
        "hidden_rhs_floor_uM": float(args.hidden_rhs_floor_uM),
        "hidden_active_rhs_rate_uM_per_h": float(
            args.hidden_active_rhs_rate_uM_per_h
        ),
        "hidden_rhs_activity_fraction": float(args.hidden_rhs_activity_fraction),
        "active_rhs_rate_uM_per_h": float(args.active_rhs_rate_uM_per_h),
        "comparable_rhs_activity_fraction": float(
            args.comparable_rhs_activity_fraction
        ),
        "active_integral_direct_count": int(
            torch.sum(source_direct_active_mask).item()
        ),
        "active_integral_group_count": int(
            torch.sum(source_group_active_mask).item()
        ),
        "initial_state_mode": args.initial_state_mode,
        "physics_substeps_per_teacher_interval": int(args.physics_substeps),
        "initial_condition_fixed_exactly": args.initial_state_mode == "fixed",
        "teacher_train_window_h": [0.0, float(args.teacher_train_max_s / 3600.0)],
        "teacher_holdout_time_h": float(teacher_time_np[holdout_mask_np][0] / 3600.0),
        "teacher_unused_gap_times_h": (teacher_time_np[unused_gap_mask_np] / 3600.0).tolist(),
        "train_metrics": {
            "direct": error_metrics(
                coarse_direct_prediction_np[teacher_mask_np],
                supervision["direct_target_np"][teacher_mask_np],
                args.mre_floor_M,
            ),
            "aggregate": error_metrics(
                coarse_group_prediction_np[teacher_mask_np],
                supervision["group_target_np"][teacher_mask_np],
                args.mre_floor_M,
            ),
            **grouped_error_metrics(
                coarse_group_prediction_np,
                supervision["group_target_np"],
                group_names,
                np.flatnonzero(teacher_mask_np),
                args.mre_floor_M,
            ),
        },
        "holdout_metrics": {
            "direct": error_metrics(
                coarse_direct_prediction_np[holdout_mask_np],
                supervision["direct_target_np"][holdout_mask_np],
                args.mre_floor_M,
            ),
            "aggregate": error_metrics(
                coarse_group_prediction_np[holdout_mask_np],
                supervision["group_target_np"][holdout_mask_np],
                args.mre_floor_M,
            ),
            **grouped_error_metrics(
                coarse_group_prediction_np,
                supervision["group_target_np"],
                group_names,
                np.flatnonzero(holdout_mask_np),
                args.mre_floor_M,
            ),
        },
        "physical_integral_metrics": nrmsr,
        "output": str(output_npz),
        "history": history,
    }
    (args.output_dir / "physical_correction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
