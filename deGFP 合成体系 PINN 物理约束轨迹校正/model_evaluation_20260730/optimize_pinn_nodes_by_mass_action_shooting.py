#!/usr/bin/env python3
"""Optimize PINN time nodes with full-mass-action multiple-shooting constraints.

This script does not ask the neural network to freely draw a smooth curve.
Instead, it treats the saved PINN states at discrete time nodes as shooting
variables and optimizes them under:

  1. teacher shared-species concentration constraints;
  2. full-elementary mass-action trapezoid transition constraints between
     adjacent nodes, C[k+1]-C[k] ~= 0.5*dt*(RHS(C[k])+RHS(C[k+1]));
  3. optional proximity to the original PINN state to avoid unbounded hidden
     species drift.

This is a fast collocation/projection test of the "PINN prior + full CRN
trajectory constraint" logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from full_mass_action_pinn import FullMassActionOperator
from train_continuous_full_crn_pinn_key import build_exchange, DEFAULT_CONTINUOUS_XML


BASE = Path(__file__).resolve().parent / "article_gapfill_20260730"
FULL = BASE / "full_elementary_pinn"


KEY20 = [
    "GTP",
    "GDP",
    "ATP",
    "ADP",
    "AMP",
    "CP",
    "PPi",
    "PO4",
    "mRNA_t",
    "Ribosome",
    "Ribosome_mRNA",
    "EFTu",
    "EFTu_GTP",
    "EFTu_GDP",
    "EFTu_EFTs",
    "EFG_GTP",
    "IF2_GTP",
    "MetRS",
    "MetRS_ATP",
    "deGFP_m",
]


def load_teacher_on_time(teacher_path: Path, time_s: np.ndarray, shared: pd.DataFrame) -> np.ndarray:
    teacher = np.load(teacher_path, allow_pickle=True)
    teacher_time = teacher["time_s"].astype(float)
    teacher_x = np.maximum(teacher["concentrations_M"].astype(float), 0.0)
    out = np.zeros((len(time_s), len(shared)), dtype=float)
    for j, row in enumerate(shared.itertuples(index=False)):
        out[:, j] = np.interp(time_s, teacher_time, teacher_x[:, int(row.species_index_teacher)])
    return out


def endpoint_metrics(x: np.ndarray, base: Path, shared: pd.DataFrame) -> dict[str, float]:
    snap = pd.read_csv(
        base / "existing_teacher_constraints" / "continuous_existing_teacher_train0to5_validate6h_all_species_6h_snapshot.csv"
    ).set_index("species_id")
    end = np.maximum(x[-1], 0.0)
    target = []
    pred = []
    for row in shared.itertuples(index=False):
        sid = str(row.species_id)
        if sid not in snap.index:
            continue
        target.append(float(snap.loc[sid, "teacher_6h_M"]))
        pred.append(float(end[int(row.species_index_full)]))
    target_a = np.array(target)
    pred_a = np.array(pred)
    err = np.abs(pred_a - target_a)
    rel = err / np.maximum(np.abs(target_a), 1e-12)
    return {
        "validate6h_mean_relative_error": float(np.mean(rel)),
        "validate6h_median_relative_error": float(np.median(rel)),
        "validate6h_p95_relative_error": float(np.quantile(rel, 0.95)),
        "validate6h_weighted_relative_error": float(np.sum(err) / max(np.sum(np.abs(target_a)), 1e-30)),
        "validate6h_r2": float(1.0 - np.sum((pred_a - target_a) ** 2) / np.sum((target_a - target_a.mean()) ** 2)),
    }


def integral_metrics_np(
    x: np.ndarray,
    rhs: np.ndarray,
    time_s: np.ndarray,
    shared_idx: np.ndarray,
    key_idx: np.ndarray,
) -> dict[str, float]:
    dt = (time_s[1:] - time_s[:-1])[:, None]
    rhs_trap = 0.5 * dt * (rhs[1:] + rhs[:-1])
    resid = x[1:] - x[:-1] - rhs_trap
    out: dict[str, float] = {}
    for label, idx in [("all3151", shared_idx), ("key20", key_idx)]:
        if len(idx) == 0:
            continue
        r = resid[:, idx]
        h = rhs_trap[:, idx]
        res_rms = np.sqrt(np.mean(r**2, axis=0)) * 1e6
        rhs_rms = np.sqrt(np.mean(h**2, axis=0)) * 1e6
        out[f"{label}_integral_residual_rms_uM_mean"] = float(np.mean(res_rms))
        out[f"{label}_integral_residual_rms_uM_median"] = float(np.median(res_rms))
        for floor in [1.0, 10.0, 100.0]:
            mask = rhs_rms > floor
            if np.any(mask):
                rel = res_rms[mask] / rhs_rms[mask]
                out[f"{label}_rhsint_gt_{floor:g}_uM_n"] = int(np.sum(mask))
                out[f"{label}_rhsint_gt_{floor:g}_uM_mean_relative_residual"] = float(np.mean(rel))
                out[f"{label}_rhsint_gt_{floor:g}_uM_median_relative_residual"] = float(np.median(rel))
                out[f"{label}_rhsint_gt_{floor:g}_uM_p95_relative_residual"] = float(np.quantile(rel, 0.95))
    return out


def build_hidden_intermediate_sets(
    assets: Path,
    shared: pd.DataFrame,
    mode: str,
    include_regex: str | None,
    exclude_regex: str | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return unmapped full-CRN states and their incident reactions."""
    species = pd.read_csv(assets / "species_index.csv").sort_values("species_index")
    species["species_id"] = species["species_id"].astype(str)
    shared_idx = set(shared["species_index_full"].astype(int).tolist())
    hidden = species[~species["species_index"].astype(int).isin(shared_idx)].copy()
    candidate_mask = pd.Series(True, index=hidden.index)
    if mode == "intermediate_like":
        candidate_mask &= hidden["species_id"].str.contains("_", regex=False)
    elif mode != "all_unmapped":
        raise ValueError(f"Unsupported hidden species mode: {mode}")
    if include_regex:
        candidate_mask &= hidden["species_id"].str.contains(include_regex, regex=True)
    if exclude_regex:
        candidate_mask &= ~hidden["species_id"].str.contains(exclude_regex, regex=True)
    selected = hidden[candidate_mask].copy()
    excluded = hidden[~candidate_mask].copy()
    hidden_idx = selected["species_index"].to_numpy(int)
    stoich = sparse.load_npz(assets / "stoichiometric_matrix_S.npz").tocsr()
    hidden_stoich = stoich[hidden_idx, :]
    hidden_rx_idx = np.flatnonzero(
        np.asarray(np.abs(hidden_stoich).sum(axis=0)).ravel() > 0
    ).astype(int)
    reactions = pd.read_csv(assets / "reaction_mass_action_inventory.csv")
    hidden_reactions = reactions[reactions["reaction_index"].isin(hidden_rx_idx)].copy()
    return hidden_idx, hidden_rx_idx, hidden_reactions, excluded


def hidden_integral_metrics_np(
    x: np.ndarray,
    rhs: np.ndarray,
    time_s: np.ndarray,
    hidden_idx: np.ndarray,
    label: str = "hidden_intermediate",
    floors_uM: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0),
) -> dict[str, float]:
    """Evaluate normalized integral residuals for a selected state subset."""
    if len(hidden_idx) == 0 or len(time_s) < 2:
        return {}
    dt = (time_s[1:] - time_s[:-1])[:, None]
    rhs_trap = 0.5 * dt * (rhs[1:] + rhs[:-1])
    resid = (x[1:] - x[:-1] - rhs_trap)[:, hidden_idx]
    hidden_rhs = rhs_trap[:, hidden_idx]
    res_rms = np.sqrt(np.mean(resid**2, axis=0)) * 1e6
    rhs_rms = np.sqrt(np.mean(hidden_rhs**2, axis=0)) * 1e6
    out = {
        f"{label}_integral_residual_rms_uM_mean": float(np.mean(res_rms)),
        f"{label}_integral_residual_rms_uM_median": float(np.median(res_rms)),
        f"{label}_integral_residual_rms_uM_p95": float(np.quantile(res_rms, 0.95)),
    }
    for floor in floors_uM:
        mask = rhs_rms > floor
        if np.any(mask):
            rel = res_rms[mask] / rhs_rms[mask]
            nrmsr = np.sqrt(
                np.mean((resid[:, mask] * 1e6 / rhs_rms[mask][None, :]) ** 2)
            )
            out[f"{label}_rhsint_gt_{floor:g}_uM_n"] = int(np.sum(mask))
            out[f"{label}_rhsint_gt_{floor:g}_uM_nrmsr"] = float(nrmsr)
            out[f"{label}_rhsint_gt_{floor:g}_uM_mean_relative_residual"] = float(np.mean(rel))
            out[f"{label}_rhsint_gt_{floor:g}_uM_median_relative_residual"] = float(np.median(rel))
            out[f"{label}_rhsint_gt_{floor:g}_uM_p95_relative_residual"] = float(np.quantile(rel, 0.95))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=FULL)
    parser.add_argument("--pinn", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-trajectory",
        type=Path,
        default=FULL / "teacher_full_species_trajectory_gamma0p5_6h_25pts.npz",
    )
    parser.add_argument("--teacher-train-max-s", type=float, default=19800.0)
    parser.add_argument("--integral-weight", type=float, default=1.0)
    parser.add_argument("--key-integral-weight", type=float, default=2.0)
    parser.add_argument("--platform-integral-weight", type=float, default=0.0)
    parser.add_argument("--platform-start-s", type=float, default=16200.0)
    parser.add_argument("--platform-end-s", type=float, default=19800.0)
    parser.add_argument("--pinn-proximity-weight", type=float, default=0.01)
    parser.add_argument("--rhs-floor-uM", type=float, default=0.01)
    parser.add_argument("--log-eps-M", type=float, default=1e-12)
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()

    dtype = torch.float64
    species = pd.read_csv(args.assets / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    mapping = pd.read_csv(args.assets.parent / "teacher_to_full_mapping" / "full_species_teacher_mapping.csv")
    shared = mapping[mapping["teacher_supervision_available"]].copy()
    shared["species_index_full"] = shared["species_index_full"].astype(int)
    shared["species_index_teacher"] = shared["species_index_teacher"].astype(int)
    shared_idx_np = shared["species_index_full"].to_numpy(int)
    key_idx_np = np.array([lookup[sid] for sid in KEY20 if sid in lookup and sid in set(shared["species_id"].astype(str))], dtype=int)

    pin = np.load(args.pinn)
    time_np = pin["time_s"].astype(float)
    if "prediction_M" in pin:
        x0_np = np.maximum(pin["prediction_M"].astype(float), 0.0)
        source_state_key = "prediction_M"
    elif "optimized_M" in pin:
        x0_np = np.maximum(pin["optimized_M"].astype(float), 0.0)
        source_state_key = "optimized_M"
    else:
        raise SystemExit("Input npz must contain either prediction_M or optimized_M.")
    teacher_np = load_teacher_on_time(args.teacher_trajectory, time_np, shared)
    teacher_mask_np = time_np <= float(args.teacher_train_max_s) + 1e-9

    x0 = torch.tensor(x0_np, dtype=dtype)
    y = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(x0, min=1e-30)) + 1e-30))
    time = torch.tensor(time_np[:, None], dtype=dtype)
    shared_idx = torch.tensor(shared_idx_np, dtype=torch.long)
    key_idx = torch.tensor(key_idx_np, dtype=torch.long)
    teacher_target = torch.tensor(teacher_np, dtype=dtype)
    teacher_mask = torch.tensor(teacher_mask_np, dtype=torch.bool)
    operator = FullMassActionOperator(args.assets, dtype=dtype)
    opt = torch.optim.Adam([y], lr=float(args.lr))
    history = []

    report_steps = set(range(0, int(args.steps) + 1, int(args.report_every))) | {int(args.steps), 1, 2, 5, 10, 25, 50}
    for step in range(int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        x = torch.nn.functional.softplus(y)
        exchange, _meta = build_exchange(time, x, species_ids, DEFAULT_CONTINUOUS_XML, dtype)
        rhs = operator(x, exchange)
        dt = time[1:] - time[:-1]
        rhs_trap = 0.5 * dt * (rhs[1:] + rhs[:-1])
        delta = x[1:] - x[:-1]
        integral_resid = delta - rhs_trap
        shared_resid = integral_resid.index_select(1, shared_idx)
        shared_rhsint = rhs_trap.index_select(1, shared_idx)
        shared_scale = torch.clamp(
            torch.sqrt(torch.mean(shared_rhsint.detach() ** 2, dim=0)),
            min=float(args.rhs_floor_uM) * 1e-6,
        )
        shared_int_loss = torch.mean(torch.clamp(shared_resid / shared_scale[None, :], -1e5, 1e5) ** 2)
        platform_int_loss = torch.tensor(0.0, dtype=dtype)
        if float(args.platform_integral_weight) > 0:
            window_mid = 0.5 * (time[1:, 0] + time[:-1, 0])
            platform_mask = (window_mid >= float(args.platform_start_s)) & (window_mid <= float(args.platform_end_s))
            if torch.any(platform_mask):
                platform_scaled = torch.clamp(shared_resid[platform_mask] / shared_scale[None, :], -1e5, 1e5)
                platform_int_loss = torch.mean(platform_scaled ** 2)
        if key_idx.numel() > 0:
            key_resid = integral_resid.index_select(1, key_idx)
            key_rhsint = rhs_trap.index_select(1, key_idx)
            key_scale = torch.clamp(
                torch.sqrt(torch.mean(key_rhsint.detach() ** 2, dim=0)),
                min=float(args.rhs_floor_uM) * 1e-6,
            )
            key_int_loss = torch.mean(torch.clamp(key_resid / key_scale[None, :], -1e5, 1e5) ** 2)
        else:
            key_int_loss = torch.tensor(0.0, dtype=dtype)
        teacher_pred = x.index_select(1, shared_idx)[teacher_mask]
        teacher_for_loss = teacher_target[teacher_mask]
        teacher_loss = torch.mean(
            torch.clamp(
                torch.log10(teacher_pred + float(args.log_eps_M))
                - torch.log10(teacher_for_loss + float(args.log_eps_M)),
                -30.0,
                30.0,
            )
            ** 2
        )
        proximity = torch.mean(
            (
                torch.log10(x + float(args.log_eps_M))
                - torch.log10(x0 + float(args.log_eps_M))
            )
            ** 2
        )
        loss = (
            float(args.integral_weight) * shared_int_loss
            + float(args.platform_integral_weight) * platform_int_loss
            + float(args.key_integral_weight) * key_int_loss
            + float(args.teacher_weight) * teacher_loss
            + float(args.pinn_proximity_weight) * proximity
        )
        if step < int(args.steps):
            loss.backward()
            torch.nn.utils.clip_grad_norm_([y], 1.0)
            opt.step()
        if step in report_steps:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach()),
                    "shared_integral_rmse": float(torch.sqrt(shared_int_loss).detach()),
                    "platform_integral_rmse": float(torch.sqrt(platform_int_loss).detach()),
                    "key_integral_rmse": float(torch.sqrt(key_int_loss).detach()),
                    "teacher_log_rmse": float(torch.sqrt(teacher_loss).detach()),
                    "proximity_log_rmse": float(torch.sqrt(proximity).detach()),
                }
            )

    with torch.no_grad():
        x = torch.nn.functional.softplus(y)
        exchange, _meta = build_exchange(time, x, species_ids, DEFAULT_CONTINUOUS_XML, dtype)
        rhs = operator(x, exchange)
    x_np = x.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    outdir = args.assets
    prefix = args.output_prefix
    np.savez_compressed(outdir / f"{prefix}_shooting_nodes.npz", time_s=time_np, optimized_M=x_np)
    summary: dict[str, object] = {
        "scope": "multiple-shooting node optimization with full-elementary mass-action integral constraints",
        "source_pinn": str(args.pinn),
        "source_state_key": source_state_key,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "teacher_weight": float(args.teacher_weight),
        "integral_weight": float(args.integral_weight),
        "platform_integral_weight": float(args.platform_integral_weight),
        "platform_start_s": float(args.platform_start_s),
        "platform_end_s": float(args.platform_end_s),
        "key_integral_weight": float(args.key_integral_weight),
        "pinn_proximity_weight": float(args.pinn_proximity_weight),
        "rhs_floor_uM": float(args.rhs_floor_uM),
        "history": history,
        "output_npz": str(outdir / f"{prefix}_shooting_nodes.npz"),
        **endpoint_metrics(x_np, args.assets.parent, shared),
        **integral_metrics_np(x_np, rhs_np, time_np, shared_idx_np, key_idx_np),
    }
    (outdir / f"{prefix}_shooting_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
