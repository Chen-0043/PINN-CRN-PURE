#!/usr/bin/env python3
"""Train the six-protein hybrid-resolution continuous PINN.

Teacher trajectories supervise only mapped observable/shared states.  The
teacher trajectory is never inserted as a fixed forcing term; every physics
evaluation recomputes elementary and effective fluxes from the PINN state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import jvp

from hybrid_resolution_pinn import DEFAULT_ASSETS, DEFAULT_POOL, HybridResolutionOperator


DEFAULT_TEACHER = DEFAULT_ASSETS / "sixprotein_hybrid_teacher_key_25pts.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "sixprotein_hybrid_pinn"


class HardICPositivePINN(nn.Module):
    """Positive full-state PINN that satisfies x(0)=x0 exactly."""

    def __init__(
        self,
        x0: torch.Tensor,
        scale: torch.Tensor,
        t_end_s: float,
        ic_gate_timescale_s: float,
        width: int,
        depth: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer("x0", x0.to(dtype=dtype))
        self.register_buffer("scale", scale.to(dtype=dtype))
        self.register_buffer("positive_mask", (x0 > 0).to(dtype=dtype))
        self.t_end_s = float(t_end_s)
        self.ic_gate_timescale_s = float(ic_gate_timescale_s)
        layers: list[nn.Module] = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, x0.numel()))
        self.net = nn.Sequential(*layers).to(dtype=dtype)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                final.weight.mul_(0.03)
                final.bias.zero_()

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(time_s[:, :1] / self.t_end_s, min=0.0, max=1.0)
        features = torch.cat(
            (
                2.0 * tau - 1.0,
                2.0 * torch.log1p(time_s[:, :1]) / np.log1p(self.t_end_s) - 1.0,
                torch.exp(-time_s[:, :1] / 60.0),
                torch.exp(-time_s[:, :1] / 600.0),
            ),
            dim=1,
        )
        z = torch.clamp(self.net(features), min=-25.0, max=25.0)
        # Release the hard initial condition on the fast biochemical timescale,
        # not over the full 24 h horizon.  Using tau here imposes an artificial
        # lower bound on ATP/GTP depletion during the first few hours.
        ic_gate = 1.0 - torch.exp(-time_s[:, :1] / self.ic_gate_timescale_s)
        positive_part = self.x0 * torch.exp(ic_gate * z)
        zero_part = ic_gate * self.scale * torch.nn.functional.softplus(z)
        return self.positive_mask * positive_part + (1.0 - self.positive_mask) * zero_part


def time_derivative(model: nn.Module, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return jvp(model, (time_s,), (torch.ones_like(time_s),))


def load_teacher(
    path: Path,
    operator: HybridResolutionOperator,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    frame = pd.read_csv(path)
    names: list[str] = []
    columns: list[str] = []
    indices: list[int] = []
    for column in frame.columns:
        if column == "time_s" or not column.endswith("_uM"):
            continue
        sid = column[:-3]
        if sid in operator.species_index:
            names.append(sid)
            columns.append(column)
            indices.append(operator.species_index[sid])
    if not names:
        raise ValueError(f"No teacher columns map to hybrid states in {path}")
    return (
        frame["time_s"].to_numpy(dtype=float),
        frame[columns].to_numpy(dtype=float) * 1e-6,
        names,
        np.asarray(indices, dtype=int),
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float64
    operator = HybridResolutionOperator(
        args.assets,
        args.pool_json,
        energy_scale=args.energy_scale,
        gamma_h=args.gamma_h,
        dtype=dtype,
    )
    time_np, target_np, teacher_names, teacher_indices_np = load_teacher(args.teacher_csv, operator)
    time_all = torch.tensor(time_np[:, None], dtype=dtype)
    target_all = torch.tensor(target_np, dtype=dtype)
    teacher_indices = torch.tensor(teacher_indices_np, dtype=torch.long)
    teacher_species_weight_np = np.ones(len(teacher_names), dtype=float)
    resource_ids = {"ATP", "ADP", "GTP", "GDP", "CTP", "UTP", "PPi", "PO4", "Glucose", "NAD", "NADH"}
    for index, sid in enumerate(teacher_names):
        if sid in resource_ids:
            teacher_species_weight_np[index] *= float(args.resource_teacher_weight)
        if sid in {"ATP", "ADP", "GTP", "GDP"}:
            teacher_species_weight_np[index] *= float(args.nucleotide_energy_weight)
        if sid == "GTP":
            teacher_species_weight_np[index] *= float(args.gtp_teacher_weight)
    teacher_species_weight_np /= max(float(np.mean(teacher_species_weight_np)), 1e-30)
    teacher_species_weight = torch.tensor(teacher_species_weight_np, dtype=dtype)
    x0 = operator.initial_state_M

    scale_np = np.maximum(x0.detach().cpu().numpy(), float(args.state_scale_floor_M))
    for column, species_index in enumerate(teacher_indices_np):
        scale_np[species_index] = max(
            scale_np[species_index],
            float(np.max(target_np[:, column])),
            float(args.state_scale_floor_M),
        )
    scale = torch.tensor(scale_np, dtype=dtype)
    teacher_scale = torch.tensor(np.maximum(np.max(np.abs(target_np), axis=0), 1e-9), dtype=dtype)
    derivative_scale = torch.clamp(
        scale / max(float(time_np[-1]), 1.0),
        min=float(args.derivative_scale_floor_M_per_s),
    )

    model = HardICPositivePINN(
        x0=x0,
        scale=scale,
        t_end_s=float(time_np[-1]),
        ic_gate_timescale_s=float(args.ic_gate_timescale_s),
        width=args.width,
        depth=args.depth,
        dtype=dtype,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr)
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step: int, name: str) -> None:
        torch.save(
            {
                "step": int(step),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "species_ids": operator.species_ids,
                "teacher_species": teacher_names,
                "configuration": vars(args),
            },
            args.output_dir / name,
        )

    for step in range(start_step, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if args.teacher_batch >= len(time_all):
            teacher_choice = torch.arange(len(time_all))
        else:
            # Early points contain the ATP/GTP depletion boundary layer. Keep
            # half of each mini-batch there and sample the rest globally.
            early_count = min(args.early_teacher_points, len(time_all))
            early_take = min(max(1, args.teacher_batch // 2), early_count)
            global_take = args.teacher_batch - early_take
            early_choice = torch.randperm(early_count, generator=generator)[:early_take]
            global_choice = torch.randperm(len(time_all), generator=generator)[:global_take]
            teacher_choice = torch.unique(torch.cat((early_choice, global_choice)))
        teacher_time = time_all.index_select(0, teacher_choice)
        teacher_target = target_all.index_select(0, teacher_choice)
        pred_teacher = model(teacher_time).index_select(1, teacher_indices)
        if args.teacher_loss_mode == "log10":
            teacher_log_residual = torch.log10(pred_teacher + args.teacher_log_eps_M) - torch.log10(
                teacher_target + args.teacher_log_eps_M
            )
        else:
            teacher_log_residual = (pred_teacher - teacher_target) / teacher_scale
        teacher_log_loss = torch.mean(teacher_log_residual.square() * teacher_species_weight)
        teacher_linear_residual = (pred_teacher - teacher_target) / teacher_scale
        teacher_linear_loss = torch.mean(teacher_linear_residual.square() * teacher_species_weight)
        teacher_loss = teacher_log_loss + args.linear_teacher_weight * teacher_linear_loss

        early_physics_count = int(round(args.physics_batch * args.early_physics_fraction))
        early_physics_count = min(max(0, early_physics_count), args.physics_batch)
        uniform_physics_count = args.physics_batch - early_physics_count
        physics_min_time_s = min(
            max(0.0, float(args.physics_min_time_s)),
            float(time_np[-1]),
        )
        collocation_parts: list[torch.Tensor] = []
        if uniform_physics_count:
            collocation_parts.append(
                torch.rand(
                    (uniform_physics_count, 1), generator=generator, dtype=dtype
                )
                * (float(time_np[-1]) - physics_min_time_s)
                + physics_min_time_s
            )
        if early_physics_count:
            # Log-uniform sampling resolves the early stiff layer without
            # allowing the unresolved sub-minute initial transient to dominate.
            log_draw = torch.rand(
                (early_physics_count, 1), generator=generator, dtype=dtype
            )
            log_min = np.log(max(physics_min_time_s, 1e-9))
            log_max = np.log(float(time_np[-1]))
            collocation_parts.append(torch.exp(log_min + log_draw * (log_max - log_min)))
        collocation_time = torch.cat(collocation_parts, dim=0)
        pred_physics, dpred_dt = time_derivative(model, collocation_time)
        physics_rhs = operator(pred_physics)
        scaled_physics = (dpred_dt - physics_rhs) / derivative_scale
        physics_loss = torch.nn.functional.smooth_l1_loss(
            scaled_physics,
            torch.zeros_like(scaled_physics),
            beta=1.0,
        )
        if step < args.teacher_warmup_steps:
            effective_physics_weight = float(args.physics_weight_start)
        else:
            ramp = min(
                1.0,
                (step - args.teacher_warmup_steps + 1) / max(1, args.physics_ramp_steps),
            )
            effective_physics_weight = float(args.physics_weight_start) + ramp * (
                float(args.physics_weight) - float(args.physics_weight_start)
            )
        loss = args.teacher_weight * teacher_loss + effective_physics_weight * physics_loss

        if step < args.steps:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            grad_norm = torch.tensor(float("nan"))
        if step in {0, args.steps} or step % max(1, args.report_every) == 0:
            entry = {
                "step": int(step),
                "total_loss": float(loss.detach()),
                "teacher_loss": float(teacher_loss.detach()),
                "teacher_log_loss": float(teacher_log_loss.detach()),
                "teacher_linear_loss": float(teacher_linear_loss.detach()),
                "physics_loss": float(physics_loss.detach()),
                "effective_physics_weight": effective_physics_weight,
                "gradient_norm_before_clip": float(grad_norm),
            }
            history.append(entry)
            with (args.output_dir / "training_progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if step > start_step and step < args.steps and step % max(1, args.checkpoint_every) == 0:
            save_checkpoint(step, "hybrid_pinn_checkpoint_latest.pt")

    with torch.no_grad():
        prediction = model(time_all).index_select(1, teacher_indices).cpu().numpy()
    nonzero = np.abs(target_np) > args.mre_floor_M
    mre_percent = float(np.mean(np.abs(prediction[nonzero] - target_np[nonzero]) / np.abs(target_np[nonzero])) * 100.0)
    scaled_rmse = float(np.sqrt(np.mean(((prediction - target_np) / teacher_scale.numpy()) ** 2)))

    prediction_frame = pd.DataFrame({"time_s": time_np})
    for column, sid in enumerate(teacher_names):
        prediction_frame[f"{sid}_teacher_uM"] = target_np[:, column] * 1e6
        prediction_frame[f"{sid}_pinn_uM"] = prediction[:, column] * 1e6
    prediction_frame.to_csv(args.output_dir / "teacher_vs_pinn.csv", index=False)
    save_checkpoint(args.steps, "hybrid_pinn_checkpoint.pt")
    is_smoke = args.steps <= 5
    per_species_rows: list[dict[str, object]] = []
    for column, sid in enumerate(teacher_names):
        reference = target_np[:, column]
        predicted = prediction[:, column]
        valid = np.abs(reference) > args.mre_floor_M
        species_mre = (
            float(np.mean(np.abs(predicted[valid] - reference[valid]) / np.abs(reference[valid])) * 100.0)
            if np.any(valid)
            else None
        )
        per_species_rows.append(
            {
                "species_id": sid,
                "evaluated_points": int(np.sum(valid)),
                "MRE_percent": species_mre,
                "RMSE_uM": float(np.sqrt(np.mean((predicted - reference) ** 2)) * 1e6),
            }
        )
    pd.DataFrame(per_species_rows).to_csv(args.output_dir / "mapped_state_metrics.csv", index=False)

    resource_names = {"ATP", "ADP", "GTP", "GDP", "CTP", "UTP", "PPi", "PO4", "Glucose", "NAD", "NADH"}
    category_metrics: dict[str, dict[str, float | int | None]] = {}
    categories = {
        "resource_pools": [i for i, sid in enumerate(teacher_names) if sid in resource_names],
        "protein_products": [i for i, sid in enumerate(teacher_names) if sid.startswith("deGFP_m__")],
        "transcripts": [i for i, sid in enumerate(teacher_names) if sid.startswith("mRNA_t__")],
    }
    for category, columns in categories.items():
        reference = target_np[:, columns]
        predicted = prediction[:, columns]
        valid = np.abs(reference) > args.mre_floor_M
        category_metrics[category] = {
            "states": len(columns),
            "evaluated_values": int(np.sum(valid)),
            "MRE_percent": (
                float(np.mean(np.abs(predicted[valid] - reference[valid]) / np.abs(reference[valid])) * 100.0)
                if np.any(valid)
                else None
            ),
            "RMSE_uM": float(np.sqrt(np.mean((predicted - reference) ** 2)) * 1e6),
        }

    simulation_time_np = np.linspace(float(time_np[0]), float(time_np[-1]), int(args.simulation_points))
    simulation_time = torch.tensor(simulation_time_np[:, None], dtype=dtype)
    full_chunks: list[np.ndarray] = []
    for begin in range(0, len(simulation_time), args.simulation_batch):
        with torch.no_grad():
            full_chunks.append(
                model(simulation_time[begin : begin + args.simulation_batch]).cpu().numpy()
            )
    full_trajectory = np.concatenate(full_chunks, axis=0)
    np.savez_compressed(
        args.output_dir / "full_state_trajectory_24h.npz",
        time_s=simulation_time_np,
        species_id=np.asarray(operator.species_ids, dtype=str),
        concentrations_M=full_trajectory,
    )

    selected = pd.DataFrame({"time_s": simulation_time_np, "time_h": simulation_time_np / 3600.0})
    for sid in teacher_names:
        selected[f"{sid}_uM"] = full_trajectory[:, operator.species_index[sid]] * 1e6
    with torch.no_grad():
        effective_flux = operator.teacher_flux(torch.tensor(full_trajectory, dtype=dtype)).cpu().numpy()
    selected["effective_glucose_ATP_regeneration_uM_per_s"] = effective_flux[:, 0] * 1e6
    selected.to_csv(args.output_dir / "selected_state_trajectory_24h.csv", index=False)

    physics_sum_sq = 0.0
    physics_abs_max = 0.0
    physics_count = 0
    raw_sum_sq = 0.0
    finite_physics = True
    for begin in range(0, len(simulation_time), args.physics_eval_batch):
        eval_time = simulation_time[begin : begin + args.physics_eval_batch]
        eval_state, eval_dxdt = time_derivative(model, eval_time)
        eval_rhs = operator(eval_state)
        residual = eval_dxdt - eval_rhs
        scaled = residual / derivative_scale
        finite_physics = finite_physics and bool(torch.isfinite(scaled).all())
        physics_sum_sq += float(torch.sum(scaled.square()).detach())
        raw_sum_sq += float(torch.sum(residual.square()).detach())
        physics_abs_max = max(physics_abs_max, float(torch.max(torch.abs(scaled)).detach()))
        physics_count += int(scaled.numel())
    physics_scaled_rms = float(np.sqrt(physics_sum_sq / max(1, physics_count)))
    physics_raw_rms_M_per_s = float(np.sqrt(raw_sum_sq / max(1, physics_count)))

    top_rows: list[dict[str, object]] = []
    for label, target_h in (("6h", 6.0), ("24h", 24.0)):
        row_index = int(np.argmin(np.abs(simulation_time_np / 3600.0 - target_h)))
        order = np.argsort(full_trajectory[row_index])[::-1][:100]
        for rank, species_index in enumerate(order, start=1):
            top_rows.append(
                {
                    "time": label,
                    "rank": rank,
                    "species_id": operator.species_ids[int(species_index)],
                    "concentration_uM": float(full_trajectory[row_index, species_index] * 1e6),
                }
            )
    pd.DataFrame(top_rows).to_csv(args.output_dir / "top_states_6h_24h.csv", index=False)

    report = {
        "status": "training_smoke_test" if is_smoke else "formal_simulation_completed",
        "steps": int(args.steps),
        "total_states": operator.n_species,
        "elementary_reactions": operator.n_elementary_reactions,
        "teacher_effective_reactions": operator.n_teacher_reactions,
        "teacher_mapped_states": len(teacher_names),
        "teacher_fixed_forcing": False,
        "physics_rhs_recomputed_from_current_pinn_state": True,
        "hard_initial_condition": True,
        "hard_initial_condition_gate_timescale_s": float(args.ic_gate_timescale_s),
        "nonnegative_state_parameterization": True,
        "mapped_state_MRE_percent_nonzero_targets": None if is_smoke else mre_percent,
        "mapped_state_scaled_RMSE": scaled_rmse,
        "category_metrics": category_metrics,
        "teacher_loss_mode": args.teacher_loss_mode,
        "teacher_species_weights": {
            sid: float(teacher_species_weight_np[index]) for index, sid in enumerate(teacher_names)
        },
        "linear_teacher_weight": float(args.linear_teacher_weight),
        "physics_weight_final": float(args.physics_weight),
        "teacher_warmup_steps": int(args.teacher_warmup_steps),
        "physics_ramp_steps": int(args.physics_ramp_steps),
        "physics_audit": {
            "evaluation_time_points": int(args.simulation_points),
            "finite": finite_physics,
            "scaled_residual_RMS": physics_scaled_rms,
            "scaled_residual_max_abs": physics_abs_max,
            "raw_residual_RMS_M_per_s": physics_raw_rms_M_per_s,
        },
        "full_state_trajectory": {
            "time_points": int(args.simulation_points),
            "states": operator.n_species,
            "path": str(args.output_dir / "full_state_trajectory_24h.npz"),
        },
        "mre_definition": "mean(abs(PINN-reference)/abs(reference))*100 over reference values above floor",
        "mre_floor_M": float(args.mre_floor_M),
        "metric_note": (
            "A two-to-five-step smoke run only verifies optimization and does not report MRE as a model result."
            if is_smoke
            else "Formal run completed; scientific acceptance depends on the reported MRE and physics audit."
        ),
        "history": history,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ic-gate-timescale-s", type=float, default=300.0)
    parser.add_argument("--teacher-batch", type=int, default=25)
    parser.add_argument("--physics-batch", type=int, default=4)
    parser.add_argument("--early-physics-fraction", type=float, default=0.5)
    parser.add_argument("--physics-min-time-s", type=float, default=60.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1e-8)
    parser.add_argument("--physics-weight-start", type=float, default=0.0)
    parser.add_argument("--teacher-warmup-steps", type=int, default=2000)
    parser.add_argument("--physics-ramp-steps", type=int, default=8000)
    parser.add_argument("--linear-teacher-weight", type=float, default=1.0)
    parser.add_argument("--resource-teacher-weight", type=float, default=4.0)
    parser.add_argument("--nucleotide-energy-weight", type=float, default=4.0)
    parser.add_argument("--gtp-teacher-weight", type=float, default=4.0)
    parser.add_argument("--early-teacher-points", type=int, default=7)
    parser.add_argument("--teacher-loss-mode", choices=("log10", "scaled_mse"), default="log10")
    parser.add_argument("--teacher-log-eps-M", type=float, default=1e-12)
    parser.add_argument("--energy-scale", type=float, default=0.8)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--mre-floor-M", type=float, default=1e-12)
    parser.add_argument("--state-scale-floor-M", type=float, default=1e-12)
    parser.add_argument("--derivative-scale-floor-M-per-s", type=float, default=1e-15)
    parser.add_argument("--simulation-points", type=int, default=97)
    parser.add_argument("--simulation-batch", type=int, default=8)
    parser.add_argument("--physics-eval-batch", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
