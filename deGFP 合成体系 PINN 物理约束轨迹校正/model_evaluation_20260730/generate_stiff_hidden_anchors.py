#!/usr/bin/env python3
"""Generate full-mass-action stiff-solver anchors for hidden intermediates.

The ODE is integrated on a local species neighborhood while all other species
follow the supplied trajectory by linear interpolation. Internal reaction fluxes
are still evaluated from the full elementary mass-action network.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from optimize_pinn_nodes_by_mass_action_shooting import FULL, build_hidden_intermediate_sets
from train_continuous_full_crn_pinn_key import DEFAULT_CONTINUOUS_XML, parse_parameters


def build_exchange_plan(species_ids: list[str], continuous_xml: Path) -> tuple[float, float, np.ndarray, np.ndarray, int | None]:
    params = parse_parameters(continuous_xml)
    gamma = float(params.get("bg_gamma", 0.5 / 3600.0))
    gamma_prod = float(params.get("bg_gamma_prod", gamma))
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    feed_idx: list[int] = []
    feed_value: list[float] = []
    for key, value in params.items():
        if not key.startswith("bg_feed_") or key == "bg_feed_deGFP_m":
            continue
        sid = key.removeprefix("bg_feed_")
        if sid == "AA" or sid not in lookup:
            continue
        feed_idx.append(int(lookup[sid]))
        feed_value.append(float(value))
    product_idx = int(lookup["deGFP_m"]) if "deGFP_m" in lookup else None
    return gamma, gamma_prod, np.array(feed_idx, dtype=int), np.array(feed_value, dtype=float), product_idx


def build_exchange_numpy(
    x: np.ndarray,
    gamma: float,
    gamma_prod: float,
    feed_idx: np.ndarray,
    feed_value: np.ndarray,
    product_idx: int | None,
) -> np.ndarray:
    exchange = np.zeros_like(x)
    if len(feed_idx) > 0:
        exchange[feed_idx] += gamma * (feed_value - x[feed_idx])
    if product_idx is not None:
        exchange[product_idx] += gamma_prod * (0.0 - x[product_idx])
    return exchange


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=FULL)
    parser.add_argument("--pinn", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--exchange-mode", choices=["continuous", "batch"], default="continuous")
    parser.add_argument("--rate-unit", choices=["M", "uM"], default="uM")
    parser.add_argument("--state-unit", choices=["M", "uM"], default="M")
    parser.add_argument("--method", choices=["BDF", "Radau", "LSODA"], default="BDF")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol-uM", type=float, default=1e-6)
    parser.add_argument("--max-step-s", type=float, default=np.inf)
    parser.add_argument("--state-transform", choices=["linear", "log_positive"], default="linear")
    parser.add_argument(
        "--log-floor-uM",
        type=float,
        default=1e-12,
        help="Positive concentration floor used by --state-transform log_positive.",
    )
    parser.add_argument(
        "--use-jac-sparsity",
        action="store_true",
        help="Provide a local mass-action dependency sparsity pattern to BDF/Radau.",
    )
    parser.add_argument("--hidden-species-mode", choices=["intermediate_like", "all_unmapped"], default="intermediate_like")
    parser.add_argument("--target-regex", default=r"RS.*(?:EFG|EFTu|RRF|RF|IF|GDP|GTP|PO4|Pept|tRNA|fMet)|(?:T7RNAP|mRNA)")
    parser.add_argument("--neighborhood", choices=["hidden_reaction_neighborhood", "target_only"], default="hidden_reaction_neighborhood")
    parser.add_argument(
        "--initial-state-mode",
        choices=["source_interval", "marched"],
        default="source_interval",
        help="Use source left state for every interval or march the hidden/local state causally across intervals.",
    )
    parser.add_argument("--max-intervals", type=int, default=4)
    parser.add_argument("--start-interval", type=int, default=0)
    parser.add_argument("--anchor-points-per-interval", type=int, default=3)
    args = parser.parse_args()

    species = pd.read_csv(args.assets / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    mapping = pd.read_csv(args.assets.parent / "teacher_to_full_mapping" / "full_species_teacher_mapping.csv")
    shared = mapping[mapping["teacher_supervision_available"]].copy()
    shared["species_index_full"] = shared["species_index_full"].astype(int)
    hidden_idx_np, hidden_rx_idx_np, _hidden_reactions, _hidden_excluded = build_hidden_intermediate_sets(
        args.assets,
        shared,
        args.hidden_species_mode,
        None,
        None,
    )
    hidden_species = np.array([species_ids[int(i)] for i in hidden_idx_np], dtype=object)
    target_mask = pd.Series(hidden_species).str.contains(args.target_regex, regex=True).to_numpy(bool)
    target_hidden_idx_np = hidden_idx_np[target_mask]
    if len(target_hidden_idx_np) == 0:
        raise SystemExit("--target-regex selected no hidden species.")

    if args.neighborhood == "hidden_reaction_neighborhood":
        stoich = sparse.load_npz(args.assets / "stoichiometric_matrix_S.npz").tocsr()
        reactant_exp = sparse.load_npz(args.assets / "reactant_exponent_matrix_A.npz").tocsr()
        target_rx_mask = np.asarray(np.abs(stoich[target_hidden_idx_np, :]).sum(axis=0)).ravel() > 0
        target_rx_idx = np.flatnonzero(target_rx_mask)
        stoich_species = np.flatnonzero(np.asarray(np.abs(stoich[:, target_rx_idx]).sum(axis=1)).ravel() > 0)
        reactant_species = np.flatnonzero(np.asarray(reactant_exp[:, target_rx_idx].sum(axis=1)).ravel() > 0)
        local_idx_np = np.union1d(np.union1d(stoich_species, reactant_species), target_hidden_idx_np).astype(int)
    else:
        local_idx_np = target_hidden_idx_np.copy()

    local_pos = {int(idx): pos for pos, idx in enumerate(local_idx_np)}
    target_local_pos = np.array([local_pos[int(idx)] for idx in target_hidden_idx_np], dtype=int)
    stoich = sparse.load_npz(args.assets / "stoichiometric_matrix_S.npz").tocsr()
    reactants = sparse.load_npz(args.assets / "reactant_exponent_matrix_A.npz").tocsr()
    k = np.load(args.assets / "rate_constants_k.npy").astype(float)
    molecularity = np.asarray(reactants.sum(axis=0)).ravel().astype(float)
    if args.rate_unit == "uM" and args.state_unit == "M":
        k = k * np.power(1e6, np.maximum(molecularity - 1.0, 0.0))
    elif args.rate_unit == "M" and args.state_unit == "uM":
        k = k * np.power(1e-6, np.maximum(molecularity - 1.0, 0.0))
    elif args.rate_unit != args.state_unit:
        raise SystemExit(f"Unsupported rate/state unit pair: {args.rate_unit}/{args.state_unit}")
    log_k = np.log(np.clip(k, 1e-300, None))
    exchange_plan = build_exchange_plan(species_ids, DEFAULT_CONTINUOUS_XML)
    jac_sparsity = None
    if args.use_jac_sparsity and args.method in {"BDF", "Radau"}:
        affected_rx_mask = np.asarray(np.abs(stoich[local_idx_np, :]).sum(axis=0)).ravel() > 0
        affected_rx_idx = np.flatnonzero(affected_rx_mask)
        rows: list[int] = []
        cols: list[int] = []
        local_idx_set = set(int(i) for i in local_idx_np)
        for rx in affected_rx_idx:
            affected_global = stoich[:, int(rx)].nonzero()[0]
            reactant_global = reactants[:, int(rx)].nonzero()[0]
            affected_local = [local_pos[int(i)] for i in affected_global if int(i) in local_idx_set]
            reactant_local = [local_pos[int(i)] for i in reactant_global if int(i) in local_idx_set]
            for row in affected_local:
                for col in reactant_local:
                    rows.append(row)
                    cols.append(col)
        diag = np.arange(len(local_idx_np), dtype=int)
        rows.extend(diag.tolist())
        cols.extend(diag.tolist())
        jac_sparsity = sparse.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(len(local_idx_np), len(local_idx_np)),
        ).tocsr()

    z = np.load(args.pinn)
    state_key = "optimized_M" if "optimized_M" in z else "prediction_M"
    time_s = z["time_s"].astype(float)
    x_source = np.maximum(z[state_key].astype(float), 0.0)
    n_intervals = len(time_s) - 1
    start_interval = max(0, int(args.start_interval))
    stop_interval = min(n_intervals, start_interval + max(1, int(args.max_intervals)))

    anchors_t: list[float] = []
    anchors_x: list[np.ndarray] = []
    interval_reports: list[dict[str, object]] = []
    log_floor_M = max(float(args.log_floor_uM) * 1e-6, 1e-300)
    marched_y = x_source[start_interval, local_idx_np].copy()

    def interpolate_source(t: float, left_i: int, right_i: int) -> np.ndarray:
        t0 = time_s[left_i]
        t1 = time_s[right_i]
        a = 0.0 if t1 == t0 else (float(t) - t0) / (t1 - t0)
        return (1.0 - a) * x_source[left_i] + a * x_source[right_i]

    for interval in range(start_interval, stop_interval):
        t0 = float(time_s[interval])
        t1 = float(time_s[interval + 1])
        if args.initial_state_mode == "marched":
            y0 = marched_y.copy()
        else:
            y0 = x_source[interval, local_idx_np].copy()
        y0_conc = np.maximum(y0, log_floor_M)
        solver_y0 = np.log(y0_conc) if args.state_transform == "log_positive" else y0

        def concentration_from_solver_state(y: np.ndarray) -> np.ndarray:
            if args.state_transform == "log_positive":
                return np.exp(np.clip(y, -700.0, 100.0))
            return np.maximum(y, 0.0)

        def rhs_local(t: float, y: np.ndarray) -> np.ndarray:
            y_conc = concentration_from_solver_state(y)
            x = interpolate_source(t, interval, interval + 1)
            x[local_idx_np] = y_conc
            log_x = np.log(np.clip(x, 1e-30, None))
            log_monomial = log_x @ reactants
            flux = np.exp(np.clip(log_monomial + log_k, -100.0, 100.0))
            if args.exchange_mode == "continuous":
                exchange = build_exchange_numpy(x, *exchange_plan)
            else:
                exchange = np.zeros_like(x)
            rhs_full = stoich @ flux + exchange
            rhs_local_conc = np.asarray(rhs_full[local_idx_np]).ravel()
            if args.state_transform == "log_positive":
                return rhs_local_conc / np.maximum(y_conc, log_floor_M)
            return rhs_local_conc

        t_eval = np.linspace(t0, t1, int(args.anchor_points_per_interval) + 2)[1:]
        sol = solve_ivp(
            rhs_local,
            (t0, t1),
            solver_y0,
            method=args.method,
            t_eval=t_eval,
            rtol=float(args.rtol),
            atol=float(args.atol_uM) * (1e-6 if args.state_transform == "linear" else 1.0),
            max_step=float(args.max_step_s),
            jac_sparsity=jac_sparsity,
        )
        interval_reports.append(
            {
                "interval": int(interval),
                "time_start_s": t0,
                "time_end_s": t1,
                "success": bool(sol.success),
                "message": str(sol.message),
                "nfev": int(sol.nfev),
                "njev": int(getattr(sol, "njev", 0)),
                "nlu": int(getattr(sol, "nlu", 0)),
            }
        )
        if not sol.success:
            break
        sol_conc = concentration_from_solver_state(sol.y)
        if args.initial_state_mode == "marched":
            marched_y = np.maximum(sol_conc[:, -1], 0.0)
        for col, t in enumerate(sol.t):
            x_anchor = interpolate_source(float(t), interval, interval + 1)
            x_anchor[local_idx_np] = np.maximum(sol_conc[:, col], 0.0)
            anchors_t.append(float(t))
            anchors_x.append(x_anchor[target_hidden_idx_np].copy())

    anchors_time = np.array(anchors_t, dtype=float)
    anchors_hidden = np.vstack(anchors_x).astype(float) if anchors_x else np.empty((0, len(target_hidden_idx_np)))
    out = args.assets / f"{args.output_prefix}_stiff_hidden_anchors.npz"
    np.savez_compressed(
        out,
        time_s=anchors_time,
        hidden_species_index=target_hidden_idx_np.astype(int),
        hidden_concentration_M=anchors_hidden,
        local_species_index=local_idx_np.astype(int),
    )
    summary = {
        "output_npz": str(out),
        "source": str(args.pinn),
        "source_state_key": state_key,
        "exchange_mode": args.exchange_mode,
        "rate_unit": args.rate_unit,
        "state_unit": args.state_unit,
        "method": args.method,
        "rtol": float(args.rtol),
        "atol_uM": float(args.atol_uM),
        "max_step_s": float(args.max_step_s),
        "state_transform": args.state_transform,
        "log_floor_uM": float(args.log_floor_uM),
        "use_jac_sparsity": bool(args.use_jac_sparsity),
        "initial_state_mode": args.initial_state_mode,
        "target_hidden_species_count": int(len(target_hidden_idx_np)),
        "local_species_count": int(len(local_idx_np)),
        "anchors_count": int(len(anchors_time)),
        "interval_reports": interval_reports,
    }
    summary_path = args.assets / f"{args.output_prefix}_stiff_hidden_anchors_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
