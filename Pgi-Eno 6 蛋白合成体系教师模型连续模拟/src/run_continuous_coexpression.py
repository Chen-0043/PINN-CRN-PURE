#!/usr/bin/env python3
"""Run a batch-validated six-protein model under the continuous boundary.

Only the reactor boundary is changed: feed substrates are
relaxed toward their initial reservoir concentrations, dialyzable products are
washed toward zero, and all six protein products leave at the same dilution
rate.  No multi-protein parameter is refitted in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_sixgene_pool_chain_v3 as sim


GENES = ["Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno"]
DEFAULT_FORMAL_CASE = "fusion_final_per_template_2p8nM"
DEFAULT_PANEL = (
    ROOT
    / "outputs"
    / "fusion_construct_final_validation_pergeneDNA_dt0p125"
    / "tx10000_e0.80"
    / "mux"
    / "panel.json"
)
SUMMARY = ROOT / "output_full_crn_background" / "formal_mech_only_20260409_210539_rechecked.json"
POOL = ROOT / "input_pool_override_sg_pure2_sixprotein_aa0p5mM.json"


def build_args(
    t_end_h: float,
    dt_s: float,
    record_every_s: float,
    *,
    panel: Path,
    pool_json: Path,
    energy_scale: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        gene_config_json=str(panel),
        summary_json=str(SUMMARY),
        dna_total_M=2.8e-9,
        t_end_h=float(t_end_h),
        dt_s=float(dt_s),
        record_every_s=float(record_every_s),
        ribo_bind_scale=1.0,
        tx_gain=1.0,
        tl_gain=1.0,
        mrna_decay_base=0.0,
        mrna_bound_decay_ratio=0.0,
        global_activity_decay_h=0.0,
        strict_crn_mode=1,
        bg_activity_mode="formal",
        bg_tx_coupling_power=2.25,
        pool_init_json=str(pool_json),
        sg_public_mode="off",
        sg_energy_mode="sg_qssa",
        sg_energy_k_atp_reg_M_s=2e-6 * float(energy_scale),
        sg_energy_k_ndk_M_s=6e-6 * float(energy_scale),
        sg_energy_K_adp_reg_M=4e-4,
        sg_energy_K_gdp_ndk_M=3e-4,
        sg_energy_K_atp_ndk_M=5e-4,
        sg_energy_K_glucose_M=5e-3,
        sg_energy_K_nad_M=2e-4,
        sg_energy_K_tpp_M=2e-5,
        sg_energy_K_mg_M=4e-3,
        sg_energy_atp_per_glucose=2.0,
        sg_tx_boost_max=0.0,
        sg_ref_glucose_M=0.05,
        sg_ref_nad_M=0.001,
        sg_ref_tpp_M=1e-4,
        sg_ref_mg_M=0.016,
        sg_ref_po4_M=0.05,
        sg_k_adp_M=1e-3,
        sg_k_gdp_M=1e-3,
        sg_k_po4_excess_M=5e-3,
        sg_tl_gtp_gate_K_M=0.0,
        sg_tl_gtp_gate_n=6.0,
    )


def build_exchange_policy(pool: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    # This is the six-protein analogue of the deGFP continuous boundary.  The
    # solver consumes explicit AA20 pools rather than the aggregate AA pool.
    feed_ids = ["ATP", "GTP", "CTP", "UTP", "Glucose", "NAD", "TPP", "Mg", *sim.AA20]
    washout_ids = ["ADP", "GDP", "PPi", "PO4", "NADH"]
    feed = {sid: float(pool[sid]) for sid in feed_ids if sid in pool}
    washout = [sid for sid in washout_ids if sid in pool]
    return feed, washout


def apply_continuous_boundary(
    pool: dict[str, float],
    genes,
    feed: dict[str, float],
    washout: list[str],
    gamma_h: float,
    dt_s: float,
) -> None:
    # Exact update for dx/dt = gamma * (x_feed - x), avoiding a time-step
    # dependent exchange approximation.
    retention = math.exp(-float(gamma_h) * float(dt_s) / 3600.0)
    for sid, target in feed.items():
        current = float(pool.get(sid, 0.0))
        pool[sid] = max(0.0, float(target) + (current - float(target)) * retention)
    for sid in washout:
        pool[sid] = max(0.0, float(pool.get(sid, 0.0)) * retention)
    for gene in genes:
        gene.protein = max(0.0, float(gene.protein) * retention)


def _record_row(
    t_s: float,
    pool: dict[str, float],
    genes,
    *,
    detailed_state: bool = False,
) -> dict[str, float]:
    row: dict[str, float] = {
        "time_s": float(t_s),
        "time_h": float(t_s / 3600.0),
        "All_proteins_uM": float(sum(g.protein for g in genes) * 1e6),
    }
    for sid in ["ATP", "ADP", "GTP", "GDP", "CTP", "UTP", "PPi", "PO4", "Glucose", "NAD", "NADH", "Ribosome", "T7RNAP"]:
        if sid in pool:
            row[f"{sid}_uM"] = float(pool[sid] * 1e6)
    for gene in genes:
        row[f"{gene.name}_uM"] = float(gene.protein * 1e6)
        row[f"mRNA_{gene.name}_uM"] = float(gene.mrna * 1e6)
        row[f"Ribosome_mRNA_{gene.name}_uM"] = float(gene.ribo_mrna * 1e6)
        row[f"TXchain_{gene.name}_total_uM"] = float(np.sum(gene.tx_pos) * 1e6)
        row[f"TLchain_{gene.name}_total_uM"] = float(np.sum(gene.tl_pos) * 1e6)
    if detailed_state:
        # Export every state actually represented by the reduced teacher.  The
        # positional chains remain aggregate constraints when mapped onto the
        # full elementary network; they are never mislabeled as one microstate.
        for sid, value in pool.items():
            row[f"teacher_pool::{sid}_M"] = float(value)
        for gene in genes:
            prefix = f"teacher_gene::{gene.name}"
            row[f"{prefix}::dna_free_M"] = float(gene.dna_free)
            row[f"{prefix}::mrna_M"] = float(gene.mrna)
            row[f"{prefix}::ribo_mrna_M"] = float(gene.ribo_mrna)
            row[f"{prefix}::pept_M"] = float(gene.pept)
            row[f"{prefix}::protein_M"] = float(gene.protein)
            for index, value in enumerate(gene.tx_pos):
                row[f"{prefix}::tx_pos::{index:04d}_M"] = float(value)
            for index, value in enumerate(gene.tl_pos):
                row[f"{prefix}::tl_pos::{index:04d}_M"] = float(value)
    return row


def run(
    t_end_h: float,
    dt_s: float,
    record_every_s: float,
    gamma_h: float,
    *,
    panel: Path = DEFAULT_PANEL,
    pool_json: Path = POOL,
    energy_scale: float = 0.8,
    formal_case: str = DEFAULT_FORMAL_CASE,
    detailed_state: bool = False,
) -> tuple[pd.DataFrame, dict]:
    args = build_args(
        t_end_h,
        dt_s,
        record_every_s,
        panel=panel,
        pool_json=pool_json,
        energy_scale=energy_scale,
    )
    params_bg, species_init = sim._read_model_defaults()
    params = dict(params_bg)
    params.update(sim.etx._load_formal_batch_params(Path(args.summary_json)))

    pool = sim._initial_pool(species_init)
    sim._apply_pool_overrides(pool, sim._load_pool_overrides(Path(args.pool_init_json)))
    initial_pool = dict(pool)
    feed, washout = build_exchange_policy(pool)
    genes = sim._load_genes(Path(args.gene_config_json), float(args.dna_total_M))
    dna_total_initial_M = float(sum(g.dna_free for g in genes))
    sim._validate_strict_crn_args(args, genes)

    dt = float(dt_s)
    t_end_s = float(t_end_h) * 3600.0
    record_every_s = max(dt, float(record_every_s))
    rows = [_record_row(0.0, pool, genes, detailed_state=detailed_state)]
    next_record = record_every_s
    t_s = 0.0
    started = time.time()
    resource_limited_steps = 0
    minimum_resource_scale = 1.0

    for _ in range(int(math.ceil(t_end_s / dt))):
        step = min(dt, t_end_s - t_s)
        if step <= 0.0:
            break
        total_delta: dict[str, float] = {}
        pool_before_step = dict(pool)
        has_resource_limiter = all(
            hasattr(sim, name)
            for name in (
                "_snapshot_dynamic_gene_state",
                "_advance_shared_ribosome_binding",
                "_limit_gene_step_to_available_resources",
            )
        )
        gene_snapshots = sim._snapshot_dynamic_gene_state(genes) if has_resource_limiter else None
        bg_factor = sim._bg_activity_factor(pool, params, args.bg_activity_mode)
        tx_factor = sim._sg_public_tx_factor(pool, args)
        if has_resource_limiter:
            sim._advance_shared_ribosome_binding(
                genes,
                pool,
                params,
                step,
                ribo_bind_scale=args.ribo_bind_scale,
            )
        for gene in genes:
            decay_h = (
                float(args.global_activity_decay_h)
                if getattr(gene, "activity_decay_h", None) is None
                else float(getattr(gene, "activity_decay_h"))
            )
            activity_factor = math.exp(-max(0.0, decay_h) * (t_s / 3600.0))
            delta = sim._step_gene(
                gene,
                pool,
                params,
                step,
                ribo_bind_scale=args.ribo_bind_scale,
                mrna_decay_base=args.mrna_decay_base,
                mrna_bound_decay_ratio=args.mrna_bound_decay_ratio,
                bg_keff_factor=bg_factor,
                bg_tx_coupling_power=args.bg_tx_coupling_power,
                activity_factor=activity_factor,
                sg_public_tx_factor=tx_factor,
                sg_tl_gtp_gate_k_M=args.sg_tl_gtp_gate_K_M,
                sg_tl_gtp_gate_n=args.sg_tl_gtp_gate_n,
                strict_crn_mode=True,
            )
            for sid, value in delta.items():
                total_delta[sid] = total_delta.get(sid, 0.0) + float(value)
        resource_scale = (
            sim._limit_gene_step_to_available_resources(
                genes,
                gene_snapshots,
                pool,
                pool_before_step,
                total_delta,
            )
            if has_resource_limiter
            else 1.0
        )
        minimum_resource_scale = min(minimum_resource_scale, resource_scale)
        if resource_scale < 1.0 - 1e-15:
            resource_limited_steps += 1
        sim._apply_delta(pool, total_delta)
        sim._apply_delta(pool, sim._apply_sg_energy_board(pool, args, step))
        apply_continuous_boundary(pool, genes, feed, washout, gamma_h, step)
        t_s += step
        if t_s + 1e-9 >= next_record or t_s >= t_end_s - 1e-9:
            rows.append(_record_row(t_s, pool, genes, detailed_state=detailed_state))
            while next_record <= t_s + 1e-9:
                next_record += record_every_s

    df = pd.DataFrame(rows)
    tail = df.loc[df["time_h"] >= max(0.0, float(t_end_h) - 24.0)].copy()
    stability: dict[str, dict[str, float]] = {}
    simulated_gene_names = [gene.name for gene in genes]
    for gene in simulated_gene_names + ["All_proteins"]:
        col = f"{gene}_uM"
        values = tail[col].to_numpy(dtype=float)
        times = tail["time_h"].to_numpy(dtype=float)
        slope = float(np.polyfit(times, values, 1)[0]) if len(values) >= 2 else float("nan")
        stability[gene] = {
            "final_uM": float(df[col].iloc[-1]),
            "tail_mean_uM": float(np.mean(values)),
            "tail_max_minus_min_uM": float(np.ptp(values)),
            "tail_abs_slope_uM_per_h": abs(slope),
        }

    summary = {
        "workflow": "formal_sixprotein_continuous_coexpression",
        "formal_case": str(formal_case),
        "batch_panel_json": str(panel),
        "batch_pool_json": str(pool_json),
        "batch_energy_scale": float(energy_scale),
        "parameter_refit_for_continuous_case": False,
        "gamma_h_inv": float(gamma_h),
        "product_outflow_gamma_h_inv": float(gamma_h),
        "t_end_h": float(t_end_h),
        "dt_s": float(dt_s),
        "record_every_s": float(record_every_s),
        "detailed_teacher_state_exported": bool(detailed_state),
        "runtime_s": float(time.time() - started),
        "dna_total_nM": float(dna_total_initial_M * 1e9),
        "dna_allocation": "read from the batch-validated panel",
        "gene_activity_decay_h_inv": {
            gene.name: (
                float(args.global_activity_decay_h)
                if getattr(gene, "activity_decay_h", None) is None
                else float(getattr(gene, "activity_decay_h"))
            )
            for gene in genes
        },
        "feed_species_M": feed,
        "washout_species": washout,
        "initial_pool_M": initial_pool,
        "stability_last_24h": stability,
        "resource_conservation_limiter": {
            "enabled": hasattr(sim, "_limit_gene_step_to_available_resources"),
            "limited_steps": int(resource_limited_steps),
            "minimum_step_scale": float(minimum_resource_scale),
            "limited_species": getattr(sim, "RESOURCE_LIMITED_IDS", []),
        },
    }
    return df, summary


def plot(df: pd.DataFrame, output: Path, gamma_h: float) -> None:
    colors = ["#286F6C", "#D55E34", "#4C67A1", "#9A5AA6", "#3A8B5B", "#C58A18"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=220)
    simulated_gene_names = [gene for gene in GENES if f"{gene}_uM" in df.columns]
    for gene, color in zip(simulated_gene_names, colors):
        axes[0].plot(df["time_h"], df[f"{gene}_uM"], lw=1.7, color=color, label=gene)
    axes[0].plot(df["time_h"], df["All_proteins_uM"], lw=2.4, color="#20252B", label="Total")
    axes[0].set(xlabel="Time (h)", ylabel="Protein concentration (uM)", title="Six-protein continuous co-expression")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    for sid, color in zip(["ATP", "GTP", "CTP", "UTP", "PPi", "PO4"], colors):
        axes[1].plot(df["time_h"], df[f"{sid}_uM"], lw=1.7, color=color, label=sid)
    axes[1].set(xlabel="Time (h)", ylabel="Pool concentration (uM)", title=f"Shared resource pools (gamma={gamma_h:g} h$^{{-1}}$)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--t-end-h", type=float, default=72.0)
    parser.add_argument("--dt-s", type=float, default=0.0625)
    parser.add_argument("--record-every-s", type=float, default=300.0)
    parser.add_argument("--panel-json", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--pool-json", type=Path, default=POOL)
    parser.add_argument("--energy-scale", type=float, default=0.8)
    parser.add_argument("--formal-case", default=DEFAULT_FORMAL_CASE)
    parser.add_argument("--detailed-state", action="store_true")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "continuous_coexpression_gamma0p5")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df, summary = run(
        args.t_end_h,
        args.dt_s,
        args.record_every_s,
        args.gamma_h,
        panel=args.panel_json,
        pool_json=args.pool_json,
        energy_scale=args.energy_scale,
        formal_case=args.formal_case,
        detailed_state=args.detailed_state,
    )
    csv_path = args.outdir / "sixprotein_continuous_timeseries.csv"
    json_path = args.outdir / "sixprotein_continuous_summary.json"
    png_path = args.outdir / "sixprotein_continuous_trajectories.png"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(df, png_path, args.gamma_h)
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path), "figure": str(png_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
