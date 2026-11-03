#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import simulate_sixgene_pool_chain_v3 as sim


GENES = ["Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno"]


def _parse_float_list(v: str) -> List[float]:
    out: List[float] = []
    for x in str(v).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("float list cannot be empty")
    return out


def _read_uM(df: pd.DataFrame, gene: str, t_s: float) -> float:
    row = df.loc[(df["time"] - t_s).abs().idxmin()]
    return float(row[f"[{gene}_m]"]) * 1e6


def _make_args(
    panel_path: Path,
    formal_summary_json: Path,
    pool_json: Path,
    outdir: Path,
    *,
    dna_total_M: float,
    t_end_h: float,
    dt_s: float,
    record_every_s: float,
    bg_tx_coupling_power: float,
    energy_scale: float,
    mrna_decay_base: float,
    mrna_bound_decay_ratio: float,
    global_activity_decay_h: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        gene_config_json=str(panel_path),
        summary_json=str(formal_summary_json),
        outdir=str(outdir),
        dna_total_M=float(dna_total_M),
        t_end_h=float(t_end_h),
        dt_s=float(dt_s),
        record_every_s=float(record_every_s),
        ribo_bind_scale=1.0,
        tx_gain=1.0,
        tl_gain=1.0,
        mrna_decay_base=float(mrna_decay_base),
        mrna_bound_decay_ratio=float(mrna_bound_decay_ratio),
        global_activity_decay_h=float(global_activity_decay_h),
        strict_crn_mode=1,
        bg_activity_mode="formal",
        bg_tx_coupling_power=float(bg_tx_coupling_power),
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


def _run_panel(panel_payload: dict, panel_outdir: Path, args_obj: SimpleNamespace) -> Tuple[pd.DataFrame, dict, float]:
    panel_outdir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_outdir / "panel.json"
    panel_path.write_text(json.dumps(panel_payload, indent=2), encoding="utf-8")
    sim._load_gene_templates.cache_clear()
    t0 = time.time()
    df, summary = sim.simulate(args_obj)
    runtime_s = time.time() - t0
    df.to_csv(panel_outdir / "timeseries.csv", index=False)
    summary_with_runtime = dict(summary)
    summary_with_runtime["runtime_s"] = float(runtime_s)
    (panel_outdir / "simulation_summary.json").write_text(
        json.dumps(summary_with_runtime, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return df, summary_with_runtime, runtime_s


def _run_named_panel(job: tuple[str, dict, Path, SimpleNamespace]) -> tuple[str, pd.DataFrame, dict, float]:
    name, panel_payload, panel_outdir, args_obj = job
    df, summary, runtime_s = _run_panel(panel_payload, panel_outdir, args_obj)
    return name, df, summary, runtime_s


def _single_panel(
    base_panel: dict,
    gene: str,
    tx_uniform: float,
    tl_uniform: float,
    dna_single_M: float,
    tx_gene_multipliers: dict[str, float],
    activity_decay_by_gene: dict[str, float],
) -> dict:
    gg = deepcopy(next(x for x in base_panel["genes"] if x["name"] == gene))
    gg["tx_scale"] = float(tx_uniform) * float(tx_gene_multipliers.get(gene, 1.0))
    gg["tl_scale"] = float(tl_uniform)
    gg["dna_M"] = float(dna_single_M)
    gg["dna_weight"] = 1.0
    if gene in activity_decay_by_gene:
        gg["activity_decay_h"] = float(activity_decay_by_gene[gene])
    return {
        "dna_total_M": float(dna_single_M),
        "dna_allocation_rule": "single_each_2p8nM",
        "genes": [gg],
    }


def _mux_panel(
    base_panel: dict,
    tx_uniform: float,
    tl_uniform: float,
    dna_total_M: float,
    *,
    dna_mode: str = "total",
    tx_gene_multipliers: dict[str, float],
    activity_decay_by_gene: dict[str, float],
) -> dict:
    out = deepcopy(base_panel)
    if dna_mode == "total":
        each = float(dna_total_M / len(out["genes"]))
        total = float(dna_total_M)
        rule = "reported_2p8nM_interpreted_as_total_equal_split"
    elif dna_mode == "per_gene":
        each = float(dna_total_M)
        total = float(dna_total_M * len(out["genes"]))
        rule = "reported_2p8nM_interpreted_as_per_template"
    else:
        raise ValueError(f"Unknown mux DNA mode: {dna_mode}")
    out["dna_total_M"] = total
    out["dna_allocation_rule"] = rule
    for g in out["genes"]:
        g["tx_scale"] = float(tx_uniform) * float(tx_gene_multipliers.get(str(g["name"]), 1.0))
        g["tl_scale"] = float(tl_uniform)
        g["dna_M"] = each
        g["dna_weight"] = 1.0
        if str(g["name"]) in activity_decay_by_gene:
            g["activity_decay_h"] = float(activity_decay_by_gene[str(g["name"])])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/unified_shared_scan"))
    p.add_argument(
        "--panel-json",
        type=Path,
        default=Path("output_sixgene_single_gene_txfit_then_all_bgtx2p25_muxTotal2p8_20260425/sixgene_panel_single_gene_txfit_absolutechains.json"),
    )
    p.add_argument(
        "--single-proxy-csv",
        type=Path,
        default=Path("output_sixgene_single_gene_proxy_paperanchored_20260424/selected_sixgene_single_gene_vs_lit_paperanchored.csv"),
    )
    p.add_argument(
        "--formal-summary-json",
        type=Path,
        default=Path("output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"),
    )
    p.add_argument(
        "--pool-json",
        type=Path,
        default=Path("input_pool_override_sg_pure2_paper_mapped_20260424.json"),
    )
    p.add_argument("--tx-list", type=str, default="3000")
    p.add_argument("--energy-list", type=str, default="0.19,0.20,0.21,0.22,0.23,0.24")
    p.add_argument("--tl-uniform", type=float, default=2800.0)
    p.add_argument("--dna-total-M", type=float, default=2.8e-9)
    p.add_argument(
        "--mux-dna-mode",
        choices=("total", "per_gene"),
        default="total",
        help="Interpret the reported 2.8 nM for All G.E. as total DNA or as 2.8 nM for each template.",
    )
    p.add_argument("--t-end-h", type=float, default=6.0)
    p.add_argument("--dt-s", type=float, default=2.0)
    p.add_argument("--record-every-s", type=float, default=10800.0)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel panel simulations per parameter combination; numerical results are unchanged.",
    )
    p.add_argument(
        "--skip-mux",
        action="store_true",
        help="Run only the six single-expression calibration conditions; final validation must omit this flag.",
    )
    p.add_argument("--bg-tx-coupling-power", type=float, default=2.25)
    p.add_argument("--mrna-decay-base", type=float, default=0.0)
    p.add_argument("--mrna-bound-decay-ratio", type=float, default=0.0)
    p.add_argument("--global-activity-decay-h", type=float, default=0.0)
    p.add_argument(
        "--tx-gene-multipliers-json",
        type=Path,
        default=None,
        help="Optional JSON object mapping each gene to a dimensionless sequence-specific TX multiplier.",
    )
    p.add_argument(
        "--activity-decay-gene-json",
        type=Path,
        default=None,
        help="Optional JSON object mapping each construct to an effective activity-decay rate in h^-1.",
    )
    p.add_argument("--score-w-mare14", type=float, default=0.7)
    p.add_argument("--score-w-pgi-eno", type=float, default=0.3)
    p.add_argument(
        "--ranking-metric",
        choices=("mare14", "legacy_weighted"),
        default="mare14",
        help="Use transparent 14-point MARE by default; legacy weighting is retained only for reproduction.",
    )
    cli = p.parse_args()

    outdir = cli.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    tx_list = _parse_float_list(cli.tx_list)
    energy_list = _parse_float_list(cli.energy_list)

    base_panel = json.loads(cli.panel_json.read_text(encoding="utf-8"))
    tx_gene_multipliers = (
        {str(key): float(value) for key, value in json.loads(cli.tx_gene_multipliers_json.read_text(encoding="utf-8")).items()}
        if cli.tx_gene_multipliers_json is not None
        else {}
    )
    activity_decay_by_gene = (
        {str(key): float(value) for key, value in json.loads(cli.activity_decay_gene_json.read_text(encoding="utf-8")).items()}
        if cli.activity_decay_gene_json is not None
        else {}
    )
    single_proxy = pd.read_csv(cli.single_proxy_csv)

    single_t3 = {str(r["gene"]): float(r["lit_3h_uM_single_proxy"]) for _, r in single_proxy.iterrows()}
    single_t6 = {str(r["gene"]): float(r["lit_6h_uM_single_proxy"]) for _, r in single_proxy.iterrows()}
    all_t3 = float(single_proxy["all_GE_total_3h_uM_proxy"].iloc[0])
    all_t6 = float(single_proxy["all_GE_total_6h_uM_proxy"].iloc[0])

    rows: List[Dict[str, float]] = []
    combo_id = 0

    for tx_uniform in tx_list:
        for energy_scale in energy_list:
            combo_id += 1
            case = f"tx{int(round(tx_uniform))}_e{energy_scale:.2f}"
            case_dir = outdir / case
            case_dir.mkdir(parents=True, exist_ok=True)

            case_start = time.time()
            per_gene: Dict[str, float] = {}
            jobs: list[tuple[str, dict, Path, SimpleNamespace]] = []

            for gene in GENES:
                panel_single = _single_panel(
                    base_panel=base_panel,
                    gene=gene,
                    tx_uniform=float(tx_uniform),
                    tl_uniform=float(cli.tl_uniform),
                    dna_single_M=float(cli.dna_total_M),
                    tx_gene_multipliers=tx_gene_multipliers,
                    activity_decay_by_gene=activity_decay_by_gene,
                )
                panel_single_dir = case_dir / f"single_{gene}"
                args_single = _make_args(
                    panel_path=panel_single_dir / "panel.json",
                    formal_summary_json=cli.formal_summary_json,
                    pool_json=cli.pool_json,
                    outdir=panel_single_dir,
                    dna_total_M=float(cli.dna_total_M),
                    t_end_h=float(cli.t_end_h),
                    dt_s=float(cli.dt_s),
                    record_every_s=float(cli.record_every_s),
                    bg_tx_coupling_power=float(cli.bg_tx_coupling_power),
                    energy_scale=float(energy_scale),
                    mrna_decay_base=float(cli.mrna_decay_base),
                    mrna_bound_decay_ratio=float(cli.mrna_bound_decay_ratio),
                    global_activity_decay_h=float(cli.global_activity_decay_h),
                )
                jobs.append((gene, panel_single, panel_single_dir, args_single))

            panel_mux = _mux_panel(
                base_panel=base_panel,
                tx_uniform=float(tx_uniform),
                tl_uniform=float(cli.tl_uniform),
                dna_total_M=float(cli.dna_total_M),
                dna_mode=str(cli.mux_dna_mode),
                tx_gene_multipliers=tx_gene_multipliers,
                activity_decay_by_gene=activity_decay_by_gene,
            )
            mux_dir = case_dir / "mux"
            args_mux = _make_args(
                panel_path=mux_dir / "panel.json",
                formal_summary_json=cli.formal_summary_json,
                pool_json=cli.pool_json,
                outdir=mux_dir,
                dna_total_M=float(cli.dna_total_M),
                t_end_h=float(cli.t_end_h),
                dt_s=float(cli.dt_s),
                record_every_s=float(cli.record_every_s),
                bg_tx_coupling_power=float(cli.bg_tx_coupling_power),
                energy_scale=float(energy_scale),
                mrna_decay_base=float(cli.mrna_decay_base),
                mrna_bound_decay_ratio=float(cli.mrna_bound_decay_ratio),
                global_activity_decay_h=float(cli.global_activity_decay_h),
            )
            if not cli.skip_mux:
                jobs.append(("mux", panel_mux, mux_dir, args_mux))

            results: dict[str, pd.DataFrame] = {}
            worker_count = max(1, min(int(cli.workers), len(jobs)))
            if worker_count == 1:
                for job in jobs:
                    name, frame, _summary, _runtime = _run_named_panel(job)
                    results[name] = frame
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {executor.submit(_run_named_panel, job): job[0] for job in jobs}
                    for future in as_completed(futures):
                        name, frame, _summary, _runtime = future.result()
                        results[name] = frame

            for gene in GENES:
                df_s = results[gene]
                s3 = _read_uM(df_s, gene, 10800.0)
                s6 = _read_uM(df_s, gene, 21600.0)
                per_gene[f"{gene}_3h"] = s3
                per_gene[f"{gene}_6h"] = s6
                per_gene[f"{gene}_ape3"] = abs(s3 - single_t3[gene]) / max(single_t3[gene], 1e-12)
                per_gene[f"{gene}_ape6"] = abs(s6 - single_t6[gene]) / max(single_t6[gene], 1e-12)

            single_mare3 = float(np.mean([per_gene[f"{g}_ape3"] for g in GENES]))
            single_mare6 = float(np.mean([per_gene[f"{g}_ape6"] for g in GENES]))
            mare12 = float((single_mare3 + single_mare6) / 2.0)
            if cli.skip_mux:
                all3 = all6 = all_ape3 = all_ape6 = float("nan")
                mare14 = float("nan")
            else:
                df_m = results["mux"]
                all3 = float(sum(_read_uM(df_m, g, 10800.0) for g in GENES))
                all6 = float(sum(_read_uM(df_m, g, 21600.0) for g in GENES))
                all_ape3 = abs(all3 - all_t3) / max(all_t3, 1e-12)
                all_ape6 = abs(all6 - all_t6) / max(all_t6, 1e-12)
                mare14 = float((12.0 * mare12 + all_ape3 + all_ape6) / 14.0)
            pgi_eno_mare = float((per_gene["Pgi_ape3"] + per_gene["Pgi_ape6"] + per_gene["Eno_ape3"] + per_gene["Eno_ape6"]) / 4.0)
            pfka_gapdh_mare = float((per_gene["PfkA_ape3"] + per_gene["PfkA_ape6"] + per_gene["GAPDH_ape3"] + per_gene["GAPDH_ape6"]) / 4.0)

            ratio_errors = [
                abs((per_gene[f"{g}_6h"] / max(per_gene[f"{g}_3h"], 1e-12)) - (single_t6[g] / single_t3[g]))
                / max(single_t6[g] / single_t3[g], 1e-12)
                for g in GENES
            ]
            if not cli.skip_mux:
                ratio_errors.append(
                    abs((all6 / max(all3, 1e-12)) - (all_t6 / all_t3)) / max(all_t6 / all_t3, 1e-12)
                )
            ratio_mare7 = float(np.mean(ratio_errors))
            if cli.skip_mux:
                score = mare12
            elif cli.ranking_metric == "mare14":
                score = mare14
            else:
                score = float(cli.score_w_mare14) * mare14 + float(cli.score_w_pgi_eno) * pgi_eno_mare

            row: Dict[str, float] = {
                "combo_id": int(combo_id),
                "case": case,
                "tx_uniform": float(tx_uniform),
                "tl_uniform": float(cli.tl_uniform),
                "energy_scale": float(energy_scale),
                "mrna_decay_base_s-1": float(cli.mrna_decay_base),
                "mrna_bound_decay_ratio": float(cli.mrna_bound_decay_ratio),
                "global_activity_decay_h-1": float(cli.global_activity_decay_h),
                "tx_gene_multipliers_json": json.dumps(tx_gene_multipliers, sort_keys=True),
                "activity_decay_by_gene_json": json.dumps(activity_decay_by_gene, sort_keys=True),
                "dt_s": float(cli.dt_s),
                "mux_dna_mode": str(cli.mux_dna_mode),
                "mux_dna_total_M": float(panel_mux["dna_total_M"]),
                "single_mare3": single_mare3,
                "single_mare6": single_mare6,
                "mare12_single": mare12,
                "mare14": mare14,
                "pgi_eno_mare": pgi_eno_mare,
                "pfka_gapdh_mare": pfka_gapdh_mare,
                "all3_uM": all3,
                "all6_uM": all6,
                "all_ape3": all_ape3,
                "all_ape6": all_ape6,
                "score": score,
                "ranking_metric": str(cli.ranking_metric),
                "calibration_only_skip_mux": bool(cli.skip_mux),
                "ratio_mare7": ratio_mare7,
                "runtime_s": float(time.time() - case_start),
            }
            for k, v in per_gene.items():
                row[k] = float(v)
            rows.append(row)

            pd.DataFrame(rows).to_csv(outdir / "shared_scan_results.csv", index=False)
            print(
                f"[done] {case} | score={score:.6f} mare12={mare12:.6f} mare14={mare14:.6f} "
                f"pgi_eno={pgi_eno_mare:.6f} all_ape3={all_ape3:.6f} all_ape6={all_ape6:.6f}",
                flush=True,
            )

    df_all = pd.DataFrame(rows)
    df_ranked = df_all.sort_values("score", ascending=True).reset_index(drop=True)
    df_ranked.to_csv(outdir / "shared_scan_results_ranked.csv", index=False)

    top_n = min(15, len(df_ranked))
    print("\nTop candidates by score:")
    print(
        df_ranked[
            [
                "case",
                "tx_uniform",
                "energy_scale",
                "score",
                "mare14",
                "pgi_eno_mare",
                "pfka_gapdh_mare",
                "all_ape3",
                "all_ape6",
            ]
        ]
        .head(top_n)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
