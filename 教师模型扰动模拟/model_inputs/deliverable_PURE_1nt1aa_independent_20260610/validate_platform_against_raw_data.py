#!/usr/bin/env python3
"""Validate single-protein TX/TL plateaus against the paper's raw data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import tellurium as te

import fit_batch_then_continuous_full_crn_background as core
from run_tx1nt_tlg1_micro3_longrun import apply_params, load_sources, set_fine_node_env


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
RAW_ROOT = PACKAGE_DIR / "raw_literature_data"
RAW_DIR = RAW_ROOT / "MGapt_deGFP_Exp_PURE_Data"
OUTDIR = PACKAGE_DIR / "results/platform_raw_validation"

TX_RAW = RAW_DIR / "2022.08.19_OnePot_PURE_mGapt_uM.csv"
TL_RAW = RAW_DIR / "2022.11.14_OnePotPURE_deGFPuM.csv"
TL_ONLY_RAW = RAW_ROOT / "Calibration_MGapt" / "2023.07.28_PUREwmRNA_updated.xlsx"
MGAPT_SOURCE = PAPER_DIR / "CRN_PURE_TX_mGaptv4_updated.xml"

PARAM_SUMMARY = PACKAGE_DIR / (
    "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"
)
HF_SUMMARY = PACKAGE_DIR / "output_full_crn_background/batch_then_continuous_summary.json"


def prepare_mgapt_source() -> Path:
    """Adapt the paper's 90-nt MGapt TX chain to the project's active IDs."""
    text = MGAPT_SOURCE.read_text(encoding="utf-8")
    text = text.replace("RNAPa", "T7RNAP")
    text = text.replace('id="mRNA" name="mRNA"', 'id="mRNA_t" name="mRNA_t"')
    text = text.replace('species="mRNA"', 'species="mRNA_t"')
    out = OUTDIR / "MGapt_90nt_T7RNAP_mRNA_t_source.xml"
    out.write_text(text, encoding="utf-8")
    return out


def simulate_tx_only(
    params: dict[str, float],
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # Match the raw MGapt-only experiment and retain the project's cleaned,
    # explicit 1-nt transcription topology.
    core.TX_1NT_SOURCE_SBML = source_path
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    ic_uM.update(
        {
            "DNA": 0.005,
            "T7RNAP": 0.1,
            "ATP": 3000.0,
            "GTP": 3000.0,
            "CTP": 1000.0,
            "UTP": 1000.0,
            "Ribosome": 0.0,
            "RS70S": 0.0,
        }
    )
    sbml, counts, _ = core.build_background_sbml(
        mode="batch",
        fitted_hf={},
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=None,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)
    selections = ["time", "[mRNA_t]", "[DNA]", "[T7RNAP]"]
    res = rr.simulate(0.0, 15.0 * 3600.0, 601, selections)
    df = pd.DataFrame(res, columns=res.colnames)
    meta = {
        "counts": counts,
        "param_apply": param_apply,
        "conditions": {
            "construct": "MGapt-only, 90 nt",
            "DNA_uM": 0.005,
            "T7RNAP_uM": 0.1,
            "ATP_uM": 3000.0,
            "GTP_uM": 3000.0,
            "CTP_uM": 1000.0,
            "UTP_uM": 1000.0,
            "translation_disabled": True,
        },
    }
    return df, meta


def simulate_combined_1nm(
    params: dict[str, float],
    hf_fit: dict[str, dict[str, float]],
    keff_map: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    # Match the 2022-11-14 MGapt-deGFP experiment, which used 1 nM DNA.
    core.TX_1NT_SOURCE_SBML = core.PACKAGE_DIR / (
        "external_PURE_CRN_models_sparse/Models/TXTL_PURE_MGapt-deGFP_model.xml"
    )
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    ic_uM["DNA"] = 0.001
    sbml, counts, _ = core.build_background_sbml(
        mode="batch",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)
    df = core.simulate_degfp(rr, t_end_s=6.0 * 3600.0, n_points=451, reset_to_origin=False)
    meta = {
        "counts": counts,
        "param_apply": param_apply,
        "conditions": {
            "construct": "MGapt-UTR1-deGFP",
            "DNA_uM": 0.001,
            "T7RNAP_uM": 1.0,
            "experiment_date": "2022-11-14",
        },
    }
    return df, meta


def simulate_translation_only(
    params: dict[str, float],
    hf_fit: dict[str, dict[str, float]],
    keff_map: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Match the 12 h PURExpress experiment with purified mRNA input."""
    core.TX_1NT_SOURCE_SBML = core.PACKAGE_DIR / (
        "external_PURE_CRN_models_sparse/Models/TXTL_PURE_MGapt-deGFP_model.xml"
    )
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    ic_uM.update({"DNA": 0.0, "T7RNAP": 0.0, "mRNA_t": 0.548})
    sbml, counts, _ = core.build_background_sbml(
        mode="batch",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    param_apply = apply_params(rr, params)
    df = core.simulate_degfp(rr, t_end_s=12.0 * 3600.0, n_points=601, reset_to_origin=False)
    meta = {
        "counts": counts,
        "param_apply": param_apply,
        "conditions": {
            "construct": "purified MGapt-UTR1-deGFP mRNA",
            "initial_mRNA_uM": 0.548,
            "DNA_uM": 0.0,
            "T7RNAP_uM": 0.0,
            "experiment_date": "2023-07-28",
            "calibration": "negative-control subtraction, then RFU/5542.4; each replicate zeroed at t0",
        },
    }
    return df, meta


def compare_window(
    exp_time_h: np.ndarray,
    exp_replicates_uM: np.ndarray,
    model_time_h: np.ndarray,
    model_uM: np.ndarray,
    start_h: float,
    end_h: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    mask = (exp_time_h >= start_h - 1e-12) & (exp_time_h <= end_h + 1e-12)
    t = exp_time_h[mask]
    reps = exp_replicates_uM[mask, :]
    exp_mean = np.mean(reps, axis=1)
    exp_sd = np.std(reps, axis=1, ddof=1)
    pred = np.interp(t, model_time_h, model_uM)
    ape = np.abs(pred - exp_mean) / exp_mean * 100.0
    table = pd.DataFrame(
        {
            "time_h": t,
            "experiment_mean_uM": exp_mean,
            "experiment_sd_uM": exp_sd,
            "model_uM": pred,
            "absolute_relative_error_percent": ape,
        }
    )
    exp_slope = float(np.polyfit(t, exp_mean, 1)[0])
    model_slope = float(np.polyfit(t, pred, 1)[0])
    exp_level = float(np.mean(exp_mean))
    model_level = float(np.mean(pred))
    metrics = {
        "window_start_h": float(start_h),
        "window_end_h": float(end_h),
        "n_timepoints": int(t.size),
        "experimental_plateau_mean_uM": exp_level,
        "model_plateau_mean_uM": model_level,
        "platform_mean_relative_error_percent": float(abs(model_level - exp_level) / exp_level * 100.0),
        "mean_absolute_relative_error_percent": float(np.mean(ape)),
        "median_absolute_relative_error_percent": float(np.median(ape)),
        "experimental_slope_uM_per_h": exp_slope,
        "model_slope_uM_per_h": model_slope,
        "experimental_relative_slope_percent_per_h": float(abs(exp_slope) / exp_level * 100.0),
        "model_relative_slope_percent_per_h": float(abs(model_slope) / max(abs(model_level), 1e-15) * 100.0),
    }
    return table, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse saved model trajectories and only rebuild comparisons/figure.",
    )
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    os.environ["PURE_MAX_STEPS"] = "600000"
    set_fine_node_env()
    params, hf_fit, keff_map = load_sources(PARAM_SUMMARY, HF_SUMMARY)

    tx_saved = OUTDIR / "tx_only_mgapt_15h_model.csv"
    combined_saved = OUTDIR / "combined_1nm_6h_model.csv"
    tl_only_saved = OUTDIR / "translation_only_0p548uM_mRNA_12h_model.csv"
    if args.reuse and tx_saved.exists() and combined_saved.exists():
        tx_model = pd.read_csv(tx_saved)
        combined_model = pd.read_csv(combined_saved)
        previous = json.loads((OUTDIR / "platform_validation_summary.json").read_text(encoding="utf-8"))
        tx_meta = {k: v for k, v in previous["transcription"].items() if k not in {"metrics", "raw_data"}}
        old_combined = previous.get("combined_expression", previous.get("translation", {}))
        combined_meta = {k: v for k, v in old_combined.items() if k not in {"metrics", "raw_data"}}
        if tl_only_saved.exists() and "translation_only" in previous:
            tl_only_model = pd.read_csv(tl_only_saved)
            tl_only_meta = {
                k: v for k, v in previous["translation_only"].items() if k not in {"metrics", "raw_data"}
            }
        else:
            tl_only_model, tl_only_meta = simulate_translation_only(params, hf_fit, keff_map)
            tl_only_model.to_csv(tl_only_saved, index=False)
    else:
        source_path = prepare_mgapt_source()
        tx_model, tx_meta = simulate_tx_only(params, source_path)
        combined_model, combined_meta = simulate_combined_1nm(params, hf_fit, keff_map)
        tl_only_model, tl_only_meta = simulate_translation_only(params, hf_fit, keff_map)
        tl_only_model.to_csv(tl_only_saved, index=False)

    tx_raw = pd.read_csv(TX_RAW)
    tx_time_h = tx_raw["timepoints"].to_numpy(dtype=float) / 3600.0
    tx_reps = tx_raw[["mRNA0", "mRNA1", "mRNA2", "mRNA3"]].to_numpy(dtype=float)
    tx_table, tx_metrics = compare_window(
        tx_time_h,
        tx_reps,
        tx_model["time"].to_numpy(dtype=float) / 3600.0,
        tx_model["[mRNA_t]"].to_numpy(dtype=float) * 1e6,
        10.0,
        15.0,
    )

    combined_raw = pd.read_csv(TL_RAW, skipinitialspace=True)
    combined_raw.columns = [str(c).strip() for c in combined_raw.columns]
    combined_time_h = combined_raw["Time"].to_numpy(dtype=float) / 60.0
    combined_reps = combined_raw[["deGFP_1", "deGFP_2", "deGFP_3"]].to_numpy(dtype=float)
    combined_table, combined_metrics = compare_window(
        combined_time_h,
        combined_reps,
        combined_model["time"].to_numpy(dtype=float) / 3600.0,
        combined_model["[deGFP_m]"].to_numpy(dtype=float) * 1e6,
        4.0,
        6.0,
    )

    tl_only_raw = pd.read_excel(TL_ONLY_RAW, sheet_name="gfp61_data")
    tl_only_time_h = np.array(
        [v.hour + v.minute / 60.0 + v.second / 3600.0 for v in tl_only_raw["Time"]],
        dtype=float,
    )
    tl_only_neg = tl_only_raw["B20"].to_numpy(dtype=float)
    tl_only_reps = tl_only_raw[["B21", "B22", "B23"]].to_numpy(dtype=float)
    tl_only_reps = (tl_only_reps - tl_only_neg[:, None]) / 5542.4
    tl_only_reps = tl_only_reps - tl_only_reps[0, :]
    tl_only_table, tl_only_metrics = compare_window(
        tl_only_time_h,
        tl_only_reps,
        tl_only_model["time"].to_numpy(dtype=float) / 3600.0,
        tl_only_model["[deGFP_m]"].to_numpy(dtype=float) * 1e6,
        8.0,
        12.0,
    )

    tx_model.to_csv(tx_saved, index=False)
    combined_model.to_csv(combined_saved, index=False)
    tl_only_model.to_csv(tl_only_saved, index=False)
    tx_table.to_csv(OUTDIR / "tx_platform_10_15h_comparison.csv", index=False)
    combined_table.to_csv(OUTDIR / "combined_expression_platform_4_6h_comparison.csv", index=False)
    tl_only_table.to_csv(OUTDIR / "translation_only_platform_8_12h_comparison.csv", index=False)

    summary = {
        "method": {
            "error_definition": "MARE = mean(|model - replicate mean| / replicate mean) * 100%",
            "near_zero_points_excluded": True,
            "transcription_platform_window_h": [10.0, 15.0],
            "translation_only_platform_window_h": [8.0, 12.0],
            "combined_expression_platform_window_h": [4.0, 6.0],
        },
        "transcription": {**tx_meta, "metrics": tx_metrics, "raw_data": str(TX_RAW)},
        "translation_only": {
            **tl_only_meta,
            "metrics": tl_only_metrics,
            "raw_data": str(TL_ONLY_RAW),
        },
        "combined_expression": {
            **combined_meta,
            "metrics": combined_metrics,
            "raw_data": str(TL_RAW),
        },
    }
    (OUTDIR / "platform_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.5), constrained_layout=True)
    tx_mean = np.mean(tx_reps, axis=1)
    tx_sd = np.std(tx_reps, axis=1, ddof=1)
    axes[0].fill_between(tx_time_h, tx_mean - tx_sd, tx_mean + tx_sd, color="#8ab6d6", alpha=0.25)
    axes[0].plot(tx_time_h, tx_mean, color="#236192", lw=1.6, label="Raw experiment mean")
    axes[0].plot(
        tx_model["time"] / 3600.0,
        tx_model["[mRNA_t]"] * 1e6,
        color="#c43c39",
        lw=1.8,
        label="Cleaned 1-nt model",
    )
    axes[0].axvspan(10.0, 15.0, color="#6f8f72", alpha=0.12, label="Platform window")
    axes[0].set(xlabel="Time (h)", ylabel="MGapt RNA (uM)", title="Transcription-only, 5 nM MGapt DNA")
    axes[0].legend(frameon=False, fontsize=8)

    tl_only_mean = np.mean(tl_only_reps, axis=1)
    tl_only_sd = np.std(tl_only_reps, axis=1, ddof=1)
    axes[1].fill_between(
        tl_only_time_h,
        tl_only_mean - tl_only_sd,
        tl_only_mean + tl_only_sd,
        color="#8ab6d6",
        alpha=0.25,
    )
    axes[1].plot(tl_only_time_h, tl_only_mean, color="#236192", lw=1.6, label="Raw experiment mean")
    axes[1].plot(
        tl_only_model["time"] / 3600.0,
        tl_only_model["[deGFP_m]"] * 1e6,
        color="#c43c39",
        lw=1.8,
        label="Cleaned 1-nt/1-aa model",
    )
    axes[1].axvspan(8.0, 12.0, color="#6f8f72", alpha=0.12, label="Platform window")
    axes[1].set(
        xlabel="Time (h)",
        ylabel="Mature deGFP (uM)",
        title="Translation-only, 0.548 uM mRNA",
    )
    axes[1].legend(frameon=False, fontsize=8)

    combined_mean = np.mean(combined_reps, axis=1)
    combined_sd = np.std(combined_reps, axis=1, ddof=1)
    axes[2].fill_between(
        combined_time_h,
        combined_mean - combined_sd,
        combined_mean + combined_sd,
        color="#8ab6d6",
        alpha=0.25,
    )
    axes[2].plot(combined_time_h, combined_mean, color="#236192", lw=1.6, label="Raw experiment mean")
    axes[2].plot(
        combined_model["time"] / 3600.0,
        combined_model["[deGFP_m]"] * 1e6,
        color="#c43c39",
        lw=1.8,
        label="Cleaned 1-nt/1-aa model",
    )
    axes[2].axvspan(4.0, 6.0, color="#6f8f72", alpha=0.12, label="Platform window")
    axes[2].set(
        xlabel="Time (h)",
        ylabel="Mature deGFP (uM)",
        title="Combined expression, 1 nM DNA",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    fig.savefig(OUTDIR / "platform_raw_data_validation.png", dpi=220)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
