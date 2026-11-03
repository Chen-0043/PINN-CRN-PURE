#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTP/UTP perturbation mechanism figure for PURE+ technical report.

Endpoint data are transcribed from the report tables. Over-optimum time-course
data are loaded from the unified 20260723 resimulation table, using the adjacent
comparison convention: 0.03× / 0.01×. No interpolation is used.
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

from perturbation_curve_style_20260728 import BAR_ALPHA, BAR_LEGEND_Y, BAR_ORDER_COLORS, COLORS, LABELS


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_RATIO_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_ratios_20260723.csv"


def setup_font():
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def endpoint_df():
    rows = [
        ("CTP", 0.00, 0.000, 0.00000, 0.00000, 2291.73, 1335.59, 84.20),
        ("CTP", 0.01, 5.707, 0.03624, 0.05739, 30.25, 17.67, 75.59),
        ("CTP", 0.03, 5.574, 0.05203, 0.08218, 16.45, 9.58, 76.14),
        ("CTP", 0.25, 5.566, 0.05581, 0.08802, 14.85, 8.65, 76.83),
        ("CTP", 1.00, 5.565, 0.05595, 0.08823, 14.80, 8.62, 76.85),
        ("CTP", 4.00, 5.565, 0.05598, 0.08828, 14.79, 8.61, 76.85),
        ("UTP", 0.00, 0.000, 0.00000, 0.00000, 2291.72, 1335.59, 84.21),
        ("UTP", 0.01, 5.570, 0.05155, 0.08144, 16.68, 9.71, 75.84),
        ("UTP", 0.03, 5.566, 0.05571, 0.08786, 14.89, 8.67, 76.82),
        ("UTP", 0.25, 5.565, 0.05593, 0.08820, 14.81, 8.62, 76.85),
        ("UTP", 1.00, 5.565, 0.05595, 0.08823, 14.80, 8.62, 76.85),
        ("UTP", 4.00, 5.565, 0.05595, 0.08823, 14.80, 8.62, 76.85),
    ]
    return pd.DataFrame(rows, columns=["perturb", "fold", "deGFP", "mRNA_t", "Ribosome_mRNA", "GTP", "GTP_TX", "PPi"])


def dense_df():
    raw = pd.read_csv(UNIFIED_RATIO_CSV)
    sub = raw[(raw["axis"].isin(["CTP", "UTP"])) & (raw["comparison"] == "0.03x/0.01x")].copy()
    return pd.DataFrame({
        "perturb": sub["axis"],
        "time": sub["time_h"],
        "mRNA_ratio": sub["mRNA_t_ratio"],
        "ribo_ratio": sub["Ribosome_mRNA_ratio"],
        "unit_tlpool": sub["TLpool_per_Ribosome_mRNA_ratio"],
        "tlpool_ratio": sub["TLchain_pos_pool_ratio"],
        "tl_prog_ratio": sub["tl_gtp_progression_ratio"],
        "degfp_ratio": sub["deGFP_ratio"],
        "total_gtp_pool": sub["total_GTP_pool_ratio"],
    }).sort_values(["perturb", "time"]).reset_index(drop=True)


def gtp_ratio_df():
    raw = pd.read_csv(UNIFIED_RATIO_CSV)
    sub = raw[(raw["axis"].isin(["CTP", "UTP"])) & (raw["comparison"] == "0.03x/0.01x")].copy()
    return pd.DataFrame({
        "perturb": sub["axis"],
        "time": sub["time_h"],
        "tx_gtp_entry": sub["tx_gtp_entry_ratio"],
        "tl_gtp_entry": sub["tl_gtp_entry_ratio"],
        "tl_gtp_progression": sub["tl_gtp_progression_ratio"],
        "total_gtp_pool": sub["total_GTP_pool_ratio"],
    }).sort_values(["perturb", "time"]).reset_index(drop=True)


def plot():
    setup_font()
    ep = endpoint_df()
    dense = dense_df()
    gr = gtp_ratio_df()

    colors = {"CTP": COLORS["CTP_condition"], "UTP": COLORS["UTP_condition"]}
    dark = "#222222"
    green = COLORS["mRNA_t"]
    purple = COLORS["Ribosome_mRNA"]
    red = COLORS["tl_gtp_progression"]
    grey = COLORS["neutral"]

    fig, axs = plt.subplots(4, 2, figsize=(13.6, 19.0))
    fig.patch.set_facecolor("white")

    # A. endpoint dose response
    ax = axs[0, 0]
    dose_labels = ["0", "0.01", "0.03", "0.25", "1", "4"]
    x = np.arange(len(dose_labels))
    for perturb in ["CTP", "UTP"]:
        sub = ep[ep.perturb == perturb].copy()
        ax.plot(x, sub["deGFP"], "-o", color=colors[perturb], lw=2.2, ms=6, label=perturb)
    ax.set_title("A  终点产量：0.01×解除断链，继续升高不再增产", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}×" for d in dose_labels])
    ax.set_ylabel("6 h deGFP (μM)")
    ax.set_xlabel("CTP 或 UTP 投料倍率")
    ax.axvspan(0.75, 1.25, color="#F3CF65", alpha=0.25, lw=0)
    ax.set_ylim(-0.2, 6.7)
    ax.grid(False)
    ax.legend(
        frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    raw_ratio = pd.read_csv(UNIFIED_RATIO_CSV)
    abs_metric_labels = [
        LABELS["deGFP"],
        LABELS["PPi"],
        "单位Ribosome_mRNA\n产物",
        LABELS["tx_gtp_entry"],
        LABELS["mRNA_t"],
        LABELS["Ribosome_mRNA"],
    ]

    def plot_absolute_bars(ax, perturb, title):
        rows_b = []
        folds = [0.01, 0.03]
        for fold in folds:
            row = ep[(ep.perturb == perturb) & (ep.fold == fold)].iloc[0]
            ratio_row = raw_ratio[
                (raw_ratio["axis"] == perturb)
                & (raw_ratio["comparison"] == "0.03x/0.01x")
                & (np.isclose(raw_ratio["time_h"], 1.0))
            ].iloc[0]
            tx_gtp_1h = ratio_row["tx_gtp_entry_low"] if fold == 0.01 else ratio_row["tx_gtp_entry_high"]
            rows_b.append((row["deGFP"], row["PPi"], row["mRNA_t"], row["Ribosome_mRNA"], tx_gtp_1h))
        vals = pd.DataFrame(rows_b, columns=["deGFP", "PPi", "mRNA_t", "Ribosome_mRNA", "tx_gtp_1h"])
        vals["product_per_ribo"] = vals["deGFP"] / vals["Ribosome_mRNA"]
        bx2 = np.arange(len(abs_metric_labels))
        left_values = np.column_stack([
            vals["deGFP"],
            vals["PPi"] / 10.0,
            vals["product_per_ribo"] / 100.0,
            vals["tx_gtp_1h"] / 1e10,
        ])
        right_values = np.column_stack([vals["mRNA_t"], vals["Ribosome_mRNA"]])
        width = 0.28
        offsets = [-0.5 * width, 0.5 * width]
        ax_right = ax.twinx()
        legend_handles = []
        for j, fold in enumerate(folds):
            bars = ax.bar(
                bx2[:4] + offsets[j], left_values[j], width,
                color=BAR_ORDER_COLORS[j], alpha=BAR_ALPHA, edgecolor="none", linewidth=0,
            )
            ax_right.bar(
                bx2[4:] + offsets[j], right_values[j], width,
                color=BAR_ORDER_COLORS[j], alpha=BAR_ALPHA, edgecolor="none", linewidth=0,
            )
            legend_handles.append(bars[0])
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(bx2)
        ax.set_xticklabels(abs_metric_labels, fontsize=8, rotation=12, ha="right")
        ax.set_ylabel(
            "显示值（deGFP/PPi: μM；单位Ribosome_mRNA产物: 无量纲；\n"
            "PPi ×0.1；单位产物 ×0.01；转录GTP入口 ×1e-10 μM/h）",
            fontsize=7.4,
        )
        ax.set_ylim(0, 12.5)
        ax_right.set_ylabel("小浓度绝对值：mRNA_t / Ribosome_mRNA (μM)")
        ax_right.set_ylim(0, 0.095)
        ax_right.spines["top"].set_visible(False)
        ax.grid(False)
        ax_right.grid(False)
        ax.legend(
            legend_handles, [f"{fold:g}×" for fold in folds],
            frameon=False, fontsize=7.5, ncol=2, loc="upper center",
            bbox_to_anchor=(0.5, BAR_LEGEND_Y),
        )

    plot_absolute_bars(axs[0, 1], "CTP", "B  CTP低剂量节流：模板少一些，但下游更接得住")
    plot_absolute_bars(axs[1, 0], "UTP", "C  UTP低剂量节流：同口径单独展示")

    # D/E. adjacent over-optimum endpoint/flux values, split by perturbation.
    over_metric_labels = [
        ("deGFP", LABELS["deGFP"]),
        ("PPi", LABELS["PPi"]),
        ("mRNA_t", LABELS["mRNA_t"]),
        ("Ribosome_mRNA", LABELS["Ribosome_mRNA"]),
        ("tl_gtp_progression", LABELS["tl_gtp_progression"]),
        ("tx_gtp_entry", LABELS["tx_gtp_entry"]),
        ("tl_gtp_entry", LABELS["tl_gtp_entry"]),
    ]

    def plot_ratio_bars(ax, perturb, title):
        e01 = ep[(ep.perturb == perturb) & (ep.fold == 0.01)].iloc[0]
        e03 = ep[(ep.perturb == perturb) & (ep.fold == 0.03)].iloc[0]
        row6 = raw_ratio[
            (raw_ratio["axis"] == perturb)
            & (raw_ratio["comparison"] == "0.03x/0.01x")
            & (np.isclose(raw_ratio["time_h"], 6.0))
        ].iloc[0]
        display_scales = {
            "deGFP": 1.0,
            "PPi": 0.1,
            "mRNA_t": 100.0,
            "Ribosome_mRNA": 100.0,
            "tl_gtp_progression": 0.003,
            "tx_gtp_entry": 1e-9,
            "tl_gtp_entry": 1e-12,
        }
        values_low = [
            e01["deGFP"] * display_scales["deGFP"],
            e01["PPi"] * display_scales["PPi"],
            e01["mRNA_t"] * display_scales["mRNA_t"],
            e01["Ribosome_mRNA"] * display_scales["Ribosome_mRNA"],
            row6["tl_gtp_progression_low"] * display_scales["tl_gtp_progression"],
            row6["tx_gtp_entry_low"] * display_scales["tx_gtp_entry"],
            row6["tl_gtp_entry_low"] * display_scales["tl_gtp_entry"],
        ]
        values_high = [
            e03["deGFP"] * display_scales["deGFP"],
            e03["PPi"] * display_scales["PPi"],
            e03["mRNA_t"] * display_scales["mRNA_t"],
            e03["Ribosome_mRNA"] * display_scales["Ribosome_mRNA"],
            row6["tl_gtp_progression_high"] * display_scales["tl_gtp_progression"],
            row6["tx_gtp_entry_high"] * display_scales["tx_gtp_entry"],
            row6["tl_gtp_entry_high"] * display_scales["tl_gtp_entry"],
        ]
        bx = np.arange(len(over_metric_labels))
        width = 0.28
        ax.bar(
            bx - width / 2, values_low, width, color=BAR_ORDER_COLORS[0], alpha=BAR_ALPHA,
            edgecolor="none", linewidth=0, label="0.01×",
        )
        ax.bar(
            bx + width / 2, values_high, width, color=BAR_ORDER_COLORS[1], alpha=BAR_ALPHA,
            edgecolor="none", linewidth=0, label="0.03×",
        )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(bx)
        ax.set_xticklabels([m[1] for m in over_metric_labels], rotation=20, ha="right")
        ax.set_ylabel(
            "6 h 显示值（deGFP/PPi/mRNA_t/Ribosome_mRNA: μM；GTP通量: μM/h；\n"
            "PPi ×0.1；mRNA_t/Ribosome_mRNA ×100；翻译GTP推进 ×0.003；转录入口 ×1e-9；翻译入口 ×1e-12）",
            fontsize=7.2,
        )
        ax.set_ylim(0, max(values_low + values_high) * 1.18)
        ax.grid(False)
        ax.legend(
            frameon=False, fontsize=8, ncol=2, loc="upper center",
            bbox_to_anchor=(0.5, BAR_LEGEND_Y),
        )

    plot_ratio_bars(axs[1, 1], "CTP", "D  CTP相邻过最优：0.01×与0.03×绝对值")
    plot_ratio_bars(axs[2, 0], "UTP", "E  UTP相邻过最优：0.01×与0.03×绝对值")

    def plot_load_progression(ax):
        sub_ctp = dense[dense.perturb == "CTP"]
        sub_utp = dense[dense.perturb == "UTP"]
        series = [
            ("mRNA_t", "mRNA_ratio", green, "o"),
            ("Ribosome_mRNA", "ribo_ratio", purple, "s"),
            (LABELS["TLchain_pos_pool"], "tlpool_ratio", COLORS["TLchain_pos_pool"], "^"),
            (LABELS["tl_gtp_progression"], "tl_prog_ratio", red, "D"),
        ]
        for label, col, color, marker in series:
            ax.plot(sub_ctp.time, sub_ctp[col], "-", marker=marker, color=color, lw=1.9 if col != "tl_prog_ratio" else 2.1, ms=4, label=f"CTP {label}")
            ax.plot(sub_utp.time, sub_utp[col], "--", marker=marker, color=color, lw=1.7 if col != "tl_prog_ratio" else 1.9, ms=3.5, alpha=0.85, label=f"UTP {label}")
        ax.axhline(1, color=dark, lw=1, ls="--", alpha=0.55)
        ax.axvspan(1.5, 2.5, color="#F3CF65", alpha=0.18, lw=0)
        ax.set_xlim(0, 6)
        ymax = dense[["mRNA_ratio", "ribo_ratio", "tlpool_ratio", "tl_prog_ratio"]].max().max()
        ax.set_ylim(0, max(2.7, ymax * 1.12))
        ax.set_xlabel("时间 (h)")
        ax.set_ylabel("0.03× / 0.01×")
        ax.grid(False)
        ax.legend(
            frameon=False, fontsize=7.2, ncol=4, loc="upper center",
            bbox_to_anchor=(0.5, BAR_LEGEND_Y),
        )

    def plot_gtp_three(ax, perturb):
        sub = gr[gr.perturb == perturb]
        ax.plot(sub.time, sub.tx_gtp_entry, "-o", color=COLORS["tx_gtp_entry"], lw=2.0, ms=4, label=LABELS["tx_gtp_entry"])
        ax.plot(sub.time, sub.tl_gtp_entry, "-s", color=COLORS["tl_gtp_entry"], lw=2.0, ms=4, label=LABELS["tl_gtp_entry"])
        ax.plot(sub.time, sub.tl_gtp_progression, "-D", color=COLORS["tl_gtp_progression"], lw=2.2, ms=4, label=LABELS["tl_gtp_progression"])
        ax.plot(sub.time, sub.total_gtp_pool, "--P", color=COLORS["total_GTP_pool"], lw=1.8, ms=4, label=LABELS["total_GTP_pool"])
        ax.axhline(1, color=dark, lw=1, ls="--", alpha=0.55)
        ax.axvspan(1.5, 2.5, color="#F3CF65", alpha=0.18, lw=0)
        ax.set_xlim(0, 6)
        ax.set_xlabel("时间 (h)")
        ax.set_ylabel("0.03× / 0.01×")
        ax.grid(False)
        ax.legend(
            frameon=False, fontsize=8, ncol=2, loc="upper center",
            bbox_to_anchor=(0.5, BAR_LEGEND_Y),
        )

    ax = axs[2, 1]
    plot_load_progression(ax)
    ax.set_title("F  CTP/UTP过最优：模板与位置链被推高，CTP幅度更强", loc="left", fontsize=12, fontweight="bold")

    ax = axs[3, 0]
    plot_gtp_three(ax, "CTP")
    ax.set_title("G  CTP过最优：推进高位后，两个GTP入口被压低", loc="left", fontsize=12, fontweight="bold")

    ax = axs[3, 1]
    plot_gtp_three(ax, "UTP")
    ax.set_title("H  UTP过最优：同方向但幅度更弱，入口仍低于1×", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.35)

    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.suptitle("CTP/UTP 单独扰动：低剂量解除断链，过最优后出现“模板/推进增强—GTP入口支撑不足”的错配",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.035, 0.995, 0.97])

    png = FIG_DIR / "ctp_utp_perturbation_mechanism_curve_reorganized_20260728.png"
    svg = FIG_DIR / "ctp_utp_perturbation_mechanism_curve_reorganized_20260728.svg"
    pdf = FIG_DIR / "ctp_utp_perturbation_mechanism_curve_reorganized_20260728.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, svg, pdf


if __name__ == "__main__":
    png, svg, pdf = plot()
    print(png)
    print(svg)
    print(pdf)
