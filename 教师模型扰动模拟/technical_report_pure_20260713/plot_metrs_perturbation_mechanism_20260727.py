#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetRS perturbation mechanism figure for PURE+ technical report, 2026-07-27.

This version uses the MetRS enzyme-only data and the adjacent over-optimal
comparison 0.125x/0.06x.  It intentionally does not use the older 0.25x/0.06x
comparison.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from perturbation_curve_style_20260728 import BAR_ALPHA, BAR_LEGEND_Y, BAR_ORDER_COLORS, COLORS, LABELS


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = ROOT / "metrs_enzyme_only_complete_20260723" / "synced_subset_metrics.csv"
RATIO_CSV = ROOT / "metrs_adjacent_ratios_20260727.csv"


def setup_font() -> None:
    for p in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def load_endpoints() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    df = df[df["component"].eq("MetRS")].copy()
    wanted = [0.0, 0.01, 0.03, 0.06, 0.125, 0.25, 1.0, 4.0]
    df = df[df["scale"].isin(wanted)].copy()
    df["scale"] = pd.Categorical(df["scale"], categories=wanted, ordered=True)
    df = df.sort_values("scale").reset_index(drop=True)
    return pd.DataFrame(
        {
            "scale": df["scale"].astype(float),
            "label": ["0×", "0.01×", "0.03×", "0.06×", "0.125×", "0.25×", "1×", "4×"],
            "deGFP_6h": df["deGFP_end_uM"].clip(lower=0),
            "deGFP_peak": df["deGFP_peak_uM"].clip(lower=0),
            "peak_time": df["deGFP_peak_time_h"],
            "mRNA_t": df["mRNA_t_end_uM"],
            "Ribosome_mRNA": df["Ribosome_mRNA_end_uM"],
            "PPi": df["PPi_end_uM"],
        }
    )


def load_adjacent_ratio() -> pd.DataFrame:
    df = pd.read_csv(RATIO_CSV)
    df = df[df["comparison"].eq("0.125x/0.06x")].copy()
    if df.empty:
        raise RuntimeError("Cannot find 0.125x/0.06x in metrs_adjacent_ratios_20260727.csv")
    return df.sort_values("time_h").reset_index(drop=True)


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22, lw=0.8)
    ax.tick_params(labelsize=9)


def main() -> None:
    setup_font()
    ep = load_endpoints()
    ratio = load_adjacent_ratio()

    blue = COLORS["deGFP"]
    orange = "#D47632"
    green = COLORS["mRNA_t"]
    purple = COLORS["Ribosome_mRNA"]
    red = COLORS["tl_gtp_progression"]
    teal = COLORS["total_GTP_pool"]
    gold = COLORS["TLchain_pos_pool"]
    grey = COLORS["neutral"]

    fig, axs = plt.subplots(2, 2, figsize=(14.6, 10.4))
    fig.patch.set_facecolor("white")

    # A. Dose-response curve.
    ax = axs[0, 0]
    x = np.arange(len(ep))
    ax.plot(x, ep["deGFP_6h"], "-o", color=blue, lw=2.4, ms=6, label=LABELS["deGFP"])
    ax.plot(x, ep["deGFP_peak"], "--s", color=COLORS["deGFP"], lw=1.8, ms=4.5, alpha=0.62, label=LABELS["deGFP_peak"])
    ax.axvspan(2.7, 3.3, color="#F3CF65", alpha=0.26, lw=0)
    ax.axvspan(3.7, 4.3, color="#E8A7A7", alpha=0.18, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ep["label"])
    ax.set_ylim(-0.25, 6.75)
    ax.set_ylabel("deGFP (μM)")
    ax.set_title("A  MetRS剂量响应：0×断链，0.06×为实用最优窗口", loc="left", fontsize=12, fontweight="bold")
    style_ax(ax)
    ax.legend(
        frameon=False, fontsize=8, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    # B. Absolute endpoint evidence, not normalized.
    ax = axs[0, 1]
    metric_labels = [
        LABELS["deGFP"],
        LABELS["PPi"],
        LABELS["mRNA_t"],
        LABELS["Ribosome_mRNA"],
        LABELS["deGFP_peak"],
        "峰值时间",
    ]
    bx = np.arange(len(metric_labels))
    bw = 0.09
    offsets = (np.arange(len(ep)) - 3.5) * bw
    ax2 = ax.twinx()
    legend_handles = []
    for j, row in ep.iterrows():
        bars = ax.bar(
            bx[[0, 1, 4, 5]] + offsets[j],
            [row["deGFP_6h"], row["PPi"] / 40.0, row["deGFP_peak"], row["peak_time"]],
            bw,
            color=BAR_ORDER_COLORS[j],
            alpha=BAR_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        ax2.bar(
            bx[2:4] + offsets[j],
            [row["mRNA_t"], row["Ribosome_mRNA"]],
            bw,
            color=BAR_ORDER_COLORS[j],
            alpha=BAR_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        legend_handles.append(bars[0])
    ax.set_xticks(bx)
    ax.set_xticklabels(metric_labels, rotation=12, ha="right")
    ax.set_ylabel("deGFP、PPi ×0.025 (μM)；峰值时间 (h)")
    ax.set_ylim(0, 7.05)
    ax2.set_ylabel("mRNA_t / Ribosome_mRNA (μM)")
    ax2.set_ylim(0, 0.72)
    ax2.spines["top"].set_visible(False)
    ax.set_title("B  绝对值证据：低MetRS峰值偏晚，0×模板与装载复合物堆积", loc="left", fontsize=12, fontweight="bold")
    ax.legend(
        legend_handles, ep["label"],
        frameon=False, fontsize=6.8, ncol=8, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style_ax(ax)
    ax.grid(False)

    # C. Adjacent over-optimal mechanism: initiation opens, TLpool/progression later falls below optimum.
    ax = axs[1, 0]
    t = ratio["time_h"]
    ax.plot(t, ratio["Met_fMet_initiation_pool_ratio"], "-o", color=COLORS["Met_fMet_initiation_pool"], lw=2.0, ms=4, label=LABELS["Met_fMet_initiation_pool"])
    ax.plot(t, ratio["TLchain_pos_pool_ratio"], "-s", color=gold, lw=2.5, ms=4.5, label=LABELS["TLchain_pos_pool"])
    ax.plot(t, ratio["tl_gtp_progression_ratio"], "--D", color=red, lw=2.1, ms=4, label=LABELS["tl_gtp_progression"])
    axr = ax.twinx()
    axr.plot(t, ratio["mRNA_t_ratio"], "-.", color=green, lw=2.0, label=LABELS["mRNA_t"])
    axr.plot(t, ratio["Ribosome_mRNA_ratio"], ":", color=purple, lw=2.6, label=LABELS["Ribosome_mRNA"])
    ax.axhline(1, color="#333333", ls="--", lw=1.0, alpha=0.55)
    ax.axvspan(0.02, 0.3, color="#DCECCB", alpha=0.48, lw=0)
    ax.axvspan(3.0, 6.0, color="#F0D79A", alpha=0.18, lw=0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.45)
    axr.set_ylim(0.70, 1.04)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("0.125× / 0.06×：起始池、TLchain_pos pool、翻译GTP推进")
    axr.set_ylabel("0.125× / 0.06×：mRNA_t、Ribosome_mRNA")
    axr.spines["top"].set_visible(False)
    ax.set_title("C  过最优早期被打开，后期推进优势消失", loc="left", fontsize=12, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style_ax(ax)

    # D. Adjacent GTP partition, with total GTP concentration.
    ax = axs[1, 1]
    ax.plot(t, ratio["tx_gtp_entry_ratio"], "-o", color=COLORS["tx_gtp_entry"], lw=2.0, ms=4, label=LABELS["tx_gtp_entry"])
    ax.plot(t, ratio["tl_gtp_entry_ratio"], "-s", color=COLORS["tl_gtp_entry"], lw=2.0, ms=4, label=LABELS["tl_gtp_entry"])
    ax.plot(t, ratio["tl_gtp_progression_ratio"], "-D", color=COLORS["tl_gtp_progression"], lw=2.2, ms=4, label=LABELS["tl_gtp_progression"])
    ax.plot(t, ratio["total_GTP_pool_ratio"], "--P", color=COLORS["total_GTP_pool"], lw=2.0, ms=4, label=LABELS["total_GTP_pool"])
    ax.axhline(1, color="#333333", ls="--", lw=1.0, alpha=0.55)
    ax.axvspan(2.0, 2.5, color="#E8A7A7", alpha=0.23, lw=0)
    ax.axvspan(3.0, 6.0, color="#DCECCB", alpha=0.18, lw=0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.45)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("0.125× / 0.06×")
    ax.set_title("D  GTP三分法：入口后期回补，但生产性推进低于最优点", loc="left", fontsize=12, fontweight="bold")
    style_ax(ax)
    ax.legend(
        frameon=False, fontsize=8, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    fig.suptitle(
        "MetRS单扰动：起始门控先打开翻译位置链，但相邻过最优后TLchain_pos pool与翻译GTP推进不能维持",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0.018, 0.035, 0.995, 0.965])

    png = FIG_DIR / "metrs_perturbation_mechanism_curve_reorganized_20260728.png"
    svg = FIG_DIR / "metrs_perturbation_mechanism_curve_reorganized_20260728.svg"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
