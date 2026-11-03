#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTP perturbation mechanism figure using the unified 20260723 time grid.

Main conventions:
- total GTP concentration = GTP + GTP_TX
- over-optimum comparison = adjacent 4× / 3×
- time-course data are from true resimulation snapshots, not interpolation
"""

from pathlib import Path
import re

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
RECORD_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_records_20260723.csv"
RATIO_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_ratios_20260723.csv"
OUT_PNG = FIG_DIR / "gtp_perturbation_mechanism_curve_reorganized_20260728.png"
OUT_SVG = FIG_DIR / "gtp_perturbation_mechanism_curve_reorganized_20260728.svg"
OUT_PDF = FIG_DIR / "gtp_perturbation_mechanism_curve_reorganized_20260728.pdf"


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


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.23)
    ax.tick_params(labelsize=8.7)


def compute_rnap_atp_wait_pool(rec):
    """Sum T7RNAP_bound_mRNA states whose next transcription step requires ATP_TX."""
    row4 = rec[(rec["axis"] == "GTP") & np.isclose(rec["scale"], 4.0)].iloc[0]
    rx = pd.read_csv(row4["source_reaction_snapshot"])
    state_to_ntp = {}
    for reactants in rx["reactants"].fillna("").astype(str):
        m = re.search(r"(T7RNAP_bound_mRNA\d+);([ACGU]TP_TX)", reactants)
        if m:
            state_to_ntp[m.group(1)] = m.group(2).replace("_TX", "")
    atp_wait_states = {state for state, ntp in state_to_ntp.items() if ntp == "ATP"}

    frames = []
    for scale in (3.0, 4.0):
        row = rec[(rec["axis"] == "GTP") & np.isclose(rec["scale"], scale)].iloc[0]
        sp = pd.read_csv(row["source_species_snapshot"])
        sub = sp[sp["species"].isin(atp_wait_states)].copy()
        pooled = sub.groupby("time_h", as_index=False)["concentration_uM"].sum()
        pooled["scale"] = scale
        frames.append(pooled)

    wide = pd.concat(frames, ignore_index=True).pivot(
        index="time_h", columns="scale", values="concentration_uM"
    ).reset_index()
    wide = wide.rename(columns={3.0: "rnap_atp_wait_3x", 4.0: "rnap_atp_wait_4x"})
    wide["rnap_atp_wait_ratio_4x_3x"] = (
        wide["rnap_atp_wait_4x"] / wide["rnap_atp_wait_3x"].where(wide["rnap_atp_wait_3x"] > 0)
    )
    return wide, len(atp_wait_states)


def main():
    setup_font()
    rec = pd.read_csv(RECORD_CSV)
    ratio = pd.read_csv(RATIO_CSV)
    end = rec[(rec["axis"] == "GTP") & np.isclose(rec["time_h"], 6.0)].copy()
    end = end.sort_values("scale").reset_index(drop=True)
    over = ratio[(ratio["axis"] == "GTP") & (ratio["comparison"] == "4x/3x")].copy()
    over = over.sort_values("time_h").reset_index(drop=True)
    rnap_wait, n_atp_wait_states = compute_rnap_atp_wait_pool(rec)

    blue = COLORS["deGFP"]
    orange = "#E68632"
    green = COLORS["mRNA_t"]
    purple = COLORS["Ribosome_mRNA"]
    red = COLORS["tl_gtp_progression"]
    teal = COLORS["total_GTP_pool"]
    grey = COLORS["neutral"]
    yellow = COLORS["PPi"]

    fig, axs = plt.subplots(3, 2, figsize=(13.8, 15.2))
    fig.patch.set_facecolor("white")

    # A. endpoint product and PPi.
    ax = axs[0, 0]
    ax2 = ax.twinx()
    x = np.arange(len(end))
    labels = [f"{v:g}×" for v in end["scale"]]
    ax.plot(x, end["deGFP"], "-o", lw=2.4, ms=6, color=blue, label=LABELS["deGFP"])
    ax2.bar(
        x, end["PPi"], width=0.55, color=yellow, alpha=BAR_ALPHA,
        edgecolor="none", linewidth=0, label=LABELS["PPi"], zorder=1,
    )
    opt_idx = int(end.index[np.isclose(end["scale"], 3.0)][0])
    over_idx = int(end.index[np.isclose(end["scale"], 4.0)][0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("6 h deGFP (μM)")
    ax2.set_ylabel("6 h PPi (μM)")
    ax.set_ylim(0, 19.5)
    ax2.set_ylim(0, max(end["PPi"]) * 1.25)
    ax.set_xlabel("GTP 初始投料倍率")
    ax.set_title("A  剂量响应：3×进入高产窗口，4×转为过量错配", loc="left", fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2, l1 + l2, frameon=False, fontsize=8, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style(ax)
    ax.grid(False)
    ax2.spines["top"].set_visible(False)

    # B. fold-change heatmap: why 3x is the best endpoint match.
    ax = axs[0, 1]
    heat_rows = [
        ("deGFP", "deGFP"),
        ("mRNA_t", "mRNA_t"),
        ("Ribosome_mRNA", LABELS["Ribosome_mRNA"]),
        ("TLchain_pos_pool", LABELS["TLchain_pos_pool"]),
        ("tl_gtp_progression", LABELS["tl_gtp_progression"]),
        ("ATP", "ATP"),
        ("ATP_TX", "ATP_TX"),
        ("total_GTP_pool", "总GTP浓度"),
        ("PPi", "PPi"),
    ]
    ref = end[np.isclose(end["scale"], 1.0)].iloc[0]
    folds = []
    for col, _ in heat_rows:
        folds.append((end[col].to_numpy(float) / float(ref[col])))
    folds = np.array(folds)
    log_folds = np.log2(np.clip(folds, 1e-4, None))
    im = ax.imshow(np.clip(log_folds, -3, 3), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(end)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(heat_rows)))
    ax.set_yticklabels([name for _, name in heat_rows])
    for i in range(folds.shape[0]):
        for j in range(folds.shape[1]):
            val = folds[i, j]
            text_color = "white" if abs(log_folds[i, j]) > 1.8 else "#222222"
            if val >= 100:
                txt = f"{val:.0f}×"
            elif val >= 10:
                txt = f"{val:.1f}×"
            elif val >= 0.01:
                txt = f"{val:.2f}×"
            else:
                txt = "<0.01×"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.4, color=text_color)
    ax.set_xlabel("GTP 初始投料倍率")
    ax.set_title("B  全网络关键节点 fold change：3×主链匹配，4×资源和释放失配", loc="left", fontsize=12, fontweight="bold")
    ax.axvline(opt_idx, color="#1B8A8F", lw=1.2, ls="--", alpha=0.75)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("log2(fold change vs 1×)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # C. endpoint traffic: template/Ribo drop, TLpool pile-up, release falls.
    ax = axs[1, 0]
    c_metric_keys = ["deGFP", "PPi", "mRNA_t", "Ribosome_mRNA", "TLchain_pos_pool"]
    c_metrics = [LABELS[key] for key in c_metric_keys]
    display_scales = {
        "deGFP": 1.0,
        "PPi": 0.2,
        "mRNA_t": 100.0,
        "Ribosome_mRNA": 100.0,
        "TLchain_pos_pool": 10.0,
    }
    e3 = end[np.isclose(end["scale"], 3.0)].iloc[0]
    e4 = end[np.isclose(end["scale"], 4.0)].iloc[0]
    v3 = [float(e3[key]) * display_scales[key] for key in c_metric_keys]
    v4 = [float(e4[key]) * display_scales[key] for key in c_metric_keys]
    cx = np.arange(len(c_metrics))
    ax.bar(
        cx - 0.18, v3, 0.34, color=BAR_ORDER_COLORS[0], alpha=BAR_ALPHA,
        edgecolor="none", linewidth=0, label="GTP 3×",
    )
    ax.bar(
        cx + 0.18, v4, 0.34, color=BAR_ORDER_COLORS[1], alpha=BAR_ALPHA,
        edgecolor="none", linewidth=0, label="GTP 4×",
    )
    ax.set_xticks(cx)
    ax.set_xticklabels(c_metrics, rotation=18, ha="right")
    ax.set_ylabel(
        "6 h 显示值（PPi ×0.2；mRNA_t/Ribosome_mRNA ×100；\n"
        "TLchain_pos pool ×10）",
        fontsize=7.6,
    )
    ax.set_title("C  3×/4×终点交通：TLchain_pos pool堆积，但模板装载下降", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(v3 + v4) * 1.12)
    ax.legend(
        frameon=False, fontsize=8, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style(ax)
    ax.grid(False)

    # D. over-optimum resource time course.
    t = over["time_h"].to_numpy(float)
    ax = axs[1, 1]
    ax.plot(t, over["ATP_ratio"], "-o", color=COLORS["ATP"], lw=2.0, ms=4, label=LABELS["ATP"])
    ax.plot(t, over["ATP_TX_ratio"], "-s", color=COLORS["ATP_TX"], lw=2.0, ms=4, label=LABELS["ATP_TX"])
    ax.plot(t, over["total_GTP_pool_ratio"], "-D", color=teal, lw=2.0, ms=4, label=LABELS["total_GTP_pool"])
    ax.plot(t, over["PPi_ratio"], "-^", color=yellow, lw=2.0, ms=4, label=LABELS["PPi"])
    ax.axhline(1, color="black", lw=1, ls="--", alpha=0.45)
    ax.axvspan(1.5, 2.25, color="#F3CF65", alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 6)
    ax.set_ylim(0.001, 180)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("GTP 4× / 3× (log)")
    ax.set_title("D  过量资源时序：ATP/ATP_TX坍塌，游离GTP池反而堆高", loc="left", fontsize=12, fontweight="bold")
    ax.legend(
        frameon=False, fontsize=8, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style(ax)

    # E. over-optimum traffic: translation pile-up plus transcription-side RNAP ATP-site blocking.
    ax = axs[2, 0]
    ax.plot(t, over["mRNA_t_ratio"], "-o", color=green, lw=1.9, ms=4, label=LABELS["mRNA_t"])
    ax.plot(t, over["Ribosome_mRNA_ratio"], "-s", color=purple, lw=1.9, ms=4, label=LABELS["Ribosome_mRNA"])
    ax.plot(t, over["TLchain_pos_pool_ratio"], "-^", color=COLORS["TLchain_pos_pool"], lw=2.0, ms=4, label=LABELS["TLchain_pos_pool"])
    ax.plot(t, over["deGFP_ratio"], "--", color=blue, lw=2.0, label=LABELS["deGFP"])
    ax.axhline(1, color="black", lw=1, ls="--", alpha=0.45)
    ax.axvspan(1.5, 2.25, color="#F3CF65", alpha=0.18, lw=0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("GTP 4× / 3×")
    ax.set_title("E  过量交通：TLchain_pos pool堆积，并伴随RNAP在A位点卡住", loc="left", fontsize=12, fontweight="bold")

    axr = ax.twinx()
    wait = rnap_wait[rnap_wait["time_h"].isin(over["time_h"])].copy()
    axr.plot(wait["time_h"], wait["rnap_atp_wait_4x"], "--P", color="#8C510A", lw=2.0, ms=4,
             label="RNAP卡A位点池 (4×)")
    axr.set_ylabel("RNAP卡A位点池 (μM, GTP 4×)", color="#8C510A")
    axr.tick_params(axis="y", labelcolor="#8C510A", labelsize=8.7)
    axr.spines["top"].set_visible(False)
    axr.set_ylim(0, max(wait["rnap_atp_wait_4x"]) * 1.25)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(
        h1 + h2, l1 + l2, frameon=False, fontsize=7.8, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style(ax)

    # F. GTP three-way partition with total GTP pool.
    ax = axs[2, 1]
    ax.plot(t, over["tx_gtp_entry_ratio"], "-o", color=COLORS["tx_gtp_entry"], lw=2.0, ms=4, label=LABELS["tx_gtp_entry"])
    ax.plot(t, over["tl_gtp_entry_ratio"], "-s", color=COLORS["tl_gtp_entry"], lw=2.0, ms=4, label=LABELS["tl_gtp_entry"])
    ax.plot(t, over["tl_gtp_progression_ratio"], "-D", color=COLORS["tl_gtp_progression"], lw=2.2, ms=4, label=LABELS["tl_gtp_progression"])
    ax.plot(t, over["total_GTP_pool_ratio"], "--P", color=teal, lw=1.9, ms=4, label=LABELS["total_GTP_pool"])
    ax.axhline(1, color="black", lw=1, ls="--", alpha=0.45)
    ax.axvspan(1.5, 2.25, color="#F3CF65", alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 6)
    ax.set_ylim(0.8, 180)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("GTP 4× / 3× (log)")
    ax.set_title("F  GTP三分法：4×把入口/推进推过头，随后变成高GTP剩余低产出", loc="left", fontsize=12, fontweight="bold")
    ax.legend(
        frameon=False, fontsize=8, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    style(ax)

    fig.suptitle("GTP 单独扰动：3×形成高产匹配，4×过量后ATP支撑坍塌并出现翻译链堵塞", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.035, 0.995, 0.97])
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_SVG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
