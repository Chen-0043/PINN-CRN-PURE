#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from perturbation_curve_style_20260728 import BAR_ALPHA, BAR_LEGEND_Y, BAR_ORDER_COLORS, COLORS, LABELS


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "technical_report_pure_20260713/tables_20260722"
FIG_DIR = ROOT / "technical_report_pure_20260713/figures"
INFO_CSV = TABLE_DIR / "total_aars_scale_information_table_20260722.csv"
TIME_CSV = TABLE_DIR / "total_aars_overoptimum_dense_time_table_0p25_vs_0p22_20260722.csv"
OUT_PNG = FIG_DIR / "total_aars_perturbation_mechanism_curve_reorganized_20260728.png"
OUT_SVG = FIG_DIR / "total_aars_perturbation_mechanism_curve_reorganized_20260728.svg"


def setup_font() -> None:
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def annotate_phase(ax, x0, x1, label, color, y=0.985):
    ax.axvspan(x0, x1, color=color, alpha=0.09, lw=0)


def main() -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(INFO_CSV)
    time = pd.read_csv(TIME_CSV)
    time_full = time[(time["time_h"] >= 0.0) & (time["time_h"] <= 6.0)].copy()

    # Colors: consistent, restrained.
    c_prod = COLORS["deGFP"]
    c_aars = COLORS["aa_tRNA_pool"]
    c_gtp = COLORS["total_GTP_pool"]
    c_tx = COLORS["tx_gtp_entry"]
    c_tl_entry = COLORS["tl_gtp_entry"]
    c_tl_prog = COLORS["tl_gtp_progression"]
    c_ribo = COLORS["Ribosome_mRNA"]
    c_gray = COLORS["neutral"]

    fig, axes = plt.subplots(3, 2, figsize=(14.8, 16.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, hspace=0.12, wspace=0.08)
    fig.suptitle("总 aaRS 扰动机制：从上料断链到 0.22× 匹配，再到 0.25× 过最优双瓶颈",
                 fontsize=18, fontweight="bold", y=1.005)

    # A. Dose-response overview.
    ax = axes[0, 0]
    x = range(len(info))
    labels = [f"{v:g}×" for v in info["total_aaRS_scale"]]
    ax.plot(x, info["deGFP_6h_uM"], "-o", color=c_prod, lw=2.5, ms=6, label=LABELS["deGFP"])
    opt_idx = info.index[info["total_aaRS_scale"].eq(0.22)][0]
    over_idx = info.index[info["total_aaRS_scale"].eq(0.25)][0]
    ax.scatter([opt_idx], [info.loc[opt_idx, "deGFP_6h_uM"]], s=100, color=c_prod, edgecolor="black", zorder=5)
    ax.scatter([over_idx], [info.loc[over_idx, "deGFP_6h_uM"]], s=100, color=c_gtp, edgecolor="black", zorder=5)
    ax.set_xticks(list(x), labels, rotation=40, ha="right")
    ax.set_ylabel("deGFP 6 h 浓度 (μM)")
    ax.set_xlabel("总 aaRS 投料倍率")
    ax.set_title("A  剂量全貌：0.22× 附近进入匹配窗口，继续增加不增产", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    # B. Low-dose insufficiency: show the missing aa-tRNA loading step directly.
    ax = axes[0, 1]
    low_scales = [0.0, 0.01, 0.03, 0.18, 0.20, 0.22, 0.25]
    low = info[info["total_aaRS_scale"].isin(low_scales)].copy()
    low = low.set_index("total_aaRS_scale").loc[low_scales].reset_index()
    width = 0.11
    offsets = [(i - (len(low) - 1) / 2) * width for i in range(len(low))]
    legend_handles = []
    metric_values = np.column_stack([
        low["deGFP_6h_uM"],
        low["PPi_6h_uM"] / 5.0,
        low["aa_tRNA_pool_6h_uM_charged_tRNA"],
    ])
    metric_x = np.arange(metric_values.shape[1])
    for i, row in low.iterrows():
        bars = ax.bar(
            metric_x + offsets[i],
            metric_values[i],
            width=width,
            color=BAR_ORDER_COLORS[i],
            alpha=BAR_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        legend_handles.append(bars[0])
    ax.set_xticks(metric_x, [LABELS["deGFP"], LABELS["PPi"], LABELS["aa_tRNA_pool"]])
    ax.set_ylabel("6 h 绝对值（μM；PPi ×0.2）")
    ax.set_ylim(0, metric_values.max() * 1.28)
    ax.set_title("B  低剂量限制：aaRS不足导致aa-tRNA上料层打不开", loc="left", fontweight="bold")
    ax.legend(
        legend_handles,
        [f"{value:g}×" for value in low["total_aaRS_scale"]],
        frameon=False,
        fontsize=7.5,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax.grid(False)

    # C. Optimum adaptation around 0.22x.
    ax = axes[1, 0]
    focus_scales = [0.18, 0.20, 0.22, 0.25]
    focus = info[info["total_aaRS_scale"].isin(focus_scales)].copy()
    focus = focus.set_index("total_aaRS_scale").loc[focus_scales].reset_index()
    base = info[info["total_aaRS_scale"].eq(0.22)].iloc[0]
    fx = range(len(focus))
    ax.plot(fx, focus["deGFP_6h_uM"] / base["deGFP_6h_uM"], "-o",
            color=c_prod, lw=2.3, ms=5.5, label=LABELS["deGFP"])
    ax.plot(fx, focus["Ribosome_mRNA_6h_uM"] / base["Ribosome_mRNA_6h_uM"], "-o",
            color=COLORS["Ribosome_mRNA"], lw=2.0, ms=5.2, label=LABELS["Ribosome_mRNA"])
    ax.plot(fx, focus["PPi_6h_uM"] / base["PPi_6h_uM"], "-o",
            color=COLORS["PPi"], lw=2.0, ms=5.2, label=LABELS["PPi"])
    ax.axhline(1, color="black", lw=1, ls=":", alpha=0.45)
    ax.axvline(2, color="#1b9e77", lw=1.2, ls="--", alpha=0.75)
    ax.set_xticks(list(fx), [f"{v:g}×" for v in focus["total_aaRS_scale"]])
    ax.set_ylabel("relative to 0.22×")
    ax.set_ylim(0.32, 1.62)
    ax.set_title("C  最优点适配性：0.22×同时兼顾核糖体承接、产物和低PPi", loc="left", fontweight="bold")
    ax.legend(
        frameon=False, fontsize=8.5, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax.grid(True, alpha=0.25)

    # D. Early loading advantage and later support collapse.
    ax = axes[1, 1]
    ax.axvspan(0, 1.0, color="#f4a582", alpha=0.14, lw=0)
    ax.axvspan(1.75, 2.25, color="#d62728", alpha=0.08, lw=0)
    ax.axvspan(2.25, 4.0, color="#2ca02c", alpha=0.08, lw=0)
    ax.plot(time_full["time_h"], time_full["aa_tRNA_pool_ratio"], "-o",
            color=c_aars, lw=2.2, ms=4.5, label=LABELS["aa_tRNA_pool"])
    ax.plot(time_full["time_h"], time_full["EFTu_GTP_aa_tRNA_ratio"], "-o",
            color=COLORS["EFTu_GTP_aa_tRNA"], lw=2.2, ms=4.5, label=LABELS["EFTu_GTP_aa_tRNA"])
    ax.plot(time_full["time_h"], time_full["translation_GTP_progression_ratio"], "-o",
            color=c_tl_prog, lw=2.2, ms=4.5, label=LABELS["tl_gtp_progression"])
    ax.plot(time_full["time_h"], time_full["total_GTP_pool_ratio_0p25_over_0p22"], "--",
            color=c_gtp, lw=2.1, label=LABELS["total_GTP_pool"])
    ax.axhline(1, color="black", lw=1, alpha=0.35)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.32)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("0.25× / 0.22×")
    ax.set_title("D  过最优早期：0.25×先打开上料层，但随后GTP/EFTu支撑塌陷", loc="left", fontweight="bold")
    ax.legend(
        frameon=False, loc="upper center", fontsize=8.5, ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax.grid(True, alpha=0.25)

    # E. TLchain spreading.
    ax = axes[2, 0]
    ax.plot(time_full["time_h"], time_full["TLchain_mid_26_225_fraction_0p22"], "-o",
            color=c_prod, lw=2.3, ms=4.5, label="0.22× 最优点")
    ax.plot(time_full["time_h"], time_full["TLchain_mid_26_225_fraction_0p25"], "-o",
            color=c_gtp, lw=2.3, ms=4.5, label="0.25× 过最优")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvspan(1.75, 2.0, color=c_gtp, alpha=0.08, lw=0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 0.93)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("中后段翻译中间态占比")
    ax.set_title("E  翻译链铺开：0.25× 没有形成 0.22× 的全链生产性推进", loc="left", fontweight="bold")
    ax.legend(
        frameon=False, loc="upper center", fontsize=8.5, ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax.grid(True, alpha=0.25)

    # F. GTP partition and recovery order.
    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax.plot(time_full["time_h"], time_full["tx_GTP_entry_ratio"], "-o",
            color=c_tx, lw=2.1, ms=4.2, label=LABELS["tx_gtp_entry"])
    ax.plot(time_full["time_h"], time_full["translation_GTP_entry_ratio"], "-o",
            color=c_tl_entry, lw=2.1, ms=4.2, label=LABELS["tl_gtp_entry"])
    ax.plot(time_full["time_h"], time_full["translation_GTP_progression_ratio"], "-o",
            color=c_tl_prog, lw=2.3, ms=4.5, label=LABELS["tl_gtp_progression"])
    ax2.plot(time_full["time_h"], time_full["mRNA_t_ratio"], "--s",
             color=COLORS["mRNA_t"], lw=1.9, ms=4.0, label=LABELS["mRNA_t"])
    ax2.plot(time_full["time_h"], time_full["Ribosome_mRNA_ratio"], "--s",
             color=c_ribo, lw=1.9, ms=4.0, label=LABELS["Ribosome_mRNA"])
    ax2.plot(time_full["time_h"], time_full["TLchain_pos_pool_ratio"], "--^",
             color=COLORS["TLchain_pos_pool"], lw=1.9, ms=4.1, label=LABELS["TLchain_pos_pool"])
    ax.axhline(1, color="black", lw=1, alpha=0.35)
    ax2.axhline(1, color="black", lw=1, alpha=0.18)
    ax.axvspan(1.75, 2.25, color=c_gtp, alpha=0.08, lw=0)
    ax.axvspan(2.25, 3.0, color=c_aars, alpha=0.08, lw=0)
    ax.axvspan(3.0, 4.0, color="#92c5de", alpha=0.13, lw=0)
    ax2.axvspan(1.75, 2.25, color=c_gtp, alpha=0.00, lw=0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.16)
    ax2.set_ylim(0.30, 2.45)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("GTP通量比值 (0.25×/0.22×)")
    ax2.set_ylabel("mRNA_t/Ribosome_mRNA/TLchain_pos pool 比值 (0.25×/0.22×)")
    ax.set_title("F  推进受阻后的连锁反应：GTP三分低谷 + 模板/装载/位置链变化", loc="left", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2, l1 + l2, frameon=False, loc="upper center", fontsize=7.8, ncol=3,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax.grid(True, alpha=0.25)

    for ax in axes.ravel():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
