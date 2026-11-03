#!/usr/bin/env python3
"""Rebuild all perturbation figures with the same three curve groups."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from perturbation_curve_style_20260728 import COLORS, LABELS
from plot_atp_full_stage_mechanism_20260723 import endpoint_table as atp_endpoint_table
from plot_ctp_utp_perturbation_mechanism_20260722 import endpoint_df as ctp_utp_endpoint_df


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIG_DIR = ROOT / "figures" / "three_group_20260728"
FIG_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_RECORDS = ROOT / "report_audit_20260723/unified_perturbation_timeseries_records_20260723.csv"
UNIFIED_RATIOS = ROOT / "report_audit_20260723/unified_perturbation_timeseries_ratios_20260723.csv"
METRS_METRICS = ROOT / "metrs_enzyme_only_complete_20260723/synced_subset_metrics.csv"
METRS_RATIOS = ROOT / "metrs_adjacent_ratios_20260727.csv"
AARS_ENDPOINTS = ROOT / "tables_20260722/total_aars_scale_information_table_20260722.csv"
AARS_TIME = ROOT / "tables_20260722/total_aars_overoptimum_dense_time_table_0p25_vs_0p22_20260722.csv"
SYNERGY = ROOT / "atp_gtp_minimal_synergy_window_20260720/all_atp_gtp_minimal_window_points.csv"


LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]


def setup_font() -> None:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")
    ax.tick_params(axis="both", colors="#000000", labelsize=8.8)


def legend_below(ax, handles=None, labels=None, *, max_cols: int = 3,
                 anchor_y: float = -0.23, fontsize: float = 7.7) -> None:
    """Place a compact, row-wise legend below the x-axis label."""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=min(max_cols, len(handles)),
        columnspacing=1.15,
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0,
    )


def set_ratio_scale(ax, frame: pd.DataFrame, columns: list[str]) -> None:
    vals = frame[columns].to_numpy(float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size and (vals.min() < 0.08 or vals.max() > 12):
        ax.set_yscale("log")


def plot_endpoint_group(ax, endpoint: pd.DataFrame, xcol: str, title: str,
                        degfp_cols: list[tuple[str, str]] | None = None) -> None:
    endpoint = endpoint.sort_values(xcol).reset_index(drop=True)
    x = np.arange(len(endpoint))
    labels = [f"{value:g}×" for value in endpoint[xcol].astype(float)]
    degfp_cols = degfp_cols or [("deGFP", LABELS["deGFP_6h"])]
    for i, (col, label) in enumerate(degfp_cols):
        ax.plot(
            x,
            endpoint[col],
            color=COLORS["deGFP"],
            lw=2.35 if i == 0 else 1.9,
            ls="-" if i == 0 else "--",
            marker="o" if i == 0 else "s",
            ms=5.2 if i == 0 else 4.2,
            alpha=1.0 if i == 0 else 0.62,
            label=label,
        )
    ax2 = ax.twinx()
    ax2.plot(x, endpoint["PPi"], color=COLORS["PPi"], lw=2.1, marker="^", ms=4.8, label=LABELS["PPi"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("扰动投料倍率")
    ax.set_ylabel("deGFP (μM)")
    ax2.set_ylabel("PPi (μM)", color="#000000")
    ax2.tick_params(axis="y", colors="#000000", labelsize=8.8)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend_below(
        ax, h1 + h2, l1 + l2, max_cols=3, anchor_y=-0.23, fontsize=8.1,
    )
    ax.set_title(f"A  deGFP 与 PPi：{title}", loc="left", fontsize=11.5, fontweight="bold")
    style_axis(ax)


def plot_ratio_group(ax, ratio: pd.DataFrame, series: list[tuple[str, str, str]], title: str,
                     legend_cols: int = 2) -> None:
    ratio = ratio.sort_values("time_h")
    plotted = []
    for i, (col, label, color) in enumerate(series):
        if col not in ratio or ratio[col].isna().all():
            continue
        ax.plot(
            ratio["time_h"], ratio[col], color=color, lw=2.0,
            ls=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
            ms=3.8, label=label,
        )
        plotted.append(col)
    set_ratio_scale(ax, ratio, plotted)
    ax.set_xlim(float(ratio["time_h"].min()), float(ratio["time_h"].max()))
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("相邻条件比值")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    legend_below(ax, max_cols=legend_cols, anchor_y=-0.23, fontsize=7.7)
    style_axis(ax)


def save_figure(fig, stem: str) -> None:
    fig.tight_layout(rect=[0.018, 0.035, 0.995, 0.945], pad=0.75)
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(path)
    plt.close(fig)


def rnap_atp_wait_ratio(records: pd.DataFrame) -> pd.DataFrame:
    row4 = records[(records["axis"].eq("GTP")) & np.isclose(records["scale"], 4.0)].iloc[0]
    reaction_path = PROJECT / str(row4["source_reaction_snapshot"])
    rx = pd.read_csv(reaction_path)
    states = set()
    for reactants in rx["reactants"].fillna("").astype(str):
        match = re.search(r"(T7RNAP_bound_mRNA\d+);ATP_TX", reactants)
        if match:
            states.add(match.group(1))
    frames = []
    for scale in (3.0, 4.0):
        row = records[(records["axis"].eq("GTP")) & np.isclose(records["scale"], scale)].iloc[0]
        species_path = PROJECT / str(row["source_species_snapshot"])
        sp = pd.read_csv(species_path)
        pooled = sp[sp["species"].isin(states)].groupby("time_h", as_index=False)["concentration_uM"].sum()
        pooled["scale"] = scale
        frames.append(pooled)
    wide = pd.concat(frames).pivot(index="time_h", columns="scale", values="concentration_uM").reset_index()
    wide["RNAP_ATP_wait_ratio"] = wide[4.0] / wide[3.0].replace(0, np.nan)
    return wide[["time_h", "RNAP_ATP_wait_ratio"]]


def standard_axis_figure(axis: str, records: pd.DataFrame, ratios: pd.DataFrame,
                         comparison: str, title: str, other: list[tuple[str, str, str]],
                         extra_other: pd.DataFrame | None = None,
                         endpoint_override: pd.DataFrame | None = None) -> None:
    if endpoint_override is None:
        endpoint = records[(records["axis"].eq(axis)) & np.isclose(records["time_h"], 6.0)].copy()
        endpoint = endpoint.rename(columns={"scale": "dose", "deGFP": "deGFP"})
    else:
        endpoint = endpoint_override.copy()
    ratio = ratios[(ratios["axis"].eq(axis)) & ratios["comparison"].eq(comparison)].copy()
    if extra_other is not None:
        ratio = ratio.merge(extra_other, on="time_h", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.985)
    plot_endpoint_group(axes[0], endpoint, "dose", "6 h 剂量终点")
    core = [
        ("tx_gtp_entry_ratio", LABELS["tx_gtp_entry"], COLORS["tx_gtp_entry"]),
        ("tl_gtp_entry_ratio", LABELS["tl_gtp_entry"], COLORS["tl_gtp_entry"]),
        ("tl_gtp_progression_ratio", LABELS["tl_gtp_progression"], COLORS["tl_gtp_progression"]),
        ("total_GTP_pool_ratio", LABELS["total_GTP_pool"], COLORS["total_GTP_pool"]),
        ("ATP_ratio", LABELS["ATP"], COLORS["ATP"]),
        ("ATP_TX_ratio", LABELS["ATP_TX"], COLORS["ATP_TX"]),
    ]
    plot_ratio_group(axes[1], ratio, core, f"B  GTP通量与能量资源：{comparison}", legend_cols=3)
    other_primary = [item for item in other if item[0] != "RNAP_ATP_wait_ratio"]
    plot_ratio_group(axes[2], ratio, other_primary, f"C  其他变量：{comparison}", legend_cols=3)
    if "RNAP_ATP_wait_ratio" in ratio:
        axr = axes[2].twinx()
        axr.plot(
            ratio["time_h"], ratio["RNAP_ATP_wait_ratio"], color=COLORS["RNAP_ATP_wait"],
            lw=1.9, ls=":", marker="^", ms=3.8, label="RNAP卡A位点池",
        )
        axr.set_yscale("log")
        axr.set_ylabel("RNAP卡A位点池比值 (log)", color="#000000")
        axr.tick_params(axis="y", colors="#000000", labelsize=8.5)
        axr.spines["top"].set_visible(False)
        h1, l1 = axes[2].get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        legend_below(
            axes[2], h1 + h2, l1 + l2, max_cols=3, anchor_y=-0.23, fontsize=7.7,
        )
    save_figure(fig, f"{axis.lower()}_three_curve_groups_20260728")


def plot_total_aars() -> None:
    endpoint = pd.read_csv(AARS_ENDPOINTS).rename(
        columns={"total_aaRS_scale": "dose", "deGFP_6h_uM": "deGFP", "PPi_6h_uM": "PPi"}
    )
    time = pd.read_csv(AARS_TIME).copy()
    fig, axes = plt.subplots(1, 4, figsize=(23.0, 5.8))
    fig.suptitle("总 aaRS 扰动：三类曲线统一归组", fontsize=15.5, fontweight="bold", y=0.985)
    plot_endpoint_group(axes[0], endpoint, "dose", "6 h 剂量终点")
    core = [
        ("tx_GTP_entry_ratio", LABELS["tx_gtp_entry"], COLORS["tx_gtp_entry"]),
        ("translation_GTP_entry_ratio", LABELS["tl_gtp_entry"], COLORS["tl_gtp_entry"]),
        ("translation_GTP_progression_ratio", LABELS["tl_gtp_progression"], COLORS["tl_gtp_progression"]),
        ("total_GTP_pool_ratio_0p25_over_0p22", LABELS["total_GTP_pool"], COLORS["total_GTP_pool"]),
    ]
    plot_ratio_group(axes[1], time, core, "B  GTP通量与能量资源：0.25×/0.22×", legend_cols=2)

    other = [
        ("aa_tRNA_pool_ratio", LABELS["aa_tRNA_pool"], COLORS["aa_tRNA_pool"]),
        ("EFTu_GTP_aa_tRNA_ratio", LABELS["EFTu_GTP_aa_tRNA"], COLORS["EFTu_GTP_aa_tRNA"]),
        ("mRNA_t_ratio", LABELS["mRNA_t"], COLORS["mRNA_t"]),
        ("Ribosome_mRNA_ratio", LABELS["Ribosome_mRNA"], COLORS["Ribosome_mRNA"]),
        ("TLchain_pos_pool_ratio", LABELS["TLchain_pos_pool"], COLORS["TLchain_pos_pool"]),
    ]
    plot_ratio_group(axes[2], time, other, "C  其他变量：0.25×/0.22×", legend_cols=3)
    ax = axes[3]
    ax.plot(time["time_h"], time["TLchain_mid_26_225_fraction_0p22"], color=COLORS["deGFP"], lw=2.2,
            ls="--", marker="o", ms=4.0, label="翻译中后段占比 0.22×")
    ax.plot(time["time_h"], time["TLchain_mid_26_225_fraction_0p25"], color=COLORS["tl_gtp_progression"], lw=2.2,
            ls=":", marker="s", ms=4.0, label="翻译中后段占比 0.25×")
    ax.set_xlim(float(time["time_h"].min()), float(time["time_h"].max()))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("中后段翻译中间态占比")
    ax.set_title("D  翻译中后段占比：单独比较两条曲线", loc="left", fontsize=11.5, fontweight="bold")
    legend_below(ax, max_cols=2, anchor_y=-0.23, fontsize=8.0)
    style_axis(ax)
    save_figure(fig, "total_aars_three_curve_groups_20260728")


def plot_metrs() -> None:
    metrics = pd.read_csv(METRS_METRICS)
    metrics = metrics[metrics["component"].eq("MetRS")].copy()
    endpoint = metrics.rename(
        columns={"scale": "dose", "deGFP_end_uM": "deGFP", "deGFP_peak_uM": "deGFP_peak", "PPi_end_uM": "PPi"}
    )
    ratio = pd.read_csv(METRS_RATIOS)
    ratio = ratio[ratio["comparison"].eq("0.125x/0.06x")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    fig.suptitle("MetRS 扰动：三类曲线统一归组", fontsize=15.5, fontweight="bold", y=0.985)
    plot_endpoint_group(
        axes[0], endpoint, "dose", "6 h 终点与峰值",
        degfp_cols=[("deGFP", LABELS["deGFP_6h"]), ("deGFP_peak", LABELS["deGFP_peak"])],
    )
    core = [
        ("tx_gtp_entry_ratio", LABELS["tx_gtp_entry"], COLORS["tx_gtp_entry"]),
        ("tl_gtp_entry_ratio", LABELS["tl_gtp_entry"], COLORS["tl_gtp_entry"]),
        ("tl_gtp_progression_ratio", LABELS["tl_gtp_progression"], COLORS["tl_gtp_progression"]),
        ("total_GTP_pool_ratio", LABELS["total_GTP_pool"], COLORS["total_GTP_pool"]),
        ("ATP_ratio", LABELS["ATP"], COLORS["ATP"]),
        ("ATP_TX_ratio", LABELS["ATP_TX"], COLORS["ATP_TX"]),
    ]
    plot_ratio_group(axes[1], ratio, core, "B  GTP通量与能量资源：0.125×/0.06×", legend_cols=3)
    other = [
        ("mRNA_t_ratio", LABELS["mRNA_t"], COLORS["mRNA_t"]),
        ("Ribosome_mRNA_ratio", LABELS["Ribosome_mRNA"], COLORS["Ribosome_mRNA"]),
        ("TLchain_pos_pool_ratio", LABELS["TLchain_pos_pool"], COLORS["TLchain_pos_pool"]),
        ("Met_fMet_initiation_pool_ratio", LABELS["Met_fMet_initiation_pool"], COLORS["Met_fMet_initiation_pool"]),
    ]
    plot_ratio_group(axes[2], ratio, other, "C  其他变量：0.125×/0.06×", legend_cols=3)
    save_figure(fig, "metrs_three_curve_groups_20260728")


def synergy_row(df: pd.DataFrame, atp: float, gtp: float) -> pd.Series:
    return df[np.isclose(df["scale_ATP"], atp) & np.isclose(df["scale_GTP"], gtp)].iloc[0]


def plot_synergy() -> None:
    df = pd.read_csv(SYNERGY)
    df = df[
        df["scale_CTP"].eq(1.0) & df["scale_UTP"].eq(1.0)
        & df["scale_MetRS"].eq(1.0) & df["scale_bulk_aars_pool"].eq(1.0)
    ]
    states = [
        ("ATP低\n1×/5×", synergy_row(df, 1, 5)),
        ("GTP少\n2×/4×", synergy_row(df, 2, 4)),
        ("协同点\n2×/5×", synergy_row(df, 2, 5)),
        ("ATP略高\n3×/5×", synergy_row(df, 3, 5)),
        ("GTP多\n2×/6×", synergy_row(df, 2, 6)),
    ]
    labels = [label for label, _ in states]
    x = np.arange(len(states))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    fig.suptitle("ATP-GTP 协同窗口：三类曲线统一归组", fontsize=15.5, fontweight="bold", y=0.985)

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(x, [r["deGFP_end_uM"] for _, r in states], "-o", color=COLORS["deGFP"], lw=2.3, label=LABELS["deGFP_6h"])
    ax2.plot(x, [r["PPi_end_uM"] for _, r in states], "-^", color=COLORS["PPi"], lw=2.1, label=LABELS["PPi"])
    ax.set_ylabel("deGFP (μM)")
    ax2.set_ylabel("PPi (μM)", color="#000000")
    ax2.tick_params(axis="y", colors="#000000", labelsize=8.8)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, max_cols=2, anchor_y=-0.23, fontsize=8)
    ax.set_title("A  deGFP 与 PPi：协同点两侧", loc="left", fontsize=11.5, fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    ax.plot(x, [r["ATP_end_uM"] for _, r in states], "-o", color=COLORS["ATP"], lw=2.1, label=LABELS["ATP"])
    ax.plot(x, [r["GTP_end_uM"] for _, r in states], "-D", color=COLORS["GTP"], lw=2.1, label=LABELS["GTP"])
    ax.set_yscale("log")
    ax.set_ylabel("6 h 浓度 (μM, log)")
    ax.set_title("B  GTP通量与能量资源：终点表可用指标", loc="left", fontsize=11.5, fontweight="bold")
    legend_below(ax, max_cols=2, anchor_y=-0.23, fontsize=8)
    style_axis(ax)

    ax = axes[2]
    ax.plot(x, [r["mRNA_t_end_uM"] for _, r in states], "-o", color=COLORS["mRNA_t"], lw=2.1, label=LABELS["mRNA_t"])
    ax.plot(x, [r["Ribosome_mRNA_end_uM"] for _, r in states], "-s", color=COLORS["Ribosome_mRNA"], lw=2.1, label=LABELS["Ribosome_mRNA"])
    ax.set_ylabel("6 h 浓度 (μM)")
    ax.set_title("C  其他变量：模板与核糖体承接", loc="left", fontsize=11.5, fontweight="bold")
    legend_below(ax, max_cols=2, anchor_y=-0.23, fontsize=8)
    style_axis(ax)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("ATP/GTP 状态")
    save_figure(fig, "atp_gtp_synergy_three_curve_groups_20260728")


def main() -> None:
    setup_font()
    records = pd.read_csv(UNIFIED_RECORDS)
    ratios = pd.read_csv(UNIFIED_RATIOS)
    gtp_extra = rnap_atp_wait_ratio(records)
    standard_axis_figure(
        "GTP", records, ratios, "4x/3x", "GTP 单独扰动：三类曲线统一归组",
        [
            ("mRNA_t_ratio", LABELS["mRNA_t"], COLORS["mRNA_t"]),
            ("Ribosome_mRNA_ratio", LABELS["Ribosome_mRNA"], COLORS["Ribosome_mRNA"]),
            ("TLchain_pos_pool_ratio", LABELS["TLchain_pos_pool"], COLORS["TLchain_pos_pool"]),
            ("RNAP_ATP_wait_ratio", "RNAP卡A位点池", COLORS["RNAP_ATP_wait"]),
        ],
        extra_other=gtp_extra,
    )
    standard_axis_figure(
        "ATP", records, ratios, "1x/0.5x", "ATP 单独扰动：三类曲线统一归组",
        [
            ("mRNA_t_ratio", LABELS["mRNA_t"], COLORS["mRNA_t"]),
            ("Ribosome_mRNA_ratio", LABELS["Ribosome_mRNA"], COLORS["Ribosome_mRNA"]),
            ("TLchain_pos_pool_ratio", LABELS["TLchain_pos_pool"], COLORS["TLchain_pos_pool"]),
            ("unit_TLpool_progression_efficiency_ratio", LABELS["unit_TLpool_progression_efficiency"], COLORS["unit_TLpool_progression_efficiency"]),
            ("charged_tRNA_pool_ratio", LABELS["charged_tRNA_pool"], COLORS["charged_tRNA_pool"]),
        ],
        endpoint_override=atp_endpoint_table().rename(
            columns={"scale": "dose", "deGFP_end_uM": "deGFP", "PPi_end_uM": "PPi"}
        ),
    )
    ntp_endpoints = ctp_utp_endpoint_df()
    for axis in ("CTP", "UTP"):
        endpoint = ntp_endpoints[ntp_endpoints["perturb"].eq(axis)].rename(columns={"fold": "dose"})
        standard_axis_figure(
            axis, records, ratios, "0.03x/0.01x", f"{axis} 单独扰动：三类曲线统一归组",
            [
                ("mRNA_t_ratio", LABELS["mRNA_t"], COLORS["mRNA_t"]),
                ("Ribosome_mRNA_ratio", LABELS["Ribosome_mRNA"], COLORS["Ribosome_mRNA"]),
                ("TLchain_pos_pool_ratio", LABELS["TLchain_pos_pool"], COLORS["TLchain_pos_pool"]),
            ],
            endpoint_override=endpoint,
        )
    plot_total_aars()
    plot_metrs()
    plot_synergy()


if __name__ == "__main__":
    main()
