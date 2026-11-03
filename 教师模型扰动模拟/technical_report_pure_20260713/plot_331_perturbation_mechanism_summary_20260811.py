#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for manuscript section 3.3.1.

The figure summarizes five perturbation axes used in the article:
GTP, ATP, total_aaRS, CTP and UTP.  It intentionally excludes the old MetRS-only
axis.  Over-optimum responses are read from the unified technical-report ratio
table, while the CTP/UTP 0× shortage endpoints are read from the latest rerun
inside ``扰动模拟-new/run_outputs/missing_ctp_utp_zero_unified_time``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "article_331_20260811"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_records_20260723.csv"
RATIOS_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_ratios_20260723.csv"
ZERO_METRICS_CSV = ROOT.parent / "run_outputs" / "missing_ctp_utp_zero_unified_time" / "synced_subset_metrics.csv"


AXIS_ORDER = ["GTP", "ATP", "total_aaRS", "CTP", "UTP"]
AXIS_LABEL = {
    "GTP": "GTP",
    "ATP": "ATP",
    "total_aaRS": "总氨酰化模块",
    "CTP": "CTP",
    "UTP": "UTP",
}
COMPARISON = {
    "GTP": "4x/3x",
    "ATP": "4x/1x",
    "total_aaRS": "0.25x/0.22x",
    "CTP": "0.03x/0.01x",
    "UTP": "0.03x/0.01x",
}
STAGES = {
    "GTP": [(0.00, 1.25, "早期建立"), (1.25, 1.50, "瓶颈转移"), (1.50, 6.00, "新平衡")],
    "ATP": [(0.00, 0.25, "早期建立"), (0.25, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
    "total_aaRS": [(0.00, 1.75, "早期建立"), (1.75, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
    "CTP": [(0.00, 1.50, "早期建立"), (1.50, 2.25, "瓶颈转移"), (2.25, 6.00, "新平衡")],
    "UTP": [(0.00, 1.50, "早期建立"), (1.50, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
}

HEATMAP_VARS = [
    ("deGFP_ratio", "deGFP"),
    ("mRNA_t_ratio", "mRNA_t"),
    ("Ribosome_mRNA_ratio", "Ribosome-\nmRNA"),
    ("TLchain_pos_pool_ratio", "翻译位置链"),
    ("charged_tRNA_pool_ratio", "aa-tRNA\npool"),
    ("ATP_ratio", "ATP"),
    ("total_GTP_pool_ratio", "总GTP池"),
    ("PPi_ratio", "PPi"),
    ("tx_gtp_entry_ratio", "转录GTP\n消耗"),
    ("tl_gtp_entry_ratio", "翻译辅助\nGTP消耗"),
    ("tl_gtp_progression_ratio", "翻译主链\nGTP消耗"),
    ("termination_release_ratio", "终止释放"),
]

FLUX_VARS = [
    ("tx_gtp_entry_ratio", "转录GTP消耗", "#2C7FB8"),
    ("tl_gtp_entry_ratio", "翻译辅助GTP消耗", "#636363"),
    ("tl_gtp_progression_ratio", "翻译主链GTP消耗", "#D95F0E"),
]

SHORTAGE_VARS = [
    ("deGFP", "deGFP"),
    ("mRNA_t", "mRNA_t"),
    ("Ribosome_mRNA", "Ribosome-\nmRNA"),
    ("GTP", "GTP"),
    ("ATP", "ATP"),
    ("PPi", "PPi"),
]

SHORTAGE_PAIR = {
    "GTP": (0.25, 3.00),
    "ATP": (0.50, 1.00),
    "total_aaRS": (0.22, 0.25),
    "CTP": (0.00, 0.01),
    "UTP": (0.00, 0.01),
}


def setup_font() -> None:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        p = Path(path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(p)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def stage_mean(frame: pd.DataFrame, start: float, end: float, col: str) -> float:
    mask = (frame["time_h"] >= start) & (frame["time_h"] <= end)
    values = frame.loc[mask, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def build_stage_table(ratios: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    rows = []
    ylabels = []
    group_labels = []
    for axis in AXIS_ORDER:
        sub = ratios[(ratios["axis"].eq(axis)) & (ratios["comparison"].eq(COMPARISON[axis]))].copy()
        for start, end, stage in STAGES[axis]:
            row = {"axis": axis, "stage": stage, "start": start, "end": end}
            for col, _label in HEATMAP_VARS:
                row[col] = stage_mean(sub, start, end, col)
            rows.append(row)
            ylabels.append(f"{AXIS_LABEL[axis]}  {stage}\n{start:g}–{end:g} h")
            group_labels.append(axis)
    table = pd.DataFrame(rows)
    matrix = table[[col for col, _label in HEATMAP_VARS]].to_numpy(float)
    return matrix, ylabels, group_labels, table


def value_to_text(ratio: float) -> str:
    if not np.isfinite(ratio):
        return ""
    if ratio >= 10:
        return f"{ratio:.0f}"
    if ratio >= 1:
        return f"{ratio:.2g}"
    if ratio >= 0.1:
        return f"{ratio:.2f}"
    return f"{ratio:.1e}"


def plot_stage_heatmap(ax, matrix: np.ndarray, ylabels: list[str], group_labels: list[str]) -> None:
    logv = np.log2(matrix)
    logv = np.clip(logv, -2.5, 2.5)
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#2166AC", "#F7F7F7", "#B2182B"])
    im = ax.imshow(logv, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(HEATMAP_VARS)))
    ax.set_xticklabels([label for _col, label in HEATMAP_VARS], fontsize=8.6)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.4)
    ax.tick_params(axis="x", rotation=0, pad=4)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A  过最优相邻扰动的时序响应：不同扰动采用各自阶段划分", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("机制变量与模块通量（相对于最优相邻条件的比值）", fontsize=10.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = value_to_text(matrix[i, j])
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="#111111")

    for cut in [2.5, 5.5, 8.5, 11.5]:
        ax.axhline(cut, color="white", lw=3.0)
    for cut in [3.5, 7.5, 10.5]:
        ax.axvline(cut, color="white", lw=2.2)

    group_color = {
        "GTP": "#1B8A8F",
        "ATP": "#56B4E9",
        "total_aaRS": "#009E73",
        "CTP": "#2C7FB8",
        "UTP": "#D95F0E",
    }
    for i, axis in enumerate(group_labels):
        ax.add_patch(plt.Rectangle((-1.35, i - 0.5), 0.22, 1.0, color=group_color[axis], clip_on=False, lw=0))

    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("log2(过最优/最优)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlim(-0.5, len(HEATMAP_VARS) - 0.5)


def plot_flux_strips(ax, stage_table: pd.DataFrame) -> None:
    flux_rows = []
    ylabels = []
    for axis in AXIS_ORDER:
        sub = stage_table[stage_table["axis"].eq(axis)].reset_index(drop=True)
        for col, label, _color in FLUX_VARS:
            flux_rows.append(sub[col].to_numpy(float))
            ylabels.append(f"{AXIS_LABEL[axis]}  {label}")
    matrix = np.asarray(flux_rows, dtype=float)
    logv = np.log2(np.clip(matrix, 1e-6, 1e6))
    logv = np.clip(logv, -2.5, 2.5)
    cmap = LinearSegmentedColormap.from_list("flux", ["#2166AC", "#F7F7F7", "#B2182B"])
    im = ax.imshow(logv, aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    ax.set_title("B  过最优后 GTP 通量分配的阶段性偏移", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["早期建立", "瓶颈转移", "新平衡"], fontsize=9.5)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("阶段窗口", fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, value_to_text(matrix[i, j]), ha="center", va="center", fontsize=6.8, color="#111111")
    for cut in [2.5, 5.5, 8.5, 11.5]:
        ax.axhline(cut, color="white", lw=2.6)
    for cut in [0.5, 1.5]:
        ax.axvline(cut, color="white", lw=1.7)

    group_color = {
        "GTP": "#1B8A8F",
        "ATP": "#56B4E9",
        "total_aaRS": "#009E73",
        "CTP": "#2C7FB8",
        "UTP": "#D95F0E",
    }
    for ai, axis in enumerate(AXIS_ORDER):
        ax.add_patch(plt.Rectangle((-0.68, ai * 3 - 0.5), 0.10, 3.0, color=group_color[axis], clip_on=False, lw=0))
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.018)
    cbar.set_label("log2(过最优/最优)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

def endpoint_row(records: pd.DataFrame, axis: str, scale: float) -> pd.Series:
    sub = records[(records["axis"].eq(axis)) & np.isclose(records["scale"], scale) & np.isclose(records["time_h"], 6.0)]
    if sub.empty:
        raise ValueError(f"Missing endpoint for {axis} {scale}")
    return sub.iloc[0]


def zero_row(zero_metrics: pd.DataFrame, axis: str) -> pd.Series:
    sub = zero_metrics[zero_metrics["component"].eq(axis)]
    if sub.empty:
        raise ValueError(f"Missing zero endpoint for {axis}")
    row = sub.iloc[0]
    return pd.Series({
        "deGFP": row["deGFP_end_uM"],
        "mRNA_t": row["mRNA_t_end_uM"],
        "Ribosome_mRNA": row["Ribosome_mRNA_end_uM"],
        "GTP": row["GTP_end_uM"],
        "ATP": row["ATP_end_uM"],
        "PPi": row["PPi_end_uM"],
    })


def build_shortage_matrix(records: pd.DataFrame, zero_metrics: pd.DataFrame) -> np.ndarray:
    values = []
    for axis in AXIS_ORDER:
        low_scale, opt_scale = SHORTAGE_PAIR[axis]
        opt = endpoint_row(records, axis, opt_scale)
        if axis in {"CTP", "UTP"} and low_scale == 0:
            low = zero_row(zero_metrics, axis)
        else:
            low = endpoint_row(records, axis, low_scale)
        row = []
        for col, _label in SHORTAGE_VARS:
            denom = float(opt[col])
            numer = float(low[col])
            if abs(denom) < 1e-12:
                row.append(np.nan)
            else:
                row.append(max(numer, 0.0) / denom)
        values.append(row)
    return np.asarray(values, dtype=float)


def plot_shortage(ax, matrix: np.ndarray) -> None:
    logv = np.log10(np.clip(matrix, 1e-6, 1e6))
    logv = np.clip(logv, -3.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("shortage", ["#08306B", "#F7F7F7", "#D94801"])
    im = ax.imshow(logv, aspect="auto", cmap=cmap, vmin=-3, vmax=1)
    ax.set_title("C  底物不足导致的断链型错配：终点相对于最优条件的浓度", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(SHORTAGE_VARS)))
    ax.set_xticklabels([label for _col, label in SHORTAGE_VARS], fontsize=9)
    ax.set_yticks(np.arange(len(AXIS_ORDER)))
    ax.set_yticklabels([AXIS_LABEL[a] for a in AXIS_ORDER], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, value_to_text(value), ha="center", va="center", fontsize=7.2, color="#111111")
    for cut in np.arange(0.5, len(AXIS_ORDER), 1):
        ax.axhline(cut, color="white", lw=1.6)
    for cut in np.arange(0.5, len(SHORTAGE_VARS), 1):
        ax.axvline(cut, color="white", lw=1.4)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("log10(不足/最优)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)


def main() -> None:
    setup_font()
    records = pd.read_csv(RECORDS_CSV)
    ratios = pd.read_csv(RATIOS_CSV)
    zero_metrics = pd.read_csv(ZERO_METRICS_CSV)

    matrix, ylabels, group_labels, stage_table = build_stage_table(ratios)
    shortage = build_shortage_matrix(records, zero_metrics)

    fig = plt.figure(figsize=(18.2, 14.4))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.15, 1.0],
        width_ratios=[1.18, 1.0],
        hspace=0.30,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "连续 PURE 体系扰动响应：资源适配、瓶颈转移与断链型错配",
        fontsize=17.5,
        fontweight="bold",
        y=0.985,
    )
    plot_stage_heatmap(ax_a, matrix, ylabels, group_labels)
    plot_flux_strips(ax_b, stage_table)
    plot_shortage(ax_c, shortage)

    fig.text(
        0.012,
        0.012,
        "注：A、B 为相邻过最优条件相对于最优相邻条件的比值；C 为不足条件相对于最优条件的终点浓度比值。"
        "CTP/UTP 不足采用最新 0× 统一时间点补跑结果；总氨酰化模块为 total_aaRS 扰动。",
        fontsize=9.2,
        color="#333333",
    )
    fig.tight_layout(rect=[0.018, 0.035, 0.99, 0.96])

    out_stem = FIG_DIR / "figure_331_perturbation_mechanism_summary_20260811"
    for suffix in ["png", "svg", "pdf"]:
        out = out_stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=260 if suffix == "png" else None, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
