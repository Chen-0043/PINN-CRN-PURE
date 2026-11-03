#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Axis-specific zero-feed response heatmap for the PURE manuscript.

The plotted values are zero-perturbation snapshots divided by each axis'
optimum-dose snapshots. MetRS is intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUN_OUTPUTS = PROJECT / "run_outputs"
FIG_DIR = ROOT / "figures" / "article_331_20260813"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_records_20260723.csv"

AXIS_ORDER = ["GTP", "ATP", "total_aaRS", "CTP", "UTP"]
AXIS_LABEL = {
    "GTP": "GTP",
    "ATP": "ATP",
    "total_aaRS": "总氨酰化模块",
    "CTP": "CTP",
    "UTP": "UTP",
}
OPTIMUM_SCALE = {
    "GTP": 3.0,
    "ATP": 0.5,
    "total_aaRS": 0.22,
    "CTP": 0.01,
    "UTP": 0.01,
}
ZERO_COMPONENT = {
    "GTP": "GTP",
    "ATP": "ATP",
    "total_aaRS": "total_aars_pool",
    "CTP": "CTP",
    "UTP": "UTP",
}
PAIR_TEXT = {
    "GTP": "0x / 3x",
    "ATP": "0x / 0.5x",
    "total_aaRS": "0x / 0.22x",
    "CTP": "0x / 0.01x",
    "UTP": "0x / 0.01x",
}
COMMON_GTP_FLUX = [
    ("tx_gtp_entry_ratio", "转录GTP掺入通量"),
    ("tl_gtp_entry_ratio", "翻译辅助GTP消耗通量"),
    ("tl_gtp_progression_ratio", "翻译主链推进GTP消耗通量"),
    ("total_GTP_pool_ratio", "总GTP浓度"),
]
AXIS_VARS = {
    "GTP": [
        ("deGFP_ratio", "deGFP"),
        *COMMON_GTP_FLUX,
        ("ATP_ratio", "ATP"),
        ("ATP_TX_ratio", "ATP_TX"),
        ("mRNA_t_ratio", "mRNA_t"),
        ("Ribosome_mRNA_ratio", "Ribosome_mRNA"),
        ("TLchain_pos_pool_ratio", "TLchain_pos pool"),
        ("RNAP_ATP_wait_ratio", "RNAP卡A位点池"),
    ],
    "ATP": [
        ("deGFP_ratio", "deGFP"),
        *COMMON_GTP_FLUX,
        ("ATP_ratio", "ATP"),
        ("ATP_TX_ratio", "ATP_TX"),
        ("mRNA_t_ratio", "mRNA_t"),
        ("Ribosome_mRNA_ratio", "Ribosome_mRNA"),
        ("TLchain_pos_pool_ratio", "TLchain_pos pool"),
        ("charged_tRNA_pool_ratio", "charged-tRNA pool"),
    ],
    "total_aaRS": [
        ("deGFP_ratio", "deGFP"),
        *COMMON_GTP_FLUX,
        ("charged_tRNA_pool_ratio", "aa-tRNA pool"),
        ("EFTu_GTP_charged_tRNA_pool_ratio", "EFTu-GTP-aa-tRNA"),
        ("mRNA_t_ratio", "mRNA_t"),
        ("Ribosome_mRNA_ratio", "Ribosome_mRNA"),
        ("TLchain_pos_pool_ratio", "TLchain_pos pool"),
    ],
    "CTP": [
        ("deGFP_ratio", "deGFP"),
        *COMMON_GTP_FLUX,
        ("ATP_ratio", "ATP"),
        ("ATP_TX_ratio", "ATP_TX"),
        ("mRNA_t_ratio", "mRNA_t"),
        ("Ribosome_mRNA_ratio", "Ribosome_mRNA"),
        ("TLchain_pos_pool_ratio", "TLchain_pos pool"),
    ],
    "UTP": [
        ("deGFP_ratio", "deGFP"),
        *COMMON_GTP_FLUX,
        ("ATP_ratio", "ATP"),
        ("ATP_TX_ratio", "ATP_TX"),
        ("mRNA_t_ratio", "mRNA_t"),
        ("Ribosome_mRNA_ratio", "Ribosome_mRNA"),
        ("TLchain_pos_pool_ratio", "TLchain_pos pool"),
    ],
}
GROUP_COLOR = {
    "GTP": "#1B8A8F",
    "ATP": "#56B4E9",
    "total_aaRS": "#009E73",
    "CTP": "#2C7FB8",
    "UTP": "#D95F0E",
}
ZERO_METRICS = {
    "GTP": RUN_OUTPUTS / "missing_atp_gtp_zero_unified_time" / "synced_subset_metrics.csv",
    "ATP": RUN_OUTPUTS / "missing_atp_gtp_zero_unified_time" / "synced_subset_metrics.csv",
    "CTP": RUN_OUTPUTS / "missing_ctp_utp_zero_unified_time" / "synced_subset_metrics.csv",
    "UTP": RUN_OUTPUTS / "missing_ctp_utp_zero_unified_time" / "synced_subset_metrics.csv",
    "total_aaRS": RUN_OUTPUTS / "missing_total_aars_zero_unified_time" / "total_aars_pool_metrics.csv",
}
GAP_ROWS = 1


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


def resolve_snapshot_path(value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    candidate = PROJECT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def read_species(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_reactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["reaction_id_str"] = df["reaction_id"].astype(str)
    df["reaction_class_str"] = df["reaction_class"].fillna("").astype(str)
    df["reactants_str"] = df["reactants"].fillna("").astype(str)
    return df


def species_sum_at(group: pd.DataFrame, selector) -> float:
    species = group["species"].astype(str)
    return float(group.loc[selector(species), "concentration_uM"].astype(float).sum())


def summarize_species(sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in sp.groupby("time_h", sort=True):
        vals = dict(zip(g["species"].astype(str), g["concentration_uM"].astype(float)))
        ribo = float(vals.get("Ribosome_mRNA", np.nan))
        tlpool = species_sum_at(g, lambda s: s.str.startswith("TLchain_pos_"))
        charged = species_sum_at(
            g,
            lambda s: (s.str.match(r"^[A-Z][a-z]{2}tRNA") | s.str.startswith("fMettRNA"))
            & ~s.str.contains("degraded", regex=False),
        )
        eftu_charged = species_sum_at(
            g,
            lambda s: s.str.startswith("EFTu_GTP_")
            & (s.str.contains("tRNA", regex=False) | s.str.contains("fMettRNA", regex=False))
            & ~s.str.contains("degraded", regex=False),
        )
        rows.append(
            {
                "time_h": float(t),
                "deGFP": float(vals.get("deGFP_m", np.nan)),
                "mRNA_t": float(vals.get("mRNA_t", np.nan)),
                "Ribosome_mRNA": ribo,
                "TLchain_pos_pool": tlpool,
                "ATP": float(vals.get("ATP", np.nan)),
                "ATP_TX": float(vals.get("ATP_TX", np.nan)),
                "GTP": float(vals.get("GTP", np.nan)),
                "GTP_TX": float(vals.get("GTP_TX", np.nan)),
                "total_GTP_pool": float(vals.get("GTP", np.nan)) + float(vals.get("GTP_TX", np.nan)),
                "PPi": float(vals.get("PPi", np.nan)),
                "charged_tRNA_pool": charged,
                "EFTu_GTP_charged_tRNA_pool": eftu_charged,
            }
        )
    return pd.DataFrame(rows)


def summarize_reactions(rx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in rx.groupby("time_h", sort=True):
        rid = g["reaction_id_str"]
        rcls = g["reaction_class_str"]
        react = g["reactants_str"]
        rate = g["rate_uM_per_h"].astype(float)
        gtp_react = react.str.contains("GTP", regex=False)
        tx = rcls.eq("transcription_or_mRNA_turnover") & gtp_react
        prog = rid.str.startswith("rxn_tl_chain_step")
        total = float(rate[gtp_react].sum())
        tx_val = float(rate[tx].sum())
        prog_val = float(rate[prog].sum())
        rows.append(
            {
                "time_h": float(t),
                "tx_gtp_entry": tx_val,
                "tl_gtp_entry": total - tx_val - prog_val,
                "tl_gtp_progression": prog_val,
            }
        )
    return pd.DataFrame(rows)


def summarize_case(species_path: Path, reaction_path: Path) -> pd.DataFrame:
    sp = summarize_species(read_species(species_path))
    rx = summarize_reactions(read_reactions(reaction_path))
    return sp.merge(rx, on="time_h", how="outer").sort_values("time_h")


def zero_paths(axis: str) -> tuple[Path, Path]:
    metrics = pd.read_csv(ZERO_METRICS[axis])
    component = ZERO_COMPONENT[axis]
    row = metrics[(metrics["component"].eq(component)) & np.isclose(metrics["scale"], 0.0)].iloc[0]
    return resolve_snapshot_path(row["species_snapshot_csv_gz"]), resolve_snapshot_path(row["reaction_snapshot_csv_gz"])


def optimum_paths(records: pd.DataFrame, axis: str) -> tuple[Path, Path]:
    opt_scale = OPTIMUM_SCALE[axis]
    row = records[(records["axis"].eq(axis)) & np.isclose(records["scale"], opt_scale)].iloc[0]
    return resolve_snapshot_path(row["source_species_snapshot"]), resolve_snapshot_path(row["source_reaction_snapshot"])


def compute_rnap_atp_wait(species_path: Path, reaction_path: Path) -> pd.DataFrame:
    rx = pd.read_csv(reaction_path)
    states: set[str] = set()
    for reactants in rx["reactants"].fillna("").astype(str):
        match = re.search(r"(T7RNAP_bound_mRNA\d+);ATP_TX", reactants)
        if match:
            states.add(match.group(1))
    sp = pd.read_csv(species_path)
    return (
        sp[sp["species"].isin(states)]
        .groupby("time_h", as_index=False)["concentration_uM"]
        .sum()
        .rename(columns={"concentration_uM": "RNAP_ATP_wait"})
    )


def ratio_for_axis(records: pd.DataFrame, axis: str) -> pd.DataFrame:
    zero_sp, zero_rx = zero_paths(axis)
    opt_sp, opt_rx = optimum_paths(records, axis)
    zero = summarize_case(zero_sp, zero_rx)
    opt = summarize_case(opt_sp, opt_rx)

    if axis == "GTP":
        zero = zero.merge(compute_rnap_atp_wait(zero_sp, zero_rx), on="time_h", how="left")
        opt = opt.merge(compute_rnap_atp_wait(opt_sp, opt_rx), on="time_h", how="left")

    keep = sorted(set(zero["time_h"]).intersection(set(opt["time_h"])))
    zero = zero[zero["time_h"].isin(keep)].copy()
    opt = opt[opt["time_h"].isin(keep)].copy()
    wide = zero.merge(opt, on="time_h", suffixes=("_zero", "_opt"), how="inner")
    out = pd.DataFrame({"time_h": wide["time_h"].astype(float)})
    cols = {
        "deGFP",
        "mRNA_t",
        "Ribosome_mRNA",
        "TLchain_pos_pool",
        "ATP",
        "ATP_TX",
        "total_GTP_pool",
        "charged_tRNA_pool",
        "EFTu_GTP_charged_tRNA_pool",
        "tx_gtp_entry",
        "tl_gtp_entry",
        "tl_gtp_progression",
        "RNAP_ATP_wait",
    }
    for col in cols:
        zcol = f"{col}_zero"
        ocol = f"{col}_opt"
        if zcol in wide and ocol in wide:
            numer = wide[zcol].astype(float).clip(lower=0.0)
            denom = wide[ocol].astype(float).clip(lower=0.0)
            ratio = pd.Series(np.nan, index=wide.index, dtype=float)
            denom_ok = denom > 1e-30
            both_zero = (~denom_ok) & (numer <= 1e-30)
            ratio.loc[denom_ok] = numer.loc[denom_ok] / denom.loc[denom_ok]
            ratio.loc[both_zero] = 1.0
            out[f"{col}_ratio"] = ratio
    return out.sort_values("time_h").reset_index(drop=True)


def build_rows(records: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    rows: list[np.ndarray] = []
    ylabels: list[str] = []
    axis_per_row: list[str] = []
    times_ref: np.ndarray | None = None
    for axis in AXIS_ORDER:
        ratio = ratio_for_axis(records, axis)
        times = ratio["time_h"].to_numpy(float)
        if times_ref is None:
            times_ref = times
        elif not np.allclose(times_ref, times):
            raise ValueError(f"Time grid mismatch for {axis}: {times}")
        for col, label in AXIS_VARS[axis]:
            if col not in ratio.columns:
                raise KeyError(f"Missing {col} for {axis}")
            rows.append(ratio[col].to_numpy(float))
            ylabels.append(label)
            axis_per_row.append(axis)
    assert times_ref is not None
    return np.asarray(rows, dtype=float), ylabels, axis_per_row, times_ref


def time_edges(times: np.ndarray) -> np.ndarray:
    mid = (times[:-1] + times[1:]) / 2
    first = times[0] - (mid[0] - times[0])
    last = times[-1] + (times[-1] - mid[-1])
    return np.r_[first, mid, last]


def insert_axis_gaps(
    matrix: np.ndarray, ylabels: list[str], axis_per_row: list[str]
) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int, int, int]]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    spans: dict[str, tuple[int, int, int, int]] = {}
    source_idx = 0
    for axis in AXIS_ORDER:
        axis_rows = sum(a == axis for a in axis_per_row)
        start = len(rows)
        for _ in range(axis_rows):
            rows.append(matrix[source_idx])
            labels.append(ylabels[source_idx])
            source_idx += 1
        end = len(rows)
        gap_start = len(rows)
        for _ in range(GAP_ROWS):
            rows.append(np.full(matrix.shape[1], np.nan))
            labels.append("")
        gap_end = len(rows)
        spans[axis] = (start, end, gap_start, gap_end)
    return np.asarray(rows, dtype=float), labels, spans


def plot_heatmap(matrix: np.ndarray, ylabels: list[str], axis_per_row: list[str], times: np.ndarray) -> plt.Figure:
    display_matrix, display_ylabels, spans = insert_axis_gaps(matrix, ylabels, axis_per_row)
    plot_matrix = np.where(np.isfinite(display_matrix), np.maximum(display_matrix, 2 ** -10), np.nan)
    logv = np.log2(plot_matrix)
    logv = np.clip(logv, -10.0, 3.0)
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#08306B", "#2166AC", "#F7F7F7", "#B2182B"])
    cmap.set_bad("#FFFFFF")

    nrows = display_matrix.shape[0]
    fig_h = max(11.2, 0.24 * nrows + 2.35)
    fig, ax = plt.subplots(figsize=(16.8, fig_h))
    mesh = ax.pcolormesh(time_edges(times), np.arange(nrows + 1), logv, cmap=cmap, vmin=-10, vmax=3, shading="flat")
    ax.set_ylim(nrows, 0)
    ax.set_xlim(0, 6)
    ax.set_xticks(np.arange(0, 6.1, 1))
    ax.set_xlabel("时间 (h)", fontsize=11.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(display_ylabels, fontsize=8.0)
    ax.tick_params(axis="y", length=0, pad=28)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title("连续 PURE 体系 0 投料相对于最优投料点的变量响应", fontsize=18, fontweight="bold", pad=18)

    for axis in AXIS_ORDER:
        row_start, row_end, gap_start, gap_end = spans[axis]
        center = (row_start + row_end) / 2
        ax.axhline(row_start, color="#D6DEE8", lw=1.6)
        ax.axhline(row_end, color="#D6DEE8", lw=1.2)
        ax.axhline(gap_end, color="#334155", lw=1.0)
        ax.add_patch(
            plt.Rectangle((-0.19, row_start), 0.16, row_end - row_start, color=GROUP_COLOR[axis], clip_on=False, lw=0)
        )
        ax.text(
            -0.94,
            center,
            f"{AXIS_LABEL[axis]}\n{PAIR_TEXT[axis]}",
            ha="right",
            va="center",
            fontsize=10.0,
            color="#475569",
            linespacing=1.25,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.035, pad=0.075, aspect=42)
    cbar.set_label("log2(0投料 / 最优投料)", fontsize=10.0)
    cbar.set_ticks([-10, -8, -6, -4, -2, 0, 1, 2, 3])
    cbar.ax.tick_params(labelsize=8.5)
    fig.text(
        0.49,
        0.018,
        "注：0 投料数据来自 扰动模拟-new/run_outputs 下最新补跑；最优点为 GTP 3x、ATP 0.5x、总氨酰化模块 0.22x、CTP/UTP 0.01x；MetRS 扰动已去除。",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#475569",
    )
    fig.tight_layout(rect=[0.13, 0.055, 0.985, 0.965])
    return fig


def main() -> None:
    setup_font()
    records = pd.read_csv(RECORDS_CSV)
    matrix, ylabels, axis_per_row, times = build_rows(records)
    fig = plot_heatmap(matrix, ylabels, axis_per_row, times)
    out_stem = FIG_DIR / "figure_331_axis_specific_zero_vs_optimum_heatmap_20260813"
    for suffix in ("png", "svg", "pdf"):
        out = out_stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=280 if suffix == "png" else None, bbox_inches="tight")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
