#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Axis-specific over-optimum response heatmap for the PURE manuscript.

Each perturbation axis uses its own adjacent over-optimum/optimum comparison
and its own variable set. MetRS is intentionally excluded.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIG_DIR = ROOT / "figures" / "article_331_20260813"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_records_20260723.csv"
RATIOS_CSV = ROOT / "report_audit_20260723" / "unified_perturbation_timeseries_ratios_20260723.csv"

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
    "ATP": "1x/0.5x",
    "total_aaRS": "0.25x/0.22x",
    "CTP": "0.03x/0.01x",
    "UTP": "0.03x/0.01x",
}
PAIR_TEXT = {
    "GTP": "3x -> 4x",
    "ATP": "0.5x -> 1x",
    "total_aaRS": "0.22x -> 0.25x",
    "CTP": "0.01x -> 0.03x",
    "UTP": "0.01x -> 0.03x",
}
STAGES = {
    "GTP": [(0.00, 1.25, "早期响应"), (1.25, 1.50, "瓶颈转移"), (1.50, 6.00, "新平衡")],
    "ATP": [(0.00, 0.25, "早期响应"), (0.25, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
    "total_aaRS": [(0.00, 1.75, "早期响应"), (1.75, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
    "CTP": [(0.00, 1.50, "早期响应"), (1.50, 2.25, "瓶颈转移"), (2.25, 6.00, "新平衡")],
    "UTP": [(0.00, 1.50, "早期响应"), (1.50, 2.00, "瓶颈转移"), (2.00, 6.00, "新平衡")],
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
GAP_ROWS = 1


class SignedPowerNorm(Normalize):
    """Symmetric nonlinear norm that gives more visual range to values near zero."""

    def __init__(self, vmin: float, vmax: float, gamma: float = 0.55, clip: bool = False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = result.data.astype(float)
        limit = max(abs(float(self.vmin)), abs(float(self.vmax)))
        scaled = np.sign(data) * np.power(np.clip(np.abs(data) / limit, 0, 1), self.gamma)
        mapped = 0.5 + 0.5 * scaled
        out = np.ma.array(mapped, mask=result.mask, copy=False)
        return out[0] if is_scalar else out

    def inverse(self, value):
        value = np.asarray(value)
        limit = max(abs(float(self.vmin)), abs(float(self.vmax)))
        scaled = (value - 0.5) * 2
        return np.sign(scaled) * limit * np.power(np.abs(scaled), 1 / self.gamma)


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


def compute_rnap_atp_wait_ratio(records: pd.DataFrame) -> pd.DataFrame:
    """Sum T7RNAP-bound states whose next transcription substrate is ATP_TX."""
    row4 = records[(records["axis"].eq("GTP")) & np.isclose(records["scale"], 4.0)].iloc[0]
    rx = pd.read_csv(resolve_snapshot_path(row4["source_reaction_snapshot"]))
    states: set[str] = set()
    for reactants in rx["reactants"].fillna("").astype(str):
        match = re.search(r"(T7RNAP_bound_mRNA\d+);ATP_TX", reactants)
        if match:
            states.add(match.group(1))

    frames = []
    for scale in (3.0, 4.0):
        row = records[(records["axis"].eq("GTP")) & np.isclose(records["scale"], scale)].iloc[0]
        sp = pd.read_csv(resolve_snapshot_path(row["source_species_snapshot"]))
        pooled = (
            sp[sp["species"].isin(states)]
            .groupby("time_h", as_index=False)["concentration_uM"]
            .sum()
            .rename(columns={"concentration_uM": scale})
        )
        frames.append(pooled)
    wide = frames[0].merge(frames[1], on="time_h", how="inner")
    wide["RNAP_ATP_wait_ratio"] = wide[4.0] / wide[3.0].where(wide[3.0] > 0)
    return wide[["time_h", "RNAP_ATP_wait_ratio"]]


def time_edges(times: np.ndarray) -> np.ndarray:
    mid = (times[:-1] + times[1:]) / 2
    first = times[0] - (mid[0] - times[0])
    last = times[-1] + (times[-1] - mid[-1])
    return np.r_[first, mid, last]


def build_rows(ratios: pd.DataFrame, records: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    rnap_wait = compute_rnap_atp_wait_ratio(records)
    rows: list[np.ndarray] = []
    ylabels: list[str] = []
    axis_per_row: list[str] = []
    times_ref: np.ndarray | None = None

    for axis in AXIS_ORDER:
        sub = ratios[(ratios["axis"].eq(axis)) & ratios["comparison"].eq(COMPARISON[axis])].copy()
        sub = sub.sort_values("time_h").reset_index(drop=True)
        if axis == "GTP":
            sub = sub.merge(rnap_wait, on="time_h", how="left")
        times = sub["time_h"].to_numpy(float)
        if times_ref is None:
            times_ref = times
        elif not np.allclose(times_ref, times):
            raise ValueError(f"Time grid mismatch for {axis}")

        for col, label in AXIS_VARS[axis]:
            if col not in sub.columns:
                raise KeyError(f"Missing {col} for {axis}")
            rows.append(sub[col].to_numpy(float))
            ylabels.append(label)
            axis_per_row.append(axis)

    assert times_ref is not None
    return np.asarray(rows, dtype=float), ylabels, axis_per_row, times_ref


def insert_axis_gaps(
    matrix: np.ndarray, ylabels: list[str], axis_per_row: list[str]
) -> tuple[np.ndarray, list[str], list[str], dict[str, tuple[int, int, int, int]]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    row_axes: list[str] = []
    spans: dict[str, tuple[int, int, int, int]] = {}
    source_idx = 0

    for axis in AXIS_ORDER:
        axis_rows = sum(a == axis for a in axis_per_row)
        start = len(rows)
        for _ in range(axis_rows):
            rows.append(matrix[source_idx])
            labels.append(ylabels[source_idx])
            row_axes.append(axis)
            source_idx += 1
        end = len(rows)
        gap_start = len(rows)
        for _ in range(GAP_ROWS):
            rows.append(np.full(matrix.shape[1], np.nan))
            labels.append("")
            row_axes.append("__gap__")
        gap_end = len(rows)
        spans[axis] = (start, end, gap_start, gap_end)

    return np.asarray(rows, dtype=float), labels, row_axes, spans


def plot_heatmap(matrix: np.ndarray, ylabels: list[str], axis_per_row: list[str], times: np.ndarray) -> plt.Figure:
    display_matrix, display_ylabels, _display_axes, spans = insert_axis_gaps(matrix, ylabels, axis_per_row)
    logv = np.log2(display_matrix)
    logv = np.clip(logv, -3.0, 3.0)
    cmap = LinearSegmentedColormap.from_list(
        "purple_blue_white_yellow_red",
        [
            (0.00, "#4B168A"),
            (0.16, "#2166AC"),
            (0.32, "#7FB3D5"),
            (0.45, "#D9EAF4"),
            (0.495, "#F8F8F8"),
            (0.505, "#F8F8F8"),
            (0.56, "#FFF1A8"),
            (0.70, "#F4A259"),
            (0.86, "#D95F02"),
            (1.00, "#B2182B"),
        ],
    )
    cmap.set_bad("#FFFFFF")
    norm = SignedPowerNorm(vmin=-3, vmax=3, gamma=0.55)

    nrows = display_matrix.shape[0]
    fig_h = max(11.2, 0.24 * nrows + 2.35)
    fig, ax = plt.subplots(figsize=(16.8, fig_h))
    y_edges = np.arange(nrows + 1)
    mesh = ax.pcolormesh(time_edges(times), y_edges, logv, cmap=cmap, norm=norm, shading="flat")
    ax.set_ylim(nrows, 0)
    ax.set_xlim(0, 6)
    ax.set_xticks(np.arange(0, 6.1, 1))
    ax.set_xlabel("时间 (h)", fontsize=11.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(display_ylabels, fontsize=8.0)
    ax.tick_params(axis="y", length=0, pad=28)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(
        "连续 PURE 体系相邻过最优投料点的变量响应",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )

    for axis in AXIS_ORDER:
        row_start, row_end, gap_start, gap_end = spans[axis]
        center = (row_start + row_end) / 2
        label_y = (gap_start + gap_end) / 2
        ax.axhline(row_start, color="#D6DEE8", lw=1.6)
        ax.axhline(row_end, color="#D6DEE8", lw=1.2)
        ax.axhline(gap_end, color="#334155", lw=1.0)
        ax.add_patch(
            plt.Rectangle(
                (-0.19, row_start),
                0.16,
                row_end - row_start,
                color=GROUP_COLOR[axis],
                clip_on=False,
                lw=0,
            )
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
        for start, end, label in STAGES[axis]:
            ax.axvline(start, ymin=1 - row_end / nrows, ymax=1 - row_start / nrows, color="#94A3B8", lw=0.7, ls="--")
            ax.text(
                (start + end) / 2,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color="#64748B",
            )
        ax.axvline(6, ymin=1 - row_end / nrows, ymax=1 - row_start / nrows, color="#94A3B8", lw=0.7, ls="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.035, pad=0.075, aspect=42)
    cbar.set_label("log2(相邻过最优点 / 最优点)", fontsize=10.0)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.tick_params(labelsize=8.5)

    fig.text(
        0.49,
        0.018,
        "注：每个扰动轴使用技术报告记录的相邻过最优/最优投料点；总氨酰化模块指 total aaRS+aa 氨酰化供给模块；MetRS 扰动已去除。",
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
    ratios = pd.read_csv(RATIOS_CSV)
    matrix, ylabels, axis_per_row, times = build_rows(ratios, records)
    fig = plot_heatmap(matrix, ylabels, axis_per_row, times)
    out_stem = FIG_DIR / "figure_331_axis_specific_overoptimum_heatmap_20260813"
    for suffix in ("png", "svg", "pdf"):
        out = out_stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=280 if suffix == "png" else None, bbox_inches="tight")
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
