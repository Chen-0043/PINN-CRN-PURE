#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATP-GTP minimal effective synergy window figure.

Data convention:
- MetRS, total aaRS, CTP and UTP are fixed at baseline.
- Points are true simulated combinations collected in
  atp_gtp_minimal_synergy_window_20260720/all_atp_gtp_minimal_window_points.csv.
- The chosen engineering optimum is ATP 2× / GTP 5×: not the mathematical
  asymptotic maximum under unlimited ATP, but the smallest effective point on
  the yield plateau with much lower residual ATP/PPi burden.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from perturbation_curve_style_20260728 import BAR_LEGEND_Y, COLORS, LABELS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "atp_gtp_minimal_synergy_window_20260720" / "all_atp_gtp_minimal_window_points.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "atp_gtp_minimal_synergy_window_curve_reorganized_20260728.png"
OUT_SVG = FIG_DIR / "atp_gtp_minimal_synergy_window_curve_reorganized_20260728.svg"
OUT_PDF = FIG_DIR / "atp_gtp_minimal_synergy_window_curve_reorganized_20260728.pdf"


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
    ax.grid(True, axis="y", alpha=0.22)
    ax.tick_params(labelsize=8.8)


def row(df, atp, gtp):
    return df[(np.isclose(df["scale_ATP"], atp)) & (np.isclose(df["scale_GTP"], gtp))].iloc[0]


def fmt(v, nd=2):
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def fmt_fold(v):
    """Format a relative value shown inside the log2-relative heatmap."""
    if abs(v) < 0.001:
        return f"{v:.1e}×"
    if abs(v) < 0.01:
        return f"{v:.4f}×"
    if abs(v) < 0.1:
        return f"{v:.3f}×"
    if abs(v) < 10:
        return f"{v:.2f}×"
    return f"{v:.1f}×"


def main():
    setup_font()
    df = pd.read_csv(DATA).copy()
    df = df[(df["scale_CTP"].eq(1.0)) & (df["scale_UTP"].eq(1.0))
            & (df["scale_MetRS"].eq(1.0)) & (df["scale_bulk_aars_pool"].eq(1.0))]
    df = df.sort_values(["scale_ATP", "scale_GTP"]).reset_index(drop=True)

    opt = row(df, 2.0, 5.0)
    atp1 = df[np.isclose(df["scale_ATP"], 1.0)].sort_values("scale_GTP")
    atp2 = df[np.isclose(df["scale_ATP"], 2.0)].sort_values("scale_GTP")
    gtp5 = df[np.isclose(df["scale_GTP"], 5.0)].sort_values("scale_ATP")

    blue = COLORS["deGFP"]
    orange = "#E68632"
    green = COLORS["mRNA_t"]
    red = COLORS["tl_gtp_progression"]
    purple = COLORS["Ribosome_mRNA"]
    teal = COLORS["total_GTP_pool"]
    grey = COLORS["neutral"]
    yellow = COLORS["PPi"]

    fig, axs = plt.subplots(1, 2, figsize=(14.2, 6.4))
    fig.patch.set_facecolor("white")

    # A. Measured two-dimensional window; no interpolation.
    ax = axs[0]
    window_points = df[
        (np.isclose(df["scale_ATP"], 1.0))
        | (np.isclose(df["scale_ATP"], 2.0))
        | (np.isclose(df["scale_GTP"], 5.0))
    ].copy()
    norm = Normalize(vmin=window_points["deGFP_end_uM"].min(), vmax=window_points["deGFP_end_uM"].max())
    sc = ax.scatter(window_points["scale_GTP"], window_points["scale_ATP"],
                    c=window_points["deGFP_end_uM"], s=225,
                    cmap="YlOrRd", norm=norm, edgecolor="#333333", linewidth=0.8, alpha=0.97)
    ax.plot(atp1["scale_GTP"], atp1["scale_ATP"], color="#6B7280", lw=2.0, ls="--", alpha=0.88,
            label="ATP固定1×：单GTP轴")
    ax.plot(atp2["scale_GTP"], atp2["scale_ATP"], color="#2563EB", lw=2.2, ls="--", alpha=0.95,
            label="ATP固定2×：扫描GTP")
    ax.plot(gtp5["scale_GTP"], gtp5["scale_ATP"], color="#059669", lw=2.2, ls="--", alpha=0.95,
            label="GTP固定5×：扫描ATP")
    ax.scatter([5], [2], s=310, facecolor="none", edgecolor=red, linewidth=2.4, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(0.05, 6.35)
    ax.set_ylim(0.8, 190)
    ax.set_xticks([0.25, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["0.25×", "1×", "2×", "3×", "4×", "5×", "6×"])
    ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 40, 80, 160])
    ax.set_yticklabels(["1×", "2×", "3×", "4×", "5×", "10×", "20×", "40×", "80×", "160×"])
    ax.set_xlabel("GTP 初始投料倍率")
    ax.set_ylabel("ATP 初始投料倍率")
    ax.set_title("A  真实扫描点：高产窗口沿 GTP≈5×展开，ATP 2×已进入平台", loc="left", fontsize=12, fontweight="bold")
    ax.legend(
        frameon=False, fontsize=8.5, loc="upper center", ncol=1,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("6 h deGFP (μM)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    style(ax)

    # D. Boundary states around the selected point.
    ax = axs[1]
    states = [
        ("ATP低 1×/5×", row(df, 1, 5)),
        ("GTP少 2×/4×", row(df, 2, 4)),
        ("协同点 2×/5×", opt),
        ("ATP略高 3×/5×", row(df, 3, 5)),
        ("GTP多 2×/6×", row(df, 2, 6)),
    ]
    metrics = [
        ("deGFP_end_uM", "deGFP", "uM"),
        ("mRNA_t_end_uM", "mRNA_t", "uM"),
        ("Ribosome_mRNA_end_uM", LABELS["Ribosome_mRNA"], "uM"),
        ("ATP_end_uM", "ATP", "uM"),
        ("GTP_end_uM", "GTP", "uM"),
        ("PPi_end_uM", "PPi", "uM"),
    ]
    mat = []
    labels_rel = []
    for _, r in states:
        vals = []
        labs = []
        for col, _, _ in metrics:
            rel = float(r[col]) / float(opt[col])
            vals.append(rel)
            labs.append(fmt_fold(rel))
        mat.append(vals)
        labels_rel.append(labs)
    mat = np.array(mat)
    log_mat = np.log2(np.clip(mat, 1e-4, None))
    im = ax.imshow(np.clip(log_mat, -3, 3), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels([label for label, _ in states])
    for i in range(len(states)):
        for j in range(len(metrics)):
            color = "white" if abs(log_mat[i, j]) > 1.8 else "#222222"
            ax.text(j, i, labels_rel[i][j], ha="center", va="center", fontsize=7.4, color=color)
    ax.add_patch(plt.Rectangle((-0.5, 1.5), len(metrics), 1, fill=False, edgecolor=red, lw=2.2))
    ax.set_title("D  最小协同点两侧：GTP少主驱动不足，GTP多/ATP多转为残余和副产物", loc="left", fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("log2(relative to ATP2×/GTP5×)", fontsize=8.2)
    cb.ax.tick_params(labelsize=7.8)

    fig.suptitle("ATP–GTP 最佳协同窗口：在基线酶用量下，ATP 2× / GTP 5× 是最小有效高产组合",
                 fontsize=15.2, fontweight="bold", y=0.992)
    fig.tight_layout(rect=[0.02, 0.04, 0.995, 0.96])
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_SVG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
