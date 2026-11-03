#!/usr/bin/env python3
"""Plot six single-protein systems and the co-expression total in one panel."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SINGLE_ROOT = ROOT / "outputs" / "original_18p88_singleprotein_continuous_72h_dt0p5"
COEXPRESSION_CSV = (
    ROOT
    / "outputs"
    / "original_18p88_teacher_continuous_72h_dt0p5"
    / "sixprotein_continuous_timeseries.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "sixprotein_coexpression_reviewer_package"
OUTPUT_SVG = OUTPUT_DIR / "图1d_六个单蛋白与六蛋白共表达总蛋白_连续72h.svg"
OUTPUT_PNG = OUTPUT_DIR / "图1d_六个单蛋白与六蛋白共表达总蛋白_连续72h.png"

GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")
COLORS = {
    "Pgi": "#287D8E",
    "PfkA": "#D95F2D",
    "TpiA": "#4C6FAE",
    "GAPDH": "#8F5AA8",
    "Pgk": "#3B8B5D",
    "Eno": "#C48719",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "Hiragino Sans GB",
                "STHeiti",
                "DejaVu Sans",
            ],
            "font.size": 12,
            "axes.labelsize": 11.5,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.31)
    fig.text(0.015, 0.965, "(d)", ha="left", va="top", fontsize=16)

    for gene in GENES:
        path = SINGLE_ROOT / gene / "singleprotein_continuous_timeseries.csv"
        data = pd.read_csv(path, usecols=["time_h", f"{gene}_uM"])
        ax.plot(
            data["time_h"],
            data[f"{gene}_uM"],
            color=COLORS[gene],
            lw=2.0,
            label=f"{gene}（单蛋白）",
        )

    coexpression = pd.read_csv(
        COEXPRESSION_CSV,
        usecols=["time_h", "All_proteins_uM"],
    )
    ax.plot(
        coexpression["time_h"],
        coexpression["All_proteins_uM"],
        color="#20252B",
        lw=2.8,
        linestyle="--",
        label="总蛋白（六蛋白共表达）",
        zorder=5,
    )

    ax.set_title("六个单蛋白体系与六蛋白共表达体系的连续模拟", pad=8)
    ax.set_xlabel("时间 (h)")
    ax.set_ylabel("蛋白质浓度 (μM)")
    ax.set_xlim(0, 72.5)
    ax.set_ylim(0, 4.8)
    ax.tick_params(axis="both", which="major", labelsize=11.5)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.5,
        handletextpad=0.55,
        fontsize=10.5,
    )

    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_PNG, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(OUTPUT_SVG)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
