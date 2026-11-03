#!/usr/bin/env python3
"""Create presentation-ready batch and continuous deGFP result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main() -> None:
    batch = pd.read_csv(OUT / "batch/batch_timeseries.csv")
    cont = pd.read_csv(OUT / "continuous/gamma_0p5_continuous_72h_timeseries.csv")

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16, "legend.fontsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].plot(batch["time"] / 3600.0, batch["[deGFP_m]"] * 1e6, color="#176B87", linewidth=2.5)
    axes[0].axhline(5.3, color="#D95F59", linestyle="--", linewidth=1.5, label="Literature target: 5.3 uM")
    axes[0].set(title="Batch PURE deGFP", xlabel="Time (h)", ylabel="deGFP concentration (uM)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(cont["time"] / 3600.0, cont["[deGFP_m]"] * 1e6, color="#C26D2D", linewidth=2.5)
    axes[1].set(title="Continuous PURE deGFP, gamma = 0.5 h^-1", xlabel="Time (h)", ylabel="Reactor deGFP concentration (uM)")
    axes[1].grid(alpha=0.2)

    fig.savefig(OUT / "batch_and_continuous_degfp.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
