#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the manuscript Section 3.3 perturbation mechanism figures.

The manuscript Figure 3 is assembled from two axis-specific heatmaps:

* Fig. 3a: zero-feed / optimum-feed responses.
* Fig. 3b: adjacent over-optimum / optimum-feed responses.

Both panels intentionally exclude the historical MetRS-only perturbation axis.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "technical_report_pure_20260713"
FIGURE_DIR = REPORT_DIR / "figures" / "article_331_20260813"

SCRIPTS = [
    REPORT_DIR / "plot_axis_specific_zero_vs_optimum_heatmap_20260813.py",
    REPORT_DIR / "plot_axis_specific_overoptimum_heatmap_20260813.py",
]


def run_script(script: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPORT_DIR / ".matplotlib-cache"))
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, check=True)


def main() -> None:
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(script)
        run_script(script)

    expected = [
        "figure_331_axis_specific_zero_vs_optimum_heatmap_20260813",
        "figure_331_axis_specific_overoptimum_heatmap_20260813",
    ]
    print("\nArticle Section 3.3 figure outputs:")
    for stem in expected:
        for suffix in ("png", "svg", "pdf"):
            path = FIGURE_DIR / f"{stem}.{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            print(path)


if __name__ == "__main__":
    main()
