#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate all current perturbation figures in this clean package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "technical_report_pure_20260713"

SCRIPTS = [
    "plot_atp_full_stage_mechanism_20260723.py",
    "plot_gtp_perturbation_mechanism_20260723.py",
    "plot_ctp_utp_perturbation_mechanism_20260722.py",
    "plot_total_aars_perturbation_mechanism_20260722.py",
    "plot_metrs_perturbation_mechanism_20260727.py",
    "plot_atp_gtp_minimal_synergy_window_20260723.py",
]


def main() -> int:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    (ROOT / ".matplotlib-cache").mkdir(exist_ok=True)

    for script in SCRIPTS:
        path = REPORT / script
        if not path.exists():
            print(f"[missing] {path}", file=sys.stderr)
            return 2
        print(f"[run] {script}")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, check=True)
    print("[done] all current perturbation figures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
