#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the latest synchronized perturbation simulations used for Fig. 3.3.1.

This entry point fixes the final unified snapshot grid:
0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5,
1.75, 2, 2.25, 2.5, 2.75, 3, 4, 5, 6 h.

It does not generate figures; ``run_all.py`` is the figure-regeneration entry.
"""

from __future__ import annotations

import os
import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
SNAPSHOT_TIME_POINTS_H = (
    "0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1,1.25,"
    "1.5,1.75,2,2.25,2.5,2.75,3,4,5,6"
)


def run_step(name: str, args: list[str], *, dry_run: bool) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    (ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
    print(f"[run] {name}", flush=True)
    print(" ".join([sys.executable, *args]), flush=True)
    if dry_run:
        return
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the latest unified-time perturbation simulations.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running simulations.")
    args = parser.parse_args()

    run_step(
        "NTP perturbations: ATP, CTP, GTP, UTP",
        [
            str(SCRIPTS / "rerun_synced_ntp_aars_perturbations_20260711.py"),
            "--components",
            "ATP,CTP,GTP,UTP",
            "--scales",
            "0,0.01,0.03,0.25,0.5,1,2,3,4",
            "--snapshot-time-points-h",
            SNAPSHOT_TIME_POINTS_H,
            "--outdir",
            str(ROOT / "run_outputs" / "latest_ntp_unified_time"),
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "total aaRS module perturbation",
        [
            str(SCRIPTS / "run_final_total_aars_pool.py"),
            "--scales",
            "0,0.01,0.03,0.18,0.20,0.22,0.25,1,4",
            "--snapshot-time-points-h",
            SNAPSHOT_TIME_POINTS_H,
            "--outdir",
            str(ROOT / "run_outputs" / "latest_total_aars_unified_time"),
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "MetRS enzyme-only perturbation",
        [
            str(SCRIPTS / "rerun_metrs_enzyme_only_fullsnap_20260713.py"),
            "--components",
            "MetRS",
            "--scales",
            "0,0.01,0.03,0.06,0.125,0.25,1,4",
            "--snapshot-time-points-h",
            SNAPSHOT_TIME_POINTS_H,
            "--outdir",
            str(ROOT / "run_outputs" / "latest_metrs_enzyme_only_unified_time"),
        ],
        dry_run=args.dry_run,
    )
    print("[done] latest perturbation simulations completed" if not args.dry_run else "[done] dry run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
