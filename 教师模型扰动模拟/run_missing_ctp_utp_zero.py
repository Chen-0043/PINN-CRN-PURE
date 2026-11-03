#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run only the missing CTP 0x and UTP 0x perturbation snapshots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAPSHOT_TIME_POINTS_H = (
    "0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1,1.25,"
    "1.5,1.75,2,2.25,2.5,2.75,3,4,5,6"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run missing CTP/UTP 0x snapshots only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "rerun_synced_ntp_aars_perturbations_20260711.py"),
        "--components",
        "CTP,UTP",
        "--scales",
        "0",
        "--snapshot-time-points-h",
        SNAPSHOT_TIME_POINTS_H,
        "--skip-baseline",
        "--outdir",
        str(ROOT / "run_outputs" / "missing_ctp_utp_zero_unified_time"),
    ]
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    (ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
