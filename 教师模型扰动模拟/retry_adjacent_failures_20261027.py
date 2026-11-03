#!/usr/bin/env python3
"""Retry the two incomplete local-optimum comparisons with tighter tolerances."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_5axis_uncertainty_20261027 as base  # noqa: E402


SOURCE = ROOT / "run_outputs" / "five_axis_uncertainty_20261027"
OUT = ROOT / "run_outputs" / "adjacent_local_optima_20261027" / "tight_retry_results.csv"


def main() -> None:
    samples = pd.read_csv(SOURCE / "parameter_multiplier_samples.csv").set_index("sample_id")
    jobs = [
        {"sample_id": 6, "condition": "ATP", "component": "ATP", "scale": 0.5},
        {"sample_id": 7, "condition": "GTP_left", "component": "GTP", "scale": 2.0},
    ]
    rows = []
    for job in jobs:
        multipliers = {k: float(v) for k, v in samples.loc[job["sample_id"]].items()}
        solved = False
        for solver_label, rtol, atol in [
            ("tight", 1e-7, 1e-13),
            ("very_tight", 1e-8, 1e-14),
        ]:
            task = {
                "analysis": "adjacent_retry",
                "solver_label": solver_label,
                "sample_id": job["sample_id"],
                "condition": job["condition"],
                "component": job["component"],
                "scale": job["scale"],
                "rtol": rtol,
                "atol": atol,
                "multipliers": multipliers,
            }
            started = time.time()
            try:
                row = base._simulate(task)
                row.update({"status": "ok", "error": ""})
                solved = True
            except Exception as exc:
                row = {
                    **task,
                    "multipliers": json.dumps(multipliers, sort_keys=True),
                    "deGFP_6h_uM": float("nan"),
                    "runtime_s": time.time() - started,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(
                f"{job['condition']} sample={job['sample_id']} {solver_label} "
                f"{row['status']} y6={row.get('deGFP_6h_uM')}",
                flush=True,
            )
            if solved:
                break
    return None


if __name__ == "__main__":
    main()
