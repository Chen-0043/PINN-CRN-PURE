#!/usr/bin/env python3
"""Run the same-condition full-network BDF or Radau comparison with a wall timeout.

The underlying solver is the archived full mass-action operator used for the
manuscript comparison.  A timeout is an observed result: no partial trajectory
is reported as a successful 0--6 h solution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
MODEL = PACKAGE / "model_evaluation_20260730"
ASSETS = MODEL / "article_gapfill_20260730" / "full_elementary_pinn"
SOLVER = MODEL / "direct_full_elementary_mass_action_solve.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["BDF", "Radau"], required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "reproduced_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"samecondition_{args.method.lower()}"
    command = [
        sys.executable,
        str(SOLVER),
        "--assets", str(ASSETS),
        "--output-prefix", str(prefix),
        "--method", args.method,
        "--rtol", "1e-6",
        "--atol-uM", "1e-6",
        "--max-time-s", "21600",
        "--n-eval", "25",
        "--exchange-mode", "continuous",
        "--use-analytic-jac",
        "--rate-unit", "uM",
        "--state-unit", "M",
    ]
    started = time.perf_counter()
    status = "completed"
    returncode = None
    try:
        completed = subprocess.run(command, check=False, timeout=args.timeout_s)
        returncode = completed.returncode
        if returncode != 0:
            status = "solver_error"
    except subprocess.TimeoutExpired:
        status = "timeout_without_complete_trajectory"
    elapsed = time.perf_counter() - started
    report = {
        "method": args.method,
        "command": command,
        "network_species": 5772,
        "network_reactions": 7061,
        "time_span_s": [0.0, 21600.0],
        "output_time_points": 25,
        "rtol": 1e-6,
        "atol_uM": 1e-6,
        "analytic_sparse_jacobian": True,
        "wall_timeout_s": args.timeout_s,
        "wall_time_s": elapsed,
        "status": status,
        "returncode": returncode,
        "trajectory_available": (prefix.with_name(prefix.name + "_direct_full_elementary_solution.npz")).exists(),
    }
    output = args.output_dir / f"samecondition_{args.method.lower()}_run.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status in {"completed", "timeout_without_complete_trajectory"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
