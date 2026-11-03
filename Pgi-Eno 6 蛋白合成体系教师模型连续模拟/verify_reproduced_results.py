#!/usr/bin/env python3
"""Compare freshly integrated 72 h trajectories with the archived paper results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REP = ROOT / "reproduced_results_verified"
REF = ROOT / "reference_results"
GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")


def compare_csv(actual: Path, expected: Path) -> dict[str, float | int]:
    a = pd.read_csv(actual)
    e = pd.read_csv(expected)
    if list(a.columns) != list(e.columns) or a.shape != e.shape:
        raise AssertionError(
            f"trajectory layout mismatch: {actual} {a.shape} vs {expected} {e.shape}"
        )
    numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    delta = np.abs(a[numeric].to_numpy(float) - e[numeric].to_numpy(float))
    return {
        "rows": int(a.shape[0]),
        "columns": int(a.shape[1]),
        "max_abs_numeric_difference": float(np.nanmax(delta)),
    }


def main() -> None:
    report: dict[str, object] = {"systems": {}}
    systems: dict[str, object] = report["systems"]  # type: ignore[assignment]
    for gene in GENES:
        actual_dir = REP / f"single_{gene}"
        expected_dir = REF / "six_single_protein_72h" / gene
        systems[f"single_{gene}"] = compare_csv(
            actual_dir / "timeseries.csv",
            expected_dir / "singleprotein_continuous_timeseries.csv",
        )
    systems["coexpression"] = compare_csv(
        REP / "coexpression" / "timeseries.csv",
        REF / "coexpression_72h" / "sixprotein_continuous_timeseries.csv",
    )
    maxima = [float(v["max_abs_numeric_difference"]) for v in systems.values()]  # type: ignore[index,union-attr]
    report["maximum_absolute_difference_across_seven_systems"] = max(maxima)
    report["status"] = "PASS" if max(maxima) <= 1e-12 else "FAIL"
    out = REP / "reproduction_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
