#!/usr/bin/env python3
"""Merge independently executed five-axis endpoint scans into one audit table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "run_outputs"
OUT = ROOT / "reproduced_results"
SCALES = [0.0, 0.01, 0.03, 0.06, 0.1, 0.22, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
COMPONENTS = ["ATP", "GTP", "CTP", "UTP", "total_aars_pool"]


def main() -> None:
    rows: list[pd.DataFrame] = []
    baseline_values: list[float] = []

    atp_path = RUNS / "article_missing_scale_degfp_endpoints_20260902" / "new_missing_degfp_endpoints_partial.csv"
    atp = pd.read_csv(atp_path)
    atp = atp[(atp.component == "ATP") & (atp.status == "ok")].copy()
    rows.append(atp[["component", "scale", "deGFP_6h_uM", "runtime_s", "status"]])

    for component in COMPONENTS[1:]:
        run_dir = RUNS / f"reproduced_parallel_{component}"
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        baseline_values.append(float(summary["baseline_1x_deGFP_6h_uM"]))
        table = pd.read_csv(run_dir / "new_missing_degfp_endpoints.csv")
        table = table[(table.component == component) & (table.status == "ok")].copy()
        rows.append(table[["component", "scale", "deGFP_6h_uM", "runtime_s", "status"]])

    if not baseline_values:
        raise RuntimeError("No completed parallel run supplied a baseline value")
    if max(baseline_values) - min(baseline_values) > 1e-12:
        raise AssertionError(f"baseline mismatch across runs: {baseline_values}")
    baseline = baseline_values[0]
    for component in COMPONENTS:
        rows.append(pd.DataFrame([{
            "component": component,
            "scale": 1.0,
            "deGFP_6h_uM": baseline,
            "runtime_s": 0.0,
            "status": "baseline_reused",
        }]))

    merged = pd.concat(rows, ignore_index=True)
    merged = merged.drop_duplicates(["component", "scale"], keep="last")
    merged = merged.sort_values(["component", "scale"], kind="stable").reset_index(drop=True)
    expected = {(c, s) for c in COMPONENTS for s in SCALES}
    observed = {(str(r.component), float(r.scale)) for r in merged.itertuples(index=False)}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise AssertionError(f"coverage mismatch; missing={missing}, extra={extra}")

    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "five_axis_6h_endpoints_fresh_run.csv", index=False)
    optima = (
        merged.sort_values(["component", "deGFP_6h_uM", "scale"], ascending=[True, False, True])
        .groupby("component", as_index=False)
        .first()[["component", "scale", "deGFP_6h_uM"]]
        .rename(columns={"scale": "optimal_scale", "deGFP_6h_uM": "optimal_deGFP_6h_uM"})
    )
    optima.to_csv(OUT / "five_axis_optima_fresh_run.csv", index=False)
    report = {
        "status": "PASS",
        "number_of_endpoint_rows": int(len(merged)),
        "number_of_freshly_integrated_nonbaseline_rows": int(len(merged) - len(COMPONENTS)),
        "shared_baseline_deGFP_6h_uM": baseline,
        "optima": optima.to_dict("records"),
    }
    (OUT / "five_axis_endpoint_reproduction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
