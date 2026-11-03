#!/usr/bin/env python3
"""Verify newly generated batch/continuous results against packaged references."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(label: str, actual: float, reference: float, abs_tol: float) -> dict:
    error = abs(float(actual) - float(reference))
    return {
        "metric": label,
        "actual": float(actual),
        "reference": float(reference),
        "absolute_error": error,
        "absolute_tolerance": abs_tol,
        "passed": error <= abs_tol,
    }


def main() -> None:
    batch = load(ROOT / "results/batch/batch_summary.json")
    batch_ref = load(ROOT / "reference_results/batch_8h/summary.json")
    cont = load(ROOT / "results/continuous/gamma_0p5_continuous_72h_summary.json")
    cont_ref = load(ROOT / "reference_results/continuous_72h/gamma_0p5_continuous_72h_summary.json")

    checks = [
        compare("batch_P2h_uM", batch["metrics"]["protein_2h_uM"], batch_ref["metrics"]["protein_2h_uM"], 0.02),
        compare("batch_P8h_uM", batch["metrics"]["protein_8h_uM"], batch_ref["metrics"]["protein_8h_uM"], 0.02),
        compare("batch_plateau_start_h", batch["metrics"]["plateau_start_h"], batch_ref["metrics"]["plateau_start_h"], 0.05),
        compare(
            "continuous_P24h_uM",
            cont["reactor_metrics_uM"]["protein_24h_uM"],
            cont_ref["reactor_metrics_uM"]["protein_24h_uM"],
            0.02,
        ),
        compare(
            "continuous_P72h_uM",
            cont["reactor_metrics_uM"]["protein_72h_uM"],
            cont_ref["reactor_metrics_uM"]["protein_72h_uM"],
            0.02,
        ),
    ]
    topology = {
        "batch_species": batch["counts"]["species_total"],
        "batch_reactions": batch["counts"]["reactions_total"],
        "continuous_species": cont["model_counts_continuous"]["species_total"],
        "continuous_reactions": cont["model_counts_continuous"]["reactions_total"],
        "passed": (
            batch["counts"]["species_total"] == 3399
            and batch["counts"]["reactions_total"] == 3696
            and cont["model_counts_continuous"]["species_total"] == 3399
            and cont["model_counts_continuous"]["reactions_total"] == 3762
        ),
    }
    report = {"metric_checks": checks, "topology_check": topology}
    report["passed"] = all(item["passed"] for item in checks) and topology["passed"]
    out = ROOT / "results/verification_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
