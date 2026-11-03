#!/usr/bin/env python3
"""Recalculate and verify the perturbation conclusions from bundled outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproduced_results"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def verify_optima() -> dict[str, object]:
    path = (
        ROOT
        / "run_outputs/article_missing_scale_degfp_endpoints_20260902"
        / "article_5axis_scale_degfp_6h_complete.csv"
    )
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows(path):
        if row.get("status", "") not in {"", "ok"}:
            continue
        grouped[row["component"]].append(
            (float(row["scale"]), float(row["deGFP_6h_uM"]))
        )
    expected = {
        "ATP": 0.5,
        "GTP": 3.0,
        "CTP": 0.01,
        "UTP": 0.01,
        "total_aars_pool": 0.22,
    }
    actual = {}
    for component, scale_expected in expected.items():
        if component not in grouped:
            raise AssertionError(f"missing perturbation axis: {component}")
        scale, concentration = max(grouped[component], key=lambda item: item[1])
        if abs(scale - scale_expected) > 1e-12:
            raise AssertionError(
                f"{component}: expected optimum {scale_expected}, got {scale}"
            )
        actual[component] = {
            "optimal_scale": scale,
            "deGFP_6h_uM": concentration,
            "evaluated_scales": len(grouped[component]),
        }
    return {"source": str(path.relative_to(ROOT)), "axes": actual}


def verify_numerical_convergence() -> dict[str, object]:
    folder = ROOT / "run_outputs/five_axis_uncertainty_20261027"
    summary = rows(folder / "convergence_summary.csv")
    gains = rows(folder / "convergence_relative_gains.csv")
    max_relative_difference = max(
        float(row["rel_diff_production_vs_very_tight"]) for row in summary
    )
    by_axis: dict[str, list[float]] = defaultdict(list)
    for row in gains:
        by_axis[row["condition"]].append(float(row["gain_percent_vs_1x"]))
    gain_spreads = {
        axis: max(values) - min(values) for axis, values in by_axis.items()
    }
    min_spread = min(gain_spreads.values())
    max_spread = max(gain_spreads.values())
    if max_relative_difference >= 1e-5:
        raise AssertionError(
            f"solver-tolerance sensitivity too large: {max_relative_difference}"
        )
    if not (1.3e-5 <= min_spread <= 1.5e-5):
        raise AssertionError(f"unexpected minimum gain spread: {min_spread}")
    if not (1.7e-4 <= max_spread <= 1.8e-4):
        raise AssertionError(f"unexpected maximum gain spread: {max_spread}")
    return {
        "source": str(folder.relative_to(ROOT)),
        "max_relative_concentration_difference": max_relative_difference,
        "gain_spread_percentage_points": gain_spreads,
        "reported_range_percentage_points": [min_spread, max_spread],
    }


def verify_local_optima() -> dict[str, object]:
    path = (
        ROOT
        / "run_outputs/adjacent_local_optima_20261027"
        / "adjacent_local_optimum_summary.csv"
    )
    table = rows(path)
    expected_axes = {"ATP", "GTP", "CTP", "UTP", "total_aars_pool"}
    if {row["axis"] for row in table} != expected_axes:
        raise AssertionError("the five perturbation axes are incomplete")
    audit = {}
    for row in table:
        if int(row["n_complete"]) != 8:
            raise AssertionError(f"{row['axis']}: incomplete parameter combinations")
        for column in [
            "all_center_is_local_max",
            "all_center_beats_baseline",
            "all_local_best_beats_baseline",
        ]:
            if not as_bool(row[column]):
                raise AssertionError(f"{row['axis']}: failed {column}")
        audit[row["axis"]] = {
            "parameter_combinations": int(row["n_complete"]),
            "optimal_scale_in_all_combinations": row["local_best_scales"],
            "center_minus_best_neighbor_uM": [
                float(row["min_center_minus_best_neighbor_uM"]),
                float(row["max_center_minus_best_neighbor_uM"]),
            ],
            "gain_vs_baseline_percent": [
                float(row["min_center_gain_vs_baseline_percent"]),
                float(row["max_center_gain_vs_baseline_percent"]),
            ],
        }
    return {"source": str(path.relative_to(ROOT)), "axes": audit}


def verify_topology_score() -> dict[str, object]:
    path = (
        ROOT
        / "topology_priority"
        / "teacher_topology_rank_real_perturbable_qssa_background_equal4_20260824.csv"
    )
    table = rows(path)
    score_column = "topology_priority_score_equal4_qssa_background"
    metric_columns = [
        "direct_reaction_count_norm",
        "one_hop_species_count_norm",
        "module_coverage_count_norm",
        "qssa_background_reaction_count_norm",
    ]
    scores = []
    max_error = 0.0
    raw_columns = [column.removesuffix("_norm") for column in metric_columns]
    bounds = {column: (min(float(row[column]) for row in table),
                       max(float(row[column]) for row in table)) for column in raw_columns}
    for row in table:
        normalized = []
        for raw, normalized_column in zip(raw_columns, metric_columns):
            lo, hi = bounds[raw]
            value = (float(row[raw])-lo)/(hi-lo) if hi>lo else 0.0
            if abs(value-float(row[normalized_column]))>1e-12:
                raise AssertionError(f"normalization mismatch: {row['object_id']} {raw}")
            normalized.append(value)
        recalculated = sum(normalized) / 4.0
        stored = float(row[score_column])
        max_error = max(max_error, abs(recalculated - stored))
        scores.append((row["object_id"], recalculated, int(row["rank_equal4_qssa_background"])))
    if max_error > 1e-12:
        raise AssertionError(f"topology score mismatch: {max_error}")
    # Equal scores retain the source table's stable order; verify that the
    # stored ranks are complete and that score never increases with rank.
    expected_order = sorted(scores, key=lambda item: item[2])
    if [item[2] for item in expected_order] != list(range(1, len(table) + 1)):
        raise AssertionError("stored topology ranks are not a complete sequence")
    if any(
        expected_order[index][1] < expected_order[index + 1][1] - 1e-15
        for index in range(len(expected_order) - 1)
    ):
        raise AssertionError(
            "stored topology rank is inconsistent with the four equal-weight scores"
        )
    return {
        "source": str(path.relative_to(ROOT)),
        "ranked_objects": len(table),
        "formula": "S=0.25*(N_rxn_norm+N_nbr_norm+N_mod_norm+N_qb_norm)",
        "max_score_recalculation_error": max_error,
        "top_12": [
            {"rank": rank, "object": name, "score": score}
            for name, score, rank in expected_order[:12]
        ],
    }


def main() -> None:
    audit = {
        "five_axis_optima": verify_optima(),
        "solver_tolerance_convergence": verify_numerical_convergence(),
        "parameter_uncertainty_local_optima": verify_local_optima(),
        "topology_priority_equal_weight_score": verify_topology_score(),
    }
    OUT.mkdir(exist_ok=True)
    destination = OUT / "perturbation_conclusion_audit.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OK: perturbation conclusions verified; audit written to {destination}")


if __name__ == "__main__":
    main()
