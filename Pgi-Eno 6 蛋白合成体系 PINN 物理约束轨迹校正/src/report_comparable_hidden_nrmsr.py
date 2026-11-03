#!/usr/bin/env python3
"""Create a fixed comparable/hidden NRMSR report for a six-protein candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RATE_THRESHOLDS = (0.004, 0.01, 0.1, 1.0)


def metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "trajectory_count": 0,
            "mean_per_trajectory_NRMSR_percent": None,
            "rms_across_trajectories_NRMSR_percent": None,
            "pooled_global_NRMSR_percent": None,
        }
    ratio = frame["NRMSR_percent"].to_numpy(float)
    residual = frame["residual_rms_uM"].to_numpy(float)
    rhs = frame["rhs_integral_rms_uM"].to_numpy(float)
    return {
        "trajectory_count": int(len(frame)),
        "mean_per_trajectory_NRMSR_percent": float(np.mean(ratio)),
        "rms_across_trajectories_NRMSR_percent": float(np.sqrt(np.mean(ratio**2))),
        "pooled_global_NRMSR_percent": float(
            np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(rhs**2)), 1e-30) * 100.0
        ),
    }


def classify_hidden_modules(species: pd.Series) -> pd.Series:
    """Assign every hidden state once using explicit network-name rules."""

    modules = pd.Series("other", index=species.index, dtype=object)
    assigned = pd.Series(False, index=species.index)
    rules = [
        ("transcription chain", r"T7RNAP|mRNA"),
        ("translation elongation chain", r"elRS70S"),
        ("termination/recycling", r"termRS70S|RF1|RF2|RF3|RRF"),
        ("translation initiation assembly", r"RS30S|IF1|IF2|IF3|fMet"),
        ("EF-Tu/EF-G factor cycle", r"EFTu|EFTs|EFG"),
        (
            "aminoacylation/tRNA supply",
            r"tRNA|(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)RS",
        ),
        ("energy regeneration", r"(?:^|_)(?:CK|MK|NDK|PPiase)(?:_|$)"),
        ("peptide release/protein maturation", r"Pept|MGapt"),
    ]
    for module, pattern in rules:
        selected = species.str.contains(pattern, regex=True, na=False) & ~assigned
        modules.loc[selected] = module
        assigned.loc[selected] = True
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args()

    comparable_path = args.candidate_dir / "corrected_nrmsr_by_constraint.csv"
    hidden_path = args.candidate_dir / "fullstate_and_hidden_nrmsr.csv"
    comparable = pd.read_csv(comparable_path)
    hidden = pd.read_csv(hidden_path)
    hidden = hidden.loc[hidden["hidden_full_elementary"].astype(bool)].copy()
    hidden["rhs_rate_rms_uM_per_h"] = hidden["rhs_integral_rms_uM"] / 0.25
    hidden["module"] = classify_hidden_modules(hidden["species_id"].astype(str))

    comparable_thresholds = {}
    hidden_thresholds = {}
    for threshold in RATE_THRESHOLDS:
        key = f"rhs_rate_gt_{threshold:g}_uM_per_h"
        comparable_thresholds[key] = metrics(
            comparable.loc[comparable["rhs_rate_rms_uM_per_h"] > threshold]
        )
        hidden_thresholds[key] = metrics(
            hidden.loc[hidden["rhs_rate_rms_uM_per_h"] > threshold]
        )

    module_rows = []
    active_hidden = hidden.loc[hidden["rhs_rate_rms_uM_per_h"] > 0.004]
    for module, frame in active_hidden.groupby("module", sort=False):
        module_rows.append({"module": module, **metrics(frame)})
    module_table = pd.DataFrame(module_rows).sort_values(
        "rms_across_trajectories_NRMSR_percent"
    )

    report = {
        "candidate": str(args.candidate_dir),
        "metric_definitions": {
            "mean_per_trajectory": "mean_i[NRMSR_i]",
            "rms_across_trajectories": "sqrt(mean_i[NRMSR_i^2])",
            "pooled_global": "RMS_{i,t}(residual)/RMS_{i,t}(integral RHS)",
            "module_scan": (
                "Post hoc exhaustive assignment of every hidden state to one explicit "
                "functional family; no module-specific training loss is used."
            ),
        },
        "all_comparable_trajectories": metrics(comparable),
        "comparable_trajectory_thresholds": comparable_thresholds,
        "all_hidden_states": metrics(hidden),
        "hidden_state_thresholds": hidden_thresholds,
        "hidden_module_scan_threshold_uM_per_h": 0.004,
        "hidden_module_count": int(len(module_table)),
        "hidden_module_ranking": module_table.to_dict(orient="records"),
    }

    summary_path = args.candidate_dir / "fixed_comparable_hidden_nrmsr_report.json"
    module_path = args.candidate_dir / "fixed_hidden_module_nrmsr_ranking.csv"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    module_table.to_csv(module_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
