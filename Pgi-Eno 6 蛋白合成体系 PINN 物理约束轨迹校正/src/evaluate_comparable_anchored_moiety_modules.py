#!/usr/bin/env python3
"""Evaluate comparable-anchored biochemical-moiety modules.

Each module is defined before looking at its error: a directly comparable free
pool is combined with every elementary-CRN state carrying the same explicitly
named biochemical moiety. The module NRMSR is computed after applying that
fixed summation operator to both the state change and the integrated RHS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid_resolution_pinn import HybridResolutionOperator


DEFAULT_ASSETS = ROOT / "outputs" / "original_18p88_full_elementary"
DEFAULT_POOL = ROOT / "input_pool_override_sg_pure2_paper_mapped_20260424.json"
DEFAULT_SUPERVISION = (
    DEFAULT_ASSETS / "sixprotein_teacher_supervision_all_metabolites_dt0p125.npz"
)

# These tokens denote an explicit chemical moiety in the generated state names.
# Token boundaries prevent, for example, ATP from matching an unrelated substring.
MOIETY_TOKENS = (
    "ATP",
    "ADP",
    "GTP",
    "GDP",
    "CTP",
    "UTP",
    "PPi",
    "PO4",
    "NAD",
    "NADH",
    "TPP",
    "EFTu",
    "EFG",
    "IF2",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--supervision", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument("--energy-scale", type=float, default=0.24)
    parser.add_argument("--gamma-h", type=float, default=0.5)
    parser.add_argument("--activity-threshold-uM-per-h", type=float, default=0.004)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.prediction.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    operator = HybridResolutionOperator(
        args.assets,
        args.pool_json,
        energy_scale=args.energy_scale,
        gamma_h=args.gamma_h,
        include_all_teacher_metabolites=True,
        dtype=torch.float64,
    )
    prediction = np.load(args.prediction, allow_pickle=True)
    time_s = prediction["time_s"].astype(float)
    state = np.maximum(prediction["prediction_M"].astype(float), 0.0)
    if state.shape != (len(time_s), operator.n_species):
        raise ValueError("Prediction dimensions do not match the hybrid operator")

    supervision = np.load(args.supervision, allow_pickle=True)
    direct_names = supervision["direct_names"].astype(str).tolist()
    direct_indices = supervision["direct_indices"].astype(int).tolist()
    direct_name_to_index = dict(zip(direct_names, direct_indices, strict=True))
    direct_index_set = set(direct_indices)

    with torch.no_grad():
        rhs = operator(torch.tensor(state, dtype=torch.float64)).cpu().numpy()
    dt_s = np.diff(time_s)[:, None]
    rhs_integral = 0.5 * dt_s * (rhs[:-1] + rhs[1:])
    residual = np.diff(state, axis=0) - rhs_integral
    interval_h = np.diff(time_s) / 3600.0

    species_ids = np.asarray(operator.species_ids, dtype=str)
    rows: list[dict[str, object]] = []
    for token in MOIETY_TOKENS:
        anchor_name = f"pool::{token}"
        if anchor_name not in direct_name_to_index:
            continue
        pattern = re.compile(rf"(^|_){re.escape(token)}($|_)")
        member_indices = {
            index
            for index, species_id in enumerate(
                species_ids[: operator.base_species_count]
            )
            if pattern.search(species_id)
        }
        member_indices.add(direct_name_to_index[anchor_name])
        indices = np.asarray(sorted(member_indices), dtype=int)

        module_rhs = rhs_integral[:, indices].sum(axis=1)
        module_residual = residual[:, indices].sum(axis=1)
        rhs_rms_M = float(np.sqrt(np.mean(module_rhs**2)))
        residual_rms_M = float(np.sqrt(np.mean(module_residual**2)))
        rhs_rate_rms_uM_per_h = float(
            np.sqrt(np.mean((module_rhs * 1e6 / interval_h) ** 2))
        )
        rows.append(
            {
                "module": token,
                "comparable_anchor": anchor_name,
                "total_state_count": int(len(indices)),
                "direct_comparable_state_count": int(
                    sum(index in direct_index_set for index in indices)
                ),
                "hidden_elementary_state_count": int(
                    sum(index not in direct_index_set for index in indices)
                ),
                "rhs_rate_rms_uM_per_h": rhs_rate_rms_uM_per_h,
                "active_at_threshold": bool(
                    rhs_rate_rms_uM_per_h > args.activity_threshold_uM_per_h
                ),
                "NRMSR_percent": float(
                    100.0 * residual_rms_M / max(rhs_rms_M, 1e-30)
                ),
            }
        )

    # A predeclared combined transcription-substrate module. CTP- and UTP-bound
    # states are disjoint, so the same fixed summation rule applies unchanged.
    pyrimidine_tokens = ("CTP", "UTP")
    pyrimidine_members: set[int] = set()
    for token in pyrimidine_tokens:
        pattern = re.compile(rf"(^|_){re.escape(token)}($|_)")
        pyrimidine_members.update(
            index
            for index, species_id in enumerate(
                species_ids[: operator.base_species_count]
            )
            if pattern.search(species_id)
        )
        pyrimidine_members.add(direct_name_to_index[f"pool::{token}"])
    indices = np.asarray(sorted(pyrimidine_members), dtype=int)
    module_rhs = rhs_integral[:, indices].sum(axis=1)
    module_residual = residual[:, indices].sum(axis=1)
    rhs_rms_M = float(np.sqrt(np.mean(module_rhs**2)))
    residual_rms_M = float(np.sqrt(np.mean(module_residual**2)))
    rhs_rate_rms_uM_per_h = float(
        np.sqrt(np.mean((module_rhs * 1e6 / interval_h) ** 2))
    )
    rows.append(
        {
            "module": "CTP+UTP transcription substrate incorporation",
            "comparable_anchor": "pool::CTP + pool::UTP",
            "total_state_count": int(len(indices)),
            "direct_comparable_state_count": int(
                sum(index in direct_index_set for index in indices)
            ),
            "hidden_elementary_state_count": int(
                sum(index not in direct_index_set for index in indices)
            ),
            "rhs_rate_rms_uM_per_h": rhs_rate_rms_uM_per_h,
            "active_at_threshold": bool(
                rhs_rate_rms_uM_per_h > args.activity_threshold_uM_per_h
            ),
            "NRMSR_percent": float(
                100.0 * residual_rms_M / max(rhs_rms_M, 1e-30)
            ),
        }
    )

    table = pd.DataFrame(rows).sort_values(
        ["active_at_threshold", "NRMSR_percent"], ascending=[False, True]
    )
    csv_path = output_dir / "comparable_anchored_moiety_module_nrmsr.csv"
    json_path = output_dir / "comparable_anchored_moiety_module_nrmsr.json"
    table.to_csv(csv_path, index=False)
    report = {
        "prediction": str(args.prediction),
        "definition": (
            "A directly comparable free pool plus every full-elementary state "
            "whose generated identifier carries the same token-delimited "
            "biochemical moiety. The fixed summation operator is applied to both "
            "state changes and integrated RHS before NRMSR is calculated."
        ),
        "interpretation_limit": (
            "A low value establishes module-level net physical consistency; it "
            "does not establish low NRMSR for every constituent hidden state."
        ),
        "activity_threshold_uM_per_h": args.activity_threshold_uM_per_h,
        "model_inventory": {
            "states": int(operator.n_species),
            "mass_action_elementary_reactions": int(
                operator.n_elementary_reactions
            ),
        },
        "modules": table.to_dict(orient="records"),
        "output_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
