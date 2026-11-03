#!/usr/bin/env python3
"""Build an auditable 3 h/6 h validation report for the six-protein model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GENES = ["Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ranked-csv", type=Path, required=True)
    p.add_argument(
        "--targets-csv",
        type=Path,
        default=Path("inputs/fusion_batch_validation/experimental_targets_with_provenance.csv"),
    )
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--comparison-ranked-csv", type=Path)
    p.add_argument("--tx-gene-multipliers-json", type=Path)
    p.add_argument("--activity-decay-gene-json", type=Path)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(args.ranked_csv).iloc[0]
    targets = pd.read_csv(args.targets_csv).set_index("gene")

    rows: list[dict[str, object]] = []
    for gene in GENES:
        for hour in (3, 6):
            observed = float(targets.loc[gene, f"lit_{hour}h_uM_single_proxy"])
            predicted = float(best[f"{gene}_{hour}h"])
            rows.append(
                {
                    "condition": gene,
                    "condition_type": "single-gene",
                    "time_h": hour,
                    "observed_uM": observed,
                    "predicted_uM": predicted,
                    "signed_error_uM": predicted - observed,
                    "absolute_error_uM": abs(predicted - observed),
                    "absolute_relative_error_percent": 100.0 * abs(predicted - observed) / observed,
                }
            )
    for hour in (3, 6):
        observed = float(targets[f"all_GE_total_{hour}h_uM_proxy"].iloc[0])
        predicted = float(best[f"all{hour}_uM"])
        rows.append(
            {
                "condition": "All G.E.",
                "condition_type": "six-template aggregate",
                "time_h": hour,
                "observed_uM": observed,
                "predicted_uM": predicted,
                "signed_error_uM": predicted - observed,
                "absolute_error_uM": abs(predicted - observed),
                "absolute_relative_error_percent": 100.0 * abs(predicted - observed) / observed,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.outdir / "batch_3h_6h_observed_vs_predicted.csv", index=False)

    single = comparison[comparison["condition_type"] == "single-gene"]
    aggregate = comparison[comparison["condition_type"] == "six-template aggregate"]
    tx_gene_multipliers = json.loads(str(best["tx_gene_multipliers_json"]))
    activity_decay_by_gene = json.loads(str(best["activity_decay_by_gene_json"]))
    if args.tx_gene_multipliers_json:
        tx_gene_multipliers = json.loads(args.tx_gene_multipliers_json.read_text(encoding="utf-8"))
    if args.activity_decay_gene_json:
        activity_decay_by_gene = json.loads(args.activity_decay_gene_json.read_text(encoding="utf-8"))

    summary: dict[str, object] = {
        "selected_case": str(best["case"]),
        "integration_step_s": float(best.get("dt_s", np.nan)) if "dt_s" in best else None,
        "shared_parameters": {
            "tx_uniform": float(best["tx_uniform"]),
            "tl_uniform": float(best["tl_uniform"]),
            "energy_scale": float(best["energy_scale"]),
        },
        "calibration_design": {
            "calibration_points": "six separate-expression systems at 3 h and 6 h (12 points)",
            "external_check_points": "six-template coexpression aggregate at 3 h and 6 h (2 points)",
            "protein_specific_fit_parameters": True,
            "gene_specific_tx_multipliers": tx_gene_multipliers,
            "gene_specific_activity_decay_h-1": activity_decay_by_gene,
            "coexpression_used_for_gene_parameter_fitting": False,
        },
        "mux_dna_interpretation": str(best["mux_dna_mode"]),
        "mux_total_dna_nM": float(best["mux_dna_total_M"]) * 1e9,
        "metrics_percent": {
            "single_gene_3h_MARE": float(single.loc[single.time_h == 3, "absolute_relative_error_percent"].mean()),
            "single_gene_6h_MARE": float(single.loc[single.time_h == 6, "absolute_relative_error_percent"].mean()),
            "six_template_aggregate_3h_APE": float(aggregate.loc[aggregate.time_h == 3, "absolute_relative_error_percent"].iloc[0]),
            "six_template_aggregate_6h_APE": float(aggregate.loc[aggregate.time_h == 6, "absolute_relative_error_percent"].iloc[0]),
            "all_14_points_MARE": float(comparison["absolute_relative_error_percent"].mean()),
            "seven_6h_over_3h_ratios_MARE": 100.0 * float(best["ratio_mare7"]),
        },
        "evidence_scope": {
            "single_gene": "six separately expressed msfGFP-enzyme fusions",
            "coexpression": "aggregate GFP fluorescence only; no protein-resolved six-gene measurements",
            "absolute_scale": "paper-anchored conversion from digitized normalized fluorescence, not a raw molar table",
            "construct_sequence": "native enzyme CDS plus reconstructed msfGFP tag DNA; exact in-house tag DNA is unavailable",
        },
    }
    if args.comparison_ranked_csv:
        other = pd.read_csv(args.comparison_ranked_csv).iloc[0]
        summary["sensitivity_comparison"] = {
            "comparison_file": str(args.comparison_ranked_csv),
            "comparison_mux_dna_interpretation": str(other["mux_dna_mode"]),
            "comparison_mare14_percent": 100.0 * float(other["mare14"]),
            "selected_mare14_percent": 100.0 * float(best["mare14"]),
            "absolute_difference_percentage_points": 100.0 * abs(float(other["mare14"]) - float(best["mare14"])),
        }
    (args.outdir / "batch_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=220)
    x = np.arange(len(GENES))
    width = 0.19
    colors = {"obs3": "#2D5F73", "pred3": "#80B7C5", "obs6": "#B4492D", "pred6": "#E7A370"}
    for offset, hour, kind, label in [
        (-1.5, 3, "observed", "Observed 3 h"),
        (-0.5, 3, "predicted", "Predicted 3 h"),
        (0.5, 6, "observed", "Observed 6 h"),
        (1.5, 6, "predicted", "Predicted 6 h"),
    ]:
        values = [
            float(comparison.loc[(comparison.condition == g) & (comparison.time_h == hour), f"{kind}_uM"].iloc[0])
            for g in GENES
        ]
        color_key = ("obs" if kind == "observed" else "pred") + str(hour)
        axes[0].bar(x + offset * width, values, width=width, color=colors[color_key], label=label)
    axes[0].set_xticks(x, GENES)
    axes[0].set_ylabel("Fluorescent fusion product (uM)")
    axes[0].set_title("Single-gene batch validation")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    obs = comparison["observed_uM"].to_numpy(float)
    pred = comparison["predicted_uM"].to_numpy(float)
    point_colors = np.where(comparison["condition_type"].eq("single-gene"), "#2D7F5E", "#B4492D")
    axes[1].scatter(obs, pred, c=point_colors, s=54, edgecolor="white", linewidth=0.7)
    low = min(obs.min(), pred.min()) * 0.92
    high = max(obs.max(), pred.max()) * 1.08
    axes[1].plot([low, high], [low, high], color="#4D5560", lw=1.2, ls="--")
    for _, row in comparison.iterrows():
        axes[1].annotate(
            f"{row['condition']} {int(row['time_h'])}h",
            (row["observed_uM"], row["predicted_uM"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )
    axes[1].set(xlabel="Observed (uM)", ylabel="Predicted (uM)", title="All 14 validation points", xlim=(low, high), ylim=(low, high))
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.outdir / "batch_validation.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
