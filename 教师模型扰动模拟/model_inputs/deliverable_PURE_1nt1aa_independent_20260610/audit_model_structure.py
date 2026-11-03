#!/usr/bin/env python3
"""Build-only audit of model transformation stages and package-local inputs."""

from __future__ import annotations

import json
from pathlib import Path

import libsbml

import fit_batch_then_continuous_full_crn_background as core
from run_tx1nt_tlg1_micro3_longrun import load_sources, set_fine_node_env


ROOT = Path(__file__).resolve().parent


def counts(model) -> dict:
    return {"species": model.getNumSpecies(), "reactions": model.getNumReactions()}


def main() -> None:
    set_fine_node_env()
    params, hf_fit, keff_map = load_sources(
        ROOT / "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json",
        ROOT / "output_full_crn_background/batch_then_continuous_summary.json",
    )
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    doc = libsbml.readSBMLFromFile(str(core.BASE_SBML))
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Cannot read packaged base SBML")

    stages = [{"stage": "base_sbml", **counts(model)}]
    core.patch_initial_conditions(model, ic_uM)
    stages.append({"stage": "patch_initial_conditions", **counts(model)})
    core.swap_transcription_to_1nt_chain(model)
    stages.append({"stage": "swap_transcription_to_1nt_chain_micro3", **counts(model)})
    core.apply_dual_ntp_pools(model)
    stages.append({"stage": "apply_dual_ntp_pools", **counts(model)})
    core.add_ntp_polymerization_trackers(model)
    stages.append({"stage": "add_ntp_polymerization_trackers", **counts(model)})
    legacy_stats = {
        "mrna_rs": core.remove_legacy_mrna_translation_entry_reactions(model),
        "rs_init": core.remove_legacy_rs_initiation_microstate_reactions(model),
        "rs_elong_term": core.remove_legacy_rs_elongation_termination_microstate_reactions(model),
        "rs_energy_side": core.remove_legacy_rs_energy_side_reactions(model),
        "rs_assembly": core.remove_legacy_rs_assembly_conformation_reactions(model),
    }
    stages.append({"stage": "remove_legacy_rs_modules", **counts(model)})
    tx_collapse = core.collapse_transcription_triplets_to_net_steps(model)
    stages.append({"stage": "tx_triplet_collapse_disabled_micro3", **counts(model)})
    core.parameterize_ribosome_binding(model)
    ribo_qssa = core.apply_ribosome_binding_qssa_if_enabled(model)
    stages.append({"stage": "parameterize_explicit_ribosome_binding", **counts(model)})
    core.split_translation_elongation_three_step(model)
    stages.append({"stage": "build_1aa_effective_translation_chain", **counts(model)})
    energy_stats = core.backgroundize_energy_regen_loops(model)
    stages.append({"stage": "backgroundize_energy_regen_loops", **counts(model)})
    side_stats = core.coarse_grain_fast_side_modules(model)
    core.add_nucleotide_family_bookkeeping_trackers(model)
    stages.append({"stage": "add_nucleotide_family_bookkeeping_trackers", **counts(model)})
    prune_stats = core.prune_unreferenced_legacy_species(model)
    stages.append({"stage": "prune_unreferenced_legacy_species", **counts(model)})

    _, batch_counts, batch_meta = core.build_background_sbml(
        mode="batch",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    _, continuous_counts, continuous_meta = core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.5,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )
    report = {
        "package_root": str(ROOT),
        "all_authoritative_inputs_inside_package": all(
            path.resolve().is_relative_to(ROOT)
            for path in [core.BASE_SBML, core.INIT_CSV, core.TX_1NT_SOURCE_SBML, core.LEGACY_RS_MIGRATION_TABLE]
        ),
        "runtime_parameter_count": len(params),
        "transformation_stages": stages,
        "transcription_triplet_collapse": tx_collapse,
        "legacy_removal": legacy_stats,
        "ribosome_binding_qssa": ribo_qssa,
        "energy_backgroundization": energy_stats,
        "fast_side_scaling": side_stats,
        "legacy_species_pruning": prune_stats,
        "final_batch_counts": batch_counts,
        "final_continuous_counts": continuous_counts,
        "continuous_feed_species": continuous_meta["continuous_feed_species"],
        "continuous_washout_species": continuous_meta["continuous_washout_species"],
        "continuous_product_outflow_species": continuous_meta["continuous_product_outflow_species"],
        "continuous_feed_uM": continuous_meta["continuous_feed_uM"],
        "batch_conservation_families": batch_meta["conservation_families"],
    }
    report["passed"] = (
        report["all_authoritative_inputs_inside_package"]
        and batch_counts["species_total"] == 3399
        and batch_counts["reactions_total"] == 3696
        and continuous_counts["species_total"] == 3399
        and continuous_counts["reactions_total"] == 3762
        and energy_stats.get("removed") == 61
        and tx_collapse.get("enabled") == 0
    )
    out = ROOT / "MODEL_STRUCTURE_AUDIT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
