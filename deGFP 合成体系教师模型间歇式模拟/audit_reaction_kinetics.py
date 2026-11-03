#!/usr/bin/env python3
"""Export reaction kinetic-law classes and parameter values for the deGFP model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import libsbml
import pandas as pd

import fit_batch_then_continuous_full_crn_background as core
from run_tx1nt_tlg1_micro3_longrun import apply_params, load_sources, set_fine_node_env


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results/reaction_kinetics_audit"
PARAM_SUMMARY = ROOT / "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"
HF_SUMMARY = ROOT / "output_full_crn_background/batch_then_continuous_summary.json"

RESERVED = {
    "time",
    "default",
    "true",
    "false",
    "piecewise",
    "max",
    "min",
    "abs",
    "pow",
    "exp",
    "log",
    "sqrt",
    "floor",
    "ceil",
}


def _refs_from_formula(formula: str, model: libsbml.Model) -> tuple[list[str], list[str]]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula or ""))
    pids = {model.getParameter(i).getId() for i in range(model.getNumParameters())}
    sids = {model.getSpecies(i).getId() for i in range(model.getNumSpecies())}
    params = sorted(t for t in tokens if t in pids and t not in RESERVED)
    species = sorted(t for t in tokens if t in sids and t not in RESERVED)
    return params, species


def _reaction_species(rxn: libsbml.Reaction) -> tuple[str, str, str]:
    reactants = []
    products = []
    modifiers = []
    for j in range(rxn.getNumReactants()):
        sr = rxn.getReactant(j)
        sto = sr.getStoichiometry() if sr.isSetStoichiometry() else 1.0
        reactants.append(f"{sto:g} {sr.getSpecies()}")
    for j in range(rxn.getNumProducts()):
        sp = rxn.getProduct(j)
        sto = sp.getStoichiometry() if sp.isSetStoichiometry() else 1.0
        products.append(f"{sto:g} {sp.getSpecies()}")
    for j in range(rxn.getNumModifiers()):
        modifiers.append(rxn.getModifier(j).getSpecies())
    return "; ".join(reactants), "; ".join(products), "; ".join(modifiers)


def _classify(rid: str, formula: str, reactants: str, products: str) -> tuple[str, str]:
    text = " ".join([rid, formula, reactants, products])
    if rid.startswith(("rxn_bg_", "rxn_feed_", "rxn_clear_")) or "bg_" in formula or "bgRes_" in text:
        return "backgroundized", "high-frequency small-molecule background/feed/clearance or bg_keff load correction"
    if rid.startswith("rxn_tl_chain_") or rid in {"rxn_tl_chain_init", "rxn_tl_chain_term", "rxn_tl_core_qssa"}:
        return "qssa_effective", "translation factorized QSSA / 1-aa effective translation chain"
    if rid == "rxn_ribosome_binding":
        return "mass_action", "explicit reversible ribosome-mRNA binding mass-action reaction"
    if "k_tl_" in formula or "K_tl_" in formula or "phi_tl_" in formula:
        return "qssa_effective", "translation QSSA-gated effective reaction"
    if rid.startswith(("rxn_ntp_xfer_", "rxn_tx_net_")):
        return "mass_action", "added mechanistic net/mass-action bookkeeping reaction"
    if "cons_" in formula or "Moiety_" in text:
        return "mass_action_with_bookkeeping", "ordinary mass-action reaction with added moiety/conservation bookkeeping"
    return "mass_action", "ordinary explicit mass-action / source SBML kinetic law"


def _parameter_table(model: libsbml.Model, fitted_params: dict[str, float]) -> pd.DataFrame:
    rows = []
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        pid = p.getId()
        value = float(p.getValue())
        if pid in fitted_params:
            source = "formal_batch_fit_overrode_sbml_default"
            effective_value = float(fitted_params[pid])
        elif pid.startswith("bg_"):
            source = "background_model"
            effective_value = value
        elif pid.startswith(("k_tl", "K_tl", "phi_tl", "N_tl")):
            source = "translation_qssa_or_chain"
            effective_value = value
        elif pid.startswith(("k_ribo", "k_ntp", "k_tx")):
            source = "mechanistic_refactor"
            effective_value = value
        else:
            source = "sbml_or_refactor_default"
            effective_value = value
        rows.append(
            {
                "parameter_id": pid,
                "sbml_value_before_roadrunner_override": value,
                "effective_value_used_in_simulation": effective_value,
                "source": source,
                "constant": bool(p.getConstant()),
            }
        )
    return pd.DataFrame(rows).sort_values("parameter_id")


def main() -> None:
    set_fine_node_env()
    params, hf_fit, keff_map = load_sources(PARAM_SUMMARY, HF_SUMMARY)
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    sbml, counts, _ = core.build_background_sbml(
        mode="batch",
        fitted_hf=hf_fit,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=1.0,
        product_outflow_h=None,
        keff_map=keff_map,
    )

    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse generated deGFP SBML")

    param_df = _parameter_table(model, params)
    param_value = dict(
        zip(param_df["parameter_id"], param_df["effective_value_used_in_simulation"])
    )

    reaction_rows = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        rid = rxn.getId()
        formula = ""
        if rxn.isSetKineticLaw() and rxn.getKineticLaw().isSetMath():
            formula = libsbml.formulaToL3String(rxn.getKineticLaw().getMath())
        reactants, products, modifiers = _reaction_species(rxn)
        params_in_formula, species_in_formula = _refs_from_formula(formula, model)
        klass, note = _classify(rid, formula, reactants, products)
        reaction_rows.append(
            {
                "reaction_id": rid,
                "class": klass,
                "note": note,
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "kinetic_law": formula,
                "parameter_ids": "; ".join(params_in_formula),
                "parameter_values": "; ".join(
                    f"{pid}={param_value.get(pid)}" for pid in params_in_formula
                ),
                "species_in_kinetic_law": "; ".join(species_in_formula),
            }
        )

    reaction_df = pd.DataFrame(reaction_rows)
    summary_df = (
        reaction_df.groupby("class")
        .size()
        .reset_index(name="reaction_count")
        .sort_values("reaction_count", ascending=False)
    )
    prefix_df = (
        reaction_df.assign(prefix=reaction_df["reaction_id"].str.extract(r"^(rxn_[A-Za-z0-9]+|R_[A-Za-z0-9]+|[A-Za-z0-9]+)", expand=False))
        .groupby(["class", "prefix"])
        .size()
        .reset_index(name="reaction_count")
        .sort_values(["class", "reaction_count"], ascending=[True, False])
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    reaction_df.to_csv(OUTDIR / "reaction_kinetics_inventory.csv", index=False)
    param_df.to_csv(OUTDIR / "parameter_values.csv", index=False)
    summary_df.to_csv(OUTDIR / "reaction_class_summary.csv", index=False)
    prefix_df.to_csv(OUTDIR / "reaction_class_by_prefix.csv", index=False)
    (OUTDIR / "generated_model_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "class_summary": summary_df.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
