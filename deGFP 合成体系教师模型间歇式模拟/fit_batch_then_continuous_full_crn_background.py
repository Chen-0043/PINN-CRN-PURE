#!/usr/bin/env python3
"""
Full-CRN workflow with high-frequency species as background fields.

Design:
1) Use the validated full mass-action COPASI model as reference (no compression).
2) Fit high-frequency species trajectories from the successful batch simulation.
3) Convert only those high-frequency species into state-driven conservative
   buffer reactions (no explicit time forcing).
4) Calibrate effective scales to match batch target (~5.3 uM plateau).
5) Reuse the same background parameters in continuous mode with feed/clearance term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import libsbml
import numpy as np
import pandas as pd
import tellurium as te
from scipy.optimize import least_squares, minimize


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_SBML = PACKAGE_DIR / "PURE_CRN_models/Models/CRN_PURE_MGaptTXwTL_with_deGFP_COPASI.xml"
INIT_CSV = PACKAGE_DIR / "PURE_CRN_models/Data_files/PURE_TXTL_initial_values_Final.csv"
TX_1NT_SOURCE_SBML = PACKAGE_DIR / "external_PURE_CRN_models_sparse/Models/TXTL_PURE_MGapt-deGFP_model.xml"

# High-frequency pool to background. Keep topology for everything else.
# Start from the minimal energy/phosphate loop to preserve accuracy.
HF_SPECIES = ["ATP", "GTP", "CTP", "UTP", "PPi", "PO4"]
AA20 = [
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
]

# Small molecules expected to exchange with reservoir in continuous mode.
# NOTE: these are candidates; the final list is filtered by model content and
# biological-feasibility policy in `select_continuous_exchange_species`.
DEFAULT_CONTINUOUS_DIALYZABLE = [
    "ATP",
    "ADP",
    "AMP",
    "GTP",
    "GDP",
    "GMP",
    "CTP",
    "CDP",
    "CMP",
    "UTP",
    "UDP",
    "UMP",
    "AA",
    "CP",
    "PO4",
    "PPi",
    "Pi",
] + AA20

# Species maintained by feed composition (others in dialyzable set are washed toward 0).
DEFAULT_CONTINUOUS_FEED = ["ATP", "GTP", "CTP", "UTP", "AA", "CP"] + AA20

# Macro-product species with explicit continuous removal (no feed) to avoid
# unbounded accumulation in long-run continuous operation.
DEFAULT_CONTINUOUS_PRODUCT_OUTFLOW = ["deGFP_m"]

# Dual-pool split for NTPs:
# - energy pool keeps original ids (ATP/GTP/CTP/UTP)
# - transcription substrate pool uses *_TX species
NTP_TX_MAP = {
    "ATP": "ATP_TX",
    "GTP": "GTP_TX",
    "CTP": "CTP_TX",
    "UTP": "UTP_TX",
}

NTP_POLY_TRACKER_MAP = {
    "ATP": "Moiety_ATP_poly",
    "ATP_TX": "Moiety_ATP_poly",
    "GTP": "Moiety_GTP_poly",
    "GTP_TX": "Moiety_GTP_poly",
    "CTP": "Moiety_CTP_poly",
    "CTP_TX": "Moiety_CTP_poly",
    "UTP": "Moiety_UTP_poly",
    "UTP_TX": "Moiety_UTP_poly",
}

NTP_FAMILY_SIGNED_TERMS = {
    "ATP_family": [
        ("ATP", 1.0),
        ("ADP", 1.0),
        ("AMP", 1.0),
        ("ATP_TX", 1.0),
        ("Moiety_ATP_poly", 1.0),
        ("Moiety_ATP_untracked_loss", 1.0),
        ("Moiety_ATP_untracked_gain", -1.0),
    ],
    "GTP_family": [
        ("GTP", 1.0),
        ("GDP", 1.0),
        ("GMP", 1.0),
        ("GTP_TX", 1.0),
        ("Moiety_GTP_poly", 1.0),
        ("Moiety_GTP_untracked_loss", 1.0),
        ("Moiety_GTP_untracked_gain", -1.0),
    ],
    "CTP_family": [
        ("CTP", 1.0),
        ("CDP", 1.0),
        ("CMP", 1.0),
        ("CTP_TX", 1.0),
        ("Moiety_CTP_poly", 1.0),
        ("bgRes_CTP", 1.0),
        ("Moiety_CTP_untracked_loss", 1.0),
        ("Moiety_CTP_untracked_gain", -1.0),
    ],
    "UTP_family": [
        ("UTP", 1.0),
        ("UDP", 1.0),
        ("UMP", 1.0),
        ("UTP_TX", 1.0),
        ("Moiety_UTP_poly", 1.0),
        ("bgRes_UTP", 1.0),
        ("Moiety_UTP_untracked_loss", 1.0),
        ("Moiety_UTP_untracked_gain", -1.0),
    ],
}

LEGACY_RS_MIGRATION_TABLE = PACKAGE_DIR / (
    "output_full_crn_background/rs_legacy_module_migration_20260609/"
    "table_RS旧模块迁移清单_337反应_20260609.csv"
)


def _reaction_has_reactant(model: "libsbml.Model", rid: str, sid: str) -> bool:
    rxn = model.getReaction(rid)
    if rxn is None:
        return False
    for j in range(rxn.getNumReactants()):
        if rxn.getReactant(j).getSpecies() == sid:
            return True
    return False


def select_continuous_exchange_species(model: "libsbml.Model") -> Tuple[List[str], List[str], Dict[str, object]]:
    present = {model.getSpecies(i).getId() for i in range(model.getNumSpecies())}
    dialyzable = [sid for sid in DEFAULT_CONTINUOUS_DIALYZABLE if sid in present]
    feed = [sid for sid in DEFAULT_CONTINUOUS_FEED if sid in present]

    # If translation elongation consumes aggregate AA pool directly, feeding both
    # AA and AA20 can become an artificial double-supply in this model.
    uses_aggregate_aa = (
        _reaction_has_reactant(model, "rxn_translation_elongation", "AA")
        or _reaction_has_reactant(model, "rxn_translation_elong_core", "AA")
    )
    if uses_aggregate_aa and "AA" in present:
        aa20_present = {sid for sid in AA20 if sid in present}
        dialyzable = [sid for sid in dialyzable if sid not in aa20_present]
        feed = [sid for sid in feed if sid not in aa20_present]

    # Keep order stable and remove duplicates.
    dialyzable = list(dict.fromkeys(dialyzable))
    feed = list(dict.fromkeys(feed))
    meta = {
        "uses_aggregate_aa_in_translation": bool(uses_aggregate_aa),
        "aa20_removed_due_to_aggregate_aa": bool(uses_aggregate_aa and "AA" in present),
    }
    return dialyzable, feed, meta


def ensure_parameter(model: "libsbml.Model", pid: str, value: float) -> "libsbml.Parameter":
    p = model.getParameter(pid)
    if p is None:
        p = model.createParameter()
        p.setId(pid)
        p.setConstant(True)
    p.setValue(float(value))
    return p


def ensure_dynamic_parameter(model: "libsbml.Model", pid: str, value: float) -> "libsbml.Parameter":
    p = model.getParameter(pid)
    if p is None:
        p = model.createParameter()
        p.setId(pid)
    p.setConstant(False)
    p.setValue(float(value))
    return p


def remove_rules_for_variable(model: "libsbml.Model", var_id: str) -> None:
    for i in range(model.getNumRules() - 1, -1, -1):
        rule = model.getRule(i)
        if hasattr(rule, "getVariable") and rule.getVariable() == var_id:
            model.removeRule(i)


def set_assignment_rule(model: "libsbml.Model", var_id: str, formula: str) -> None:
    remove_rules_for_variable(model, var_id)
    rule = model.createAssignmentRule()
    rule.setVariable(var_id)
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(f"Failed to parse assignment formula for {var_id}: {formula}")
    rule.setMath(ast)


def set_rate_rule(model: "libsbml.Model", var_id: str, formula: str) -> None:
    remove_rules_for_variable(model, var_id)
    rule = model.createRateRule()
    rule.setVariable(var_id)
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(f"Failed to parse rate formula for {var_id}: {formula}")
    rule.setMath(ast)


def sanitize_id(raw: str) -> str:
    out = []
    for ch in str(raw):
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    sid = "".join(out)
    if not sid:
        sid = "x"
    if sid[0].isdigit():
        sid = f"x_{sid}"
    return sid


def remove_reaction_if_exists(model: "libsbml.Model", rid: str) -> None:
    rxn = model.getReaction(rid)
    if rxn is not None:
        model.removeReaction(rxn.getId())


def ensure_species_concentration(
    model: "libsbml.Model",
    sid: str,
    compartment_id: str,
    init_conc: float = 0.0,
) -> "libsbml.Species":
    sp = model.getSpecies(sid)
    if sp is None:
        sp = model.createSpecies()
        sp.setId(sid)
    sp.setCompartment(compartment_id)
    sp.setBoundaryCondition(False)
    sp.setConstant(False)
    sp.setHasOnlySubstanceUnits(False)
    sp.setInitialConcentration(float(init_conc))
    if sp.isSetInitialAmount():
        sp.unsetInitialAmount()
    return sp


def add_reaction_with_formula(
    model: "libsbml.Model",
    rid: str,
    reactants: List[Tuple[str, float]],
    products: List[Tuple[str, float]],
    formula: str,
    reversible: bool = False,
) -> "libsbml.Reaction":
    remove_reaction_if_exists(model, rid)
    rxn = model.createReaction()
    rxn.setId(rid)
    rxn.setReversible(bool(reversible))
    for sid, sto in reactants:
        sr = rxn.createReactant()
        sr.setSpecies(sid)
        sr.setStoichiometry(float(sto))
    for sid, sto in products:
        sp = rxn.createProduct()
        sp.setSpecies(sid)
        sp.setStoichiometry(float(sto))
    kl = rxn.createKineticLaw()
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(f"Failed to parse kinetic formula for {rid}: {formula}")
    kl.setMath(ast)
    return rxn


def _reaction_species_ids(rxn: "libsbml.Reaction") -> List[str]:
    ids: List[str] = []
    for j in range(rxn.getNumReactants()):
        ids.append(rxn.getReactant(j).getSpecies())
    for j in range(rxn.getNumProducts()):
        ids.append(rxn.getProduct(j).getSpecies())
    return ids


def _set_or_create_product_stoichiometry(rxn: "libsbml.Reaction", sid: str, sto: float) -> None:
    for j in range(rxn.getNumProducts()):
        sp = rxn.getProduct(j)
        if sp.getSpecies() == sid:
            sp.setStoichiometry(float(sto))
            return
    sp = rxn.createProduct()
    sp.setSpecies(sid)
    sp.setStoichiometry(float(sto))


def _remove_product_if_exists(rxn: "libsbml.Reaction", sid: str) -> None:
    for j in range(rxn.getNumProducts() - 1, -1, -1):
        if rxn.getProduct(j).getSpecies() == sid:
            rxn.removeProduct(j)


def _is_transcription_reaction(rxn: "libsbml.Reaction") -> bool:
    # In this model, transcription chain reactions carry T7RNAP-bound states.
    return any(sid.startswith("T7RNAP") for sid in _reaction_species_ids(rxn))


def _reaction_species_ids_all(rxn: "libsbml.Reaction") -> List[str]:
    ids: List[str] = []
    for j in range(rxn.getNumReactants()):
        ids.append(rxn.getReactant(j).getSpecies())
    for j in range(rxn.getNumProducts()):
        ids.append(rxn.getProduct(j).getSpecies())
    for j in range(rxn.getNumModifiers()):
        ids.append(rxn.getModifier(j).getSpecies())
    return ids


def _sync_species_from_source(target: "libsbml.Model", source: "libsbml.Model", sid: str) -> None:
    s_src = source.getSpecies(sid)
    if s_src is None:
        return
    s_tgt = target.getSpecies(sid)
    if s_tgt is None:
        ret = target.addSpecies(s_src.clone())
        if ret != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(f"Failed adding transcription species {sid}, code={ret}")
        return
    # Keep current calibrated initial concentrations for existing species.
    s_tgt.setCompartment(s_src.getCompartment())
    s_tgt.setBoundaryCondition(s_src.getBoundaryCondition())
    s_tgt.setConstant(s_src.getConstant())
    s_tgt.setHasOnlySubstanceUnits(s_src.getHasOnlySubstanceUnits())


def swap_transcription_to_1nt_chain(
    model: "libsbml.Model",
    source_path: Path = TX_1NT_SOURCE_SBML,
) -> Dict[str, int]:
    if not source_path.exists():
        raise FileNotFoundError(f"1nt transcription source not found: {source_path}")

    doc_src = libsbml.readSBMLFromFile(str(source_path))
    src = doc_src.getModel()
    if src is None:
        raise RuntimeError(f"Failed to parse source transcription model: {source_path}")

    def is_tx_reaction(rxn: "libsbml.Reaction") -> bool:
        return any(sid.startswith("T7RNAP") for sid in _reaction_species_ids_all(rxn))

    removed = 0
    for i in range(model.getNumReactions() - 1, -1, -1):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        if is_tx_reaction(rxn):
            model.removeReaction(i)
            removed += 1

    src_tx_ids: List[str] = []
    needed_species: set[str] = set()
    for i in range(src.getNumReactions()):
        rxn = src.getReaction(i)
        if rxn is None:
            continue
        if not is_tx_reaction(rxn):
            continue
        src_tx_ids.append(rxn.getId())
        needed_species.update(_reaction_species_ids_all(rxn))

    for sid in sorted(needed_species):
        _sync_species_from_source(model, src, sid)

    existing = {model.getReaction(i).getId() for i in range(model.getNumReactions())}
    added = 0
    for rid in src_tx_ids:
        rxn_src = src.getReaction(rid)
        if rxn_src is None:
            continue
        rid_use = rid if rid not in existing else f"tx1nt_{rid}"
        clone = rxn_src.clone()
        clone.setId(rid_use)
        ret = model.addReaction(clone)
        if ret != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise RuntimeError(f"Failed adding 1nt transcription reaction {rid_use}, code={ret}")
        existing.add(rid_use)
        added += 1

    return {
        "tx_removed": int(removed),
        "tx_added": int(added),
        "has_T7RNAP_bound_mRNA0805": int(model.getSpecies("T7RNAP_bound_mRNA0805") is not None),
    }


def parameterize_ribosome_binding(model: "libsbml.Model") -> None:
    rxn = model.getReaction("rxn_ribosome_binding")
    if rxn is None or (not rxn.isSetKineticLaw()):
        return
    base_f = 1.0e5
    base_r = 0.3
    scale = float(os.getenv("PURE_RIBO_BIND_SCALE", "1.0"))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    # Scale both directions together to keep Kd = k_r/k_f unchanged while
    # reducing stiffness from ultra-fast reversible occupancy equilibration.
    ensure_parameter(model, "k_ribo_bind_f", base_f * scale)
    ensure_parameter(model, "k_ribo_bind_r", base_r * scale)
    kl = rxn.getKineticLaw()
    formula = "default * k_ribo_bind_f * mRNA_t * Ribosome - default * k_ribo_bind_r * Ribosome_mRNA"
    ast = libsbml.parseL3Formula(formula)
    if ast is not None:
        kl.setMath(ast)


def remove_legacy_mrna_translation_entry_reactions(model: "libsbml.Model") -> Dict[str, int]:
    """
    Remove the orphaned legacy translation-initiation branch that still consumes
    `mRNA` through explicit RS30S/RS50S/RS70S microstate reactions.

    Current production/TL backbone uses:
      mRNA_t + Ribosome <-> Ribosome_mRNA -> TLchain...
    so the old `mRNA`-driven RS* branch is disconnected from the active TX->TL
    path and only adds inconsistent duplicate topology.
    """
    removed = 0
    for i in range(model.getNumReactions() - 1, -1, -1):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        sids = _reaction_species_ids_all(rxn)
        if "mRNA" not in sids:
            continue
        if any(sid.startswith(("RS30S", "RS50S", "RS70S")) for sid in sids):
            model.removeReaction(i)
            removed += 1
    return {"legacy_mrna_rs_reactions_removed": int(removed)}


def remove_legacy_rs_initiation_microstate_reactions(model: "libsbml.Model") -> Dict[str, int]:
    """
    Remove only the legacy RS* initiation branch after its effective function
    has been migrated into the new Ribosome/Ribosome_mRNA -> TLchain_pos_000
    gate.

    We intentionally do *not* remove RS* elongation/termination microstates
    here; this stage only migrates the old initiation logic.
    """
    removed = 0
    init_markers = {
        "IF1",
        "IF2",
        "IF2_GTP",
        "IF2_GDP",
        "IF3",
        "fMettRNAfMetCAU",
        "IF2_GTP_fMettRNAfMetCAU",
    }
    for i in range(model.getNumReactions() - 1, -1, -1):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        sids = _reaction_species_ids_all(rxn)
        if not any(sid.startswith(("RS30S", "RS70S")) for sid in sids):
            continue
        if not any(sid in init_markers for sid in sids):
            continue
        model.removeReaction(i)
        removed += 1
    return {"legacy_rs_init_reactions_removed": int(removed)}


def _load_legacy_rs_reaction_ids_by_category() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not LEGACY_RS_MIGRATION_TABLE.exists():
        return out
    try:
        with LEGACY_RS_MIGRATION_TABLE.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rid = str(row.get("反应ID", "")).strip()
                cat = str(row.get("功能分类", "")).strip()
                if not rid or not cat:
                    continue
                out.setdefault(cat, []).append(rid)
    except Exception:
        return {}
    return out


def _remove_reactions_by_ids(model: "libsbml.Model", reaction_ids: List[str]) -> Dict[str, object]:
    removed: List[str] = []
    missing: List[str] = []
    for rid in reaction_ids:
        rxn = model.getReaction(rid)
        if rxn is None:
            missing.append(rid)
            continue
        model.removeReaction(rid)
        removed.append(rid)
    return {
        "removed": int(len(removed)),
        "missing": int(len(missing)),
        "removed_ids_sample": removed[:20],
        "missing_ids_sample": missing[:20],
    }


def remove_legacy_rs_elongation_termination_microstate_reactions(model: "libsbml.Model") -> Dict[str, object]:
    """
    Remove the remaining legacy RS* elongation/translocation and
    termination/recycling microstates after their effective role has been
    migrated into the explicit TL chain step gate (phi_elong / phi_term).
    """
    cats = _load_legacy_rs_reaction_ids_by_category()
    reaction_ids = (
        cats.get("翻译延长/转位微状态", [])
        + cats.get("翻译终止/核糖体回收微状态", [])
    )
    stats = _remove_reactions_by_ids(model, reaction_ids)
    stats["categories"] = ["翻译延长/转位微状态", "翻译终止/核糖体回收微状态"]
    return stats


def remove_legacy_rs_energy_side_reactions(model: "libsbml.Model") -> Dict[str, object]:
    """
    Remove old PO4/PPi coupling side branches that still hang on RS* microstates.
    These pools are already represented by the unified backgrounded small-molecule
    fields and should not remain attached to dead legacy translation states.
    """
    cats = _load_legacy_rs_reaction_ids_by_category()
    reaction_ids = cats.get("能量/副产物耦合旧侧支", [])
    stats = _remove_reactions_by_ids(model, reaction_ids)
    stats["categories"] = ["能量/副产物耦合旧侧支"]
    return stats


def remove_legacy_rs_assembly_conformation_reactions(model: "libsbml.Model") -> Dict[str, object]:
    """
    Remove old RS30S/RS50S/RS70S assembly and conformation reactions after the
    active translation backbone has been migrated to the unified Ribosome pool.
    """
    cats = _load_legacy_rs_reaction_ids_by_category()
    reaction_ids = cats.get("核糖体亚基装配/构象旧模块", [])
    stats = _remove_reactions_by_ids(model, reaction_ids)
    stats["categories"] = ["核糖体亚基装配/构象旧模块"]
    return stats


def _env_flag(name: str, default: str = "1") -> bool:
    raw = str(os.getenv(name, default)).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _species_is_referenced_anywhere(model: "libsbml.Model", sid: str) -> bool:
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        for j in range(rxn.getNumReactants()):
            if rxn.getReactant(j).getSpecies() == sid:
                return True
        for j in range(rxn.getNumProducts()):
            if rxn.getProduct(j).getSpecies() == sid:
                return True
        for j in range(rxn.getNumModifiers()):
            if rxn.getModifier(j).getSpecies() == sid:
                return True
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule is None:
            continue
        if hasattr(rule, "getVariable") and rule.getVariable() == sid:
            return True
        if rule.isSetMath():
            txt = libsbml.formulaToL3String(rule.getMath())
            if re.search(rf"\b{re.escape(sid)}\b", txt):
                return True
    for i in range(model.getNumInitialAssignments()):
        ia = model.getInitialAssignment(i)
        if ia is None:
            continue
        if ia.getSymbol() == sid:
            return True
        if ia.isSetMath():
            txt = libsbml.formulaToL3String(ia.getMath())
            if re.search(rf"\b{re.escape(sid)}\b", txt):
                return True
    for i in range(model.getNumEvents()):
        ev = model.getEvent(i)
        if ev is None:
            continue
        if ev.isSetTrigger() and ev.getTrigger().isSetMath():
            txt = libsbml.formulaToL3String(ev.getTrigger().getMath())
            if re.search(rf"\b{re.escape(sid)}\b", txt):
                return True
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            if ea is None:
                continue
            if ea.getVariable() == sid:
                return True
            if ea.isSetMath():
                txt = libsbml.formulaToL3String(ea.getMath())
                if re.search(rf"\b{re.escape(sid)}\b", txt):
                    return True
    return False


def prune_unreferenced_legacy_species(model: "libsbml.Model") -> Dict[str, object]:
    """
    Remove old-chain-only species that are no longer referenced by any reaction,
    rule, initial assignment, or event after the new-chain migration has been
    completed.
    """
    if not _env_flag("PURE_PRUNE_LEGACY_SPECIES", "1"):
        return {"enabled": False, "removed": 0, "sample": []}

    def is_legacy_sid(sid: str) -> bool:
        if sid == "mRNA":
            return True
        return sid.startswith(("RS30S", "RS50S", "RS70S"))

    remove_ids: List[str] = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp is None:
            continue
        sid = sp.getId()
        if not is_legacy_sid(sid):
            continue
        if _species_is_referenced_anywhere(model, sid):
            continue
        remove_ids.append(sid)

    removed: List[str] = []
    for sid in remove_ids:
        sp = model.getSpecies(sid)
        if sp is None:
            continue
        model.removeSpecies(sid)
        removed.append(sid)

    return {
        "enabled": True,
        "removed": int(len(removed)),
        "sample": removed[:50],
    }


def apply_ribosome_binding_qssa_if_enabled(model: "libsbml.Model") -> Dict[str, object]:
    """
    Optional logic-level stiffness reduction:
      - remove explicit fast reversible ribosome-binding reaction
      - keep k_f/k_r parameters as Kd reference for runtime QSSA projection
    """
    mode = os.getenv("PURE_RIBO_BIND_MODE", "explicit").strip().lower()
    if mode not in {"qssa", "algebraic", "projected"}:
        return {"enabled": False, "removed": False}
    removed = False
    if model.getReaction("rxn_ribosome_binding") is not None:
        model.removeReaction("rxn_ribosome_binding")
        removed = True
    # Ensure reference parameters exist for runtime QSSA projection.
    ensure_parameter(model, "k_ribo_bind_f", float(model.getParameter("k_ribo_bind_f").getValue()) if model.getParameter("k_ribo_bind_f") is not None else 1.0e5)
    ensure_parameter(model, "k_ribo_bind_r", float(model.getParameter("k_ribo_bind_r").getValue()) if model.getParameter("k_ribo_bind_r") is not None else 0.3)
    return {"enabled": True, "removed": removed}


def apply_dual_ntp_pools(model: "libsbml.Model") -> None:
    frac_param = ensure_parameter(model, "bg_ntp_tx_init_frac", 0.25)
    tx_frac = float(np.clip(frac_param.getValue(), 0.01, 0.8))

    for ntp, ntp_tx in NTP_TX_MAP.items():
        sp_energy = model.getSpecies(ntp)
        if sp_energy is None:
            continue
        comp_id = sp_energy.getCompartment()
        e0 = float(sp_energy.getInitialConcentration()) if sp_energy.isSetInitialConcentration() else 0.0

        sp_tx = ensure_species_concentration(
            model=model,
            sid=ntp_tx,
            compartment_id=comp_id,
            init_conc=0.0,
        )
        tx0 = max(1e-18, e0 * tx_frac)
        e_new = max(1e-18, e0 - tx0)
        sp_energy.setInitialConcentration(e_new)
        sp_tx.setInitialConcentration(tx0)

        kf_pid = f"k_ntp_xfer_{ntp}_f"
        kr_pid = f"k_ntp_xfer_{ntp}_r"
        ensure_parameter(model, kf_pid, 2.0e-3)
        ensure_parameter(model, kr_pid, 2.0e-3)
        vol_term = comp_id
        add_reaction_with_formula(
            model=model,
            rid=f"rxn_ntp_xfer_f_{ntp}",
            reactants=[(ntp, 1.0)],
            products=[(ntp_tx, 1.0)],
            formula=f"{kf_pid}*{ntp}*{vol_term}",
            reversible=False,
        )
        add_reaction_with_formula(
            model=model,
            rid=f"rxn_ntp_xfer_r_{ntp}",
            reactants=[(ntp_tx, 1.0)],
            products=[(ntp, 1.0)],
            formula=f"{kr_pid}*{ntp_tx}*{vol_term}",
            reversible=False,
        )

    # Route transcription-chain NTP consumption through *_TX pools.
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        rid = rxn.getId()
        if rid.startswith("rxn_ntp_xfer_"):
            continue
        if not _is_transcription_reaction(rxn):
            continue

        changed = False
        for j in range(rxn.getNumReactants()):
            sr = rxn.getReactant(j)
            sid = sr.getSpecies()
            if sid in NTP_TX_MAP and model.getSpecies(NTP_TX_MAP[sid]) is not None:
                sr.setSpecies(NTP_TX_MAP[sid])
                changed = True

        if changed and rxn.isSetKineticLaw():
            kl = rxn.getKineticLaw()
            base_formula = libsbml.formulaToL3String(kl.getMath())
            for sid, tx_sid in NTP_TX_MAP.items():
                base_formula = re.sub(rf"\b{re.escape(sid)}\b", tx_sid, base_formula)
            ast = libsbml.parseL3Formula(base_formula)
            if ast is not None:
                kl.setMath(ast)


def add_ntp_polymerization_trackers(model: "libsbml.Model") -> None:
    """
    Track nucleotide moieties polymerized into RNA so family conservation checks
    remain algebraically closed after dual-pool substitution.
    """
    tracker_base = {
        "Moiety_ATP_poly": "ATP_TX",
        "Moiety_GTP_poly": "GTP_TX",
        "Moiety_CTP_poly": "CTP_TX",
        "Moiety_UTP_poly": "UTP_TX",
    }
    fallback_comp = model.getCompartment(0).getId() if model.getNumCompartments() > 0 else "default"
    for tracker_sid, base_sid in tracker_base.items():
        base_sp = model.getSpecies(base_sid)
        comp_id = base_sp.getCompartment() if base_sp is not None else fallback_comp
        ensure_species_concentration(model, tracker_sid, compartment_id=comp_id, init_conc=0.0)

    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None or (not _is_transcription_reaction(rxn)):
            continue
        increments: Dict[str, float] = {}
        for j in range(rxn.getNumReactants()):
            sr = rxn.getReactant(j)
            sid = sr.getSpecies()
            tracker_sid = NTP_POLY_TRACKER_MAP.get(sid)
            if tracker_sid is None:
                continue
            increments[tracker_sid] = increments.get(tracker_sid, 0.0) + float(sr.getStoichiometry())
        for tracker_sid, sto in increments.items():
            if sto > 0.0:
                _set_or_create_product_stoichiometry(rxn, tracker_sid, sto)


def collapse_transcription_triplets_to_net_steps(model: "libsbml.Model") -> Dict[str, int]:
    """
    Collapse transcription micro-triplets:
      mRNA(i)+NTP -> mRNA(i)_NTP -> mRNA(i+1)_PPi -> mRNA(i+1)+PPi
    into one net step:
      mRNA(i)+NTP -> mRNA(i+1)+PPi

    The net-rate uses the original first sub-step formula multiplied by
    `k_tx_net_scale` so we can tune to preserve batch accuracy.
    """
    step_mode = os.getenv("PURE_TX_STEP_MODE", "micro3").strip().lower()
    if step_mode not in {"net1", "collapsed", "qssa_net"}:
        return {"enabled": 0, "collapsed_triplets": 0, "reactions_removed": 0, "reactions_added": 0}

    scale0 = float(os.getenv("PURE_TX_NET_SCALE", "0.35"))
    if (not np.isfinite(scale0)) or scale0 <= 0.0:
        scale0 = 0.35
    ensure_parameter(model, "k_tx_net_scale", scale0)

    re_state = re.compile(r"^T7RNAP_bound_mRNA(\d{4})$")
    re_ntp_bound = re.compile(r"^T7RNAP_bound_mRNA(\d{4})_(ATP|GTP|CTP|UTP)$")
    re_ppi_bound = re.compile(r"^T7RNAP_bound_mRNA(\d{4})_PPi$")

    tx_rxn_ids: List[str] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is not None and _is_transcription_reaction(rxn):
            tx_rxn_ids.append(rxn.getId())

    by_reactant: Dict[str, List[str]] = {}
    for rid in tx_rxn_ids:
        rxn = model.getReaction(rid)
        if rxn is None:
            continue
        for j in range(rxn.getNumReactants()):
            sid = rxn.getReactant(j).getSpecies()
            by_reactant.setdefault(sid, []).append(rid)

    def _reaction_formula(rid: str) -> str:
        rxn = model.getReaction(rid)
        if rxn is None or (not rxn.isSetKineticLaw()):
            return ""
        return libsbml.formulaToL3String(rxn.getKineticLaw().getMath())

    def _build_balance(rids: List[str]) -> Dict[str, float]:
        bal: Dict[str, float] = {}
        for rid in rids:
            rxn = model.getReaction(rid)
            if rxn is None:
                continue
            for j in range(rxn.getNumReactants()):
                sr = rxn.getReactant(j)
                sid = sr.getSpecies()
                bal[sid] = bal.get(sid, 0.0) - float(sr.getStoichiometry())
            for j in range(rxn.getNumProducts()):
                sp = rxn.getProduct(j)
                sid = sp.getSpecies()
                bal[sid] = bal.get(sid, 0.0) + float(sp.getStoichiometry())
        return {k: v for k, v in bal.items() if abs(v) > 1e-12}

    triplets: List[Tuple[str, str, str, str, str, str, str]] = []
    # tuple = (rid_a, rid_b, rid_c, state_i, ntp_sid, state_ip1, formula_a)
    for rid_a in tx_rxn_ids:
        rxn_a = model.getReaction(rid_a)
        if rxn_a is None:
            continue
        rcts = [rxn_a.getReactant(j).getSpecies() for j in range(rxn_a.getNumReactants())]
        prds = [rxn_a.getProduct(j).getSpecies() for j in range(rxn_a.getNumProducts())]

        state_i = None
        for sid in rcts:
            if re_state.match(sid):
                state_i = sid
                break
        if state_i is None:
            continue

        ntp_sid = None
        for sid in rcts:
            if sid in {"ATP", "GTP", "CTP", "UTP", "ATP_TX", "GTP_TX", "CTP_TX", "UTP_TX"}:
                ntp_sid = sid
                break
        if ntp_sid is None:
            continue

        ntp_bound = None
        for sid in prds:
            m = re_ntp_bound.match(sid)
            if m is not None and int(m.group(1)) == int(re_state.match(state_i).group(1)):
                ntp_bound = sid
                break
        if ntp_bound is None:
            continue

        rid_b = None
        state_ip1_ppi = None
        for cand in by_reactant.get(ntp_bound, []):
            rxn_b = model.getReaction(cand)
            if rxn_b is None:
                continue
            b_prds = [rxn_b.getProduct(j).getSpecies() for j in range(rxn_b.getNumProducts())]
            for sid in b_prds:
                mpp = re_ppi_bound.match(sid)
                if mpp is None:
                    continue
                idx_i = int(re_state.match(state_i).group(1))
                if int(mpp.group(1)) == (idx_i + 1):
                    rid_b = cand
                    state_ip1_ppi = sid
                    break
            if rid_b is not None:
                break
        if rid_b is None or state_ip1_ppi is None:
            continue

        rid_c = None
        state_ip1 = None
        for cand in by_reactant.get(state_ip1_ppi, []):
            rxn_c = model.getReaction(cand)
            if rxn_c is None:
                continue
            c_prds = [rxn_c.getProduct(j).getSpecies() for j in range(rxn_c.getNumProducts())]
            for sid in c_prds:
                ms = re_state.match(sid)
                if ms is not None and int(ms.group(1)) == int(re_ppi_bound.match(state_ip1_ppi).group(1)):
                    rid_c = cand
                    state_ip1 = sid
                    break
            if rid_c is not None:
                break
        if rid_c is None or state_ip1 is None:
            continue

        fa = _reaction_formula(rid_a)
        if not fa:
            continue
        triplets.append((rid_a, rid_b, rid_c, state_i, ntp_sid, state_ip1, fa))

    if not triplets:
        return {"enabled": 1, "collapsed_triplets": 0, "reactions_removed": 0, "reactions_added": 0}

    used_reactions: set[str] = set()
    added = 0
    for rid_a, rid_b, rid_c, state_i, ntp_sid, state_ip1, formula_a in triplets:
        if rid_a in used_reactions or rid_b in used_reactions or rid_c in used_reactions:
            continue
        used_reactions.update({rid_a, rid_b, rid_c})

        bal = _build_balance([rid_a, rid_b, rid_c])
        if state_i not in bal or ntp_sid not in bal or state_ip1 not in bal:
            continue
        if bal[state_i] >= -1e-12 or bal[ntp_sid] >= -1e-12 or bal[state_ip1] <= 1e-12:
            continue

        reactants: List[Tuple[str, float]] = []
        products: List[Tuple[str, float]] = []
        for sid, sto in sorted(bal.items()):
            if sto < 0:
                reactants.append((sid, -float(sto)))
            elif sto > 0:
                products.append((sid, float(sto)))

        idx_i = int(re_state.match(state_i).group(1))
        ntp_tag = ntp_sid.replace("_TX", "")
        rid_net = f"rxn_tx_net_{idx_i:04d}_{ntp_tag}"
        formula = f"({formula_a}) * k_tx_net_scale"
        add_reaction_with_formula(
            model=model,
            rid=rid_net,
            reactants=reactants,
            products=products,
            formula=formula,
            reversible=False,
        )
        added += 1

    removed = 0
    for rid in sorted(used_reactions):
        if model.getReaction(rid) is not None:
            model.removeReaction(rid)
            removed += 1

    return {
        "enabled": 1,
        "collapsed_triplets": int(added),
        "reactions_removed": int(removed),
        "reactions_added": int(added),
    }


def _is_energy_regen_core_species(sid: str) -> bool:
    cores = ("CK", "NDK", "MK", "AK", "PPiase")
    return sid in cores or any(sid.startswith(f"{c}_") for c in cores)


def _is_energy_regen_context_species(sid: str) -> bool:
    pool = {
        "ATP", "ADP", "AMP",
        "GTP", "GDP", "GMP",
        "CTP", "CDP", "CMP",
        "UTP", "UDP", "UMP",
        "ATP_TX", "GTP_TX", "CTP_TX", "UTP_TX",
        "PO4", "PPi", "Pi",
        "CP", "Cr",
    }
    if sid in pool:
        return True
    if sid.startswith("Moiety_"):
        return True
    return _is_energy_regen_core_species(sid)


def backgroundize_energy_regen_loops(model: "libsbml.Model") -> Dict[str, object]:
    """
    Remove explicit fast nucleotide-regeneration/phosphate enzyme loops and let
    ATP/GTP/PPi/PO4 background closure carry this high-frequency module.
    """
    mode = os.getenv("PURE_ENERGY_REGEN_MODE", "explicit").strip().lower()
    if mode not in {"bg", "background", "remove", "implicit"}:
        return {"enabled": False, "removed": 0, "sample": []}

    remove_ids: List[str] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        sids = _reaction_species_ids_all(rxn)
        if not sids:
            continue
        if not any(_is_energy_regen_core_species(sid) for sid in sids):
            continue
        if all(_is_energy_regen_context_species(sid) for sid in sids):
            remove_ids.append(rxn.getId())

    for rid in remove_ids:
        if model.getReaction(rid) is not None:
            model.removeReaction(rid)

    return {
        "enabled": True,
        "removed": int(len(remove_ids)),
        "sample": remove_ids[:20],
    }


def add_nucleotide_family_bookkeeping_trackers(model: "libsbml.Model") -> None:
    """
    Balance residual family-wise stoichiometric mismatch in each reaction.

    For each nucleotide family and each reaction, define:
        delta = (sum family products) - (sum family reactants)
    Then append one inert bookkeeping product so that family balance closes:
        delta < 0 -> add Moiety_*_untracked_loss with sto = -delta
        delta > 0 -> add Moiety_*_untracked_gain with sto = +delta

    Conservation checks then use:
        family_total = tracked_members + untracked_loss - untracked_gain
    """
    family_defs = {
        "ATP_family": {
            "members": {"ATP", "ADP", "AMP", "ATP_TX", "Moiety_ATP_poly"},
            "loss_sid": "Moiety_ATP_untracked_loss",
            "gain_sid": "Moiety_ATP_untracked_gain",
            "comp_sid": "ATP_TX",
        },
        "GTP_family": {
            "members": {"GTP", "GDP", "GMP", "GTP_TX", "Moiety_GTP_poly"},
            "loss_sid": "Moiety_GTP_untracked_loss",
            "gain_sid": "Moiety_GTP_untracked_gain",
            "comp_sid": "GTP_TX",
        },
        "CTP_family": {
            "members": {"CTP", "CDP", "CMP", "CTP_TX", "Moiety_CTP_poly"},
            "loss_sid": "Moiety_CTP_untracked_loss",
            "gain_sid": "Moiety_CTP_untracked_gain",
            "comp_sid": "CTP_TX",
        },
        "UTP_family": {
            "members": {"UTP", "UDP", "UMP", "UTP_TX", "Moiety_UTP_poly"},
            "loss_sid": "Moiety_UTP_untracked_loss",
            "gain_sid": "Moiety_UTP_untracked_gain",
            "comp_sid": "UTP_TX",
        },
    }
    fallback_comp = model.getCompartment(0).getId() if model.getNumCompartments() > 0 else "default"

    for info in family_defs.values():
        base_sp = model.getSpecies(info["comp_sid"])
        comp_id = base_sp.getCompartment() if base_sp is not None else fallback_comp
        ensure_species_concentration(model, info["loss_sid"], compartment_id=comp_id, init_conc=0.0)
        ensure_species_concentration(model, info["gain_sid"], compartment_id=comp_id, init_conc=0.0)

    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        for info in family_defs.values():
            members = info["members"]
            delta = 0.0
            for j in range(rxn.getNumProducts()):
                sp = rxn.getProduct(j)
                if sp.getSpecies() in members:
                    delta += float(sp.getStoichiometry())
            for j in range(rxn.getNumReactants()):
                sp = rxn.getReactant(j)
                if sp.getSpecies() in members:
                    delta -= float(sp.getStoichiometry())

            loss_sid = info["loss_sid"]
            gain_sid = info["gain_sid"]
            if delta < -1e-15:
                _set_or_create_product_stoichiometry(rxn, loss_sid, -delta)
                _remove_product_if_exists(rxn, gain_sid)
            elif delta > 1e-15:
                _set_or_create_product_stoichiometry(rxn, gain_sid, delta)
                _remove_product_if_exists(rxn, loss_sid)
            else:
                _remove_product_if_exists(rxn, loss_sid)
                _remove_product_if_exists(rxn, gain_sid)


def split_translation_elongation_three_step(model: "libsbml.Model") -> None:
    old = model.getReaction("rxn_translation_elongation")
    if old is None:
        return

    ribo_mrna = model.getSpecies("Ribosome_mRNA")
    if ribo_mrna is None:
        return

    ensure_parameter(model, "k_tl_init", 0.015)
    ensure_parameter(model, "k_tl_elong", 0.007)
    ensure_parameter(model, "k_tl_term", 0.05)
    ensure_parameter(model, "K_tl_AA", 5.0e-5)
    ensure_parameter(model, "K_tl_ATP", 5.0e-5)
    ensure_parameter(model, "K_tl_GTP", 5.0e-5)
    ensure_parameter(model, "K_tl_eps", 1.0e-18)
    # Optional factorized QSSA gates for translation-factor fast loops.
    # For the migrated fine model we keep the initiation gate deliberately
    # low-dimensional: most legacy RS70S initiation occupancy/load effect is
    # absorbed by the background load field, while only the most essential
    # biochemical availability signal is retained here as a light gate.
    ensure_parameter(model, "K_tl_IF2_tot", 1.0e-8)
    ensure_parameter(model, "K_tl_IF1_tot", 1.0e-8)
    ensure_parameter(model, "K_tl_IF3_tot", 1.0e-8)
    ensure_parameter(model, "K_tl_fMet_tot", 1.0e-8)
    ensure_parameter(model, "K_tl_EFTu_tot", 5.0e-8)
    ensure_parameter(model, "K_tl_EFG_tot", 5.0e-8)
    # Soft floor avoids hard shutdown when one fast sub-pool is transiently low.
    ensure_parameter(model, "phi_tl_init_floor", 0.25)
    ensure_parameter(model, "phi_tl_elong_floor", 0.25)
    ensure_parameter(model, "phi_tl_term_floor", 0.25)

    def sid_or_zero(sid: str) -> str:
        return sid if model.getSpecies(sid) is not None else "0"

    def soft_sat(expr: str, km_pid: str, floor_pid: str) -> str:
        return f"({floor_pid} + (1 - {floor_pid}) * (({expr})/({km_pid} + ({expr}))))"

    tl_mode = os.getenv("PURE_TL_QSSA_MODE", "legacy_migrated_light").strip().lower()
    gtp_sat = "(GTP / (K_tl_GTP + GTP))"
    aa_sat = "(AA / (K_tl_AA + AA))"
    atp_sat = "(ATP / (K_tl_ATP + ATP))"
    if tl_mode in {"factorized", "legacy_migrated", "legacy_migrated_light", "init_migrated"}:
        if1_tot = f"({sid_or_zero('IF1')})"
        if2_tot = (
            f"({sid_or_zero('IF2')} + {sid_or_zero('IF2_GTP')} + "
            f"{sid_or_zero('IF2_GDP')} + {sid_or_zero('IF2_GTP_fMettRNAfMetCAU')})"
        )
        if3_tot = f"({sid_or_zero('IF3')})"
        fmet_tot = f"({sid_or_zero('fMettRNAfMetCAU')} + {sid_or_zero('IF2_GTP_fMettRNAfMetCAU')})"
        eftu_tot = f"({sid_or_zero('EFTu')} + {sid_or_zero('EFTu_GTP')})"
        efg_tot = f"({sid_or_zero('EFG')} + {sid_or_zero('EFG_GTP')})"
        if tl_mode == "factorized":
            init_gate = (
                f"({soft_sat(if1_tot, 'K_tl_IF1_tot', 'phi_tl_init_floor')} * "
                f"{soft_sat(if2_tot, 'K_tl_IF2_tot', 'phi_tl_init_floor')} * "
                f"{soft_sat(if3_tot, 'K_tl_IF3_tot', 'phi_tl_init_floor')} * "
                f"{soft_sat(fmet_tot, 'K_tl_fMet_tot', 'phi_tl_init_floor')})"
            )
        else:
            # Background-load field already absorbs most RS70S occupancy /
            # initiation-assembly effects. Keep only a light IF2-fMet gate here
            # to represent the irreplaceable biochemical trigger for productive
            # initiation without reintroducing the full legacy stiffness.
            init_gate = (
                f"({soft_sat(if2_tot, 'K_tl_IF2_tot', 'phi_tl_init_floor')} * "
                f"{soft_sat(fmet_tot, 'K_tl_fMet_tot', 'phi_tl_init_floor')})"
            )
        elong_gate = (
            f"({soft_sat(eftu_tot, 'K_tl_EFTu_tot', 'phi_tl_elong_floor')} * "
            f"{soft_sat(efg_tot, 'K_tl_EFG_tot', 'phi_tl_elong_floor')})"
        )
        term_gate = soft_sat(efg_tot, "K_tl_EFG_tot", "phi_tl_term_floor")
    else:
        init_gate = "1"
        elong_gate = "1"
        term_gate = "1"

    k_init_eff = f"(default * k_tl_init * {gtp_sat} * {init_gate})"
    k_elong_eff = f"(default * k_tl_elong * {aa_sat} * {atp_sat} * {gtp_sat} * {elong_gate})"
    k_term_eff = f"(default * k_tl_term * {gtp_sat} * {term_gate})"

    tl_pathway_mode = os.getenv("PURE_TL_PATHWAY_MODE", "core_qssa").strip().lower()
    if tl_pathway_mode in {"step_chain", "chain", "net_step"}:
        # Build a per-position net elongation chain:
        # Ribosome_mRNA --(init)--> TLpos_000 --(step)--> ... --(step)--> TLpos_N --(term)-->
        # Ribosome + mRNA_t + PeptGpept
        # Each step aggregates EFTu/EFG/GTP-hydrolysis/translocation microsteps.
        aa_total = 226.0
        atp_total = 226.0
        gtp_total = 452.0
        for j in range(old.getNumReactants()):
            sr = old.getReactant(j)
            sid = sr.getSpecies()
            sto = float(sr.getStoichiometry())
            if sid == "AA":
                aa_total = sto
            elif sid == "ATP":
                atp_total = sto
            elif sid == "GTP":
                gtp_total = sto
        n_steps_env = os.getenv("PURE_TL_CHAIN_STEPS", "").strip()
        group_env = os.getenv("PURE_TL_CHAIN_GROUP_AA", "").strip()
        group_aa = 1.0
        if group_env:
            try:
                group_aa = max(1.0, float(group_env))
            except Exception:
                group_aa = 1.0
        if n_steps_env:
            try:
                n_steps = max(1, int(float(n_steps_env)))
            except Exception:
                n_steps = max(1, int(round(aa_total / group_aa)))
        else:
            # QSSA-style grouped net-elongation: merge `group_aa` micro-cycles into one step.
            n_steps = max(1, int(math.ceil(aa_total / group_aa)))

        chain_scale_env = os.getenv("PURE_TL_CHAIN_SCALE", "").strip()
        chain_scale0 = 1.0
        if chain_scale_env:
            try:
                chain_scale0 = max(1e-6, float(chain_scale_env))
            except Exception:
                chain_scale0 = 1.0

        ensure_parameter(model, "N_tl_steps", float(n_steps))
        ensure_parameter(model, "N_tl_group_aa", float(group_aa))
        ensure_parameter(model, "k_tl_chain_scale", chain_scale0)
        aa_per = aa_total / float(n_steps)
        atp_per = atp_total / float(n_steps)
        gtp_per = gtp_total / float(n_steps)

        chain_kin_mode = os.getenv("PURE_TL_CHAIN_KINETICS", "effective").strip().lower()
        use_mass_action_chain = chain_kin_mode in {"mass_action", "ma", "elementary"}
        k_step_eff = f"({k_elong_eff} * N_tl_steps * k_tl_chain_scale)"
        if use_mass_action_chain:
            # Calibrate mass-action chain k from the effective-chain timescale at
            # the current initial pools, so switching kinetics does not blow up scale.
            sp_aa = model.getSpecies("AA")
            sp_atp = model.getSpecies("ATP")
            sp_gtp = model.getSpecies("GTP")
            aa0 = float(sp_aa.getInitialConcentration()) if sp_aa is not None and sp_aa.isSetInitialConcentration() else 1e-18
            atp0 = float(sp_atp.getInitialConcentration()) if sp_atp is not None and sp_atp.isSetInitialConcentration() else 1e-18
            gtp0 = float(sp_gtp.getInitialConcentration()) if sp_gtp is not None and sp_gtp.isSetInitialConcentration() else 1e-18
            aa0 = max(aa0, 1e-18)
            atp0 = max(atp0, 1e-18)
            gtp0 = max(gtp0, 1e-18)
            p_k_elong = model.getParameter("k_tl_elong")
            k_tl_elong_val = float(p_k_elong.getValue()) if p_k_elong is not None else 0.007
            k_step_target = max(1e-18, k_tl_elong_val * float(n_steps) * chain_scale0)
            denom = 1.0
            if model.getSpecies("AA") is not None and aa_per > 0.0:
                denom *= aa0 ** float(aa_per)
            if model.getSpecies("ATP") is not None and atp_per > 0.0:
                denom *= atp0 ** float(atp_per)
            if model.getSpecies("GTP") is not None and gtp_per > 0.0:
                denom *= gtp0 ** float(gtp_per)
            k_ma_default = k_step_target / max(denom, 1e-300)
            k_ma_default = float(np.clip(k_ma_default, 1e-12, 1e18))
            k_ma_env = os.getenv("PURE_TL_CHAIN_MA_K", "").strip()
            if k_ma_env:
                try:
                    k_ma_default = max(1e-12, float(k_ma_env))
                except Exception:
                    pass
            ensure_parameter(model, "k_tl_chain_ma", k_ma_default)

        def _pow_term(sid: str, expo: float) -> str:
            if model.getSpecies(sid) is None or expo <= 0.0:
                return ""
            if abs(float(expo) - 1.0) < 1e-12:
                return sid
            return f"pow({sid}, {float(expo):.12g})"

        # Remove previous coarse channels and any existing chain reactions.
        remove_reaction_if_exists(model, "rxn_translation_elongation")
        remove_reaction_if_exists(model, "rxn_translation_elong_core")
        remove_reaction_if_exists(model, "rxn_translation_init")
        remove_reaction_if_exists(model, "rxn_translation_term")
        for i in range(model.getNumReactions() - 1, -1, -1):
            rid = model.getReaction(i).getId()
            if rid.startswith("rxn_tl_chain_"):
                model.removeReaction(i)

        comp_id = ribo_mrna.getCompartment()
        chain_sids: List[str] = []
        for i in range(n_steps + 1):
            sid = f"TLchain_pos_{i:03d}"
            ensure_species_concentration(model, sid=sid, compartment_id=comp_id, init_conc=0.0)
            chain_sids.append(sid)

        add_reaction_with_formula(
            model=model,
            rid="rxn_tl_chain_load",
            reactants=[("Ribosome_mRNA", 1.0)],
            products=[(chain_sids[0], 1.0)],
            formula=f"Ribosome_mRNA * ({k_init_eff})",
            reversible=False,
        )

        for i in range(n_steps):
            reactants: List[Tuple[str, float]] = [(chain_sids[i], 1.0)]
            if model.getSpecies("AA") is not None and aa_per > 0.0:
                reactants.append(("AA", aa_per))
            if model.getSpecies("ATP") is not None and atp_per > 0.0:
                reactants.append(("ATP", atp_per))
            if model.getSpecies("GTP") is not None and gtp_per > 0.0:
                reactants.append(("GTP", gtp_per))

            if use_mass_action_chain:
                terms = [chain_sids[i]]
                aa_term = _pow_term("AA", aa_per)
                atp_term = _pow_term("ATP", atp_per)
                gtp_term = _pow_term("GTP", gtp_per)
                if aa_term:
                    terms.append(aa_term)
                if atp_term:
                    terms.append(atp_term)
                if gtp_term:
                    terms.append(gtp_term)
                step_formula = " * ".join(terms + ["k_tl_chain_ma"])
            else:
                step_formula = f"{chain_sids[i]} * ({k_step_eff})"

            add_reaction_with_formula(
                model=model,
                rid=f"rxn_tl_chain_step_{i:03d}",
                reactants=reactants,
                products=[(chain_sids[i + 1], 1.0)],
                formula=step_formula,
                reversible=False,
            )

        term_products: List[Tuple[str, float]] = []
        if model.getSpecies("Ribosome") is not None:
            term_products.append(("Ribosome", 1.0))
        if model.getSpecies("mRNA_t") is not None:
            term_products.append(("mRNA_t", 1.0))
        if model.getSpecies("PeptGpept") is not None:
            term_products.append(("PeptGpept", 1.0))
        add_reaction_with_formula(
            model=model,
            rid="rxn_tl_chain_term",
            reactants=[(chain_sids[-1], 1.0)],
            products=term_products,
            formula=f"{chain_sids[-1]} * ({k_term_eff})",
            reversible=False,
        )
        return

    remove_reaction_if_exists(model, "rxn_translation_elongation")
    remove_reaction_if_exists(model, "rxn_translation_elong_core")
    remove_reaction_if_exists(model, "rxn_translation_init")
    add_reaction_with_formula(
        model=model,
        rid="rxn_translation_elong_core",
        reactants=[("Ribosome_mRNA", 1.0), ("AA", 226.0), ("ATP", 226.0), ("GTP", 452.0)],
        products=[("Ribosome", 1.0), ("mRNA_t", 1.0), ("PeptGpept", 1.0)],
        formula=(
            "Ribosome_mRNA * ("
            f"(({k_init_eff}) * ({k_elong_eff}) * ({k_term_eff})) / "
            f"(K_tl_eps + ({k_init_eff})*({k_elong_eff}) + ({k_init_eff})*({k_term_eff}) + ({k_elong_eff})*({k_term_eff}))"
            ")"
        ),
        reversible=False,
    )
    remove_reaction_if_exists(model, "rxn_translation_term")


def _is_protected_slow_species_id(sid: str) -> bool:
    if sid in {
        "DNA",
        "mRNA",
        "mRNA_t",
        "Ribosome",
        "Ribosome_mRNA",
        "RS70S",
        "deGFP_m",
        "PeptGpept",
    }:
        return True
    return sid.startswith("T7RNAP") or sid.startswith("TLchain_pos")


def coarse_grain_fast_side_modules(model: "libsbml.Model") -> Dict[str, object]:
    """
    Reduce stiffness from side fast loops (mainly aaRS/tRNA cycles) without
    altering the low-frequency TX/TL backbone topology.

    Strategy:
      - detect parameters used *only* by reactions in aaRS/tRNA-rich side modules
      - apply a uniform time-scale factor to those parameters
      - keep protected TX/TL-core-connected reactions untouched
    """
    scale = float(os.getenv("PURE_FAST_SIDE_SCALE", "1.0"))
    min_k = float(os.getenv("PURE_FAST_SIDE_MIN_K", "10.0"))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if not np.isfinite(min_k) or min_k < 0.0:
        min_k = 0.0
    if abs(scale - 1.0) < 1e-15:
        return {
            "enabled": False,
            "scale": 1.0,
            "min_k": min_k,
            "scaled_param_count": 0,
            "scaled_params": [],
        }

    token_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    param_ids = {model.getParameter(i).getId() for i in range(model.getNumParameters())}
    rxn_specs: List[Dict[str, object]] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn is None or (not rxn.isSetKineticLaw()):
            continue
        formula = libsbml.formulaToL3String(rxn.getKineticLaw().getMath())
        tokens = set(token_re.findall(formula))
        used_params = {tok for tok in tokens if tok in param_ids}
        if not used_params:
            continue
        species_ids = _reaction_species_ids_all(rxn)
        has_aars_or_trna = any(
            ("tRNA" in sid) or (("RS" in sid) and sid != "RS70S") for sid in species_ids
        )
        touches_slow = any(_is_protected_slow_species_id(sid) for sid in species_ids)
        rxn_specs.append(
            {
                "rid": rxn.getId(),
                "used_params": used_params,
                "has_aars_or_trna": bool(has_aars_or_trna),
                "touches_slow": bool(touches_slow),
            }
        )

    param_to_rxn: Dict[str, List[Dict[str, object]]] = {}
    for spec in rxn_specs:
        for pid in spec["used_params"]:
            param_to_rxn.setdefault(pid, []).append(spec)

    scaled_params: List[str] = []
    for pid, specs in param_to_rxn.items():
        p = model.getParameter(pid)
        if p is None:
            continue
        kval = float(p.getValue())
        if abs(kval) < float(min_k):
            continue
        # Only scale parameters exclusively used in side fast loops and not
        # touching protected TX/TL core reactions.
        if all(bool(s["has_aars_or_trna"]) and (not bool(s["touches_slow"])) for s in specs):
            p.setValue(float(kval) * float(scale))
            scaled_params.append(pid)

    return {
        "enabled": True,
        "scale": float(scale),
        "min_k": float(min_k),
        "scaled_param_count": int(len(scaled_params)),
        "scaled_params": sorted(set(scaled_params)),
    }


def add_conservation_assignment_rules(model: "libsbml.Model") -> Dict[str, List[str]]:
    """
    Add explicit algebraic conservation constraints for nucleotide families.
    Returns the family->members map actually used in this model.
    """
    used: Dict[str, List[str]] = {}
    for fam, signed_terms in NTP_FAMILY_SIGNED_TERMS.items():
        present_terms = [(sid, coef) for sid, coef in signed_terms if model.getSpecies(sid) is not None]
        if len(present_terms) < 2:
            continue
        total0 = 0.0
        expr_terms: List[str] = []
        label_terms: List[str] = []
        for sid, coef in present_terms:
            sp = model.getSpecies(sid)
            c0 = float(sp.getInitialConcentration()) if sp is not None and sp.isSetInitialConcentration() else 0.0
            total0 += float(coef) * c0
            if abs(float(coef) - 1.0) < 1e-15:
                expr_terms.append(sid)
                label_terms.append(f"+{sid}")
            elif abs(float(coef) + 1.0) < 1e-15:
                expr_terms.append(f"(-{sid})")
                label_terms.append(f"-{sid}")
            else:
                expr_terms.append(f"({float(coef)}*{sid})")
                sign = "+" if float(coef) >= 0.0 else ""
                label_terms.append(f"{sign}{float(coef)}*{sid}")
        total0 = max(abs(total0), 1e-18)
        ensure_parameter(model, f"cons0_{fam}", total0)
        ensure_dynamic_parameter(model, f"cons_err_{fam}", 0.0)
        ensure_dynamic_parameter(model, f"cons_rel_{fam}", 0.0)
        sum_expr = " + ".join(expr_terms)
        set_assignment_rule(model, f"cons_err_{fam}", f"({sum_expr}) - cons0_{fam}")
        # Keep sign here; abs is handled in Python diagnostics/penalty.
        set_assignment_rule(model, f"cons_rel_{fam}", f"(cons_err_{fam})/max(1e-18, abs(cons0_{fam}))")
        used[fam] = label_terms
    return used


def apply_mechanistic_refactor(model: "libsbml.Model") -> None:
    tx_mode = os.getenv("PURE_TX_PATHWAY_MODE", "").strip().lower()
    if tx_mode in {"1nt", "nt1", "full_nt"}:
        swap_transcription_to_1nt_chain(model)
    apply_dual_ntp_pools(model)
    add_ntp_polymerization_trackers(model)
    # Compare safely against the previously successful finest continuous baseline:
    # old runs did not remove the RS* legacy branches by default.
    if _env_flag("PURE_REMOVE_LEGACY_MRNA_RS", "1"):
        remove_legacy_mrna_translation_entry_reactions(model)
    if _env_flag("PURE_REMOVE_LEGACY_RS_INIT", "1"):
        remove_legacy_rs_initiation_microstate_reactions(model)
    # Migration audit 2026-06-10:
    # removing the remaining RS* elong/term + energy-side + assembly branches
    # preserved both batch and continuous deGFP trajectories under the current
    # new-chain parameterization, so these legacy modules are now removed by
    # default and only kept when explicitly disabled for regression checks.
    if _env_flag("PURE_REMOVE_LEGACY_RS_ELONG_TERM", "1"):
        remove_legacy_rs_elongation_termination_microstate_reactions(model)
    if _env_flag("PURE_REMOVE_LEGACY_RS_ENERGY_SIDE", "1"):
        remove_legacy_rs_energy_side_reactions(model)
    if _env_flag("PURE_REMOVE_LEGACY_RS_ASSEMBLY", "1"):
        remove_legacy_rs_assembly_conformation_reactions(model)
    collapse_transcription_triplets_to_net_steps(model)
    parameterize_ribosome_binding(model)
    apply_ribosome_binding_qssa_if_enabled(model)
    split_translation_elongation_three_step(model)
    backgroundize_energy_regen_loops(model)
    coarse_grain_fast_side_modules(model)
    add_nucleotide_family_bookkeeping_trackers(model)
    prune_unreferenced_legacy_species(model)


def add_background_buffer_reactions(
    model: "libsbml.Model",
    sid: str,
    reservoir_sid: str,
    lambda_formula: str,
    theta_var: str,
) -> None:
    rid_out = f"rxn_bg_buf_out_{sanitize_id(sid)}"
    rid_in = f"rxn_bg_buf_in_{sanitize_id(sid)}"
    remove_reaction_if_exists(model, rid_out)
    remove_reaction_if_exists(model, rid_in)

    sp = model.getSpecies(sid)
    if sp is None:
        raise ValueError(f"Missing species for background buffer: {sid}")
    comp_id = sp.getCompartment()
    vol_term = comp_id

    # S -> Res, mass-action drain toward buffer reservoir.
    r_out = model.createReaction()
    r_out.setId(rid_out)
    r_out.setReversible(False)
    rr = r_out.createReactant()
    rr.setSpecies(sid)
    rr.setStoichiometry(1.0)
    pp = r_out.createProduct()
    pp.setSpecies(reservoir_sid)
    pp.setStoichiometry(1.0)
    kl_out = r_out.createKineticLaw()
    ast_out = libsbml.parseL3Formula(f"({lambda_formula})*{sid}*{vol_term}")
    if ast_out is None:
        raise ValueError(f"Failed to parse background-out kinetic law for {sid}")
    kl_out.setMath(ast_out)

    # Res -> S, mass-action refill from the same conserved reservoir pool.
    r_in = model.createReaction()
    r_in.setId(rid_in)
    r_in.setReversible(False)
    rr2 = r_in.createReactant()
    rr2.setSpecies(reservoir_sid)
    rr2.setStoichiometry(1.0)
    pp2 = r_in.createProduct()
    pp2.setSpecies(sid)
    pp2.setStoichiometry(1.0)
    kl_in = r_in.createKineticLaw()
    ast_in = libsbml.parseL3Formula(
        f"({lambda_formula})*{theta_var}*{reservoir_sid}*{vol_term}"
    )
    if ast_in is None:
        raise ValueError(f"Failed to parse background-in kinetic law for {sid}")
    kl_in.setMath(ast_in)


def add_background_family_exchange_reactions(
    model: "libsbml.Model",
    sid: str,
    counter_sid: str,
    phosphate_sid: str | None,
    water_sid: str | None,
    lambda_formula: str,
    theta_var: str,
) -> None:
    rid_out = f"rxn_bg_fx_out_{sanitize_id(sid)}"
    rid_in = f"rxn_bg_fx_in_{sanitize_id(sid)}"
    remove_reaction_if_exists(model, rid_out)
    remove_reaction_if_exists(model, rid_in)

    sp = model.getSpecies(sid)
    cp = model.getSpecies(counter_sid)
    if sp is None or cp is None:
        raise ValueError(f"Missing species for family exchange: {sid}, {counter_sid}")
    comp_id = sp.getCompartment()
    vol_term = comp_id

    # Triphosphate -> diphosphate (+ phosphate, + optional water bookkeeping).
    r_out = model.createReaction()
    r_out.setId(rid_out)
    r_out.setReversible(False)
    rr = r_out.createReactant()
    rr.setSpecies(sid)
    rr.setStoichiometry(1.0)
    if water_sid is not None and model.getSpecies(water_sid) is not None:
        rrw = r_out.createReactant()
        rrw.setSpecies(water_sid)
        rrw.setStoichiometry(1.0)
    pp = r_out.createProduct()
    pp.setSpecies(counter_sid)
    pp.setStoichiometry(1.0)
    if phosphate_sid is not None and model.getSpecies(phosphate_sid) is not None:
        ppp = r_out.createProduct()
        ppp.setSpecies(phosphate_sid)
        ppp.setStoichiometry(1.0)
    kl_out = r_out.createKineticLaw()
    ast_out = libsbml.parseL3Formula(f"({lambda_formula})*{sid}*{vol_term}")
    if ast_out is None:
        raise ValueError(f"Failed to parse family-out kinetic law for {sid}")
    kl_out.setMath(ast_out)

    # Diphosphate (+ phosphate) -> triphosphate (+ optional water bookkeeping).
    r_in = model.createReaction()
    r_in.setId(rid_in)
    r_in.setReversible(False)
    rr2 = r_in.createReactant()
    rr2.setSpecies(counter_sid)
    rr2.setStoichiometry(1.0)
    if phosphate_sid is not None and model.getSpecies(phosphate_sid) is not None:
        rrp = r_in.createReactant()
        rrp.setSpecies(phosphate_sid)
        rrp.setStoichiometry(1.0)
    pp2 = r_in.createProduct()
    pp2.setSpecies(sid)
    pp2.setStoichiometry(1.0)
    if water_sid is not None and model.getSpecies(water_sid) is not None:
        ppw = r_in.createProduct()
        ppw.setSpecies(water_sid)
        ppw.setStoichiometry(1.0)
    kl_in = r_in.createKineticLaw()
    ast_in = libsbml.parseL3Formula(
        f"({lambda_formula})*{theta_var}*{counter_sid}*{vol_term}"
    )
    if ast_in is None:
        raise ValueError(f"Failed to parse family-in kinetic law for {sid}")
    kl_in.setMath(ast_in)


def add_exchange_reactions(
    model: "libsbml.Model",
    sid: str,
    gamma_pid: str,
    feed_pid: str,
) -> None:
    rid_in = f"rxn_ex_in_{sanitize_id(sid)}"
    rid_out = f"rxn_ex_out_{sanitize_id(sid)}"
    remove_reaction_if_exists(model, rid_in)
    remove_reaction_if_exists(model, rid_out)

    # Inflow: -> S, rate = gamma * S_feed
    rin = model.createReaction()
    rin.setId(rid_in)
    rin.setReversible(False)
    p = rin.createProduct()
    p.setSpecies(sid)
    p.setStoichiometry(1.0)
    sp = model.getSpecies(sid)
    comp_id = sp.getCompartment() if sp is not None else (model.getCompartment(0).getId() if model.getNumCompartments() > 0 else "default")
    # KineticLaw units in SBML are amount/time; for concentration-state species we
    # multiply by compartment volume to realize dC/dt = gamma*(C_feed - C).
    vol_term = comp_id
    kl_in = rin.createKineticLaw()
    ast_in = libsbml.parseL3Formula(f"{gamma_pid}*{feed_pid}*{vol_term}")
    if ast_in is None:
        raise ValueError(f"Failed to parse inflow kinetic law for {sid}")
    kl_in.setMath(ast_in)

    # Outflow: S -> , rate = gamma * S
    rout = model.createReaction()
    rout.setId(rid_out)
    rout.setReversible(False)
    r = rout.createReactant()
    r.setSpecies(sid)
    r.setStoichiometry(1.0)
    kl_out = rout.createKineticLaw()
    ast_out = libsbml.parseL3Formula(f"{gamma_pid}*{sid}*{vol_term}")
    if ast_out is None:
        raise ValueError(f"Failed to parse outflow kinetic law for {sid}")
    kl_out.setMath(ast_out)


def load_initial_uM_table(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path, header=None, index_col=0)
    ic = {str(k): float(v) for k, v in df[1].to_dict().items()}
    # Same validated overrides used by the successful baseline script.
    ic["Ribosome"] = 3.0e-6 * 1e6
    ic["DNA"] = 5.0e-9 * 1e6
    ic["T7RNAP"] = 1.0e-6 * 1e6
    ic["ATP"] = 3.75e-3 * 1e6
    ic["GTP"] = 2.5e-3 * 1e6
    return ic


def patch_initial_conditions(model: "libsbml.Model", ic_uM: Dict[str, float]) -> int:
    patched = 0
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        name = sp.getName()
        target_uM = None
        if sid in ic_uM:
            target_uM = float(ic_uM[sid])
        elif name in ic_uM:
            target_uM = float(ic_uM[name])
        elif sid == "RS70S" and "Ribosome" in ic_uM:
            target_uM = float(ic_uM["Ribosome"])
        if target_uM is None:
            continue

        sp.setInitialConcentration(target_uM * 1.0e-6)  # uM -> M
        if sp.isSetInitialAmount():
            sp.unsetInitialAmount()
        patched += 1
    return patched


def get_model_counts(model: "libsbml.Model") -> Dict[str, int]:
    return {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
    }


def build_patched_full_sbml() -> Tuple[str, Dict[str, float], Dict[str, int], List[str]]:
    doc = libsbml.readSBMLFromFile(str(BASE_SBML))
    model = doc.getModel()
    if model is None:
        raise RuntimeError(f"Failed to parse model from {BASE_SBML}")

    ic_uM = load_initial_uM_table(INIT_CSV)
    patch_initial_conditions(model, ic_uM)
    apply_mechanistic_refactor(model)
    counts = get_model_counts(model)

    present_hf = [sid for sid in HF_SPECIES if model.getSpecies(sid) is not None]
    return libsbml.writeSBMLToString(doc), ic_uM, counts, present_hf


def configure_rr(rr) -> None:
    rr.setIntegrator("cvode")
    integ = rr.getIntegrator()
    integ.setValue("variable_step_size", True)
    integ.setValue("stiff", True)
    integ.setValue("absolute_tolerance", 1e-12)
    integ.setValue("relative_tolerance", 1e-6)
    # Let CVODE pick its own startup/min step under stiff regimes; fixed tiny
    # startup/min-step constraints can trigger t=0 error-test failures.
    # Use a moderate cap so numerically bad candidates fail fast during fitting.
    max_steps = int(os.getenv("PURE_MAX_STEPS", "120000"))
    for key, value in (("maximum_num_steps", max_steps),):
        try:
            integ.setValue(key, value)
        except Exception:
            pass


def simulate_reference_batch(rr, present_hf: List[str], t_end_s: float = 8 * 3600, n_points: int = 600) -> pd.DataFrame:
    sels = ["time", "[deGFP_m]", "[mRNA_t]"] + [f"[{sid}]" for sid in present_hf]
    res = rr.simulate(0, t_end_s, n_points, sels)
    return pd.DataFrame(res, columns=res.colnames)


def fit_single_trace_exp(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    y0 = float(y[0])
    y_end = float(y[-1])
    trend = "down" if y_end < y0 else "up"

    if trend == "down":
        lo_inf = 0.0
        hi_inf = max(y0 * 1.05, 1e-18)
        x0_inf = float(np.clip(y_end, lo_inf, hi_inf))
    else:
        lo_inf = min(y0, float(np.min(y)))
        hi_inf = max(float(np.max(y)) * 2.0, y0 + 1e-18)
        x0_inf = float(np.clip(y_end, lo_inf, hi_inf))

    x0 = np.array([x0_inf, 1e-4], dtype=float)  # [y_inf, k]
    lb = np.array([lo_inf, 1e-9], dtype=float)
    ub = np.array([hi_inf, 1e-1], dtype=float)

    eps_rel = max(1e-12, 0.02 * abs(y0))
    # Use only concentration-state weighting; avoid explicit time landmarks.
    phase_w = np.ones_like(t)

    def residual(theta: np.ndarray) -> np.ndarray:
        y_inf, k = float(theta[0]), float(theta[1])
        y_hat = y_inf + (y0 - y_inf) * np.exp(-k * t)
        return np.sqrt(phase_w) * (y_hat - y) / (np.abs(y) + eps_rel)

    res_single = least_squares(residual, x0=x0, bounds=(lb, ub), max_nfev=300)
    y_inf_s, k_s = float(res_single.x[0]), float(res_single.x[1])
    y_hat_s = y_inf_s + (y0 - y_inf_s) * np.exp(-k_s * t)
    rmse_single = float(np.sqrt(np.mean((y_hat_s - y) ** 2)))

    # Two-timescale profile to capture fast equilibration + slow depletion phases.
    x0_bi = np.array([y_inf_s, 0.7, min(5e-3, max(5e-5, 10.0 * k_s)), max(5e-8, 0.2 * k_s)], dtype=float)
    lb_bi = np.array([lo_inf, 0.0, 1e-7, 1e-9], dtype=float)   # [y_inf, a_fast, k_fast, k_slow]
    ub_bi = np.array([hi_inf, 1.0, 1e-1, 1e-2], dtype=float)
    x0_bi = np.clip(x0_bi, lb_bi + 1e-15, ub_bi - 1e-15)

    def residual_bi(theta: np.ndarray) -> np.ndarray:
        y_inf, a_fast, k_fast, k_slow = map(float, theta)
        y_hat = y_inf + (y0 - y_inf) * (
            a_fast * np.exp(-k_fast * t) + (1.0 - a_fast) * np.exp(-k_slow * t)
        )
        return np.sqrt(phase_w) * (y_hat - y) / (np.abs(y) + eps_rel)

    res_bi = least_squares(residual_bi, x0=x0_bi, bounds=(lb_bi, ub_bi), max_nfev=500)
    y_inf_b, a_fast_b, k1_b, k2_b = map(float, res_bi.x)
    # Keep explicit fast/slow ordering for SBML readability.
    if k1_b >= k2_b:
        k_fast_b, k_slow_b = k1_b, k2_b
        a_fast_ord = a_fast_b
    else:
        k_fast_b, k_slow_b = k2_b, k1_b
        a_fast_ord = 1.0 - a_fast_b
    y_hat_b = y_inf_b + (y0 - y_inf_b) * (
        a_fast_ord * np.exp(-k_fast_b * t) + (1.0 - a_fast_ord) * np.exp(-k_slow_b * t)
    )
    rmse_bi = float(np.sqrt(np.mean((y_hat_b - y) ** 2)))

    use_bi = rmse_bi < 0.98 * rmse_single
    if use_bi:
        return {
            "profile": "bi_exp",
            "y0_M": y0,
            "yinf_M": y_inf_b,
            "a_fast": a_fast_ord,
            "k_fast_s_inv": k_fast_b,
            "k_slow_s_inv": k_slow_b,
            "k_s_inv": a_fast_ord * k_fast_b + (1.0 - a_fast_ord) * k_slow_b,
            "trend": trend,
            "rmse_M": rmse_bi,
            "rmse_single_M": rmse_single,
        }

    return {
        "profile": "single_exp",
        "y0_M": y0,
        "yinf_M": y_inf_s,
        "k_s_inv": k_s,
        "trend": trend,
        "rmse_M": rmse_single,
    }


def fit_background_from_reference(df_ref: pd.DataFrame, present_hf: List[str]) -> Dict[str, Dict[str, float]]:
    t = df_ref["time"].to_numpy(dtype=float)
    fitted: Dict[str, Dict[str, float]] = {}
    for sid in present_hf:
        col = f"[{sid}]"
        if col not in df_ref.columns:
            continue
        y = df_ref[col].to_numpy(dtype=float)
        fitted[sid] = fit_single_trace_exp(t=t, y=y)
    return fitted


def fit_keff_map_from_flux_ratio(df_ref: pd.DataFrame, df_bg: pd.DataFrame) -> Dict[str, float]:
    t_ref = df_ref["time"].to_numpy(dtype=float)
    p_ref = df_ref["[deGFP_m]"].to_numpy(dtype=float)
    t_bg = df_bg["time"].to_numpy(dtype=float)
    p_bg = df_bg["[deGFP_m]"].to_numpy(dtype=float)

    # Align baseline background trajectory to reference times.
    p_bg_itp = np.interp(t_ref, t_bg, p_bg)
    v_ref = np.gradient(p_ref, t_ref)
    v_bg = np.gradient(p_bg_itp, t_ref)

    def state_col(df: pd.DataFrame, sid: str, default: float) -> np.ndarray:
        col = f"[{sid}]"
        if col not in df.columns:
            return np.full_like(t_ref, float(default), dtype=float)
        return np.maximum(df[col].to_numpy(dtype=float), 1e-18)

    atp = state_col(df_ref, "ATP", 1e-3)
    gtp = state_col(df_ref, "GTP", 1e-4)
    po4 = state_col(df_ref, "PO4", 1e-8)
    ppi = state_col(df_ref, "PPi", 1e-12)

    eps = max(1e-14, 0.02 * np.max(np.abs(v_bg)))
    raw_ratio = np.where(np.abs(v_bg) > eps, v_ref / v_bg, np.nan)
    ratio = np.clip(raw_ratio, 1e-3, 20.0)

    valid = np.isfinite(ratio)
    if np.count_nonzero(valid) < 20:
        # Conservative fallback: keep correction near neutral.
        return {
            "r_inf": 0.1,
            "r_0": 1.0,
            "K_ATP_M": max(float(np.nanmedian(atp)), 1e-7),
            "K_GTP_M": max(float(np.nanmedian(gtp)), 1e-8),
            "KI_PO4_M": max(float(np.nanmedian(po4)), 1e-7),
            "KI_PPi_M": max(float(np.nanmedian(ppi)), 1e-12),
            "fit_rmse": float("nan"),
        }

    ratio_v = ratio[valid]
    atp_v = atp[valid]
    gtp_v = gtp[valid]
    po4_v = po4[valid]
    ppi_v = ppi[valid]
    # State-based weighting: emphasize low-energy/high-waste states
    # without using absolute-time triggers.
    atp_ref = max(float(np.median(atp_v)), 1e-12)
    gtp_ref = max(float(np.median(gtp_v)), 1e-12)
    po4_ref = max(float(np.median(po4_v)), 1e-12)
    ppi_ref = max(float(np.median(ppi_v)), 1e-15)
    energy_avail = (atp_v / (atp_v + atp_ref)) * (gtp_v / (gtp_v + gtp_ref))
    waste_load = (po4_v / (po4_v + po4_ref)) + (ppi_v / (ppi_v + ppi_ref))
    w = 1.0 + 1.5 * (1.0 - energy_avail) + 0.5 * waste_load

    def model(theta: np.ndarray) -> np.ndarray:
        r_inf, r0, k_atp, k_gtp, ki_po4, ki_ppi = map(float, theta)
        drive_atp = atp_v / (atp_v + k_atp)
        drive_gtp = gtp_v / (gtp_v + k_gtp)
        inhib_po4 = 1.0 / (1.0 + po4_v / ki_po4)
        inhib_ppi = 1.0 / (1.0 + ppi_v / ki_ppi)
        return r_inf + (r0 - r_inf) * drive_atp * drive_gtp * inhib_po4 * inhib_ppi

    def resid(theta: np.ndarray) -> np.ndarray:
        pred = np.clip(model(theta), 1e-4, 20.0)
        target = np.clip(ratio_v, 1e-4, 20.0)
        return np.sqrt(w) * (np.log(pred) - np.log(target))

    x0 = np.array(
        [
            0.05,
            float(np.clip(np.percentile(ratio_v, 90), 0.2, 20.0)),
            max(float(np.median(atp_v)), 1e-7),
            max(float(np.median(gtp_v)), 1e-8),
            max(float(np.median(po4_v)), 1e-7),
            max(float(np.median(ppi_v)), 1e-12),
        ],
        dtype=float,
    )
    lb = np.array([0.01, 0.1, 1e-7, 1e-9, 1e-7, 1e-12], dtype=float)
    ub = np.array([2.0, 20.0, 1e-2, 1e-2, 5e-2, 1e-4], dtype=float)
    res = least_squares(resid, x0=x0, bounds=(lb, ub), max_nfev=600)

    r_inf, r0, k_atp, k_gtp, ki_po4, ki_ppi = map(float, res.x)
    pred = np.clip(model(res.x), 1e-4, 20.0)
    return {
        "r_inf": r_inf,
        "r_0": r0,
        "K_ATP_M": k_atp,
        "K_GTP_M": k_gtp,
        "KI_PO4_M": ki_po4,
        "KI_PPi_M": ki_ppi,
        "fit_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(np.clip(ratio_v, 1e-4, 20.0))) ** 2))),
    }


def build_background_sbml(
    mode: str,
    fitted_hf: Dict[str, Dict[str, float]],
    ic_uM: Dict[str, float],
    gamma_h: float,
    feed_multiplier: float,
    product_outflow_h: float | None = None,
    keff_map: Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, int], Dict[str, object]]:
    if mode not in {"batch", "continuous"}:
        raise ValueError("mode must be 'batch' or 'continuous'")

    doc = libsbml.readSBMLFromFile(str(BASE_SBML))
    model = doc.getModel()
    if model is None:
        raise RuntimeError(f"Failed to parse model from {BASE_SBML}")

    patch_initial_conditions(model, ic_uM)
    apply_mechanistic_refactor(model)
    conservation_families: Dict[str, List[str]] = {}

    ensure_parameter(model, "bg_k_scale", 1.0)
    # Minimal effective-availability corrections for coarse-grained energy fields.
    ensure_parameter(model, "bg_ATP_inf_scale", 1.0)
    ensure_parameter(model, "bg_GTP_inf_scale", 1.0)
    ensure_parameter(model, "bg_gamma", float(gamma_h) / 3600.0)
    if mode == "continuous":
        gamma_prod_h = float(gamma_h) if product_outflow_h is None else float(product_outflow_h)
        ensure_parameter(model, "bg_gamma_prod", gamma_prod_h / 3600.0)

    # State-driven load coordinate for high-frequency background fields.
    # All terms are dimensionless except explicit in/out rates (1/s), so the
    # resulting load clock is unit-consistent.
    ensure_parameter(model, "bg_load_K_M2", 3.0e-13)  # ~O(mRNA*Ribosome), in M^2
    ensure_parameter(model, "bg_load_gain", 1.0)  # dimensionless instantaneous weight
    ensure_parameter(model, "bg_load_mem_in", 1.0e-4)  # 1/s
    ensure_parameter(model, "bg_load_mem_out", 5.0e-5)  # 1/s, prevents unbounded memory drift
    ensure_parameter(model, "bg_load_mem_gain", 1.0)  # dimensionless memory weight
    ensure_parameter(model, "bg_load_clock_gain", 1.0)  # dimensionless multiplier
    if model.getSpecies("Ribosome") is not None and model.getSpecies("mRNA_t") is not None:
        load_expr = "(mRNA_t*Ribosome)/(bg_load_K_M2 + mRNA_t*Ribosome)"
    elif model.getSpecies("RS70S") is not None and model.getSpecies("mRNA_t") is not None:
        # Backward-compatible fallback only if the new mainline ribosome pool is absent.
        load_expr = "(mRNA_t*RS70S)/(bg_load_K_M2 + mRNA_t*RS70S)"
    else:
        # Conservative fallback if expected load proxy species are absent.
        ensure_parameter(model, "bg_load_K_M", 1.0e-8)
        load_expr = "mRNA_t/(bg_load_K_M + mRNA_t)"
    ensure_dynamic_parameter(model, "bgLoadMem", 0.0)
    set_rate_rule(
        model,
        "bgLoadMem",
        f"(bg_load_mem_in*({load_expr})) - (bg_load_mem_out*bgLoadMem)",
    )
    load_drive = f"(1 + (bg_load_gain*({load_expr})) + (bg_load_mem_gain*bgLoadMem))"
    ensure_dynamic_parameter(model, "bg_load_clock", 1.0)
    set_assignment_rule(
        model,
        "bg_load_clock",
        f"(bg_load_clock_gain*{load_drive})",
    )

    # State-dependent effective drive field for translation elongation channel.
    if keff_map is not None:
        ensure_dynamic_parameter(model, "bg_keff", 1.0)
        ensure_parameter(model, "bg_keff_scale", 1.0)
        ensure_parameter(model, "bg_keff_inf_scale", 1.0)
        ensure_parameter(model, "bg_keff_KATP_scale", 1.0)
        ensure_parameter(model, "bg_keff_KGTP_scale", 1.0)
        ensure_parameter(model, "bg_keff_KIPO4_scale", 1.0)
        ensure_parameter(model, "bg_keff_KIPPi_scale", 1.0)
        ensure_parameter(model, "bg_keff_r_inf", float(keff_map["r_inf"]))
        ensure_parameter(model, "bg_keff_r0", float(keff_map["r_0"]))
        ensure_parameter(model, "bg_keff_K_ATP", float(keff_map["K_ATP_M"]))
        ensure_parameter(model, "bg_keff_K_GTP", float(keff_map["K_GTP_M"]))
        ensure_parameter(model, "bg_keff_KI_PO4", float(keff_map["KI_PO4_M"]))
        ensure_parameter(model, "bg_keff_KI_PPi", float(keff_map["KI_PPi_M"]))
        set_assignment_rule(
            model,
            "bg_keff",
            "(bg_keff_inf_scale*bg_keff_r_inf) + "
            "(bg_keff_r0 - (bg_keff_inf_scale*bg_keff_r_inf)) * "
            "((ATP/(ATP + (bg_keff_K_ATP*bg_keff_KATP_scale))) * "
            "(GTP/(GTP + (bg_keff_K_GTP*bg_keff_KGTP_scale))) * "
            "(1/(1 + PO4/(bg_keff_KI_PO4*bg_keff_KIPO4_scale))) * "
            "(1/(1 + PPi/(bg_keff_KI_PPi*bg_keff_KIPPi_scale))))",
        )

        rxn = model.getReaction("rxn_translation_elong_core")
        if rxn is None:
            rxn = model.getReaction("rxn_translation_elongation")
        if rxn is not None and rxn.isSetKineticLaw():
            kl = rxn.getKineticLaw()
            base_formula = libsbml.formulaToL3String(kl.getMath())
            new_formula = f"({base_formula}) * (bg_keff_scale * bg_keff)"
            ast = libsbml.parseL3Formula(new_formula)
            if ast is not None:
                kl.setMath(ast)

    hf_meta: Dict[str, object] = {
        "consumed_like": [],
        "waste_like": [],
        "continuous_feed_species": [],
        "continuous_washout_species": [],
        "continuous_product_outflow_species": [],
        "continuous_feed_uM": {},
        "continuous_product_outflow_h_inv": (
            float(gamma_h) if product_outflow_h is None else float(product_outflow_h)
        ),
        "conservation_families": conservation_families,
    }

    for sid, fp in fitted_hf.items():
        sp = model.getSpecies(sid)
        if sp is None:
            continue

        # Keep species dynamic (not boundary-clamped) so low-frequency network
        # and background closure remain coupled through mass-action reactions.
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
        remove_rules_for_variable(model, sid)

        y0 = float(fp["y0_M"])
        yinf = float(fp["yinf_M"])
        k = max(1e-12, float(fp["k_s_inv"]))
        trend = str(fp["trend"])
        profile = str(fp.get("profile", "single_exp"))

        ensure_parameter(model, f"bg_{sid}_0", y0)
        ensure_parameter(model, f"bg_{sid}_inf", yinf)
        ensure_parameter(model, f"bg_{sid}_k", k)
        if profile == "bi_exp":
            ensure_parameter(model, f"bg_{sid}_a", float(np.clip(float(fp["a_fast"]), 0.0, 1.0)))
            ensure_parameter(model, f"bg_{sid}_k1", max(1e-12, float(fp["k_fast_s_inv"])))
            ensure_parameter(model, f"bg_{sid}_k2", max(1e-12, float(fp["k_slow_s_inv"])))

        if trend == "down":
            feed_uM = float(ic_uM.get(sid, y0 * 1e6)) * float(feed_multiplier)
            hf_meta["consumed_like"].append(sid)
        else:
            feed_uM = 0.0
            hf_meta["waste_like"].append(sid)
        ensure_parameter(model, f"bg_{sid}_feed", feed_uM * 1e-6)  # uM -> M

        inf_expr = f"bg_{sid}_inf"
        if sid == "ATP":
            inf_expr = "(bg_ATP_inf_scale*bg_ATP_inf)"
        elif sid == "GTP":
            inf_expr = "(bg_GTP_inf_scale*bg_GTP_inf)"
        if profile == "bi_exp":
            keff = f"((bg_{sid}_a*bg_{sid}_k1 + (1 - bg_{sid}_a)*bg_{sid}_k2)*bg_k_scale)"
        else:
            keff = f"(bg_{sid}_k*bg_k_scale)"

        # For ATP/GTP, use biologically meaningful family exchange instead of
        # external hidden reservoirs: ATP <-> ADP, GTP <-> GDP.
        family_counter = None
        if sid == "ATP" and model.getSpecies("ADP") is not None:
            family_counter = "ADP"
        elif sid == "GTP" and model.getSpecies("GDP") is not None:
            family_counter = "GDP"

        if family_counter is not None:
            cp = model.getSpecies(family_counter)
            cp0 = (
                float(cp.getInitialConcentration())
                if (cp is not None and cp.isSetInitialConcentration())
                else 1e-12
            )
            cp0 = max(cp0, 1e-12)
            theta0 = max(yinf / cp0, 1e-12)
            ensure_parameter(model, f"bg_{sid}_theta0", theta0)
            ensure_dynamic_parameter(model, f"bg_{sid}_theta", theta0)
            if sid == "ATP":
                theta_expr = f"max(1e-12, bg_ATP_inf_scale*bg_{sid}_theta0)"
            elif sid == "GTP":
                theta_expr = f"max(1e-12, bg_GTP_inf_scale*bg_{sid}_theta0)"
            else:
                theta_expr = f"max(1e-12, bg_{sid}_theta0)"
            set_assignment_rule(model, f"bg_{sid}_theta", theta_expr)
            lam_expr = f"(({keff})*bg_load_clock) / (1 + bg_{sid}_theta)"
            # Optional explicit hydrolysis bookkeeping for family exchange:
            # ATP/GTP + H2O <-> ADP/GDP + PO4
            # Set PURE_BG_FAMILY_EXCHANGE_BOOKKEEP=0 to recover the legacy
            # simplified exchange ATP<->ADP / GTP<->GDP used in older runs.
            use_exchange_bookkeep = str(os.getenv("PURE_BG_FAMILY_EXCHANGE_BOOKKEEP", "1")).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            phosphate_sid = None
            water_sid = None
            if use_exchange_bookkeep:
                phosphate_sid = "PO4" if model.getSpecies("PO4") is not None else None
                if phosphate_sid is not None:
                    water_sid = "H2O_bg"
                    ensure_species_concentration(
                        model,
                        sid=water_sid,
                        compartment_id=sp.getCompartment(),
                        init_conc=float(os.getenv("PURE_BG_WATER_INIT_M", "55.0")),
                    )
            add_background_family_exchange_reactions(
                model=model,
                sid=sid,
                counter_sid=family_counter,
                phosphate_sid=phosphate_sid,
                water_sid=water_sid,
                lambda_formula=lam_expr,
                theta_var=f"bg_{sid}_theta",
            )
        else:
            # Conservative hidden reservoir: sid <-> bgRes_sid.
            # This keeps a stoichiometric buffer pool for each backgrounded species.
            pool_total = max(max(y0, yinf) * 4.0 + max(y0, yinf, 1e-18), y0 + 1e-18)
            res0 = max(pool_total - y0, 1e-18)
            res_sid = f"bgRes_{sanitize_id(sid)}"
            ensure_species_concentration(
                model,
                sid=res_sid,
                compartment_id=sp.getCompartment(),
                init_conc=res0,
            )
            res_sp = model.getSpecies(res_sid)
            if res_sp is not None:
                res_sp.setBoundaryCondition(False)
                res_sp.setConstant(False)

            ensure_parameter(model, f"bg_{sid}_total", pool_total)
            ensure_dynamic_parameter(model, f"bg_{sid}_theta", max(yinf / max(pool_total - yinf, 1e-18), 0.0))
            set_assignment_rule(
                model,
                f"bg_{sid}_theta",
                f"max(0, ({inf_expr}) / max(1e-18, (bg_{sid}_total - ({inf_expr}))))",
            )
            lam_expr = f"(({keff})*bg_load_clock) / (1 + bg_{sid}_theta)"
            add_background_buffer_reactions(
                model=model,
                sid=sid,
                reservoir_sid=res_sid,
                lambda_formula=lam_expr,
                theta_var=f"bg_{sid}_theta",
            )

    if mode == "continuous":
        dialyzable_species, feed_species, exchange_policy_meta = select_continuous_exchange_species(model)
        hf_meta["continuous_exchange_policy"] = exchange_policy_meta
        # Add reactor-level exchange reactions for dialyzable small molecules
        # (including backgrounded HF species), preserving mass-action ODE structure.
        hf_set = set(fitted_hf.keys())
        for sid in dialyzable_species:
            sp = model.getSpecies(sid)
            if sp is None:
                continue
            feed_uM = 0.0
            if sid in feed_species and sid in ic_uM:
                feed_uM = float(ic_uM[sid]) * float(feed_multiplier)
                hf_meta["continuous_feed_species"].append(sid)
            elif sid in feed_species:
                # Fallback to model initial concentration when CSV lacks this key
                # (e.g., aggregate AA pool in this model).
                init_M = float(sp.getInitialConcentration()) if sp.isSetInitialConcentration() else 0.0
                feed_uM = max(0.0, init_M * 1.0e6) * float(feed_multiplier)
                hf_meta["continuous_feed_species"].append(sid)
            else:
                hf_meta["continuous_washout_species"].append(sid)
            if sid in hf_meta["continuous_feed_species"]:
                hf_meta["continuous_feed_uM"][sid] = float(feed_uM)
            feed_pid = f"bg_feed_{sanitize_id(sid)}"
            ensure_parameter(model, feed_pid, feed_uM * 1.0e-6)  # uM -> M
            add_exchange_reactions(
                model=model,
                sid=sid,
                gamma_pid="bg_gamma",
                feed_pid=feed_pid,
            )

        # Add explicit outflow channel for selected macro products.
        for sid in DEFAULT_CONTINUOUS_PRODUCT_OUTFLOW:
            sp = model.getSpecies(sid)
            if sp is None or sid in hf_set:
                continue
            feed_pid = f"bg_feed_{sanitize_id(sid)}"
            ensure_parameter(model, feed_pid, 0.0)
            add_exchange_reactions(
                model=model,
                sid=sid,
                gamma_pid="bg_gamma_prod",
                feed_pid=feed_pid,
            )
            hf_meta["continuous_product_outflow_species"].append(sid)

    # Build algebraic conservation rules after all background species/reactions
    # are materialized (e.g., bgRes_CTP/UTP), so the check is structurally closed.
    conservation_families = add_conservation_assignment_rules(model)
    hf_meta["conservation_families"] = conservation_families

    bg_count = len(fitted_hf)
    bg_res_count = len([sid for sid in fitted_hf.keys() if model.getSpecies(f"bgRes_{sanitize_id(sid)}") is not None])
    boundary_count = sum(1 for i in range(model.getNumSpecies()) if model.getSpecies(i).getBoundaryCondition())
    counts_out = {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
        "background_species_count": bg_count,
        "background_reservoir_species_count": bg_res_count,
        "boundary_species_count": boundary_count,
        "dynamic_species_count": model.getNumSpecies() - boundary_count,
    }
    return libsbml.writeSBMLToString(doc), counts_out, hf_meta


def simulate_degfp(rr, t_end_s: float, n_points: int, reset_to_origin: bool = True) -> pd.DataFrame:
    if reset_to_origin:
        rr.resetToOrigin()
    sel = ["time", "[deGFP_m]", "[mRNA_t]", "[ATP]", "[ADP]", "[GTP]", "[GDP]", "[CP]", "[PO4]", "[PPi]"]
    try:
        gp = list(rr.getGlobalParameterIds())
    except Exception:
        gp = []
    for pid in gp:
        if str(pid).startswith("cons_err_") or str(pid).startswith("cons_rel_"):
            sel.append(str(pid))
    sel = list(dict.fromkeys(sel))
    res = rr.simulate(0, t_end_s, n_points, sel)
    return pd.DataFrame(res, columns=res.colnames)


def batch_metrics(df: pd.DataFrame) -> Dict[str, float]:
    t_h = df["time"].to_numpy() / 3600.0
    p_uM = df["[deGFP_m]"].to_numpy() * 1e6
    p2 = float(np.interp(2.0, t_h, p_uM))
    p4 = float(np.interp(4.0, t_h, p_uM))
    p8 = float(np.interp(8.0, t_h, p_uM))
    peak = float(np.max(p_uM)) if p_uM.size else 0.0

    if p_uM.size >= 12:
        # Stricter plateau detector:
        # 1) local slope is small in absolute unit (uM/h),
        # 2) signal already sits in a narrow neighborhood around terminal level.
        # This avoids premature plateau flags from overly broad relative-slope thresholds.
        slope = np.gradient(p_uM, t_h)
        final_uM = float(p_uM[-1])
        slope_thr_abs = max(1e-6, float(os.getenv("PURE_PLATEAU_SLOPE_THR_UMPH", "5e-3")))
        band_rel = max(0.0, float(os.getenv("PURE_PLATEAU_BAND_REL", "5e-3")))
        band_abs = max(0.0, float(os.getenv("PURE_PLATEAU_BAND_ABS_UM", "0.0")))
        min_start_h = max(0.0, float(os.getenv("PURE_PLATEAU_MIN_START_H", "0.5")))
        band = max(band_abs, band_rel * max(abs(final_uM), 1e-12))

        stable = (np.abs(slope) <= slope_thr_abs) & (np.abs(p_uM - final_uM) <= band)
        run_len = max(6, int(0.04 * p_uM.size))
        plateau_start_idx = None
        for i in range(0, p_uM.size - run_len + 1):
            if t_h[i] < min_start_h:
                continue
            if np.all(stable[i : i + run_len]):
                plateau_start_idx = i
                break
        if plateau_start_idx is None:
            plateau_start_idx = int(0.75 * p_uM.size)
    else:
        plateau_start_idx = int(0.75 * p_uM.size) if p_uM.size else 0

    post = p_uM[plateau_start_idx:] if plateau_start_idx < p_uM.size else p_uM[-1:]
    post_t = t_h[plateau_start_idx:] if plateau_start_idx < t_h.size else t_h[-1:]
    plateau_level = float(np.median(post)) if post.size else float(p_uM[-1] if p_uM.size else 0.0)
    drift = float(np.max(np.abs(post - plateau_level))) if post.size else 0.0
    if post.size >= 5:
        plateau_slope = float(np.polyfit(post_t, post, 1)[0])
    else:
        plateau_slope = 0.0

    return {
        "protein_2h_uM": p2,
        "protein_4h_uM": p4,
        "protein_8h_uM": p8,
        "protein_peak_uM": peak,
        "plateau_level_uM": plateau_level,
        "plateau_start_h": float(t_h[plateau_start_idx]) if t_h.size else 0.0,
        "plateau_max_abs_drift_uM": drift,
        "plateau_abs_slope_uM_per_h": float(abs(plateau_slope)),
    }


def append_cumulative_degfp_equivalent(df: pd.DataFrame, gamma_prod_h_inv: float) -> pd.DataFrame:
    df_out = df.copy()
    if "time" not in df_out.columns or "[deGFP_m]" not in df_out.columns:
        return df_out
    t_s = df_out["time"].to_numpy(dtype=float)
    p_M = np.clip(df_out["[deGFP_m]"].to_numpy(dtype=float), 0.0, np.inf)
    gamma_s_inv = max(0.0, float(gamma_prod_h_inv)) / 3600.0
    wash_rate = gamma_s_inv * p_M
    washed = np.zeros_like(p_M)
    if t_s.size > 1:
        dt = np.diff(t_s)
        washed[1:] = np.cumsum(0.5 * (wash_rate[:-1] + wash_rate[1:]) * dt)
    df_out["[deGFP_washout_cum]"] = washed
    df_out["[deGFP_cum_equiv]"] = p_M + washed
    return df_out


def continuous_metrics(df: pd.DataFrame, protein_col: str = "[deGFP_m]") -> Dict[str, float]:
    if protein_col not in df.columns:
        raise KeyError(f"Missing protein column for continuous metrics: {protein_col}")
    t_h = df["time"].to_numpy(dtype=float) / 3600.0
    p_uM = df[protein_col].to_numpy(dtype=float) * 1e6
    return {
        "protein_2h_uM": float(np.interp(2.0, t_h, p_uM)),
        "protein_4h_uM": float(np.interp(4.0, t_h, p_uM)),
        "protein_8h_uM": float(np.interp(8.0, t_h, p_uM)),
        "protein_24h_uM": float(np.interp(24.0, t_h, p_uM)),
        "protein_48h_uM": float(np.interp(48.0, t_h, p_uM)),
    }


def objective_batch(metrics: Dict[str, float], target_uM: float) -> float:
    rel = lambda x: x / max(target_uM, 1e-12)
    slope_rel = metrics["plateau_abs_slope_uM_per_h"] / max(target_uM, 1e-12)
    overshoot = max(0.0, metrics["protein_peak_uM"] - target_uM)
    return float(
        10.0 * rel(metrics["plateau_level_uM"] - target_uM) ** 2
        + 24.0 * rel(metrics["plateau_max_abs_drift_uM"]) ** 2
        + 10.0 * slope_rel**2
        + 3.0 * rel(overshoot) ** 2
    )


def fit_batch_effective_scales(
    batch_rr,
    target_uM: float,
    maxiter: int = 24,
    fit_points: int = 260,
    n_starts: int = 4,
    random_samples: int = 40,
    random_seed: int = 20260325,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    # Base fit dimensions (log10-scales):
    # [k_scale, ATP_inf_scale, GTP_inf_scale, keff_scale, keff_inf_scale,
    #  keff_KATP_scale, keff_KGTP_scale, keff_KIPO4_scale]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    param_ids = set(list(batch_rr.getGlobalParameterIds()))
    has_keff = "bg_keff_scale" in param_ids
    has_mem_in = "bg_load_mem_in" in param_ids
    has_mem_out = "bg_load_mem_out" in param_ids
    has_mem_gain = "bg_load_mem_gain" in param_ids
    has_load_gain = "bg_load_gain" in param_ids
    has_load_km2 = "bg_load_K_M2" in param_ids

    base_mem_in = float(batch_rr["bg_load_mem_in"]) if has_mem_in else 1.0
    base_mem_out = float(batch_rr["bg_load_mem_out"]) if has_mem_out else 1.0
    base_mem_gain = float(batch_rr["bg_load_mem_gain"]) if has_mem_gain else 1.0
    base_load_gain = float(batch_rr["bg_load_gain"]) if has_load_gain else 1.0
    base_load_km2 = float(batch_rr["bg_load_K_M2"]) if has_load_km2 else 1.0

    # Optional memory scales appended to theta:
    # log10(mem_in_scale), log10(mem_out_scale), log10(mem_gain_scale),
    # log10(load_gain_scale), log10(load_km2_scale)
    extras = []
    if has_mem_in:
        extras.append(0.0)
    if has_mem_out:
        extras.append(0.0)
    if has_mem_gain:
        extras.append(0.0)
    if has_load_gain:
        extras.append(0.0)
    if has_load_km2:
        extras.append(0.0)
    if extras:
        x0 = np.concatenate([x0, np.asarray(extras, dtype=float)])

    lo = [-1.0, -0.3, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0]
    hi = [1.0, 0.3, 0.7, 1.0, 0.7, 1.0, 1.0, 1.0]
    if has_mem_in:
        lo.append(-4.0)
        hi.append(4.0)
    if has_mem_out:
        lo.append(-4.0)
        hi.append(4.0)
    if has_mem_gain:
        lo.append(-3.0)
        hi.append(3.0)
    if has_load_gain:
        lo.append(-1.5)
        hi.append(1.5)
    if has_load_km2:
        lo.append(-1.5)
        hi.append(1.5)
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)

    def unpack(theta: np.ndarray) -> Dict[str, float]:
        th = np.asarray(theta, dtype=float)
        th = np.clip(th, lo_arr, hi_arr)
        vals = {
            "k_scale": float(10.0 ** th[0]),
            "ATP_inf_scale": float(10.0 ** th[1]),
            "GTP_inf_scale": float(10.0 ** th[2]),
            "keff_scale": float(10.0 ** th[3]),
            "keff_inf_scale": float(10.0 ** th[4]),
            "keff_KATP_scale": float(10.0 ** th[5]),
            "keff_KGTP_scale": float(10.0 ** th[6]),
            "keff_KIPO4_scale": float(10.0 ** th[7]),
        }
        idx = 8
        if has_mem_in:
            vals["mem_in_scale"] = float(10.0 ** th[idx])
            idx += 1
        if has_mem_out:
            vals["mem_out_scale"] = float(10.0 ** th[idx])
            idx += 1
        if has_mem_gain:
            vals["mem_gain_scale"] = float(10.0 ** th[idx])
            idx += 1
        if has_load_gain:
            vals["load_gain_scale"] = float(10.0 ** th[idx])
            idx += 1
        if has_load_km2:
            vals["load_km2_scale"] = float(10.0 ** th[idx])
        return vals

    cache: Dict[Tuple[float, ...], float] = {}

    def obj(theta: np.ndarray) -> float:
        vals = unpack(theta)
        key = tuple(round(float(v), 8) for v in vals.values())
        if key in cache:
            return cache[key]
        batch_rr.resetToOrigin()
        batch_rr["bg_k_scale"] = vals["k_scale"]
        batch_rr["bg_ATP_inf_scale"] = vals["ATP_inf_scale"]
        batch_rr["bg_GTP_inf_scale"] = vals["GTP_inf_scale"]
        if has_mem_in:
            batch_rr["bg_load_mem_in"] = max(1e-12, base_mem_in * vals.get("mem_in_scale", 1.0))
        if has_mem_out:
            batch_rr["bg_load_mem_out"] = max(1e-12, base_mem_out * vals.get("mem_out_scale", 1.0))
        if has_mem_gain:
            batch_rr["bg_load_mem_gain"] = max(1e-12, base_mem_gain * vals.get("mem_gain_scale", 1.0))
        if has_load_gain:
            batch_rr["bg_load_gain"] = max(1e-8, base_load_gain * vals.get("load_gain_scale", 1.0))
        if has_load_km2:
            batch_rr["bg_load_K_M2"] = max(1e-20, base_load_km2 * vals.get("load_km2_scale", 1.0))
        if has_keff:
            batch_rr["bg_keff_scale"] = vals["keff_scale"]
            if "bg_keff_inf_scale" in param_ids:
                batch_rr["bg_keff_inf_scale"] = vals["keff_inf_scale"]
            if "bg_keff_KATP_scale" in param_ids:
                batch_rr["bg_keff_KATP_scale"] = vals["keff_KATP_scale"]
            if "bg_keff_KGTP_scale" in param_ids:
                batch_rr["bg_keff_KGTP_scale"] = vals["keff_KGTP_scale"]
            if "bg_keff_KIPO4_scale" in param_ids:
                batch_rr["bg_keff_KIPO4_scale"] = vals["keff_KIPO4_scale"]
            if "bg_keff_KIPPi_scale" in param_ids:
                batch_rr["bg_keff_KIPPi_scale"] = 1.0
        try:
            df = simulate_degfp(batch_rr, t_end_s=8 * 3600, n_points=fit_points, reset_to_origin=False)
            m = batch_metrics(df)
            val = objective_batch(m, target_uM=target_uM)
            # Mild regularization to avoid unphysical extreme compensation.
            reg = 0.01 * (
                np.log10(max(vals["ATP_inf_scale"], 1e-12)) ** 2
                + np.log10(max(vals["GTP_inf_scale"], 1e-12)) ** 2
                + np.log10(max(vals["keff_scale"], 1e-12)) ** 2
                + np.log10(max(vals["keff_inf_scale"], 1e-12)) ** 2
                + np.log10(max(vals["keff_KATP_scale"], 1e-12)) ** 2
                + np.log10(max(vals["keff_KGTP_scale"], 1e-12)) ** 2
                + np.log10(max(vals["keff_KIPO4_scale"], 1e-12)) ** 2
            )
            if has_mem_in:
                reg += 0.005 * (np.log10(max(vals.get("mem_in_scale", 1.0), 1e-12)) ** 2)
            if has_mem_out:
                reg += 0.005 * (np.log10(max(vals.get("mem_out_scale", 1.0), 1e-12)) ** 2)
            if has_mem_gain:
                reg += 0.005 * (np.log10(max(vals.get("mem_gain_scale", 1.0), 1e-12)) ** 2)
            if has_load_gain:
                reg += 0.002 * (np.log10(max(vals.get("load_gain_scale", 1.0), 1e-12)) ** 2)
            if has_load_km2:
                reg += 0.002 * (np.log10(max(vals.get("load_km2_scale", 1.0), 1e-12)) ** 2)
            val = float(val + reg)
        except Exception:
            val = 1e12
        cache[key] = float(val)
        return float(val)

    n_extra = len(x0) - 8
    extras0 = np.zeros(max(0, n_extra), dtype=float)
    core_starts = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.array([0.3, 0.0, 0.2, 0.0, -0.6, -0.3, 0.2, 0.4], dtype=float),
        np.array([0.5, 0.0, 0.3, 0.1, -0.5, 0.2, -0.2, 0.5], dtype=float),
        np.array([-0.2, 0.0, 0.1, 0.2, -0.3, 0.1, 0.0, 0.2], dtype=float),
    ]
    starts_all = [np.concatenate([c, extras0.copy()]) for c in core_starts]
    starts_all = [np.clip(s, lo_arr, hi_arr) for s in starts_all]

    # Global pre-search to avoid local minima on high-dimensional stiff objectives.
    rng = np.random.default_rng(int(random_seed))
    candidates: List[Tuple[float, np.ndarray]] = []
    for s in starts_all:
        candidates.append((float(obj(s)), np.asarray(s, dtype=float)))
    for _ in range(max(0, int(random_samples))):
        s = rng.uniform(lo_arr, hi_arr)
        candidates.append((float(obj(s)), np.asarray(s, dtype=float)))
    candidates.sort(key=lambda x: x[0])
    n_refine = max(1, min(int(n_starts), len(candidates)))
    starts = [np.asarray(candidates[i][1], dtype=float) for i in range(n_refine)]

    best_x = x0
    best_val = float("inf")
    for s in starts:
        res = minimize(
            obj,
            x0=s,
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-4},
        )
        val = float(obj(res.x))
        if val < best_val:
            best_val = val
            best_x = np.asarray(res.x, dtype=float)

    vals_opt = unpack(best_x)
    k_opt = vals_opt["k_scale"]
    atp_opt = vals_opt["ATP_inf_scale"]
    gtp_opt = vals_opt["GTP_inf_scale"]
    keff_opt = vals_opt["keff_scale"]
    keff_inf_opt = vals_opt["keff_inf_scale"]
    keff_katp_opt = vals_opt["keff_KATP_scale"]
    keff_kgtp_opt = vals_opt["keff_KGTP_scale"]
    keff_kipo4_opt = vals_opt["keff_KIPO4_scale"]
    mem_in_scale_opt = vals_opt.get("mem_in_scale", 1.0)
    mem_out_scale_opt = vals_opt.get("mem_out_scale", 1.0)
    mem_gain_scale_opt = vals_opt.get("mem_gain_scale", 1.0)
    load_gain_scale_opt = vals_opt.get("load_gain_scale", 1.0)
    load_km2_scale_opt = vals_opt.get("load_km2_scale", 1.0)

    batch_rr.resetToOrigin()
    batch_rr["bg_k_scale"] = k_opt
    batch_rr["bg_ATP_inf_scale"] = atp_opt
    batch_rr["bg_GTP_inf_scale"] = gtp_opt
    if has_mem_in:
        batch_rr["bg_load_mem_in"] = max(1e-12, base_mem_in * mem_in_scale_opt)
    if has_mem_out:
        batch_rr["bg_load_mem_out"] = max(1e-12, base_mem_out * mem_out_scale_opt)
    if has_mem_gain:
        batch_rr["bg_load_mem_gain"] = max(1e-12, base_mem_gain * mem_gain_scale_opt)
    if has_load_gain:
        batch_rr["bg_load_gain"] = max(1e-8, base_load_gain * load_gain_scale_opt)
    if has_load_km2:
        batch_rr["bg_load_K_M2"] = max(1e-20, base_load_km2 * load_km2_scale_opt)
    if has_keff:
        batch_rr["bg_keff_scale"] = keff_opt
        if "bg_keff_inf_scale" in param_ids:
            batch_rr["bg_keff_inf_scale"] = keff_inf_opt
        if "bg_keff_KATP_scale" in param_ids:
            batch_rr["bg_keff_KATP_scale"] = keff_katp_opt
        if "bg_keff_KGTP_scale" in param_ids:
            batch_rr["bg_keff_KGTP_scale"] = keff_kgtp_opt
        if "bg_keff_KIPO4_scale" in param_ids:
            batch_rr["bg_keff_KIPO4_scale"] = keff_kipo4_opt
        if "bg_keff_KIPPi_scale" in param_ids:
            batch_rr["bg_keff_KIPPi_scale"] = 1.0
    params = {
        "k_scale": k_opt,
        "ATP_inf_scale": atp_opt,
        "GTP_inf_scale": gtp_opt,
        "keff_scale": keff_opt if has_keff else 1.0,
        "keff_inf_scale": keff_inf_opt if has_keff else 1.0,
        "keff_KATP_scale": keff_katp_opt if has_keff else 1.0,
        "keff_KGTP_scale": keff_kgtp_opt if has_keff else 1.0,
        "keff_KIPO4_scale": keff_kipo4_opt if has_keff else 1.0,
        "keff_KIPPi_scale": 1.0,
        "mem_in_scale": mem_in_scale_opt if has_mem_in else 1.0,
        "mem_out_scale": mem_out_scale_opt if has_mem_out else 1.0,
        "mem_gain_scale": mem_gain_scale_opt if has_mem_gain else 1.0,
        "mem_in": (base_mem_in * mem_in_scale_opt) if has_mem_in else None,
        "mem_out": (base_mem_out * mem_out_scale_opt) if has_mem_out else None,
        "mem_gain": (base_mem_gain * mem_gain_scale_opt) if has_mem_gain else None,
        "load_gain_scale": load_gain_scale_opt if has_load_gain else 1.0,
        "load_km2_scale": load_km2_scale_opt if has_load_km2 else 1.0,
        "load_gain": (base_load_gain * load_gain_scale_opt) if has_load_gain else None,
        "load_K_M2": (base_load_km2 * load_km2_scale_opt) if has_load_km2 else None,
    }
    try:
        df_final = simulate_degfp(batch_rr, t_end_s=8 * 3600, n_points=500, reset_to_origin=False)
        m_final = batch_metrics(df_final)
    except Exception:
        # Keep the workflow robust: if the optimizer's final point is numerically unstable,
        # fall back to neutral scales so the caller still gets a valid baseline output.
        batch_rr.resetToOrigin()
        neutral = {
            "bg_k_scale": 1.0,
            "bg_ATP_inf_scale": 1.0,
            "bg_GTP_inf_scale": 1.0,
            "bg_keff_scale": 1.0,
            "bg_keff_inf_scale": 1.0,
            "bg_keff_KATP_scale": 1.0,
            "bg_keff_KGTP_scale": 1.0,
            "bg_keff_KIPO4_scale": 1.0,
            "bg_keff_KIPPi_scale": 1.0,
            "bg_load_mem_in": base_mem_in,
            "bg_load_mem_out": base_mem_out,
            "bg_load_mem_gain": base_mem_gain,
            "bg_load_gain": base_load_gain,
            "bg_load_K_M2": base_load_km2,
        }
        for pid, pv in neutral.items():
            if pid in param_ids:
                batch_rr[pid] = pv
        params = {
            "k_scale": 1.0,
            "ATP_inf_scale": 1.0,
            "GTP_inf_scale": 1.0,
            "keff_scale": 1.0,
            "keff_inf_scale": 1.0,
            "keff_KATP_scale": 1.0,
            "keff_KGTP_scale": 1.0,
            "keff_KIPO4_scale": 1.0,
            "keff_KIPPi_scale": 1.0,
            "mem_in_scale": 1.0,
            "mem_out_scale": 1.0,
            "mem_gain_scale": 1.0,
            "mem_in": base_mem_in if has_mem_in else None,
            "mem_out": base_mem_out if has_mem_out else None,
            "mem_gain": base_mem_gain if has_mem_gain else None,
            "load_gain_scale": 1.0,
            "load_km2_scale": 1.0,
            "load_gain": base_load_gain if has_load_gain else None,
            "load_K_M2": base_load_km2 if has_load_km2 else None,
        }
        df_final = simulate_degfp(batch_rr, t_end_s=8 * 3600, n_points=500, reset_to_origin=False)
        m_final = batch_metrics(df_final)
    return params, m_final, df_final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PACKAGE_DIR / "results/core_workflow",
        help="Output directory for summary/csv artifacts.",
    )
    parser.add_argument("--target-uM", type=float, default=5.3, help="Batch deGFP target at 2h (uM).")
    parser.add_argument("--gamma-h", type=float, default=0.5, help="Continuous exchange rate gamma (1/h).")
    parser.add_argument(
        "--product-gamma-h",
        type=float,
        default=None,
        help=(
            "Continuous outflow rate for macro-products (1/h). "
            "Default: use --gamma-h."
        ),
    )
    parser.add_argument(
        "--feed-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for continuous feed concentrations of dialyzable feed species.",
    )
    parser.add_argument("--fit-maxiter", type=int, default=24, help="Nelder-Mead max iterations.")
    parser.add_argument("--fit-points", type=int, default=260, help="Time points per fit evaluation.")
    parser.add_argument(
        "--fit-starts",
        type=int,
        default=4,
        help="Number of deterministic initial points for Nelder-Mead (1-4).",
    )
    parser.add_argument(
        "--fit-random-samples",
        type=int,
        default=40,
        help="Random global pre-search samples before local Nelder-Mead refinement.",
    )
    parser.add_argument(
        "--fit-random-seed",
        type=int,
        default=20260325,
        help="Random seed for global pre-search sampling.",
    )
    parser.add_argument(
        "--baseline-fit-maxiter",
        type=int,
        default=8,
        help="Max iterations for the pre-keff baseline fit (set <=0 to skip baseline fitting).",
    )
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Only run formal batch fitting and skip continuous simulation.",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: reference full model (successful baseline).
    sbml_full, ic_uM, counts_full, present_hf = build_patched_full_sbml()
    rr_ref = te.loadSBMLModel(sbml_full)
    configure_rr(rr_ref)
    df_ref = simulate_reference_batch(rr_ref, present_hf=present_hf, t_end_s=8 * 3600, n_points=700)
    ref_m = batch_metrics(df_ref[["time", "[deGFP_m]"]].assign(**{"[mRNA_t]": df_ref["[mRNA_t]"]}))

    # Step 2: fit high-frequency background parameters from reference trajectories.
    fitted_hf = fit_background_from_reference(df_ref, present_hf=present_hf)

    # Step 3a: batch background baseline (without k_eff correction).
    sbml_batch_bg, counts_bg, hf_meta = build_background_sbml(
        mode="batch",
        fitted_hf=fitted_hf,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=float(args.feed_multiplier),
        product_outflow_h=None,
        keff_map=None,
    )
    rr_batch = te.loadSBMLModel(sbml_batch_bg)
    configure_rr(rr_batch)
    baseline_fit_iter = int(args.baseline_fit_maxiter)
    if baseline_fit_iter > 0:
        eff_params0, b_metrics0, df_batch0 = fit_batch_effective_scales(
            rr_batch,
            target_uM=float(args.target_uM),
            maxiter=baseline_fit_iter,
            fit_points=max(120, int(args.fit_points // 2)),
            n_starts=max(1, min(int(args.fit_starts), 4)),
            random_samples=max(0, int(args.fit_random_samples // 2)),
            random_seed=int(args.fit_random_seed),
        )
    else:
        rr_batch.resetToOrigin()
        df_batch0 = simulate_degfp(rr_batch, t_end_s=8 * 3600, n_points=max(180, int(args.fit_points // 2)))
        b_metrics0 = batch_metrics(df_batch0)
        eff_params0 = {
            "k_scale": 1.0,
            "ATP_inf_scale": 1.0,
            "GTP_inf_scale": 1.0,
            "keff_scale": 1.0,
            "keff_inf_scale": 1.0,
            "keff_KATP_scale": 1.0,
            "keff_KGTP_scale": 1.0,
            "keff_KIPO4_scale": 1.0,
            "keff_KIPPi_scale": 1.0,
            "mem_in_scale": 1.0,
            "mem_out_scale": 1.0,
            "mem_gain_scale": 1.0,
            "load_gain_scale": 1.0,
            "load_km2_scale": 1.0,
        }

    # Step 3b: learn state-dependent k_eff from reference-vs-background flux ratio.
    keff_map = fit_keff_map_from_flux_ratio(df_ref=df_ref, df_bg=df_batch0)

    # Step 3c: rebuild batch background with k_eff channel and re-fit.
    sbml_batch_bg2, counts_bg, hf_meta = build_background_sbml(
        mode="batch",
        fitted_hf=fitted_hf,
        ic_uM=ic_uM,
        gamma_h=0.0,
        feed_multiplier=float(args.feed_multiplier),
        product_outflow_h=None,
        keff_map=keff_map,
    )
    rr_batch2 = te.loadSBMLModel(sbml_batch_bg2)
    configure_rr(rr_batch2)
    eff_params, b_metrics, df_batch = fit_batch_effective_scales(
        rr_batch2,
        target_uM=float(args.target_uM),
        maxiter=int(args.fit_maxiter),
        fit_points=int(args.fit_points),
        n_starts=max(1, min(int(args.fit_starts), 4)),
        random_samples=max(0, int(args.fit_random_samples)),
        random_seed=int(args.fit_random_seed),
    )

    c_metrics = None
    c_metrics_reactor = None
    df_cont = None
    hf_meta_cont = None
    counts_bg_cont = None
    if not args.batch_only:
        # Step 4: continuous background model with same fitted parameters + split exchange
        # for dialyzable small molecules (beyond the backgrounded HF subset).
        sbml_cont_bg, counts_bg_cont, hf_meta_cont = build_background_sbml(
            mode="continuous",
            fitted_hf=fitted_hf,
            ic_uM=ic_uM,
            gamma_h=float(args.gamma_h),
            feed_multiplier=float(args.feed_multiplier),
            product_outflow_h=(
                float(args.product_gamma_h) if args.product_gamma_h is not None else None
            ),
            keff_map=keff_map,
        )
        rr_cont = te.loadSBMLModel(sbml_cont_bg)
        configure_rr(rr_cont)
        rr_cont["bg_k_scale"] = eff_params["k_scale"]
        rr_cont["bg_ATP_inf_scale"] = eff_params["ATP_inf_scale"]
        rr_cont["bg_GTP_inf_scale"] = eff_params["GTP_inf_scale"]
        if "bg_keff_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_scale"] = eff_params.get("keff_scale", 1.0)
        if "bg_keff_inf_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_inf_scale"] = eff_params.get("keff_inf_scale", 1.0)
        if "bg_keff_KATP_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_KATP_scale"] = eff_params.get("keff_KATP_scale", 1.0)
        if "bg_keff_KGTP_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_KGTP_scale"] = eff_params.get("keff_KGTP_scale", 1.0)
        if "bg_keff_KIPO4_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_KIPO4_scale"] = eff_params.get("keff_KIPO4_scale", 1.0)
        if "bg_keff_KIPPi_scale" in list(rr_cont.getGlobalParameterIds()):
            rr_cont["bg_keff_KIPPi_scale"] = eff_params.get("keff_KIPPi_scale", 1.0)
        if "bg_load_mem_in" in list(rr_cont.getGlobalParameterIds()) and eff_params.get("mem_in") is not None:
            rr_cont["bg_load_mem_in"] = float(eff_params["mem_in"])
        if "bg_load_mem_out" in list(rr_cont.getGlobalParameterIds()) and eff_params.get("mem_out") is not None:
            rr_cont["bg_load_mem_out"] = float(eff_params["mem_out"])
        if "bg_load_mem_gain" in list(rr_cont.getGlobalParameterIds()) and eff_params.get("mem_gain") is not None:
            rr_cont["bg_load_mem_gain"] = float(eff_params["mem_gain"])
        if "bg_load_gain" in list(rr_cont.getGlobalParameterIds()) and eff_params.get("load_gain") is not None:
            rr_cont["bg_load_gain"] = float(eff_params["load_gain"])
        if "bg_load_K_M2" in list(rr_cont.getGlobalParameterIds()) and eff_params.get("load_K_M2") is not None:
            rr_cont["bg_load_K_M2"] = float(eff_params["load_K_M2"])

        df_cont = simulate_degfp(rr_cont, t_end_s=48 * 3600, n_points=900, reset_to_origin=False)
        gamma_prod_h_inv = float(
            (hf_meta_cont or {}).get(
                "continuous_product_outflow_h_inv",
                float(args.product_gamma_h) if args.product_gamma_h is not None else float(args.gamma_h),
            )
        )
        df_cont = append_cumulative_degfp_equivalent(df_cont, gamma_prod_h_inv=gamma_prod_h_inv)
        c_metrics_reactor = continuous_metrics(df_cont, protein_col="[deGFP_m]")
        c_metrics = continuous_metrics(df_cont, protein_col="[deGFP_cum_equiv]")

    # Save outputs.
    ref_csv = out_dir / "reference_batch_full_crn.csv"
    batch_csv = out_dir / "batch_fitted_timeseries.csv"
    cont_csv = out_dir / "continuous_same_kinetics_timeseries.csv"
    df_ref.to_csv(ref_csv, index=False)
    df_batch.to_csv(batch_csv, index=False)
    if df_cont is not None:
        df_cont.to_csv(cont_csv, index=False)

    summary = {
        "workflow": "full CRN + reference-informed high-frequency backgrounds",
        "source_model": str(BASE_SBML),
        "counts_full_model": counts_full,
        "counts_background_model": counts_bg,
        "counts_background_model_continuous": counts_bg_cont,
        "background_species": present_hf,
        "background_groups": hf_meta,
        "batch_reference_full_model_metrics": ref_m,
        "batch_fit": {
            "target_2h_uM": float(args.target_uM),
            "baseline_without_keff_metrics": b_metrics0,
            "baseline_without_keff_params": eff_params0,
            "keff_map": keff_map,
            "effective_params": eff_params,
            "metrics": b_metrics,
        },
        "continuous_process": (
            {
                "gamma_h_inv": float(args.gamma_h),
                "feed_multiplier": float(args.feed_multiplier),
                "exchange_policy": (hf_meta_cont or {}).get("continuous_exchange_policy", {}),
                "dialyzable_exchange_species": sorted(
                    list((hf_meta_cont or {}).get("continuous_feed_species", []))
                    + list((hf_meta_cont or {}).get("continuous_washout_species", []))
                ),
                "feed_species_uM": (hf_meta_cont or {}).get("continuous_feed_uM", {}),
                "product_outflow_species": sorted(
                    list((hf_meta_cont or {}).get("continuous_product_outflow_species", []))
                ),
                "product_outflow_gamma_h_inv": (hf_meta_cont or {}).get("continuous_product_outflow_h_inv"),
                "metrics_basis": "deGFP_cum_equiv = deGFP_m + integral(gamma_prod * deGFP_m dt)",
                "metrics": c_metrics,
                "reactor_concentration_metrics": c_metrics_reactor,
            }
            if not args.batch_only
            else None
        ),
        "hf_fit_params": fitted_hf,
        "output_files": {
            "reference_batch_csv": str(ref_csv),
            "batch_csv": str(batch_csv),
            "continuous_csv": str(cont_csv) if df_cont is not None else None,
        },
    }
    out_json = out_dir / "batch_then_continuous_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Full-CRN background workflow finished.")
    print(
        "[OK] Full model counts: species={species_total}, reactions={reactions_total}".format(
            **counts_full
        )
    )
    print(
        "[OK] Background model counts: backgrounded={background_species_count}, dynamic={dynamic_species_count}".format(
            **counts_bg
        )
    )
    print(
        "[OK] Reference batch full model: P2h={:.3f} uM, drift={:.4f} uM".format(
            ref_m["protein_2h_uM"], ref_m["plateau_max_abs_drift_uM"]
        )
    )
    print(
        "[OK] Batch background fit: k_scale={:.4f}, ATP_inf_scale={:.3f}, "
        "GTP_inf_scale={:.3f}, keff_scale={:.3f}, keff_inf_scale={:.3f}, "
        "keff_KATP_scale={:.3f}, keff_KGTP_scale={:.3f}, "
        "keff_KIPO4_scale={:.3f}, load_gain_scale={:.3f}, load_km2_scale={:.3f}, "
        "P2h={:.3f} uM, drift={:.4f} uM".format(
            eff_params["k_scale"],
            eff_params["ATP_inf_scale"],
            eff_params["GTP_inf_scale"],
            eff_params.get("keff_scale", 1.0),
            eff_params.get("keff_inf_scale", 1.0),
            eff_params.get("keff_KATP_scale", 1.0),
            eff_params.get("keff_KGTP_scale", 1.0),
            eff_params.get("keff_KIPO4_scale", 1.0),
            eff_params.get("load_gain_scale", 1.0),
            eff_params.get("load_km2_scale", 1.0),
            b_metrics["protein_2h_uM"],
            b_metrics["plateau_max_abs_drift_uM"],
        )
    )
    if c_metrics is not None:
        print(
            "[OK] Continuous same params (cumulative-equivalent): P4h={:.3f} uM, P48h={:.3f} uM".format(
                c_metrics["protein_4h_uM"], c_metrics["protein_48h_uM"]
            )
        )
        if c_metrics_reactor is not None:
            print(
                "[OK] Continuous reactor concentration: P4h={:.3f} uM, P48h={:.3f} uM".format(
                    c_metrics_reactor["protein_4h_uM"], c_metrics_reactor["protein_48h_uM"]
                )
            )
    else:
        print("[OK] Batch-only mode: continuous simulation skipped.")
    print(f"[OK] Summary: {out_json}")


if __name__ == "__main__":
    main()
