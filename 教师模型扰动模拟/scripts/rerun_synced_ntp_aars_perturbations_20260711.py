#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TMP_MPL = Path("/private/tmp/codex_mplconfig_synced_perturb_20260711")
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPL))

import libsbml
import numpy as np
import pandas as pd
import tellurium as te


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = PACKAGE_ROOT / "model_inputs" / "deliverable_PURE_1nt1aa_independent_20260610"
OUTDIR_DEFAULT = PACKAGE_ROOT / "run_outputs" / "synced_ntp_aars_perturb_20260711"

sys.path.insert(0, str(DELIVERABLE))
import run_tx1nt_tlg1_micro3_longrun as run_cont  # noqa: E402
import fit_batch_then_continuous_full_crn_background as core  # noqa: E402


NTPS = ["ATP", "CTP", "GTP", "UTP"]
AARS = [
    "AlaRS",
    "ArgRS",
    "AsnRS",
    "AspRS",
    "CysRS",
    "GlnRS",
    "GluRS",
    "GlyRS",
    "HisRS",
    "IleRS",
    "LeuRS",
    "LysRS",
    "MetRS",
    "PheRS",
    "ProRS",
    "SerRS",
    "ThrRS",
    "TrpRS",
    "TyrRS",
    "ValRS",
]
NTP_FAMILY_IDS = {
    "ATP": ["ATP", "ADP", "AMP", "ATP_TX", "Moiety_ATP_poly"],
    "CTP": ["CTP", "CTP_TX", "Moiety_CTP_poly"],
    "GTP": ["GTP", "GDP", "GMP", "GTP_TX", "Moiety_GTP_poly"],
    "UTP": ["UTP", "UTP_TX", "Moiety_UTP_poly"],
}
KEY_SPECIES = [
    "deGFP_m",
    "mRNA_t",
    "ATP",
    "ADP",
    "AMP",
    "GTP",
    "GDP",
    "GMP",
    "CTP",
    "UTP",
    "AA",
    "CP",
    "PO4",
    "PPi",
    "DNA",
    "T7RNAP",
    "Ribosome",
    "Ribosome_mRNA",
    "IF1",
    "IF2",
    "IF3",
    "EFG",
    "EFTu",
    "MetRS",
    "fMettRNAfMetCAU",
]
AA20_SET = set(core.AA20)


def parse_snapshot_time_points(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if any(v < 0 for v in vals):
        raise ValueError("Snapshot time points must be non-negative")
    return vals


def nearest_snapshot_indices(t_s: np.ndarray, target_h: List[float]) -> List[int]:
    if not target_h:
        return []
    idxs: List[int] = []
    for h in target_h:
        idxs.append(int(np.argmin(np.abs(t_s - float(h) * 3600.0))))
    return sorted(set(idxs))


def classify_reaction(rid: str, react: List[str], prod: List[str], mods: List[str]) -> str:
    allsp = react + prod + mods
    text = ";".join(allsp)
    if rid.startswith("rxn_ex_in_deGFP") or rid.startswith("rxn_ex_out_deGFP"):
        return "product_exchange"
    if rid.startswith("rxn_ex_in_") or rid.startswith("rxn_ex_out_"):
        if any(s in AA20_SET for s in allsp):
            return "amino_acid_exchange"
        if any(s in {"ATP", "GTP", "CTP", "UTP", "ADP", "GDP"} for s in allsp):
            return "ntp_exchange"
        return "environment_exchange"
    if "DNA" in text or "T7RNAP" in text or "mRNA_t" in text:
        return "transcription_or_mRNA_turnover"
    if "EFG" in text and ("GTP" in allsp or "GDP" in allsp or "PO4" in allsp):
        return "EFG_GTP_translocation_cycle"
    if "EFTu" in text and ("GTP" in allsp or "GDP" in allsp or "PO4" in allsp):
        return "EFTu_GTP_delivery_cycle"
    if "IF2" in text and ("GTP" in allsp or "GDP" in allsp or "PO4" in allsp):
        return "IF2_GTP_initiation_cycle"
    if ("RF3" in text or "RRF" in text) and ("GTP" in allsp or "GDP" in allsp or "PO4" in allsp):
        return "release_recycling_GTP_cycle"
    if ("GTP" in allsp or "GDP" in allsp) and "PO4" in allsp:
        return "other_GTP_hydrolysis"
    if "RS" in text and ("tRNA" in text or "AMP" in text or "PPi" in allsp or "ATP" in allsp):
        return "aa_tRNA_charging_network"
    if "tRNA" in text and ("AMP" in text or "PPi" in allsp or any(aa in allsp for aa in AA20_SET)):
        return "aa_tRNA_charging_network"
    if "Ribosome" in text:
        return "ribosome_translation_assembly"
    if any(s in {"ATP", "ADP", "CP", "PO4", "PPi"} for s in allsp):
        return "ATP_CP_PPi_energy_waste_network"
    if any(s in AA20_SET for s in allsp):
        return "amino_acid_local_binding"
    return "other_mass_action_network"


def reaction_table_from_sbml(sbml: str) -> pd.DataFrame:
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    rows: List[Dict[str, object]] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        rid = rxn.getId()
        react = [rxn.getReactant(j).getSpecies() for j in range(rxn.getNumReactants())]
        prod = [rxn.getProduct(j).getSpecies() for j in range(rxn.getNumProducts())]
        mods = [rxn.getModifier(j).getSpecies() for j in range(rxn.getNumModifiers())]
        rows.append(
            {
                "reaction_index": int(i),
                "reaction_id": rid,
                "reactants": ";".join(react),
                "products": ";".join(prod),
                "modifiers": ";".join(mods),
                "reaction_class": classify_reaction(rid, react, prod, mods),
            }
        )
    return pd.DataFrame(rows)


def parse_scales(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("No scales provided")
    if any(v < 0 for v in vals):
        raise ValueError("Scales must be non-negative")
    return vals


def tag_float(x: float) -> str:
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_environment() -> None:
    os.chdir(DELIVERABLE)
    run_cont.set_fine_node_env()


def load_sources() -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    set_environment()
    params_path = DELIVERABLE / "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json"
    hf_path = DELIVERABLE / "output_full_crn_background/batch_then_continuous_summary.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))["batch_best_params"]
    hfs = json.loads(hf_path.read_text(encoding="utf-8"))
    hf_fit = hfs["hf_fit_params"]
    keff_map = hfs["batch_fit"]["keff_map"]
    ic_uM = core.load_initial_uM_table(core.INIT_CSV)
    return params, hf_fit, keff_map, ic_uM


def apply_params(rr, params: Dict[str, float]) -> Dict[str, int]:
    pids = set(rr.getGlobalParameterIds())
    applied = 0
    missing = 0
    for pid, val in params.items():
        if pid in pids:
            rr[pid] = float(val)
            applied += 1
        else:
            missing += 1
    return {"applied": applied, "missing": missing}


def endpoint_species_uM(rr) -> Dict[str, float]:
    ids = list(rr.getFloatingSpeciesIds())
    vals = np.asarray(rr.getFloatingSpeciesConcentrations(), dtype=float)
    return {sid: float(val * 1e6) for sid, val in zip(ids, vals)}


def scale_species_initial(model: "libsbml.Model", sid: str, scale: float) -> bool:
    sp = model.getSpecies(sid)
    if sp is None:
        return False
    if sp.isSetInitialConcentration():
        sp.setInitialConcentration(float(sp.getInitialConcentration()) * float(scale))
        return True
    if sp.isSetInitialAmount():
        sp.setInitialAmount(float(sp.getInitialAmount()) * float(scale))
        return True
    return False


def set_parameter_value(model: "libsbml.Model", pid: str, value: float) -> bool:
    p = model.getParameter(pid)
    if p is None:
        return False
    p.setValue(float(value))
    return True


def scale_parameter_value(model: "libsbml.Model", pid: str, scale: float) -> bool:
    p = model.getParameter(pid)
    if p is None:
        return False
    p.setValue(float(p.getValue()) * float(scale))
    return True


def disable_nonphysical_qssa_floors(model: "libsbml.Model", component: str) -> List[str]:
    changed: List[str] = []
    if component == "MetRS":
        if set_parameter_value(model, "phi_tl_init_floor", 0.0):
            changed.append("phi_tl_init_floor")
    return changed


def synchronize_ntp_perturbation(model: "libsbml.Model", component: str, scale: float) -> Dict[str, object]:
    species_scaled: List[str] = []
    params_scaled: List[str] = []
    params_set: Dict[str, float] = {}

    for sid in NTP_FAMILY_IDS[component]:
        if scale_species_initial(model, sid, scale):
            species_scaled.append(sid)

    if component in {"CTP", "UTP"}:
        res_sid = f"bgRes_{component}"
        if scale_species_initial(model, res_sid, scale):
            species_scaled.append(res_sid)

    for pid in [f"bg_{component}_feed", f"bg_feed_{component}", f"bg_{component}_0"]:
        if scale_parameter_value(model, pid, scale):
            params_scaled.append(pid)

    if component in {"ATP", "GTP"}:
        scale_pid = f"bg_{component}_inf_scale"
        if set_parameter_value(model, scale_pid, scale):
            params_set[scale_pid] = float(scale)
    else:
        for pid in [f"bg_{component}_inf", f"bg_{component}_total"]:
            if scale_parameter_value(model, pid, scale):
                params_scaled.append(pid)

    return {
        "kind": "ntp_family_total_and_feed",
        "species_scaled": species_scaled,
        "params_scaled": params_scaled,
        "params_set": params_set,
    }


def synchronize_aars_perturbation(model: "libsbml.Model", component: str, scale: float) -> Dict[str, object]:
    species_scaled: List[str] = []
    params_scaled: List[str] = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        if component in sid and scale_species_initial(model, sid, scale):
            species_scaled.append(sid)
    # Important reduced-model closure:
    # the current translation chain consumes aggregate AA directly
    # (rxn_tl_chain_step_* use AA, ATP, GTP), rather than codon-specific
    # aaRS-charged tRNA species. Therefore, to make single-aaRS perturbations
    # meaningfully propagate to translation output in this reduced model, we
    # must synchronize the aggregate AA translation bottleneck as well.
    # This is an approximation, but it is structurally consistent with the
    # current lowered translation mechanism and restores the expected 0x->0
    # behavior for essential aaRS perturbations.
    if scale_species_initial(model, "AA", scale):
        species_scaled.append("AA")
    for pid in ["bg_feed_AA", "bg_AA_feed"]:
        if scale_parameter_value(model, pid, scale):
            params_scaled.append(pid)
    return {
        "kind": "aars_total_enzyme_pool_plus_aggregate_AA_limiter",
        "species_scaled": species_scaled,
        "params_scaled": params_scaled,
        "params_set": {},
    }


def apply_synced_perturbation(
    model: "libsbml.Model",
    component: str,
    scale: float,
) -> Dict[str, object]:
    qssa_floor_zeroed = disable_nonphysical_qssa_floors(model, component)
    if component in NTPS:
        detail = synchronize_ntp_perturbation(model, component, scale)
    elif component in AARS:
        detail = synchronize_aars_perturbation(model, component, scale)
    else:
        raise ValueError(f"Unsupported component for synced rerun: {component}")
    detail["qssa_floor_zeroed"] = qssa_floor_zeroed
    return detail


def build_perturbed_model(
    *,
    component: str,
    scale: float,
    base_ic_uM: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    gamma_h: float,
    keff_map: Dict[str, float],
) -> Tuple["libsbml.Model", Dict[str, int], Dict[str, object], Dict[str, object]]:
    set_environment()
    sbml, counts, meta = core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=base_ic_uM,
        gamma_h=float(gamma_h),
        feed_multiplier=1.0,
        product_outflow_h=float(gamma_h),
        keff_map=keff_map,
    )
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse generated SBML for perturbation rerun")
    perturb_meta = apply_synced_perturbation(model, component, scale)
    return model, counts, meta, perturb_meta


def build_baseline_rr(
    *,
    base_ic_uM: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    gamma_h: float,
    keff_map: Dict[str, float],
) -> Tuple[object, Dict[str, int], Dict[str, object], str]:
    set_environment()
    sbml, counts, meta = core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=base_ic_uM,
        gamma_h=float(gamma_h),
        feed_multiplier=1.0,
        product_outflow_h=float(gamma_h),
        keff_map=keff_map,
    )
    rr = te.loadSBMLModel(sbml)
    core.configure_rr(rr)
    rr.resetToOrigin()
    return rr, counts, meta, sbml


def build_baseline_model(
    *,
    base_ic_uM: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    gamma_h: float,
    keff_map: Dict[str, float],
) -> Tuple["libsbml.Model", Dict[str, int], Dict[str, object]]:
    set_environment()
    sbml, counts, meta = core.build_background_sbml(
        mode="continuous",
        fitted_hf=hf_fit,
        ic_uM=base_ic_uM,
        gamma_h=float(gamma_h),
        feed_multiplier=1.0,
        product_outflow_h=float(gamma_h),
        keff_map=keff_map,
    )
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse generated baseline SBML")
    return model, counts, meta


def build_perturbed_rr(
    *,
    component: str,
    scale: float,
    base_ic_uM: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    gamma_h: float,
    keff_map: Dict[str, float],
) -> Tuple[object, Dict[str, int], Dict[str, object], Dict[str, object], str]:
    model, counts, meta, perturb_meta = build_perturbed_model(
        component=component,
        scale=scale,
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    doc = model.getSBMLDocument()
    sbml_perturbed = libsbml.writeSBMLToString(doc)
    rr = te.loadSBMLModel(sbml_perturbed)
    core.configure_rr(rr)
    rr.resetToOrigin()
    for pid in perturb_meta.get("qssa_floor_zeroed", []):
        if pid in set(rr.getGlobalParameterIds()):
            rr[pid] = 0.0
    return rr, counts, meta, perturb_meta, sbml_perturbed


def build_selection_columns(rr, *, include_all_species: bool = False) -> List[str]:
    species_ids = list(rr.getFloatingSpeciesIds())
    if include_all_species:
        selected = species_ids
    else:
        species_id_set = set(species_ids)
        selected = [sid for sid in KEY_SPECIES if sid in species_id_set]
    return ["time"] + [f"[{sid}]" for sid in selected]


def subset_key_species_timeseries(full_ts: pd.DataFrame) -> pd.DataFrame:
    keep = ["time"] + [f"[{sid}]" for sid in KEY_SPECIES if f"[{sid}]" in full_ts.columns]
    return full_ts[keep].copy()


def build_sparse_snapshot_bundle(
    *,
    rr,
    full_ts: pd.DataFrame,
    reaction_table: pd.DataFrame,
    component: str,
    scale: float,
    requested_time_points_h: List[float],
) -> Dict[str, object]:
    if not requested_time_points_h:
        return {
            "species_rows": [],
            "reaction_rows": [],
            "class_rows": [],
            "requested_time_points_h": "",
            "actual_time_points_h": [],
        }

    t_s = full_ts["time"].to_numpy(dtype=float)
    snapshot_idxs = nearest_snapshot_indices(t_s, requested_time_points_h)
    species_ids = list(rr.getFloatingSpeciesIds())
    reaction_ids = list(rr.getReactionIds())
    reaction_meta = reaction_table.set_index("reaction_id")

    species_rows: List[Dict[str, object]] = []
    reaction_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    requested_text = ",".join("%g" % x for x in requested_time_points_h)

    species_cols = [f"[{sid}]" for sid in species_ids]
    for ti in snapshot_idxs:
        time_s = float(t_s[ti])
        time_h = float(time_s / 3600.0)
        state_row = full_ts.iloc[ti]

        for sid, col in zip(species_ids, species_cols):
            if col not in full_ts.columns:
                continue
            species_rows.append(
                {
                    "component": component,
                    "scale": float(scale),
                    "time_s": time_s,
                    "time_h": time_h,
                    "requested_time_points_h": requested_text,
                    "species": sid,
                    "concentration_uM": float(state_row[col]) * 1e6,
                }
            )

        for sid, col in zip(species_ids, species_cols):
            if col in full_ts.columns:
                rr[sid] = float(state_row[col])
        rates = np.asarray(rr.getReactionRates(), dtype=float)
        class_abs: Dict[str, float] = {}
        class_signed: Dict[str, float] = {}
        for j, rid in enumerate(reaction_ids):
            meta = reaction_meta.loc[rid]
            reaction_class = str(meta["reaction_class"])
            signed_rate = float(rates[j] * 1e6 * 3600.0)
            abs_rate = abs(signed_rate)
            reaction_rows.append(
                {
                    "component": component,
                    "scale": float(scale),
                    "time_s": time_s,
                    "time_h": time_h,
                    "requested_time_points_h": requested_text,
                    "reaction_index": int(meta["reaction_index"]),
                    "reaction_id": rid,
                    "reaction_class": reaction_class,
                    "rate_uM_per_h": signed_rate,
                    "abs_rate_uM_per_h": abs_rate,
                    "reactants": str(meta["reactants"]),
                    "products": str(meta["products"]),
                    "modifiers": str(meta["modifiers"]),
                }
            )
            class_abs[reaction_class] = class_abs.get(reaction_class, 0.0) + abs_rate
            class_signed[reaction_class] = class_signed.get(reaction_class, 0.0) + signed_rate

        for reaction_class in sorted(class_abs):
            class_rows.append(
                {
                    "component": component,
                    "scale": float(scale),
                    "time_s": time_s,
                    "time_h": time_h,
                    "requested_time_points_h": requested_text,
                    "reaction_class": reaction_class,
                    "class_abs_rate_uM_per_h": float(class_abs[reaction_class]),
                    "class_signed_rate_uM_per_h": float(class_signed[reaction_class]),
                }
            )

    return {
        "species_rows": species_rows,
        "reaction_rows": reaction_rows,
        "class_rows": class_rows,
        "requested_time_points_h": requested_text,
        "actual_time_points_h": [float(t_s[i] / 3600.0) for i in snapshot_idxs],
    }


def run_rr_and_collect(
    *,
    rr,
    counts: Dict[str, int],
    params: Dict[str, float],
    component: str,
    scale: float,
    perturbation_kind: str,
    qssa_floors_zeroed: List[str],
    n_scaled_species: int,
    scaled_species: List[str],
    scaled_params: List[str],
    set_params: Dict[str, float],
    t_end_h: float,
    n_points: int,
    gamma_h: float,
    reaction_table: Optional[pd.DataFrame] = None,
    snapshot_time_points_h: Optional[List[float]] = None,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float], Optional[Dict[str, object]]]:
    snapshot_time_points_h = snapshot_time_points_h or []
    param_apply = apply_params(rr, params)
    for pid in qssa_floors_zeroed:
        if pid in set(rr.getGlobalParameterIds()):
            rr[pid] = 0.0

    include_all_species = bool(snapshot_time_points_h)
    selections = build_selection_columns(rr, include_all_species=include_all_species)
    t0 = time.time()
    res = rr.simulate(0.0, float(t_end_h) * 3600.0, int(n_points), selections)
    runtime_s = float(time.time() - t0)
    full_ts = pd.DataFrame(res, columns=res.colnames)
    ts = subset_key_species_timeseries(full_ts) if include_all_species else full_ts
    endpoint = endpoint_species_uM(rr)

    t_h = ts["time"].to_numpy(dtype=float) / 3600.0
    degfp = ts["[deGFP_m]"].to_numpy(dtype=float) * 1e6 if "[deGFP_m]" in ts else np.array([np.nan])
    row: Dict[str, object] = {
        "component": component,
        "scale": float(scale),
        "gamma_h": float(gamma_h),
        "t_end_h": float(t_end_h),
        "runtime_s": runtime_s,
        "species_total": int(counts["species_total"]),
        "reactions_total": int(counts["reactions_total"]),
        "param_applied": int(param_apply["applied"]),
        "param_missing": int(param_apply["missing"]),
        "perturbation_kind": perturbation_kind,
        "qssa_floors_zeroed": ",".join(qssa_floors_zeroed),
        "n_scaled_species": int(n_scaled_species),
        "deGFP_end_uM": float(degfp[-1]),
        "deGFP_peak_uM": float(np.nanmax(degfp)),
        "deGFP_peak_time_h": float(t_h[int(np.nanargmax(degfp))]) if np.isfinite(degfp).any() else np.nan,
        "scaled_species": ",".join(scaled_species),
        "scaled_params": ",".join(scaled_params),
        "set_params": json.dumps(set_params, ensure_ascii=False, sort_keys=True),
    }
    for sid in KEY_SPECIES:
        col = f"[{sid}]"
        if col in ts:
            vals = ts[col].to_numpy(dtype=float) * 1e6
            row[f"{sid}_end_uM"] = float(vals[-1])
            row[f"{sid}_peak_uM"] = float(np.nanmax(vals))

    sparse_bundle = None
    if snapshot_time_points_h:
        if reaction_table is None:
            raise ValueError("reaction_table is required when snapshot_time_points_h is provided")
        sparse_bundle = build_sparse_snapshot_bundle(
            rr=rr,
            full_ts=full_ts,
            reaction_table=reaction_table,
            component=component,
            scale=float(scale),
            requested_time_points_h=snapshot_time_points_h,
        )
        row["snapshot_requested_time_points_h"] = sparse_bundle["requested_time_points_h"]
        row["snapshot_actual_time_points_h"] = json.dumps(sparse_bundle["actual_time_points_h"], ensure_ascii=False)
        row["snapshot_n_time_points"] = int(len(sparse_bundle["actual_time_points_h"]))

    return row, ts, endpoint, sparse_bundle


def simulate_baseline(
    *,
    params: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    base_ic_uM: Dict[str, float],
    gamma_h: float,
    t_end_h: float,
    n_points: int,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float]]:
    rr, counts, _meta, _sbml = build_baseline_rr(
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    row, ts, endpoint, _sparse_bundle = run_rr_and_collect(
        rr=rr,
        counts=counts,
        params=params,
        component="__baseline__",
        scale=1.0,
        perturbation_kind="shared_baseline_1x",
        qssa_floors_zeroed=[],
        n_scaled_species=0,
        scaled_species=[],
        scaled_params=[],
        set_params={},
        t_end_h=float(t_end_h),
        n_points=int(n_points),
        gamma_h=float(gamma_h),
    )
    return row, ts, endpoint


def simulate_condition(
    *,
    component: str,
    scale: float,
    base_ic_uM: Dict[str, float],
    params: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    gamma_h: float,
    t_end_h: float,
    n_points: int,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float]]:
    rr, counts, _meta, perturb_meta, _sbml = build_perturbed_rr(
        component=component,
        scale=scale,
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    row, ts, endpoint, _sparse_bundle = run_rr_and_collect(
        rr=rr,
        counts=counts,
        params=params,
        component=component,
        scale=float(scale),
        perturbation_kind=str(perturb_meta["kind"]),
        qssa_floors_zeroed=list(perturb_meta.get("qssa_floor_zeroed", [])),
        n_scaled_species=int(len(perturb_meta.get("species_scaled", []))),
        scaled_species=list(perturb_meta.get("species_scaled", [])),
        scaled_params=list(perturb_meta.get("params_scaled", [])),
        set_params=dict(perturb_meta.get("params_set", {})),
        t_end_h=float(t_end_h),
        n_points=int(n_points),
        gamma_h=float(gamma_h),
    )
    return row, ts, endpoint


def simulate_baseline_with_sparse_snapshots(
    *,
    params: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    base_ic_uM: Dict[str, float],
    gamma_h: float,
    t_end_h: float,
    n_points: int,
    snapshot_time_points_h: List[float],
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float], Dict[str, object], pd.DataFrame]:
    rr, counts, _meta, sbml = build_baseline_rr(
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    reaction_table = reaction_table_from_sbml(sbml)
    row, ts, endpoint, sparse_bundle = run_rr_and_collect(
        rr=rr,
        counts=counts,
        params=params,
        component="__baseline__",
        scale=1.0,
        perturbation_kind="shared_baseline_1x",
        qssa_floors_zeroed=[],
        n_scaled_species=0,
        scaled_species=[],
        scaled_params=[],
        set_params={},
        t_end_h=float(t_end_h),
        n_points=int(n_points),
        gamma_h=float(gamma_h),
        reaction_table=reaction_table,
        snapshot_time_points_h=snapshot_time_points_h,
    )
    assert sparse_bundle is not None
    return row, ts, endpoint, sparse_bundle, reaction_table


def simulate_condition_with_sparse_snapshots(
    *,
    component: str,
    scale: float,
    base_ic_uM: Dict[str, float],
    params: Dict[str, float],
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    gamma_h: float,
    t_end_h: float,
    n_points: int,
    snapshot_time_points_h: List[float],
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float], Dict[str, object], pd.DataFrame]:
    rr, counts, _meta, perturb_meta, sbml = build_perturbed_rr(
        component=component,
        scale=scale,
        base_ic_uM=base_ic_uM,
        hf_fit=hf_fit,
        gamma_h=gamma_h,
        keff_map=keff_map,
    )
    reaction_table = reaction_table_from_sbml(sbml)
    row, ts, endpoint, sparse_bundle = run_rr_and_collect(
        rr=rr,
        counts=counts,
        params=params,
        component=component,
        scale=float(scale),
        perturbation_kind=str(perturb_meta["kind"]),
        qssa_floors_zeroed=list(perturb_meta.get("qssa_floor_zeroed", [])),
        n_scaled_species=int(len(perturb_meta.get("species_scaled", []))),
        scaled_species=list(perturb_meta.get("species_scaled", [])),
        scaled_params=list(perturb_meta.get("params_scaled", [])),
        set_params=dict(perturb_meta.get("params_set", {})),
        t_end_h=float(t_end_h),
        n_points=int(n_points),
        gamma_h=float(gamma_h),
        reaction_table=reaction_table,
        snapshot_time_points_h=snapshot_time_points_h,
    )
    assert sparse_bundle is not None
    return row, ts, endpoint, sparse_bundle, reaction_table


def get_initial_uM(model: "libsbml.Model", sid: str) -> float:
    sp = model.getSpecies(sid)
    if sp is None:
        return float("nan")
    if sp.isSetInitialConcentration():
        return float(sp.getInitialConcentration()) * 1e6
    if sp.isSetInitialAmount():
        return float(sp.getInitialAmount())
    return float("nan")


def get_param(model: "libsbml.Model", pid: str) -> float:
    p = model.getParameter(pid)
    if p is None:
        return float("nan")
    return float(p.getValue())


def run_correctness_checks(
    *,
    outdir: Path,
    hf_fit: Dict[str, Dict[str, float]],
    keff_map: Dict[str, float],
    base_ic_uM: Dict[str, float],
    gamma_h: float,
) -> pd.DataFrame:
    checks = [
        ("ATP", 0.0),
        ("CTP", 0.0),
        ("GTP", 0.0),
        ("UTP", 0.0),
        ("MetRS", 0.0),
        ("__BASELINE__", 1.0),
    ]
    rows: List[Dict[str, object]] = []
    for component, scale in checks:
        if component == "__BASELINE__":
            model, counts, _meta = build_baseline_model(
                base_ic_uM=base_ic_uM,
                hf_fit=hf_fit,
                gamma_h=gamma_h,
                keff_map=keff_map,
            )
            perturb_meta = {"kind": "shared_baseline_1x", "qssa_floor_zeroed": []}
        else:
            model, counts, _meta, perturb_meta = build_perturbed_model(
                component=component,
                scale=scale,
                base_ic_uM=base_ic_uM,
                hf_fit=hf_fit,
                keff_map=keff_map,
                gamma_h=gamma_h,
            )
        row = {
            "check_name": ("baseline_x1" if component == "__BASELINE__" else f"{component}_x{tag_float(scale)}"),
            "component": component,
            "scale": float(scale),
            "species_total": int(counts["species_total"]),
            "reactions_total": int(counts["reactions_total"]),
            "perturbation_kind": str(perturb_meta["kind"]),
            "qssa_floors_zeroed": ",".join(perturb_meta.get("qssa_floor_zeroed", [])),
            "ATP_init_uM": get_initial_uM(model, "ATP"),
            "ADP_init_uM": get_initial_uM(model, "ADP"),
            "AMP_init_uM": get_initial_uM(model, "AMP"),
            "CTP_init_uM": get_initial_uM(model, "CTP"),
            "UTP_init_uM": get_initial_uM(model, "UTP"),
            "bgRes_CTP_init_uM": get_initial_uM(model, "bgRes_CTP"),
            "bgRes_UTP_init_uM": get_initial_uM(model, "bgRes_UTP"),
            "bg_feed_ATP_M": get_param(model, "bg_feed_ATP"),
            "bg_ATP_feed_M": get_param(model, "bg_ATP_feed"),
            "bg_feed_CTP_M": get_param(model, "bg_feed_CTP"),
            "bg_CTP_feed_M": get_param(model, "bg_CTP_feed"),
            "bg_feed_GTP_M": get_param(model, "bg_feed_GTP"),
            "bg_GTP_feed_M": get_param(model, "bg_GTP_feed"),
            "bg_feed_UTP_M": get_param(model, "bg_feed_UTP"),
            "bg_UTP_feed_M": get_param(model, "bg_UTP_feed"),
            "bg_ATP_inf_scale": get_param(model, "bg_ATP_inf_scale"),
            "bg_GTP_inf_scale": get_param(model, "bg_GTP_inf_scale"),
            "bg_CTP_inf_M": get_param(model, "bg_CTP_inf"),
            "bg_UTP_inf_M": get_param(model, "bg_UTP_inf"),
            "phi_tl_init_floor": get_param(model, "phi_tl_init_floor"),
            "phi_tl_elong_floor": get_param(model, "phi_tl_elong_floor"),
            "phi_tl_term_floor": get_param(model, "phi_tl_term_floor"),
        }
        metrs_species = [sid for sid in perturb_meta.get("species_scaled", []) if "MetRS" in sid]
        if metrs_species:
            row["MetRS_related_max_init_uM"] = float(
                np.nanmax([get_initial_uM(model, sid) for sid in metrs_species])
            )
        else:
            row["MetRS_related_max_init_uM"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "correctness_checks.csv", index=False)
    return df


def write_sparse_snapshot_files(
    *,
    outdir: Path,
    label: str,
    component: str,
    scale: float,
    sparse_bundle: Dict[str, object],
    reaction_table: pd.DataFrame,
) -> Dict[str, object]:
    snap_root = outdir / "sparse_snapshots"
    species_path = snap_root / "species" / f"species_snapshot_{label}.csv.gz"
    reaction_path = snap_root / "reactions" / f"reaction_snapshot_{label}.csv.gz"
    class_path = snap_root / "reaction_classes" / f"reaction_class_snapshot_{label}.csv.gz"
    reaction_map_path = snap_root / "reaction_class_map.csv"

    pd.DataFrame(sparse_bundle["species_rows"]).to_csv(species_path, index=False, compression="gzip")
    pd.DataFrame(sparse_bundle["reaction_rows"]).to_csv(reaction_path, index=False, compression="gzip")
    pd.DataFrame(sparse_bundle["class_rows"]).to_csv(class_path, index=False, compression="gzip")
    if not reaction_map_path.exists():
        reaction_table.to_csv(reaction_map_path, index=False)

    return {
        "component": component,
        "scale": float(scale),
        "label": label,
        "is_shared_baseline": bool(component == "__baseline__" and np.isclose(scale, 1.0)),
        "is_alias": False,
        "alias_of_label": "",
        "requested_time_points_h": str(sparse_bundle.get("requested_time_points_h", "")),
        "actual_time_points_h": json.dumps(sparse_bundle.get("actual_time_points_h", []), ensure_ascii=False),
        "snapshot_n_time_points": int(len(sparse_bundle.get("actual_time_points_h", []))),
        "species_snapshot_csv_gz": str(species_path),
        "reaction_snapshot_csv_gz": str(reaction_path),
        "reaction_class_snapshot_csv_gz": str(class_path),
        "reaction_class_map_csv": str(reaction_map_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun synchronized perturbations for NTP and aaRS on the new-link model.")
    ap.add_argument("--components", type=str, default="ATP,CTP,GTP,UTP," + ",".join(AARS))
    ap.add_argument("--scales", type=str, default="0.25,1,4")
    ap.add_argument("--gamma-h", type=float, default=0.5)
    ap.add_argument("--t-end-h", type=float, default=6.0)
    ap.add_argument("--n-points", type=int, default=121)
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    ap.add_argument("--save-timeseries", action="store_true")
    ap.add_argument("--snapshot-time-points-h", type=str, default="")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "timeseries").mkdir(exist_ok=True)
    (args.outdir / "endpoints").mkdir(exist_ok=True)

    components = [x.strip() for x in args.components.split(",") if x.strip()]
    unsupported = [x for x in components if x not in set(NTPS + AARS)]
    if unsupported:
        raise ValueError(f"Unsupported components: {unsupported}")
    scales = parse_scales(args.scales)
    snapshot_time_points_h = parse_snapshot_time_points(args.snapshot_time_points_h)
    if any(x > float(args.t_end_h) + 1e-12 for x in snapshot_time_points_h):
        raise ValueError("All snapshot time points must be within [0, t_end_h]")

    if snapshot_time_points_h:
        snap_root = args.outdir / "sparse_snapshots"
        (snap_root / "species").mkdir(parents=True, exist_ok=True)
        (snap_root / "reactions").mkdir(parents=True, exist_ok=True)
        (snap_root / "reaction_classes").mkdir(parents=True, exist_ok=True)

    params, hf_fit, keff_map, base_ic_uM = load_sources()
    manifest = {
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "components": components,
        "scales": scales,
        "gamma_h": float(args.gamma_h),
        "t_end_h": float(args.t_end_h),
        "n_points": int(args.n_points),
        "snapshot_time_points_h": snapshot_time_points_h,
        "semantics": {
            "NTP": "family-total + feed + background closure synchronized",
            "aaRS": "total enzyme pool synchronized across enzyme-containing species, and aggregate AA translation bottleneck/feed are co-scaled to match the reduced translation chain",
            "QSSA_floors": "phi_tl_init/elong/term floors set to 0",
        },
    }
    write_json(args.outdir / "run_manifest.json", manifest)

    correctness_df = run_correctness_checks(
        outdir=args.outdir,
        hf_fit=hf_fit,
        keff_map=keff_map,
        base_ic_uM=base_ic_uM,
        gamma_h=float(args.gamma_h),
    )

    rows: List[Dict[str, object]] = []
    endpoint_rows: List[Dict[str, object]] = []
    snapshot_manifest_rows: List[Dict[str, object]] = []
    if not args.skip_baseline and snapshot_time_points_h:
        baseline_row, baseline_ts, baseline_endpoint, baseline_sparse_bundle, baseline_reaction_table = simulate_baseline_with_sparse_snapshots(
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            base_ic_uM=base_ic_uM,
            gamma_h=float(args.gamma_h),
            t_end_h=float(args.t_end_h),
            n_points=int(args.n_points),
            snapshot_time_points_h=snapshot_time_points_h,
        )
        baseline_snapshot_meta = write_sparse_snapshot_files(
            outdir=args.outdir,
            label="baseline_shared_1x",
            component="__baseline__",
            scale=1.0,
            sparse_bundle=baseline_sparse_bundle,
            reaction_table=baseline_reaction_table,
        )
    elif not args.skip_baseline:
        baseline_row, baseline_ts, baseline_endpoint = simulate_baseline(
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            base_ic_uM=base_ic_uM,
            gamma_h=float(args.gamma_h),
            t_end_h=float(args.t_end_h),
            n_points=int(args.n_points),
        )
        baseline_snapshot_meta = None
    if not args.skip_baseline:
        baseline_ep_path = args.outdir / "endpoints" / "endpoint_baseline_shared_1x.csv"
        pd.DataFrame({"species": list(baseline_endpoint.keys()), "final_uM": list(baseline_endpoint.values())}).to_csv(
            baseline_ep_path, index=False
        )
        if args.save_timeseries:
            baseline_ts_path = args.outdir / "timeseries" / "timeseries_baseline_shared_1x.csv"
            baseline_ts.to_csv(baseline_ts_path, index=False)
        total = int(len(components) * len([s for s in scales if not np.isclose(s, 1.0)]) + 1)
        done = 0
        print(f"[RUN] {done + 1}/{total} baseline_shared_1x", flush=True)
        baseline_record = dict(baseline_row)
        baseline_record["status"] = "ok"
        baseline_record["component"] = "__baseline__"
        baseline_record["endpoint_csv"] = str(baseline_ep_path)
        if args.save_timeseries:
            baseline_record["timeseries_csv"] = str(baseline_ts_path)
        if baseline_snapshot_meta is not None:
            baseline_record["species_snapshot_csv_gz"] = baseline_snapshot_meta["species_snapshot_csv_gz"]
            baseline_record["reaction_snapshot_csv_gz"] = baseline_snapshot_meta["reaction_snapshot_csv_gz"]
            baseline_record["reaction_class_snapshot_csv_gz"] = baseline_snapshot_meta["reaction_class_snapshot_csv_gz"]
            snapshot_manifest_rows.append(baseline_snapshot_meta)
        rows.append(baseline_record)
        for sid, val in baseline_endpoint.items():
            endpoint_rows.append({"component": "__baseline__", "scale": 1.0, "species": sid, "final_uM": val})
        done += 1
        print(
            f"[OK] baseline_shared_1x deGFP={baseline_record['deGFP_end_uM']:.6g} "
            f"ATP={baseline_record.get('ATP_end_uM', np.nan):.6g} GTP={baseline_record.get('GTP_end_uM', np.nan):.6g} "
            f"CTP={baseline_record.get('CTP_end_uM', np.nan):.6g} UTP={baseline_record.get('UTP_end_uM', np.nan):.6g} "
            f"runtime={baseline_record['runtime_s']:.1f}s",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(args.outdir / "synced_subset_metrics_partial.csv", index=False)
    else:
        total = int(len(components) * len([s for s in scales if not np.isclose(s, 1.0)]))
        done = 0

    for component in components:
        for scale in scales:
            if np.isclose(float(scale), 1.0):
                if args.skip_baseline:
                    continue
                row = dict(baseline_row)
                row["component"] = component
                row["scale"] = 1.0
                row["status"] = "ok"
                row["perturbation_kind"] = "shared_baseline_1x"
                row["endpoint_csv"] = str(baseline_ep_path)
                if args.save_timeseries:
                    row["timeseries_csv"] = str(baseline_ts_path)
                if baseline_snapshot_meta is not None:
                    row["species_snapshot_csv_gz"] = baseline_snapshot_meta["species_snapshot_csv_gz"]
                    row["reaction_snapshot_csv_gz"] = baseline_snapshot_meta["reaction_snapshot_csv_gz"]
                    row["reaction_class_snapshot_csv_gz"] = baseline_snapshot_meta["reaction_class_snapshot_csv_gz"]
                    snapshot_manifest_rows.append(
                        {
                            "component": component,
                            "scale": 1.0,
                            "label": f"{component}_x1_shared_baseline_alias",
                            "is_shared_baseline": False,
                            "is_alias": True,
                            "alias_of_label": "baseline_shared_1x",
                            "requested_time_points_h": baseline_snapshot_meta["requested_time_points_h"],
                            "actual_time_points_h": baseline_snapshot_meta["actual_time_points_h"],
                            "snapshot_n_time_points": baseline_snapshot_meta["snapshot_n_time_points"],
                            "species_snapshot_csv_gz": baseline_snapshot_meta["species_snapshot_csv_gz"],
                            "reaction_snapshot_csv_gz": baseline_snapshot_meta["reaction_snapshot_csv_gz"],
                            "reaction_class_snapshot_csv_gz": baseline_snapshot_meta["reaction_class_snapshot_csv_gz"],
                            "reaction_class_map_csv": baseline_snapshot_meta["reaction_class_map_csv"],
                        }
                    )
                rows.append(row)
                for sid, val in baseline_endpoint.items():
                    endpoint_rows.append({"component": component, "scale": 1.0, "species": sid, "final_uM": val})
                pd.DataFrame(rows).to_csv(args.outdir / "synced_subset_metrics_partial.csv", index=False)
                continue
            label = f"{component}_x{tag_float(scale)}"
            done += 1
            print(f"[RUN] {done}/{total} {label}", flush=True)
            if snapshot_time_points_h:
                row, ts, endpoint, sparse_bundle, reaction_table = simulate_condition_with_sparse_snapshots(
                    component=component,
                    scale=scale,
                    base_ic_uM=base_ic_uM,
                    params=params,
                    hf_fit=hf_fit,
                    keff_map=keff_map,
                    gamma_h=float(args.gamma_h),
                    t_end_h=float(args.t_end_h),
                    n_points=int(args.n_points),
                    snapshot_time_points_h=snapshot_time_points_h,
                )
            else:
                row, ts, endpoint = simulate_condition(
                    component=component,
                    scale=scale,
                    base_ic_uM=base_ic_uM,
                    params=params,
                    hf_fit=hf_fit,
                    keff_map=keff_map,
                    gamma_h=float(args.gamma_h),
                    t_end_h=float(args.t_end_h),
                    n_points=int(args.n_points),
                )
            row["status"] = "ok"
            ep_path = args.outdir / "endpoints" / f"endpoint_{label}.csv"
            pd.DataFrame({"species": list(endpoint.keys()), "final_uM": list(endpoint.values())}).to_csv(ep_path, index=False)
            row["endpoint_csv"] = str(ep_path)
            if args.save_timeseries:
                ts_path = args.outdir / "timeseries" / f"timeseries_{label}.csv"
                ts.to_csv(ts_path, index=False)
                row["timeseries_csv"] = str(ts_path)
            if snapshot_time_points_h:
                snapshot_meta = write_sparse_snapshot_files(
                    outdir=args.outdir,
                    label=label,
                    component=component,
                    scale=float(scale),
                    sparse_bundle=sparse_bundle,
                    reaction_table=reaction_table,
                )
                row["species_snapshot_csv_gz"] = snapshot_meta["species_snapshot_csv_gz"]
                row["reaction_snapshot_csv_gz"] = snapshot_meta["reaction_snapshot_csv_gz"]
                row["reaction_class_snapshot_csv_gz"] = snapshot_meta["reaction_class_snapshot_csv_gz"]
                snapshot_manifest_rows.append(snapshot_meta)
            rows.append(row)
            for sid, val in endpoint.items():
                endpoint_rows.append({"component": component, "scale": float(scale), "species": sid, "final_uM": val})
            print(
                f"[OK] {label} deGFP={row['deGFP_end_uM']:.6g} ATP={row.get('ATP_end_uM', np.nan):.6g} "
                f"GTP={row.get('GTP_end_uM', np.nan):.6g} CTP={row.get('CTP_end_uM', np.nan):.6g} "
                f"UTP={row.get('UTP_end_uM', np.nan):.6g} runtime={row['runtime_s']:.1f}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(args.outdir / "synced_subset_metrics_partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    if not metrics.empty and not args.skip_baseline:
        metrics["baseline_deGFP_end_uM"] = float(baseline_row["deGFP_end_uM"])
        metrics["deGFP_delta_vs_component_1x_uM"] = metrics["deGFP_end_uM"] - metrics["baseline_deGFP_end_uM"]
        metrics["deGFP_fold_vs_component_1x"] = metrics["deGFP_end_uM"] / metrics["baseline_deGFP_end_uM"]
    metrics.to_csv(args.outdir / "synced_subset_metrics.csv", index=False)

    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(args.outdir / "synced_subset_endpoints.csv", index=False)
    if snapshot_manifest_rows:
        pd.DataFrame(snapshot_manifest_rows).to_csv(args.outdir / "sparse_snapshot_manifest.csv", index=False)

    snapshot_cols = [
        "check_name",
        "ATP_init_uM",
        "CTP_init_uM",
        "bgRes_CTP_init_uM",
        "bgRes_UTP_init_uM",
        "bg_feed_ATP_M",
        "bg_feed_CTP_M",
        "bg_feed_GTP_M",
        "bg_feed_UTP_M",
        "bg_ATP_inf_scale",
        "bg_GTP_inf_scale",
        "phi_tl_init_floor",
    ]
    snapshot_cols = [c for c in snapshot_cols if c in correctness_df.columns]

    summary = {
        "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_csv": str(args.outdir / "synced_subset_metrics.csv"),
        "endpoints_csv": str(args.outdir / "synced_subset_endpoints.csv"),
        "correctness_checks_csv": str(args.outdir / "correctness_checks.csv"),
        "correctness_snapshot": correctness_df[snapshot_cols].to_dict(orient="records"),
    }
    if snapshot_manifest_rows:
        summary["sparse_snapshot_manifest_csv"] = str(args.outdir / "sparse_snapshot_manifest.csv")
    write_json(args.outdir / "run_summary.json", summary)
    print(f"[DONE] {args.outdir / 'synced_subset_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
