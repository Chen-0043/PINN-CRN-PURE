#!/usr/bin/env python3
"""
Route-1 multi-protein simulation on top of the validated full-CRN background model.

What this script does:
1) Rebuilds the background-enabled batch model (same HF background policy as current workflow).
2) Reuses fitted batch scales from the formal policy-fit snapshot.
3) Adds multiple coarse-grained expression channels that share global resources.
4) Runs one multiplex scenario + per-gene single controls.
5) Reports crosstalk metrics (multiplex yield suppression vs single).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import libsbml
import numpy as np
import pandas as pd

# Tellurium stack may import matplotlib; pin writable cache dir to avoid
# repeated font-cache rebuild in restricted HOME environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import tellurium as te

import fit_batch_then_continuous_full_crn_background as wf


DEFAULT_FORMAL_SUMMARY = Path("output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json")
DEFAULT_HF_SUMMARY = Path("output_full_crn_background/batch_then_continuous_summary.json")
DEFAULT_OUTDIR = Path("output_route1_multigene")


@dataclass(frozen=True)
class GeneConfig:
    name: str
    aa_length: int
    tl_scale: float = 1.0


DEFAULT_GENES: List[GeneConfig] = [
    # Approximate E. coli enzyme lengths (aa). Used for translation resource load.
    GeneConfig(name="Pgi", aa_length=549, tl_scale=1.0),
    GeneConfig(name="PfkA", aa_length=320, tl_scale=1.0),
    GeneConfig(name="TpiA", aa_length=255, tl_scale=1.0),
    GeneConfig(name="GAPDH", aa_length=335, tl_scale=1.0),
    GeneConfig(name="Pgk", aa_length=394, tl_scale=1.0),
    GeneConfig(name="Eno", aa_length=432, tl_scale=1.0),
]


def _ensure_species(
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
    sp.setInitialConcentration(float(init_conc))
    if sp.isSetInitialAmount():
        sp.unsetInitialAmount()
    sp.setBoundaryCondition(False)
    sp.setConstant(False)
    sp.setHasOnlySubstanceUnits(False)
    return sp


def _set_reaction_formula(rxn: "libsbml.Reaction", formula: str) -> None:
    kl = rxn.getKineticLaw()
    if kl is None:
        kl = rxn.createKineticLaw()
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise ValueError(f"Invalid formula: {formula}")
    kl.setMath(ast)


def _create_reaction(
    model: "libsbml.Model",
    rid: str,
    reactants: Dict[str, float],
    products: Dict[str, float],
    formula: str,
    reversible: bool = False,
) -> None:
    wf.remove_reaction_if_exists(model, rid)
    rxn = model.createReaction()
    rxn.setId(rid)
    rxn.setReversible(bool(reversible))
    rxn.setFast(False)

    for sid, sto in reactants.items():
        ref = rxn.createReactant()
        ref.setSpecies(sid)
        ref.setStoichiometry(float(sto))
        ref.setConstant(True)

    for sid, sto in products.items():
        ref = rxn.createProduct()
        ref.setSpecies(sid)
        ref.setStoichiometry(float(sto))
        ref.setConstant(True)

    _set_reaction_formula(rxn, formula)


def _load_formal_batch_params(summary_json: Path) -> Dict[str, float]:
    data = json.loads(summary_json.read_text())
    return {str(k): float(v) for k, v in data.get("batch_best_params", {}).items()}


def _apply_params_if_present(rr, params: Dict[str, float]) -> None:
    pids = set(rr.getGlobalParameterIds())
    for pid, val in params.items():
        if pid in pids:
            rr[pid] = float(val)


def _set_fine_node_env_defaults() -> None:
    # Keep this aligned with the deGFP 1nt/micro3 + 1aa model settings.
    os.environ.setdefault("PURE_TX_PATHWAY_MODE", "1nt")
    os.environ.setdefault("PURE_TX_STEP_MODE", "micro3")
    os.environ.setdefault("PURE_ENERGY_REGEN_MODE", "background")
    os.environ.setdefault("PURE_TL_PATHWAY_MODE", "step_chain")
    os.environ.setdefault("PURE_TL_QSSA_MODE", "factorized")
    os.environ.setdefault("PURE_TL_CHAIN_GROUP_AA", "1.0")
    os.environ.setdefault("PURE_TL_CHAIN_SCALE", "30.0")
    os.environ.setdefault("PURE_RIBO_BIND_MODE", "explicit")
    os.environ.setdefault("PURE_RIBO_BIND_SCALE", "1.0")
    # Exchange bookkeeping: default to the new explicit family bookkeeping mode.
    os.environ.setdefault("PURE_BG_FAMILY_EXCHANGE_BOOKKEEP", "1")


def _load_hf_and_keff_from_summary(hf_summary_json: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    payload = json.loads(Path(hf_summary_json).read_text())
    fitted_hf_raw = payload.get("hf_fit_params", {})
    keff_raw = payload.get("batch_fit", {}).get("keff_map", {})
    fitted_hf: Dict[str, Dict[str, float]] = {}
    for sid, fp in dict(fitted_hf_raw).items():
        entry: Dict[str, float] = {}
        for k, v in dict(fp).items():
            try:
                entry[str(k)] = float(v)
            except Exception:
                # keep non-numeric fields such as profile/trend for build_background_sbml
                entry[str(k)] = v
        fitted_hf[str(sid)] = entry
    keff_map = {str(k): float(v) for k, v in dict(keff_raw).items()}
    if len(fitted_hf) == 0:
        raise RuntimeError(f"Missing hf_fit_params in {hf_summary_json}")
    if len(keff_map) == 0:
        raise RuntimeError(f"Missing batch_fit.keff_map in {hf_summary_json}")
    return fitted_hf, keff_map


def _configure_rr(rr) -> None:
    wf.configure_rr(rr)
    integ = rr.getIntegrator()
    # Slightly relaxed tolerances for larger multigene scenarios.
    atol = float(os.environ.get("PURE_RR_ATOL", "1e-10"))
    rtol = float(os.environ.get("PURE_RR_RTOL", "1e-5"))
    max_steps = int(float(os.environ.get("PURE_RR_MAX_STEPS", "1000000")))
    for key, value in (
        ("absolute_tolerance", atol),
        ("relative_tolerance", rtol),
        ("maximum_num_steps", max_steps),
    ):
        try:
            integ.setValue(key, value)
        except Exception:
            pass


def _rebuild_background_batch_model(
    gamma_h: float,
    fit_points: int,
    seed_params: Dict[str, float] | None = None,
    hf_summary_json: Path | None = None,
    force_fine_node_env: bool = False,
) -> Tuple[str, Dict[str, int], Dict[str, object]]:
    if bool(force_fine_node_env):
        _set_fine_node_env_defaults()

    sbml_ref, ic_uM, _, present_hf = wf.build_patched_full_sbml()

    if hf_summary_json is not None:
        fitted_hf_all, keff_map = _load_hf_and_keff_from_summary(Path(hf_summary_json))
        # Keep only the HF species present in the currently built reference model.
        fitted_hf = {sid: fitted_hf_all[sid] for sid in present_hf if sid in fitted_hf_all}
        if len(fitted_hf) == 0:
            raise RuntimeError(
                f"hf summary {hf_summary_json} has no overlap with present_hf species ({present_hf})"
            )
        sbml_bg, counts_bg, hf_meta = wf.build_background_sbml(
            mode="batch",
            fitted_hf=fitted_hf,
            ic_uM=ic_uM,
            gamma_h=gamma_h,
            feed_multiplier=1.0,
            keff_map=keff_map,
        )
        hf_meta = dict(hf_meta)
        hf_meta["keff_map"] = {k: float(v) for k, v in keff_map.items()}
        hf_meta["hf_source"] = str(Path(hf_summary_json))
        hf_meta["hf_mode"] = "locked_from_summary"
        return sbml_bg, counts_bg, hf_meta

    rr_ref = te.loadSBMLModel(sbml_ref)
    _configure_rr(rr_ref)
    if seed_params:
        _apply_params_if_present(rr_ref, seed_params)
    df_ref = wf.simulate_reference_batch(rr_ref, present_hf, t_end_s=8 * 3600.0, n_points=fit_points)
    fitted_hf = wf.fit_background_from_reference(df_ref, present_hf)

    sbml_bg0, _, _ = wf.build_background_sbml(
        mode="batch",
        fitted_hf=fitted_hf,
        ic_uM=ic_uM,
        gamma_h=gamma_h,
        feed_multiplier=1.0,
        keff_map=None,
    )
    rr_bg0 = te.loadSBMLModel(sbml_bg0)
    _configure_rr(rr_bg0)
    if seed_params:
        _apply_params_if_present(rr_bg0, seed_params)
    df_bg0 = wf.simulate_degfp(rr_bg0, t_end_s=8 * 3600.0, n_points=fit_points, reset_to_origin=True)
    keff_map = wf.fit_keff_map_from_flux_ratio(df_ref, df_bg0)

    sbml_bg, counts_bg, hf_meta = wf.build_background_sbml(
        mode="batch",
        fitted_hf=fitted_hf,
        ic_uM=ic_uM,
        gamma_h=gamma_h,
        feed_multiplier=1.0,
        keff_map=keff_map,
    )
    hf_meta = dict(hf_meta)
    hf_meta["keff_map"] = {k: float(v) for k, v in keff_map.items()}
    return sbml_bg, counts_bg, hf_meta


def _extend_model_with_multigene_channels(
    sbml_text: str,
    genes: List[GeneConfig],
    tx_share: Dict[str, float],
    degfp_tx_share: float,
) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse SBML")

    mRNA_ref = model.getSpecies("mRNA_t")
    if mRNA_ref is None:
        raise RuntimeError("Base model missing mRNA_t species")
    compartment_id = mRNA_ref.getCompartment()

    counts_before = {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
    }

    # Partition transcription output of the existing channel.
    wf.ensure_parameter(model, "gene_share_degfp", float(degfp_tx_share))
    r343 = model.getReaction("r343")
    if r343 is None:
        raise RuntimeError("Base model missing reaction r343")
    _set_reaction_formula(r343, "k_forward7 * T7RNAP_bound_mRNA0113 * gene_share_degfp")

    has_bg_keff = model.getParameter("bg_keff_scale") is not None and model.getParameter("bg_keff") is not None
    keff_factor = " * (bg_keff_scale * bg_keff)" if has_bg_keff else ""
    wf.ensure_parameter(model, "k_ribo_bind_f", 195464.87719773027)
    wf.ensure_parameter(model, "k_ribo_bind_r", 0.34584637957254144)
    wf.ensure_parameter(model, "k_tl_init", 0.026998731015091766)
    wf.ensure_parameter(model, "k_tl_elong", 0.009476441238810956)
    wf.ensure_parameter(model, "k_tl_term", 0.055568416391920346)
    wf.ensure_parameter(model, "K_tl_AA", 3.2535684250485034e-05)
    wf.ensure_parameter(model, "K_tl_ATP", 6.11033863918126e-05)
    wf.ensure_parameter(model, "K_tl_GTP", 3.113087089656521e-05)
    wf.ensure_parameter(model, "K_tl_eps", 1.0e-18)

    protein_species_by_gene: Dict[str, str] = {}
    for g in genes:
        gid = wf.sanitize_id(g.name)
        share_val = float(tx_share.get(g.name, 0.0))
        wf.ensure_parameter(model, f"gene_share_{gid}", share_val)
        wf.ensure_parameter(model, f"gene_tl_scale_{gid}", float(g.tl_scale))
        wf.ensure_parameter(model, f"gene_len_scale_{gid}", 226.0 / float(max(1, g.aa_length)))

        sid_mrna = f"mRNA_t_{gid}"
        sid_bound = f"Ribosome_mRNA_{gid}"
        sid_pept = f"Pept_{gid}"
        sid_prot = f"{gid}_m"
        protein_species_by_gene[g.name] = sid_prot

        _ensure_species(model, sid_mrna, compartment_id, 0.0)
        _ensure_species(model, sid_bound, compartment_id, 0.0)
        _ensure_species(model, sid_pept, compartment_id, 0.0)
        _ensure_species(model, sid_prot, compartment_id, 0.0)

        _create_reaction(
            model=model,
            rid=f"rxn_tx_partition_{gid}",
            reactants={"T7RNAP_bound_mRNA0113": 1.0},
            products={"T7RNAP": 1.0, "DNA": 1.0, "mRNA_i": 1.0, sid_mrna: 1.0},
            formula=f"k_forward7 * T7RNAP_bound_mRNA0113 * gene_share_{gid}",
            reversible=False,
        )

        _create_reaction(
            model=model,
            rid=f"rxn_ribosome_binding_{gid}",
            reactants={sid_mrna: 1.0, "Ribosome": 1.0},
            products={sid_bound: 1.0},
            formula=(
                f"default * k_ribo_bind_f * {sid_mrna} * Ribosome"
                f" - default * k_ribo_bind_r * {sid_bound}"
            ),
            reversible=True,
        )

        aa_need = max(1, int(g.aa_length))
        gtp_need = 2 * aa_need
        _create_reaction(
            model=model,
            rid=f"rxn_translation_elongation_{gid}",
            reactants={sid_bound: 1.0, "AA": float(aa_need), "ATP": float(aa_need), "GTP": float(gtp_need)},
            products={"Ribosome": 1.0, sid_mrna: 1.0, sid_pept: 1.0},
            formula=(
                f"gene_tl_scale_{gid} * gene_len_scale_{gid} * {sid_bound} * ("
                "("
                "(default * k_tl_init * (GTP / (K_tl_GTP + GTP))) * "
                "(default * k_tl_elong * "
                "(AA / (K_tl_AA + AA)) * "
                "(ATP / (K_tl_ATP + ATP)) * "
                "(GTP / (K_tl_GTP + GTP))) * "
                "(default * k_tl_term * (GTP / (K_tl_GTP + GTP)))"
                ") / ("
                "K_tl_eps + "
                "(default * k_tl_init * (GTP / (K_tl_GTP + GTP))) * "
                "(default * k_tl_elong * "
                "(AA / (K_tl_AA + AA)) * "
                "(ATP / (K_tl_ATP + ATP)) * "
                "(GTP / (K_tl_GTP + GTP))) + "
                "(default * k_tl_init * (GTP / (K_tl_GTP + GTP))) * "
                "(default * k_tl_term * (GTP / (K_tl_GTP + GTP))) + "
                "(default * k_tl_elong * "
                "(AA / (K_tl_AA + AA)) * "
                "(ATP / (K_tl_ATP + ATP)) * "
                "(GTP / (K_tl_GTP + GTP))) * "
                "(default * k_tl_term * (GTP / (K_tl_GTP + GTP)))"
                ")"
                ")"
                f"{keff_factor}"
            ),
            reversible=False,
        )

        _create_reaction(
            model=model,
            rid=f"reaction_protein_folding_{gid}",
            reactants={sid_pept: 1.0},
            products={sid_prot: 1.0},
            formula=f"0.001667 * {sid_pept}",
            reversible=False,
        )

    counts_after = {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
    }
    counts_delta = {
        "species_added": counts_after["species_total"] - counts_before["species_total"],
        "reactions_added": counts_after["reactions_total"] - counts_before["reactions_total"],
    }
    return libsbml.writeSBMLToString(doc), protein_species_by_gene, counts_delta


def _simulate(rr, protein_species: Dict[str, str], t_end_h: float, points: int) -> pd.DataFrame:
    sel = ["time", "[ATP]", "[GTP]", "[PPi]", "[PO4]", "[Ribosome]", "[deGFP_m]", "[mRNA_t]"]
    for _, sid in protein_species.items():
        sel.append(f"[{sid}]")
        sel.append(f"[mRNA_t_{wf.sanitize_id(sid[:-2])}]")  # "<gene>_m" -> "mRNA_t_<gene>"
    sel = list(dict.fromkeys(sel))
    t_end_s = float(t_end_h) * 3600.0
    n_points = int(points)
    try:
        res = rr.simulate(0.0, t_end_s, n_points, sel)
        return pd.DataFrame(res, columns=res.colnames)
    except RuntimeError:
        # Retry once with looser tolerances when CVODE gets stuck in stiff pockets.
        integ = rr.getIntegrator()
        for key, value in (
            ("absolute_tolerance", 1e-9),
            ("relative_tolerance", 3e-5),
            ("maximum_num_steps", 2000000),
        ):
            try:
                integ.setValue(key, value)
            except Exception:
                pass
        rr.resetToOrigin()
        res = rr.simulate(0.0, t_end_s, n_points, sel)
        return pd.DataFrame(res, columns=res.colnames)


def _protein_metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    t_h = df["time"].to_numpy(dtype=float) / 3600.0
    y_uM = np.maximum(df[col].to_numpy(dtype=float), 0.0) * 1e6

    def interp(xh: float) -> float:
        return float(np.interp(float(xh), t_h, y_uM))

    if t_h.size > 1:
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(y_uM, t_h))
        else:
            auc = float(np.trapz(y_uM, t_h))
    else:
        auc = 0.0
    return {
        "protein_2h_uM": interp(2.0),
        "protein_4h_uM": interp(4.0),
        "protein_6h_uM": interp(6.0),
        "protein_8h_uM": interp(8.0),
        "protein_peak_uM": float(np.max(y_uM)) if y_uM.size else 0.0,
        "protein_auc_uM_h": auc,
    }


def _resource_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sid in ("ATP", "GTP", "PPi", "PO4", "Ribosome"):
        col = f"[{sid}]"
        if col not in df.columns:
            continue
        y = np.asarray(df[col].to_numpy(dtype=float), dtype=float)
        out[f"{sid}_min_uM"] = float(np.min(y) * 1e6) if y.size else 0.0
        out[f"{sid}_max_uM"] = float(np.max(y) * 1e6) if y.size else 0.0
    return out


def _run_scenario(
    sbml_bg: str,
    genes: List[GeneConfig],
    tx_share: Dict[str, float],
    degfp_tx_share: float,
    batch_params: Dict[str, float],
    t_end_h: float,
    points: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float], Dict[str, int]]:
    sbml_ext, prot_map, counts_delta = _extend_model_with_multigene_channels(
        sbml_text=sbml_bg,
        genes=genes,
        tx_share=tx_share,
        degfp_tx_share=degfp_tx_share,
    )
    rr = te.loadSBMLModel(sbml_ext)
    _configure_rr(rr)
    _apply_params_if_present(rr, batch_params)
    df = _simulate(rr, prot_map, t_end_h=t_end_h, points=points)

    protein_metrics = {g.name: _protein_metrics(df, f"[{prot_map[g.name]}]") for g in genes}
    resource_metrics = _resource_metrics(df)
    return df, protein_metrics, resource_metrics, counts_delta


def _build_equal_share(genes: List[GeneConfig]) -> Dict[str, float]:
    if not genes:
        return {}
    s = 1.0 / float(len(genes))
    return {g.name: s for g in genes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Route-1 multi-protein simulation with shared-resource crosstalk.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_FORMAL_SUMMARY, help="Formal fit summary JSON.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--gamma-h", type=float, default=1.0, help="Batch gamma parameter (h^-1) for background build.")
    parser.add_argument("--fit-points", type=int, default=220, help="Points for reference/background fitting runs.")
    parser.add_argument("--t-end-h", type=float, default=8.0, help="Simulation horizon in hours.")
    parser.add_argument("--points", type=int, default=280, help="Output points per scenario.")
    parser.add_argument("--degfp-share", type=float, default=0.0, help="Transcription flux share kept on native deGFP channel.")
    parser.add_argument("--skip-single-controls", action="store_true", help="Skip per-gene single controls.")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    genes = list(DEFAULT_GENES)
    batch_params = _load_formal_batch_params(args.summary_json)
    sbml_bg, counts_bg, hf_meta = _rebuild_background_batch_model(
        gamma_h=float(args.gamma_h),
        fit_points=int(args.fit_points),
        seed_params=batch_params,
    )

    multiplex_share = _build_equal_share(genes)
    df_multi, metrics_multi, resource_multi, counts_delta = _run_scenario(
        sbml_bg=sbml_bg,
        genes=genes,
        tx_share=multiplex_share,
        degfp_tx_share=float(args.degfp_share),
        batch_params=batch_params,
        t_end_h=float(args.t_end_h),
        points=int(args.points),
    )
    df_multi.to_csv(outdir / "route1_multiplex_timeseries.csv", index=False)

    single_metrics: Dict[str, Dict[str, float]] = {}
    if not args.skip_single_controls:
        for g in genes:
            single_share = {x.name: (1.0 if x.name == g.name else 0.0) for x in genes}
            df_single, m_single, _, _ = _run_scenario(
                sbml_bg=sbml_bg,
                genes=genes,
                tx_share=single_share,
                degfp_tx_share=float(args.degfp_share),
                batch_params=batch_params,
                t_end_h=float(args.t_end_h),
                points=int(args.points),
            )
            df_single.to_csv(outdir / f"route1_single_{wf.sanitize_id(g.name)}_timeseries.csv", index=False)
            single_metrics[g.name] = m_single[g.name]

    crosstalk_rows = []
    for g in genes:
        m = metrics_multi[g.name]
        s = single_metrics.get(g.name)
        if s is not None:
            single_p6 = float(s["protein_6h_uM"])
            multi_p6 = float(m["protein_6h_uM"])
            ratio = multi_p6 / max(single_p6, 1e-12)
            suppression = 1.0 - ratio
        else:
            single_p6 = float("nan")
            ratio = float("nan")
            suppression = float("nan")
        crosstalk_rows.append(
            {
                "gene": g.name,
                "aa_length": g.aa_length,
                "tx_share_multiplex": float(multiplex_share[g.name]),
                "protein_6h_uM_multiplex": float(m["protein_6h_uM"]),
                "protein_6h_uM_single": single_p6,
                "multiplex_to_single_ratio": ratio,
                "suppression_fraction": suppression,
            }
        )
    df_crosstalk = pd.DataFrame(crosstalk_rows).sort_values("suppression_fraction", ascending=False, na_position="last")
    df_crosstalk.to_csv(outdir / "route1_crosstalk_metrics.csv", index=False)

    summary = {
        "workflow": "route1_multigene_shared_resource",
        "assumptions": {
            "shared_transcription_flux_partition": True,
            "shared_translation_resources": ["Ribosome", "AA", "ATP", "GTP"],
            "explicit_assembly_kon_koff": False,
            "gene_set": [{**g.__dict__} for g in genes],
            "degfp_tx_share": float(args.degfp_share),
            "multiplex_tx_share": multiplex_share,
        },
        "counts_background_batch": counts_bg,
        "counts_multigene_delta": counts_delta,
        "batch_params_source": str(args.summary_json),
        "batch_params_used": batch_params,
        "hf_meta": hf_meta,
        "metrics": {
            "multiplex_protein_metrics": metrics_multi,
            "multiplex_resource_metrics": resource_multi,
            "single_protein_metrics": single_metrics,
        },
    }
    (outdir / "route1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("Saved:")
    print(outdir / "route1_summary.json")
    print(outdir / "route1_multiplex_timeseries.csv")
    print(outdir / "route1_crosstalk_metrics.csv")
    if not args.skip_single_controls:
        for g in genes:
            print(outdir / f"route1_single_{wf.sanitize_id(g.name)}_timeseries.csv")


if __name__ == "__main__":
    main()
