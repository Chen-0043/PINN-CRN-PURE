#!/usr/bin/env python3
"""
Segmented 6-gene co-expression simulator with explicit 1nt TX and 1aa TL
position chains, integrated by a positivity-preserving chain solver.

This is a diagnostic/production candidate for the multi-protein route where
generic CVODE/RoadRunner integration of thousands of ultra-low-concentration
chain intermediates becomes either inaccurate (oneStep) or too slow (simulate).

The model keeps the overlapping fitted kinetic parameters and public resource
pool logic, but integrates the unidirectional TX/TL chains directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import libsbml
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

import simulate_route1_multigene_background as route1
import simulate_route1_multigene_explicit_tx_hybrid as hyb
import simulate_route1_twogene_explicit_tx_lines as etx


AA20 = list(etx.AA20)
NTP_FOR_NT = {"A": "ATP", "C": "CTP", "G": "GTP", "U": "UTP"}
RESOURCE_LIMITED_IDS = ["ATP", "GTP", "CTP", "UTP", *AA20]


@dataclass
class GeneChainState:
    name: str
    gid: str
    tx_scale: float
    tl_scale: float
    dna_free: float
    rna_nt: str
    tx_ntp_ids: List[str]
    aa_counts: Dict[str, int]
    aa_total: int
    tx_pos: np.ndarray
    activity_decay_h: float | None = None
    mrna: float = 0.0
    ribo_mrna: float = 0.0
    tl_pos: np.ndarray | None = None
    pept: float = 0.0
    protein: float = 0.0


def _snapshot_dynamic_gene_state(genes: List[GeneChainState]) -> List[Dict[str, object]]:
    """Copy only mutable state used by one integration step."""
    snapshots: List[Dict[str, object]] = []
    for gene in genes:
        snapshots.append(
            {
                "dna_free": float(gene.dna_free),
                "tx_pos": gene.tx_pos.copy(),
                "mrna": float(gene.mrna),
                "ribo_mrna": float(gene.ribo_mrna),
                "tl_pos": None if gene.tl_pos is None else gene.tl_pos.copy(),
                "pept": float(gene.pept),
                "protein": float(gene.protein),
            }
        )
    return snapshots


def _limit_gene_step_to_available_resources(
    genes: List[GeneChainState],
    snapshots: List[Dict[str, object]],
    pool: Dict[str, float],
    pool_before_step: Dict[str, float],
    delta: Dict[str, float],
) -> float:
    """Scale a tentative multi-gene step when shared substrates are exhausted.

    The chain solvers advance state before their shared-pool deltas are applied.
    Simply clipping a negative pool therefore creates product without the
    corresponding ATP/GTP/NTP/amino-acid consumption.  A single common scale
    preserves competition among all genes and keeps state progress consistent
    with the amount of shared substrate that was actually available.
    """
    scale = 1.0
    for sid in RESOURCE_LIMITED_IDS:
        demand = -float(delta.get(sid, 0.0))
        if demand <= 0.0:
            continue
        available = max(0.0, float(pool_before_step.get(sid, 0.0)))
        scale = min(scale, available / demand)
    scale = float(np.clip(scale, 0.0, 1.0))
    if scale >= 1.0 - 1e-15:
        return 1.0

    scalar_fields = ("dna_free", "mrna", "ribo_mrna", "pept", "protein")
    for gene, old in zip(genes, snapshots):
        for field in scalar_fields:
            old_value = float(old[field])
            new_value = float(getattr(gene, field))
            setattr(gene, field, old_value + scale * (new_value - old_value))
        old_tx = np.asarray(old["tx_pos"], dtype=float)
        gene.tx_pos = old_tx + scale * (gene.tx_pos - old_tx)
        old_tl = old["tl_pos"]
        if old_tl is not None and gene.tl_pos is not None:
            old_tl_array = np.asarray(old_tl, dtype=float)
            gene.tl_pos = old_tl_array + scale * (gene.tl_pos - old_tl_array)

    # Ribosome binding is the only foreground operation that mutates the pool
    # before the aggregate delta is applied, so blend it over the same step.
    old_ribosome = float(pool_before_step.get("Ribosome", 0.0))
    pool["Ribosome"] = old_ribosome + scale * (float(pool.get("Ribosome", 0.0)) - old_ribosome)
    for sid in list(delta):
        delta[sid] = float(delta[sid]) * scale
    return scale


@lru_cache(maxsize=1)
def _read_model_defaults_cached() -> Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]:
    sbml_bg, _counts_bg, _hf_meta = route1._rebuild_background_batch_model(
        gamma_h=1.0,
        fit_points=120,
        hf_summary_json=Path("output_full_crn_background/batch_then_continuous_summary.json"),
        force_fine_node_env=True,
    )
    doc = libsbml.readSBMLFromString(sbml_bg)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse background SBML")
    params: Dict[str, float] = {}
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        if p is not None:
            params[p.getId()] = float(p.getValue())
    species: Dict[str, float] = {}
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp is None:
            continue
        sid = sp.getId()
        if sp.isSetInitialConcentration():
            species[sid] = float(sp.getInitialConcentration())
        elif sp.isSetInitialAmount():
            species[sid] = float(sp.getInitialAmount())
    return tuple(sorted(params.items())), tuple(sorted(species.items()))


def _read_model_defaults() -> Tuple[Dict[str, float], Dict[str, float]]:
    params_items, species_items = _read_model_defaults_cached()
    return dict(params_items), dict(species_items)


@lru_cache(maxsize=1)
def _gene_setup_background_sbml() -> str:
    sbml_bg, _counts_bg, _hf_meta = route1._rebuild_background_batch_model(
        gamma_h=1.0,
        fit_points=120,
        hf_summary_json=Path("output_full_crn_background/batch_then_continuous_summary.json"),
        force_fine_node_env=True,
    )
    return sbml_bg


def _param(params: Dict[str, float], name: str, fallback: float) -> float:
    return float(params.get(name, fallback))


def _soft_sat(x: float, km: float, floor: float) -> float:
    x = max(0.0, float(x))
    return float(floor + (1.0 - floor) * (x / (float(km) + x + 1e-300)))


def _sat(x: float, km: float) -> float:
    x = max(0.0, float(x))
    return float(x / (float(km) + x + 1e-300))


def _hill_on(x: float, k: float, n: float) -> float:
    x = max(0.0, float(x))
    k = max(1e-300, float(k))
    n = max(1.0, float(n))
    xn = x**n
    kn = k**n
    return float(xn / (xn + kn + 1e-300))


def _aa_drive(pool: Dict[str, float], aa_counts: Dict[str, int], k_aa: float) -> float:
    total = float(sum(max(0, int(v)) for v in aa_counts.values()))
    if total <= 0:
        return _sat(pool.get("AA", 0.0), k_aa)
    log_acc = 0.0
    for aa in AA20:
        cnt = int(aa_counts.get(aa, 0))
        if cnt <= 0:
            continue
        s = max(1e-300, _sat(pool.get(aa, 0.0), k_aa))
        log_acc += (float(cnt) / total) * math.log(s)
    return float(math.exp(log_acc))


def _validate_strict_crn_args(args: argparse.Namespace, genes: List[GeneChainState]) -> None:
    tol = 1e-12

    def _require_close(label: str, value: float, target: float = 1.0) -> None:
        if abs(float(value) - float(target)) > tol:
            raise RuntimeError(
                f"strict CRN mode requires {label}={target:g}, got {float(value):.12g}"
            )

    def _require_zero(label: str, value: float) -> None:
        if abs(float(value)) > tol:
            raise RuntimeError(f"strict CRN mode requires {label}=0, got {float(value):.12g}")

    _require_close("ribo_bind_scale", float(args.ribo_bind_scale), 1.0)
    _require_close("tx_gain", float(args.tx_gain), 1.0)
    _require_close("tl_gain", float(args.tl_gain), 1.0)
    if float(args.mrna_decay_base) < 0.0 or float(args.mrna_bound_decay_ratio) < 0.0:
        raise RuntimeError("strict CRN mode requires nonnegative mRNA degradation parameters")
    if float(args.global_activity_decay_h) < 0.0:
        raise RuntimeError("strict CRN mode requires nonnegative activity decay")
    if any(g.activity_decay_h is not None and float(g.activity_decay_h) < 0.0 for g in genes):
        raise RuntimeError("strict CRN mode requires nonnegative gene-specific activity decay")
    _require_zero("sg_tx_boost_max", float(args.sg_tx_boost_max))
    _require_zero("sg_tl_gtp_gate_K_M", float(args.sg_tl_gtp_gate_K_M))

    sg_mode = str(args.sg_public_mode).strip().lower()
    if sg_mode not in {"off", "none", "disabled"}:
        raise RuntimeError(
            "strict CRN mode requires sg_public_mode to be off/none, "
            f"got {args.sg_public_mode!r}"
        )

    # Gene-specific chain factors are allowed in strict foreground mode as long
    # as they are treated as route-specific parameters rather than shared
    # numerical bridge gains.


def _advance_chain_backward_euler(
    x: np.ndarray,
    trans_rates: np.ndarray,
    source: float,
    sink_rate: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Positive triangular backward-Euler solve for a unidirectional chain."""
    n = len(x) - 1
    y = np.empty_like(x)
    if n <= 0:
        y[0] = (float(x[0]) + dt * float(source)) / (1.0 + dt * float(sink_rate))
        return y, np.zeros(0, dtype=float), float(sink_rate) * float(y[0])

    # Solve the same lower-bidiagonal backward-Euler system in compiled code.
    # This is algebraically identical to the former Python recurrence and is
    # substantially faster for the 1-nt/1-aa chains used by six fusion genes.
    diagonal = np.concatenate(
        (1.0 + dt * np.asarray(trans_rates, dtype=float), np.asarray([1.0 + dt * float(sink_rate)]))
    )
    banded = np.zeros((2, n + 1), dtype=float)
    banded[0] = diagonal
    banded[1, :n] = -dt * np.asarray(trans_rates, dtype=float)
    rhs = np.asarray(x, dtype=float).copy()
    rhs[0] += dt * float(source)
    y = solve_banded(
        (1, 0), banded, rhs, overwrite_ab=True, overwrite_b=True, check_finite=False
    )
    flux = np.asarray(trans_rates, dtype=float) * y[:-1]
    sink_flux = float(sink_rate) * float(y[n])
    return y, flux, sink_flux


@lru_cache(maxsize=32)
def _load_gene_templates(panel_json_str: str, dna_total_M: float) -> Tuple[Dict[str, object], ...]:
    panel_json = Path(panel_json_str)
    panel_payload = json.loads(panel_json.read_text(encoding="utf-8"))
    raw_panel_genes = panel_payload.get("genes", []) if isinstance(panel_payload, dict) else panel_payload
    activity_decay_by_name = {
        str(item["name"]): float(item["activity_decay_h"])
        for item in raw_panel_genes
        if isinstance(item, dict) and item.get("activity_decay_h") is not None
    }
    old_tx = os.environ.get("PURE_TX_PATHWAY_MODE")
    os.environ["PURE_TX_PATHWAY_MODE"] = "1nt"
    try:
        genes_raw, panel_total = hyb._load_gene_panel(panel_json)
        genes_raw = hyb._autofill_known_cds(genes_raw)
        genes_raw = hyb._apply_tx_compression(genes_raw, 1.0)
    finally:
        if old_tx is None:
            os.environ.pop("PURE_TX_PATHWAY_MODE", None)
        else:
            os.environ["PURE_TX_PATHWAY_MODE"] = old_tx

    dna_total = float(panel_total if panel_total is not None else dna_total_M)
    dna_by_name = hyb._allocate_gene_dna(genes_raw, active_total_M=dna_total)
    gene_cfgs = hyb._build_gene_cfgs(genes_raw)

    # _build_gene_tx_rna_sequence only needs a model for fallback template
    # attributes; the CDS-filled panel follows the direct CDS path.
    sbml_bg = _gene_setup_background_sbml()
    model = libsbml.readSBMLFromString(sbml_bg).getModel()
    if model is None:
        raise RuntimeError("Failed to parse background model for gene setup")

    out: List[Dict[str, object]] = []
    for g in gene_cfgs:
        gid = route1.wf.sanitize_id(g.name)
        rna, _meta = etx._build_gene_tx_rna_sequence(model, g)
        aa_counts, aa_meta = etx._build_gene_aa20_profile(g)
        aa_total = max(1, int(aa_meta["aa_total"]))
        tx_ntp_ids = [NTP_FOR_NT[ch] for ch in rna]
        out.append(
            {
                "name": g.name,
                "gid": gid,
                "tx_scale": float(g.tx_scale),
                "tl_scale": float(g.tl_scale),
                "dna_free": float(max(0.0, dna_by_name.get(g.name, 0.0))),
                "rna_nt": rna,
                "tx_ntp_ids": tuple(tx_ntp_ids),
                "aa_counts": tuple(sorted((k, int(v)) for k, v in aa_counts.items())),
                "aa_total": int(aa_total),
                "activity_decay_h": activity_decay_by_name.get(g.name),
            }
        )
    return tuple(out)


def _load_genes(panel_json: Path, dna_total_M: float) -> List[GeneChainState]:
    templates = _load_gene_templates(str(panel_json.resolve()), float(dna_total_M))
    out: List[GeneChainState] = []
    for tpl in templates:
        rna = str(tpl["rna_nt"])
        aa_total = int(tpl["aa_total"])
        out.append(
            GeneChainState(
                name=str(tpl["name"]),
                gid=str(tpl["gid"]),
                tx_scale=float(tpl["tx_scale"]),
                tl_scale=float(tpl["tl_scale"]),
                dna_free=float(tpl["dna_free"]),
                rna_nt=rna,
                tx_ntp_ids=list(tpl["tx_ntp_ids"]),
                aa_counts={str(k): int(v) for k, v in tpl["aa_counts"]},
                aa_total=aa_total,
                tx_pos=np.zeros(len(rna) + 1, dtype=float),
                tl_pos=np.zeros(aa_total + 1, dtype=float),
                activity_decay_h=(
                    None if tpl.get("activity_decay_h") is None else float(tpl["activity_decay_h"])
                ),
            )
        )
    return out


def _initial_pool(species_init: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "ATP",
        "ADP",
        "GTP",
        "CTP",
        "UTP",
        "GDP",
        "PPi",
        "PO4",
        "Glucose",
        "NAD",
        "NADH",
        "TPP",
        "Mg",
        "T7RNAP",
        "Ribosome",
        "IF2",
        "IF2_GTP",
        "EFTu",
        "EFTu_GTP",
        "EFG",
        "EFG_GTP",
        "AA",
        *AA20,
    ]
    return {k: float(max(0.0, species_init.get(k, 0.0))) for k in keys}


def _load_pool_overrides(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pool_overrides_M" in payload:
        payload = payload["pool_overrides_M"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid pool override JSON: {path}")
    out: Dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = float(v)
        except Exception as exc:
            raise RuntimeError(f"Invalid override value for {k!r} in {path}: {v!r}") from exc
    return out


def _apply_pool_overrides(pool: Dict[str, float], overrides: Dict[str, float]) -> Dict[str, float]:
    applied: Dict[str, float] = {}
    for sid, value in overrides.items():
        if sid in pool:
            pool[sid] = max(0.0, float(value))
            applied[sid] = float(pool[sid])
    return applied


def _bg_keff_factor_formal(pool: Dict[str, float], params: Dict[str, float]) -> float:
    """Mirror the background SBML assignment rule for bg_keff."""
    atp = max(0.0, float(pool.get("ATP", 0.0)))
    gtp = max(0.0, float(pool.get("GTP", 0.0)))
    po4 = max(0.0, float(pool.get("PO4", 0.0)))
    ppi = max(0.0, float(pool.get("PPi", 0.0)))
    inf_scale = _param(params, "bg_keff_inf_scale", 1.0)
    r_inf = _param(params, "bg_keff_r_inf", 2.0)
    r0 = _param(params, "bg_keff_r0", 0.370831925223904)
    k_atp = _param(params, "bg_keff_K_ATP", 1.0e-7) * _param(params, "bg_keff_KATP_scale", 1.0)
    k_gtp = _param(params, "bg_keff_K_GTP", 3.07510079072268e-4) * _param(params, "bg_keff_KGTP_scale", 1.0)
    ki_po4 = _param(params, "bg_keff_KI_PO4", 0.0499996986708244) * _param(params, "bg_keff_KIPO4_scale", 1.0)
    ki_ppi = _param(params, "bg_keff_KI_PPi", 1.30199093609539e-8) * _param(params, "bg_keff_KIPPi_scale", 1.0)
    keff = (
        inf_scale * r_inf
        + (r0 - inf_scale * r_inf)
        * (atp / (atp + k_atp + 1e-300))
        * (gtp / (gtp + k_gtp + 1e-300))
        * (1.0 / (1.0 + po4 / (ki_po4 + 1e-300)))
        * (1.0 / (1.0 + ppi / (ki_ppi + 1e-300)))
    )
    scale = _param(params, "bg_keff_scale", 1.0)
    return float(max(0.0, scale * keff))


def _bg_activity_factor(pool: Dict[str, float], params: Dict[str, float], mode: str) -> float:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"unity", "none", "off", "disabled"}:
        return 1.0
    if mode_norm in {"formal", "pure_bg", "bg_keff"}:
        return _bg_keff_factor_formal(pool, params)
    raise RuntimeError(f"Unknown bg activity mode: {mode}")


def _sg_public_tx_factor(pool: Dict[str, float], args: argparse.Namespace) -> float:
    mode = str(args.sg_public_mode).strip().lower()
    if mode in {"off", "none", "disabled"}:
        return 1.0
    if mode not in {"paper_norm", "sg_pure", "shared_gate"}:
        raise RuntimeError(f"Unknown sg public mode: {args.sg_public_mode}")

    def _paper_ratio(sid: str, ref: float) -> float:
        if ref <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, pool.get(sid, 0.0) / ref)))

    gate = (
        _paper_ratio("Glucose", float(args.sg_ref_glucose_M))
        * _paper_ratio("NAD", float(args.sg_ref_nad_M))
        * _paper_ratio("TPP", float(args.sg_ref_tpp_M))
        * _paper_ratio("Mg", float(args.sg_ref_mg_M))
    )
    boost = 1.0 + max(0.0, float(args.sg_tx_boost_max)) * gate
    adp = max(0.0, float(pool.get("ADP", 0.0)))
    gdp = max(0.0, float(pool.get("GDP", 0.0)))
    po4_excess = max(0.0, float(pool.get("PO4", 0.0)) - float(args.sg_ref_po4_M))
    debt = (
        1.0 / (1.0 + adp / (float(args.sg_k_adp_M) + 1e-300))
        * 1.0 / (1.0 + gdp / (float(args.sg_k_gdp_M) + 1e-300))
        * 1.0 / (1.0 + po4_excess / (float(args.sg_k_po4_excess_M) + 1e-300))
    )
    return float(max(0.0, boost * debt))


def _sg_energy_board_fluxes(pool: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    mode = str(getattr(args, "sg_energy_mode", "off")).strip().lower()
    if mode in {"off", "none", "disabled"}:
        return {"gate_reg": 0.0, "gate_ndk": 0.0, "r_atp_reg": 0.0, "r_ndk": 0.0}
    if mode not in {"minimal", "qssa", "sg_qssa"}:
        raise RuntimeError(f"Unknown sg energy mode: {args.sg_energy_mode}")

    gate_reg = (
        _sat(pool.get("Glucose", 0.0), float(args.sg_energy_K_glucose_M))
        * _sat(pool.get("NAD", 0.0), float(args.sg_energy_K_nad_M))
        * _sat(pool.get("TPP", 0.0), float(args.sg_energy_K_tpp_M))
        * _sat(pool.get("Mg", 0.0), float(args.sg_energy_K_mg_M))
        * _sat(pool.get("ADP", 0.0), float(args.sg_energy_K_adp_reg_M))
    )
    gate_ndk = (
        _sat(pool.get("GDP", 0.0), float(args.sg_energy_K_gdp_ndk_M))
        * _sat(pool.get("ATP", 0.0), float(args.sg_energy_K_atp_ndk_M))
        * _sat(pool.get("Mg", 0.0), float(args.sg_energy_K_mg_M))
    )
    r_atp_reg = float(max(0.0, float(args.sg_energy_k_atp_reg_M_s) * gate_reg))
    r_ndk = float(max(0.0, float(args.sg_energy_k_ndk_M_s) * gate_ndk))
    return {
        "gate_reg": float(gate_reg),
        "gate_ndk": float(gate_ndk),
        "r_atp_reg": r_atp_reg,
        "r_ndk": r_ndk,
    }


def _apply_sg_energy_board(pool: Dict[str, float], args: argparse.Namespace, dt: float) -> Dict[str, float]:
    flux = _sg_energy_board_fluxes(pool, args)
    delta: Dict[str, float] = {}

    atp_yield_per_glucose = max(1e-12, float(args.sg_energy_atp_per_glucose))

    atp_reg_amt = max(0.0, float(flux["r_atp_reg"]) * float(dt))
    atp_reg_amt = min(
        atp_reg_amt,
        max(0.0, pool.get("ADP", 0.0)),
        max(0.0, pool.get("Glucose", 0.0)) * atp_yield_per_glucose,
    )
    if atp_reg_amt > 0.0:
        delta["ATP"] = delta.get("ATP", 0.0) + atp_reg_amt
        delta["ADP"] = delta.get("ADP", 0.0) - atp_reg_amt
        delta["Glucose"] = delta.get("Glucose", 0.0) - (atp_reg_amt / atp_yield_per_glucose)

    ndk_amt = max(0.0, float(flux["r_ndk"]) * float(dt))
    ndk_amt = min(
        ndk_amt,
        max(0.0, pool.get("GDP", 0.0) + delta.get("GDP", 0.0)),
        max(0.0, pool.get("ATP", 0.0) + delta.get("ATP", 0.0)),
    )
    if ndk_amt > 0.0:
        delta["ATP"] = delta.get("ATP", 0.0) - ndk_amt
        delta["ADP"] = delta.get("ADP", 0.0) + ndk_amt
        delta["GTP"] = delta.get("GTP", 0.0) + ndk_amt
        delta["GDP"] = delta.get("GDP", 0.0) - ndk_amt

    return delta


def _apply_delta(pool: Dict[str, float], delta: Dict[str, float]) -> None:
    for sid, dv in delta.items():
        pool[sid] = max(0.0, float(pool.get(sid, 0.0)) + float(dv))


def _advance_shared_ribosome_binding(
    genes: List[GeneChainState],
    pool: Dict[str, float],
    params: Dict[str, float],
    dt: float,
    *,
    ribo_bind_scale: float,
    max_substep_s: float = 0.05,
) -> None:
    """Advance all mRNA-ribosome binding reactions against one shared pool.

    The former per-gene explicit update allowed every template to see the full
    free-ribosome concentration before the summed pool delta was clipped.  That
    violated competition when several genes bound ribosomes in the same step
    and made the result strongly step-size dependent.  This synchronized
    update evaluates all binding proposals from the same state and scales only
    their aggregate positive demand when free ribosomes are limiting.
    """
    if not genes or dt <= 0.0:
        return
    k_bind_f = _param(params, "k_ribo_bind_f", etx.DEFAULT_MECH_QSSA_PARAMS["k_ribo_bind_f"])
    k_bind_r = _param(params, "k_ribo_bind_r", etx.DEFAULT_MECH_QSSA_PARAMS["k_ribo_bind_r"])
    n_substeps = max(1, int(math.ceil(float(dt) / float(max_substep_s))))
    h = float(dt) / float(n_substeps)

    for _ in range(n_substeps):
        ribosome = max(0.0, float(pool.get("Ribosome", 0.0)))
        rates = np.asarray(
            [
                float(ribo_bind_scale)
                * (k_bind_f * max(0.0, g.mrna) * ribosome - k_bind_r * max(0.0, g.ribo_mrna))
                for g in genes
            ],
            dtype=float,
        )

        # Apply dissociation first.  Its released ribosome is available to all
        # association reactions in this same synchronized substep.
        for g, rate in zip(genes, rates):
            if rate >= 0.0:
                continue
            amount = min(max(0.0, -float(rate) * h), max(0.0, g.ribo_mrna))
            g.ribo_mrna -= amount
            g.mrna += amount
            ribosome += amount

        proposals = np.asarray(
            [min(max(0.0, float(rate) * h), max(0.0, g.mrna)) for g, rate in zip(genes, rates)],
            dtype=float,
        )
        total = float(np.sum(proposals))
        if total > ribosome and total > 0.0:
            proposals *= ribosome / total
        for g, amount in zip(genes, proposals):
            amount = float(amount)
            g.mrna -= amount
            g.ribo_mrna += amount
            ribosome -= amount
        pool["Ribosome"] = max(0.0, ribosome)


def _step_gene(
    g: GeneChainState,
    pool: Dict[str, float],
    params: Dict[str, float],
    dt: float,
    *,
    ribo_bind_scale: float,
    mrna_decay_base: float,
    mrna_bound_decay_ratio: float,
    bg_keff_factor: float,
    bg_tx_coupling_power: float,
    activity_factor: float,
    sg_public_tx_factor: float,
    sg_tl_gtp_gate_k_M: float,
    sg_tl_gtp_gate_n: float,
    strict_crn_mode: bool,
) -> Dict[str, float]:
    delta: Dict[str, float] = {}

    k_forward = _param(params, "k_forward", 9.406062)
    k4 = _param(params, "k_forward4", 2.676172)
    k5 = _param(params, "k_forward5", 100.066259)
    k6 = _param(params, "k_forward6", 1306.617911)
    k7 = _param(params, "k_forward7", 45.998466)
    eps = 1e-30

    gtp = pool.get("GTP", 0.0)
    activity = float(max(0.0, activity_factor))
    tx_shared = float(max(0.0, sg_public_tx_factor))
    tx_bg = float(max(0.0, bg_keff_factor)) ** float(max(0.0, bg_tx_coupling_power))
    r0 = activity * tx_shared * tx_bg * k_forward * pool.get("T7RNAP", 0.0) * g.dna_free * gtp * g.tx_scale
    init_amt = max(0.0, r0 * dt)
    init_amt = min(init_amt, g.dna_free, pool.get("T7RNAP", 0.0), pool.get("GTP", 0.0))

    ntp_rates = np.empty(len(g.tx_ntp_ids), dtype=float)
    for i, ntp in enumerate(g.tx_ntp_ids):
        ntp_conc = pool.get(ntp, 0.0)
        denom = (
            1.0 / (k4 * ntp_conc * g.tx_scale * tx_shared * tx_bg + eps)
            + 1.0 / (k5 * g.tx_scale * tx_shared * tx_bg + eps)
            + 1.0 / (k6 * g.tx_scale * tx_shared * tx_bg + eps)
        )
        ntp_rates[i] = 1.0 / denom

    tx_new, tx_flux, release_flux = _advance_chain_backward_euler(
        g.tx_pos, ntp_rates, init_amt / max(dt, eps), k7 * g.tx_scale * tx_shared * tx_bg, dt
    )
    release_amt = max(0.0, release_flux * dt)

    # TX resource bookkeeping.
    g.dna_free = max(0.0, g.dna_free - init_amt + release_amt)
    delta["T7RNAP"] = delta.get("T7RNAP", 0.0) - init_amt + release_amt
    delta["GTP"] = delta.get("GTP", 0.0) - init_amt
    delta["GDP"] = delta.get("GDP", 0.0) + init_amt
    delta["PO4"] = delta.get("PO4", 0.0) + init_amt
    for ntp in ("ATP", "GTP", "CTP", "UTP"):
        used = float(np.sum(tx_flux[[i for i, x in enumerate(g.tx_ntp_ids) if x == ntp]]) * dt)
        if used:
            delta[ntp] = delta.get(ntp, 0.0) - used
            delta["PPi"] = delta.get("PPi", 0.0) + used
    g.tx_pos = np.maximum(tx_new, 0.0)
    g.mrna += release_amt

    # Translation chain gates.
    k_tl_init = _param(params, "k_tl_init", etx.DEFAULT_MECH_QSSA_PARAMS["k_tl_init"])
    k_tl_elong = _param(params, "k_tl_elong", etx.DEFAULT_MECH_QSSA_PARAMS["k_tl_elong"])
    k_tl_term = _param(params, "k_tl_term", etx.DEFAULT_MECH_QSSA_PARAMS["k_tl_term"])
    k_aa = _param(params, "K_tl_AA", etx.DEFAULT_MECH_QSSA_PARAMS["K_tl_AA"])
    k_atp = _param(params, "K_tl_ATP", etx.DEFAULT_MECH_QSSA_PARAMS["K_tl_ATP"])
    k_gtp = _param(params, "K_tl_GTP", etx.DEFAULT_MECH_QSSA_PARAMS["K_tl_GTP"])
    gtp_sat = _sat(pool.get("GTP", 0.0), k_gtp)
    atp_sat = _sat(pool.get("ATP", 0.0), k_atp)
    aa_drive = _aa_drive(pool, g.aa_counts, k_aa)
    if2_tot = pool.get("IF2", 0.0) + pool.get("IF2_GTP", 0.0)
    eftu_tot = pool.get("EFTu", 0.0) + pool.get("EFTu_GTP", 0.0)
    efg_tot = pool.get("EFG", 0.0) + pool.get("EFG_GTP", 0.0)
    if strict_crn_mode:
        init_gate = _sat(if2_tot, 1.0e-8)
        elong_gate = _sat(eftu_tot, 5.0e-8) * _sat(efg_tot, 5.0e-8)
        term_gate = _sat(efg_tot, 5.0e-8)
    else:
        init_gate = _soft_sat(if2_tot, 1.0e-8, 0.25)
        elong_gate = _soft_sat(eftu_tot, 5.0e-8, 0.25) * _soft_sat(efg_tot, 5.0e-8, 0.25)
        term_gate = _soft_sat(efg_tot, 5.0e-8, 0.25)

    n_steps = g.aa_total
    load_rate = g.ribo_mrna * (activity * g.tl_scale * k_tl_init * gtp_sat * init_gate) * bg_keff_factor
    sg_tl_gate = float(max(0.0, sg_public_tx_factor))
    if float(sg_tl_gtp_gate_k_M) > 0.0:
        sg_tl_gate *= _hill_on(pool.get("GTP", 0.0), float(sg_tl_gtp_gate_k_M), float(sg_tl_gtp_gate_n))
    load_rate *= sg_tl_gate
    load_amt = min(g.ribo_mrna, max(0.0, load_rate * dt))
    g.ribo_mrna -= load_amt

    step_rate = (
        g.tl_scale
        * float(n_steps)
        * k_tl_elong
        * aa_drive
        * atp_sat
        * gtp_sat
        * elong_gate
        * bg_keff_factor
        * activity
        * sg_tl_gate
    )
    term_rate = (
        activity
        * g.tl_scale
        * k_tl_term
        * gtp_sat
        * term_gate
        * bg_keff_factor
        * sg_tl_gate
    )
    tl_rates = np.full(n_steps, step_rate, dtype=float)
    tl_new, tl_flux, term_flux = _advance_chain_backward_euler(
        g.tl_pos if g.tl_pos is not None else np.zeros(n_steps + 1),
        tl_rates,
        load_amt / max(dt, eps),
        term_rate,
        dt,
    )
    term_amt = max(0.0, term_flux * dt)
    g.tl_pos = np.maximum(tl_new, 0.0)

    # TL resource bookkeeping. Existing SBML uses average composition per 1aa
    # step when group_aa=1, so mirror that exactly.
    tl_step_total = float(np.sum(tl_flux) * dt)
    atp_per = 1.0
    gtp_per = 2.0
    delta["ATP"] = delta.get("ATP", 0.0) - atp_per * tl_step_total
    delta["ADP"] = delta.get("ADP", 0.0) + atp_per * tl_step_total
    delta["GTP"] = delta.get("GTP", 0.0) - gtp_per * tl_step_total
    delta["GDP"] = delta.get("GDP", 0.0) + gtp_per * tl_step_total
    delta["PO4"] = delta.get("PO4", 0.0) + (atp_per + gtp_per) * tl_step_total
    for aa, cnt in g.aa_counts.items():
        if cnt > 0:
            delta[aa] = delta.get(aa, 0.0) - (float(cnt) / float(n_steps)) * tl_step_total
    delta["Ribosome"] = delta.get("Ribosome", 0.0) + term_amt
    g.mrna += term_amt
    g.pept += term_amt

    # Decay and folding.
    mrna_decay = min(g.mrna, max(0.0, mrna_decay_base * g.mrna * dt))
    g.mrna -= mrna_decay
    bound_decay = min(
        g.ribo_mrna,
        max(0.0, mrna_decay_base * mrna_bound_decay_ratio * g.ribo_mrna * dt),
    )
    g.ribo_mrna -= bound_decay
    delta["Ribosome"] = delta.get("Ribosome", 0.0) + bound_decay
    fold = min(g.pept, max(0.0, 0.001667 * g.pept * dt))
    g.pept -= fold
    g.protein += fold

    return delta


def simulate(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    params_bg, species_init = _read_model_defaults()
    batch_params = etx._load_formal_batch_params(Path(args.summary_json))
    params = dict(params_bg)
    params.update(batch_params)

    pool = _initial_pool(species_init)
    pool_overrides = _load_pool_overrides(Path(args.pool_init_json)) if args.pool_init_json else {}
    pool_initial_default = {k: float(v) for k, v in sorted(pool.items())}
    pool_overrides_applied = _apply_pool_overrides(pool, pool_overrides)
    pool_initial_effective = {k: float(v) for k, v in sorted(pool.items())}
    genes = _load_genes(Path(args.gene_config_json), float(args.dna_total_M))
    for g in genes:
        g.tx_scale *= float(args.tx_gain)
        g.tl_scale *= float(args.tl_gain)
    strict_crn_mode = bool(int(getattr(args, "strict_crn_mode", 0)))
    if strict_crn_mode:
        _validate_strict_crn_args(args, genes)
    dt = float(args.dt_s)
    t_end = float(args.t_end_h) * 3600.0
    record_every = max(dt, float(args.record_every_s))
    next_record = 0.0
    rows: List[Dict[str, float]] = []

    def record(t: float) -> None:
        sg_energy_flux = _sg_energy_board_fluxes(pool, args)
        row: Dict[str, float] = {
            "time": float(t),
            "[ATP]": pool.get("ATP", 0.0),
            "[ADP]": pool.get("ADP", 0.0),
            "[GTP]": pool.get("GTP", 0.0),
            "[GDP]": pool.get("GDP", 0.0),
            "[CTP]": pool.get("CTP", 0.0),
            "[UTP]": pool.get("UTP", 0.0),
            "[PPi]": pool.get("PPi", 0.0),
            "[PO4]": pool.get("PO4", 0.0),
            "[Glucose]": pool.get("Glucose", 0.0),
            "[NAD]": pool.get("NAD", 0.0),
            "[NADH]": pool.get("NADH", 0.0),
            "[Ribosome]": pool.get("Ribosome", 0.0),
            "[T7RNAP]": pool.get("T7RNAP", 0.0),
            "bg_keff_factor": _bg_activity_factor(pool, params, str(args.bg_activity_mode)),
            "sg_public_tx_factor": _sg_public_tx_factor(pool, args),
            "sg_energy_gate_reg": float(sg_energy_flux["gate_reg"]),
            "sg_energy_gate_ndk": float(sg_energy_flux["gate_ndk"]),
            "sg_energy_r_atp_reg": float(sg_energy_flux["r_atp_reg"]),
            "sg_energy_r_ndk": float(sg_energy_flux["r_ndk"]),
        }
        for g in genes:
            row[f"[{g.gid}_m]"] = float(g.protein)
            row[f"[mRNA_t_{g.gid}]"] = float(g.mrna)
            row[f"[Ribosome_mRNA_{g.gid}]"] = float(g.ribo_mrna)
            row[f"[DNA_{g.gid}]"] = float(g.dna_free)
            row[f"[TXchain_{g.gid}_total]"] = float(np.sum(g.tx_pos))
            row[f"[TLchain_{g.gid}_total]"] = float(np.sum(g.tl_pos)) if g.tl_pos is not None else 0.0
        rows.append(row)

    t = 0.0
    record(t)
    n_steps = int(math.ceil(t_end / dt))
    resource_limited_steps = 0
    minimum_resource_scale = 1.0
    for _ in range(n_steps):
        step = min(dt, t_end - t)
        if step <= 0:
            break
        total_delta: Dict[str, float] = {}
        pool_before_step = dict(pool)
        gene_snapshots = _snapshot_dynamic_gene_state(genes)
        bg_keff_factor = _bg_activity_factor(pool, params, str(args.bg_activity_mode))
        sg_tx_factor = _sg_public_tx_factor(pool, args)
        _advance_shared_ribosome_binding(
            genes,
            pool,
            params,
            step,
            ribo_bind_scale=float(args.ribo_bind_scale),
        )
        for g in genes:
            decay_h = (
                float(args.global_activity_decay_h)
                if g.activity_decay_h is None
                else float(g.activity_decay_h)
            )
            activity_factor = math.exp(-max(0.0, decay_h) * (t / 3600.0))
            d = _step_gene(
                g,
                pool,
                params,
                step,
                ribo_bind_scale=float(args.ribo_bind_scale),
                mrna_decay_base=float(args.mrna_decay_base),
                mrna_bound_decay_ratio=float(args.mrna_bound_decay_ratio),
                bg_keff_factor=bg_keff_factor,
                bg_tx_coupling_power=float(args.bg_tx_coupling_power),
                activity_factor=activity_factor,
                sg_public_tx_factor=sg_tx_factor,
                sg_tl_gtp_gate_k_M=float(args.sg_tl_gtp_gate_K_M),
                sg_tl_gtp_gate_n=float(args.sg_tl_gtp_gate_n),
                strict_crn_mode=bool(strict_crn_mode),
            )
            for sid, dv in d.items():
                total_delta[sid] = total_delta.get(sid, 0.0) + float(dv)
        resource_scale = _limit_gene_step_to_available_resources(
            genes,
            gene_snapshots,
            pool,
            pool_before_step,
            total_delta,
        )
        minimum_resource_scale = min(minimum_resource_scale, resource_scale)
        if resource_scale < 1.0 - 1e-15:
            resource_limited_steps += 1
        _apply_delta(pool, total_delta)
        if str(getattr(args, "sg_energy_mode", "off")).strip().lower() not in {"off", "none", "disabled"}:
            _apply_delta(pool, _apply_sg_energy_board(pool, args, step))
        t += step
        if t + 1e-12 >= next_record + record_every or t >= t_end - 1e-12:
            next_record = t
            record(t)

    df = pd.DataFrame(rows)
    summary = {
        "workflow": "sixgene_pool_chain_v3",
        "description": "Direct positivity-preserving 1nt TX / 1aa TL chain simulator with shared public pools.",
        "t_end_h": float(args.t_end_h),
        "dt_s": float(args.dt_s),
        "record_every_s": float(args.record_every_s),
        "tx_gain": float(args.tx_gain),
        "tl_gain": float(args.tl_gain),
        "ribo_bind_scale": float(args.ribo_bind_scale),
        "mrna_decay_base": float(args.mrna_decay_base),
        "mrna_bound_decay_ratio": float(args.mrna_bound_decay_ratio),
        "n_genes": int(len(genes)),
        "genes": [
            {
                "name": g.name,
                "gid": g.gid,
                "tx_nt_steps": int(len(g.rna_nt)),
                "tl_aa_steps": int(g.aa_total),
                "tx_scale": float(g.tx_scale),
                "tl_scale": float(g.tl_scale),
                "activity_decay_h": (
                    float(args.global_activity_decay_h)
                    if g.activity_decay_h is None
                    else float(g.activity_decay_h)
                ),
            }
            for g in genes
        ],
        "final_uM": {g.name: float(g.protein * 1e6) for g in genes},
        "final_mRNA_uM": {g.name: float(g.mrna * 1e6) for g in genes},
        "pool_init_default_M": pool_initial_default,
        "pool_init_overrides_M": {k: float(v) for k, v in sorted(pool_overrides_applied.items())},
        "pool_init_effective_M": pool_initial_effective,
        "pool_final_M": {k: float(v) for k, v in sorted(pool.items())},
        "bg_activity_mode": str(args.bg_activity_mode),
        "bg_keff_factor_final": float(_bg_activity_factor(pool, params, str(args.bg_activity_mode))),
        "bg_tx_coupling_power": float(args.bg_tx_coupling_power),
        "sg_public_mode": str(args.sg_public_mode),
        "sg_public_tx_factor_final": float(_sg_public_tx_factor(pool, args)),
        "sg_energy_mode": str(getattr(args, "sg_energy_mode", "off")),
        "sg_energy_gate_reg_final": float(_sg_energy_board_fluxes(pool, args)["gate_reg"]),
        "sg_energy_gate_ndk_final": float(_sg_energy_board_fluxes(pool, args)["gate_ndk"]),
        "sg_energy_r_atp_reg_final_M_s": float(_sg_energy_board_fluxes(pool, args)["r_atp_reg"]),
        "sg_energy_r_ndk_final_M_s": float(_sg_energy_board_fluxes(pool, args)["r_ndk"]),
        "sg_energy_k_atp_reg_M_s": float(getattr(args, "sg_energy_k_atp_reg_M_s", 0.0)),
        "sg_energy_k_ndk_M_s": float(getattr(args, "sg_energy_k_ndk_M_s", 0.0)),
        "sg_energy_K_adp_reg_M": float(getattr(args, "sg_energy_K_adp_reg_M", 0.0)),
        "sg_energy_K_gdp_ndk_M": float(getattr(args, "sg_energy_K_gdp_ndk_M", 0.0)),
        "sg_energy_K_atp_ndk_M": float(getattr(args, "sg_energy_K_atp_ndk_M", 0.0)),
        "sg_energy_K_glucose_M": float(getattr(args, "sg_energy_K_glucose_M", 0.0)),
        "sg_energy_K_nad_M": float(getattr(args, "sg_energy_K_nad_M", 0.0)),
        "sg_energy_K_tpp_M": float(getattr(args, "sg_energy_K_tpp_M", 0.0)),
        "sg_energy_K_mg_M": float(getattr(args, "sg_energy_K_mg_M", 0.0)),
        "sg_energy_atp_per_glucose": float(getattr(args, "sg_energy_atp_per_glucose", 0.0)),
        "sg_tx_boost_max": float(args.sg_tx_boost_max),
        "sg_ref_glucose_M": float(args.sg_ref_glucose_M),
        "sg_ref_nad_M": float(args.sg_ref_nad_M),
        "sg_ref_tpp_M": float(args.sg_ref_tpp_M),
        "sg_ref_mg_M": float(args.sg_ref_mg_M),
        "sg_ref_po4_M": float(args.sg_ref_po4_M),
        "sg_k_adp_M": float(args.sg_k_adp_M),
        "sg_k_gdp_M": float(args.sg_k_gdp_M),
        "sg_k_po4_excess_M": float(args.sg_k_po4_excess_M),
        "sg_tl_gtp_gate_K_M": float(args.sg_tl_gtp_gate_K_M),
        "sg_tl_gtp_gate_n": float(args.sg_tl_gtp_gate_n),
        "global_activity_decay_h": float(args.global_activity_decay_h),
        "strict_crn_mode": bool(strict_crn_mode),
        "resource_conservation_limiter": {
            "enabled": True,
            "limited_steps": int(resource_limited_steps),
            "minimum_step_scale": float(minimum_resource_scale),
            "limited_species": RESOURCE_LIMITED_IDS,
        },
    }
    return df, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gene-config-json", default="output_tmp_reimagined_v2_solverfix_directbg_1h_20260423/sixgene_panel_safe.json")
    p.add_argument("--summary-json", default="output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json")
    p.add_argument("--outdir", default="output_sixgene_pool_chain_v3")
    p.add_argument("--dna-total-M", type=float, default=5e-9)
    p.add_argument("--t-end-h", type=float, default=3.0)
    p.add_argument("--dt-s", type=float, default=1.0)
    p.add_argument("--record-every-s", type=float, default=60.0)
    p.add_argument("--ribo-bind-scale", type=float, default=1.0)
    p.add_argument("--tx-gain", type=float, default=1.0)
    p.add_argument("--tl-gain", type=float, default=1.0)
    p.add_argument("--mrna-decay-base", type=float, default=1e-6)
    p.add_argument("--mrna-bound-decay-ratio", type=float, default=0.2)
    p.add_argument("--global-activity-decay-h", type=float, default=0.0)
    p.add_argument("--strict-crn-mode", type=int, default=0)
    p.add_argument("--bg-activity-mode", default="formal")
    p.add_argument("--bg-tx-coupling-power", type=float, default=0.0)
    p.add_argument("--pool-init-json", default=None)
    p.add_argument("--sg-public-mode", default="off")
    p.add_argument("--sg-energy-mode", default="off")
    p.add_argument("--sg-energy-k-atp-reg-M-s", type=float, default=0.0)
    p.add_argument("--sg-energy-k-ndk-M-s", type=float, default=0.0)
    p.add_argument("--sg-energy-K-adp-reg-M", type=float, default=4e-4)
    p.add_argument("--sg-energy-K-gdp-ndk-M", type=float, default=3e-4)
    p.add_argument("--sg-energy-K-atp-ndk-M", type=float, default=5e-4)
    p.add_argument("--sg-energy-K-glucose-M", type=float, default=5e-3)
    p.add_argument("--sg-energy-K-nad-M", type=float, default=2e-4)
    p.add_argument("--sg-energy-K-tpp-M", type=float, default=2e-5)
    p.add_argument("--sg-energy-K-mg-M", type=float, default=4e-3)
    p.add_argument("--sg-energy-atp-per-glucose", type=float, default=2.0)
    p.add_argument("--sg-tx-boost-max", type=float, default=0.0)
    p.add_argument("--sg-ref-glucose-M", type=float, default=0.05)
    p.add_argument("--sg-ref-nad-M", type=float, default=0.001)
    p.add_argument("--sg-ref-tpp-M", type=float, default=1e-4)
    p.add_argument("--sg-ref-mg-M", type=float, default=0.016)
    p.add_argument("--sg-ref-po4-M", type=float, default=0.05)
    p.add_argument("--sg-k-adp-M", type=float, default=1e-3)
    p.add_argument("--sg-k-gdp-M", type=float, default=1e-3)
    p.add_argument("--sg-k-po4-excess-M", type=float, default=5e-3)
    p.add_argument("--sg-tl-gtp-gate-K-M", type=float, default=0.0)
    p.add_argument("--sg-tl-gtp-gate-n", type=float, default=6.0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, summary = simulate(args)
    df.to_csv(outdir / "pool_chain_timeseries.csv", index=False)
    (outdir / "pool_chain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
