#!/usr/bin/env python3
"""Direct full-elementary mass-action solve with optional continuous exchange.

No teacher trajectory is used as a constraint.  A PINN/refined node file can be
supplied only to reuse its time grid for output/evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from generate_stiff_hidden_anchors import build_exchange_numpy, build_exchange_plan
from optimize_pinn_nodes_by_mass_action_shooting import hidden_integral_metrics_np
from train_continuous_full_crn_pinn_key import DEFAULT_CONTINUOUS_XML, parse_parameters


BASE = Path(__file__).resolve().parent / "article_gapfill_20260730"
FULL = BASE / "full_elementary_pinn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=FULL)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--method", choices=["BDF", "Radau", "LSODA"], default="BDF")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol-uM", type=float, default=1e-6)
    parser.add_argument("--max-step-s", type=float, default=np.inf)
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument("--max-time-s", type=float, default=21600.0)
    parser.add_argument("--exchange-mode", choices=["batch", "continuous"], default="continuous")
    parser.add_argument("--chunk-s", type=float, default=0.0, help=">0 restarts the stiff solver every chunk-s seconds.")
    parser.add_argument(
        "--chunk-report-stride",
        type=int,
        default=1,
        help="Save every Nth successful chunk report; failed chunks are always saved.",
    )
    parser.add_argument(
        "--initial-state-npz",
        type=Path,
        default=None,
        help="Optional species_id/concentrations_M npz used only for the initial state at --initial-state-time-s.",
    )
    parser.add_argument("--initial-state-time-s", type=float, default=0.0)
    parser.add_argument(
        "--initial-missing-mode",
        choices=["asset", "zero"],
        default="asset",
        help="For species absent from --initial-state-npz, keep asset x0 or set to zero.",
    )
    parser.add_argument("--pinn-grid-npz", type=Path, default=None, help="Optional PINN/refined file used only for t_eval.")
    parser.add_argument("--n-eval", type=int, default=313, help="Used when --pinn-grid-npz is absent; <=0 uses solver internal steps.")
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="For rate-constant screening, keep only the final state and skip integral residual metrics.",
    )
    parser.add_argument("--use-jac-sparsity", action="store_true")
    parser.add_argument(
        "--use-analytic-jac",
        action="store_true",
        help="Provide the analytic sparse Jacobian for BDF/Radau mass-action solves.",
    )
    parser.add_argument(
        "--rate-unit",
        choices=["M", "uM"],
        default="M",
        help="Unit basis of mass-action rate constants. uM rescales k by 1e6^(molecularity-1) for M-valued states.",
    )
    parser.add_argument(
        "--state-unit",
        choices=["M", "uM"],
        default="M",
        help="Numerical state unit used by the ODE solver. uM keeps calibrated uM-rate constants well scaled and converts outputs back to M.",
    )
    parser.add_argument(
        "--scale-aars-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for aminoacyl-tRNA synthetase elementary reactions.",
    )
    parser.add_argument(
        "--scale-aars-amp-forward",
        type=float,
        default=1.0,
        help="Diagnostic scale for AARS aminoacyl-adenylate/PPi forming forward reactions.",
    )
    parser.add_argument(
        "--scale-aars-aaamp-leak",
        type=float,
        default=1.0,
        help="Diagnostic scale for AARS_AAAMP -> AARS + free AAAMP leak reactions.",
    )
    parser.add_argument(
        "--scale-aars-trna-transfer",
        type=float,
        default=1.0,
        help="Diagnostic scale for AARS aminoacyl-transfer steps that form charged tRNA intermediates.",
    )
    parser.add_argument(
        "--scale-transcription-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for T7RNAP elementary transcription reactions.",
    )
    parser.add_argument(
        "--scale-translation-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for ribosome/translation elementary reactions.",
    )
    parser.add_argument(
        "--scale-translation-productive",
        type=float,
        default=1.0,
        help="Diagnostic scale for translation steps that increase the Pept index.",
    )
    parser.add_argument(
        "--scale-translation-initiation",
        type=float,
        default=1.0,
        help="Diagnostic scale for 30S/50S/IF/mRNA initiation and 70S loading reactions.",
    )
    parser.add_argument(
        "--scale-init-mrna-binding",
        type=float,
        default=1.0,
        help="Diagnostic scale for initiation mRNA association reactions.",
    )
    parser.add_argument(
        "--scale-init-mrna-unbinding",
        type=float,
        default=1.0,
        help="Diagnostic scale for initiation mRNA dissociation reactions.",
    )
    parser.add_argument(
        "--scale-init-50s-joining",
        type=float,
        default=1.0,
        help="Diagnostic scale for 50S joining to 30S initiation complexes.",
    )
    parser.add_argument(
        "--scale-init-50s-splitting",
        type=float,
        default=1.0,
        help="Diagnostic scale for reverse 70S initiation complex splitting.",
    )
    parser.add_argument(
        "--scale-init-if2-hydrolysis",
        type=float,
        default=1.0,
        help="Diagnostic scale for IF2-GTP hydrolysis in 70S initiation complexes.",
    )
    parser.add_argument(
        "--scale-init-if-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for IF/IF2-GDP release from 70S initiation complexes.",
    )
    parser.add_argument(
        "--scale-eftu-delivery",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFTu-GTP-aa-tRNA delivery/binding reactions to elongating ribosomes.",
    )
    parser.add_argument(
        "--scale-eftu-free-nucleotide-binding",
        type=float,
        default=1.0,
        help="Diagnostic paired scale for free EFTu GDP/GTP association and dissociation reactions.",
    )
    parser.add_argument(
        "--scale-eftu-hydrolysis-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFTu GTP hydrolysis and EFTu-GDP release after aa-tRNA delivery.",
    )
    parser.add_argument(
        "--scale-eftu-delivery-reverse",
        type=float,
        default=1.0,
        help="Diagnostic scale for reverse EFTu-GTP-aa-tRNA delivery unbinding reactions.",
    )
    parser.add_argument(
        "--scale-eftu-gtp-hydrolysis",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFTu_GTP -> EFTu_GDP_PO4 reactions on elongating ribosomes.",
    )
    parser.add_argument(
        "--scale-eftu-po4-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFTu_GDP_PO4 -> EFTu_GDP + PO4 reactions on elongating ribosomes.",
    )
    parser.add_argument(
        "--scale-eftu-gdp-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFTu_GDP release from elongating ribosome complexes.",
    )
    parser.add_argument(
        "--scale-efg-translocation",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFG-associated translocation reactions.",
    )
    parser.add_argument(
        "--scale-efg-free-nucleotide-binding",
        type=float,
        default=1.0,
        help="Diagnostic paired scale for free EFG GDP/GTP association and dissociation reactions.",
    )
    parser.add_argument(
        "--scale-efg-binding-reverse",
        type=float,
        default=1.0,
        help="Diagnostic scale for reverse EFG-GTP unbinding from elongating ribosomes.",
    )
    parser.add_argument(
        "--scale-efg-gtp-hydrolysis",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFG_GTP -> EFG_GDP_PO4 reactions on ribosomes.",
    )
    parser.add_argument(
        "--scale-efg-free-ribosome-hydrolysis",
        type=float,
        default=1.0,
        help="Diagnostic scale for off-path RS50S/RS70S_EFG_GTP hydrolysis, separate from elongating ribosome EFG hydrolysis.",
    )
    parser.add_argument(
        "--scale-efg-po4-translocation",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFG_GDP_PO4 -> translocated EFG_GDP + PO4 reactions.",
    )
    parser.add_argument(
        "--scale-efg-gdp-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for EFG_GDP release after translocation.",
    )
    parser.add_argument(
        "--scale-translation-termination",
        type=float,
        default=1.0,
        help="Diagnostic scale for RF/RRF/termination ribosome recycling reactions.",
    )
    parser.add_argument(
        "--scale-if2-free-nucleotide-binding",
        type=float,
        default=1.0,
        help="Diagnostic paired scale for free IF2 GDP/GTP association and dissociation reactions.",
    )
    parser.add_argument(
        "--scale-rf3-free-nucleotide-binding",
        type=float,
        default=1.0,
        help="Diagnostic paired scale for free RF3 GDP/GTP association and dissociation reactions.",
    )
    parser.add_argument(
        "--scale-protein-release",
        type=float,
        default=1.0,
        help="Diagnostic scale for the final Pept0225 -> deGFP_m release step.",
    )
    parser.add_argument(
        "--scale-energy-regen-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for CK/MK/NDK energy-regeneration elementary reactions.",
    )
    parser.add_argument(
        "--scale-ck-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for creatine kinase elementary reactions.",
    )
    parser.add_argument(
        "--scale-mk-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for myokinase/adenylate kinase elementary reactions.",
    )
    parser.add_argument(
        "--scale-ndk-family",
        type=float,
        default=1.0,
        help="Diagnostic scale for nucleoside diphosphate kinase elementary reactions.",
    )
    parser.add_argument(
        "--use-background-buffers",
        action="store_true",
        help="Apply exact conserved-reservoir background buffer fluxes present in the formal XML for CTP/UTP/PO4/PPi.",
    )
    parser.add_argument(
        "--use-atp-gtp-background-fx",
        action="store_true",
        help="Also apply formal backgroundized ATP<->ADP+PO4 and GTP<->GDP+PO4 effective fluxes.",
    )
    parser.add_argument(
        "--background-buffer-scale",
        type=float,
        default=1.0,
        help="Additional diagnostic multiplier for the formal background buffer fluxes.",
    )
    parser.add_argument(
        "--scale-exchange-feed",
        type=float,
        default=1.0,
        help="Diagnostic scale for bg_feed_* continuous exchange rate constants.",
    )
    parser.add_argument(
        "--scale-exchange-atp",
        type=float,
        default=1.0,
        help="Additional diagnostic scale for ATP continuous exchange rate constant.",
    )
    parser.add_argument(
        "--scale-exchange-gtp",
        type=float,
        default=1.0,
        help="Additional diagnostic scale for GTP continuous exchange rate constant.",
    )
    parser.add_argument(
        "--scale-product-washout",
        type=float,
        default=1.0,
        help="Diagnostic scale for deGFP_m product washout exchange rate constant.",
    )
    return parser.parse_args()


def build_full_jac_sparsity(stoich: sparse.csr_matrix, reactants: sparse.csr_matrix) -> sparse.csr_matrix:
    n_species, n_reactions = stoich.shape
    rows: list[int] = []
    cols: list[int] = []
    for rx in range(n_reactions):
        affected = stoich[:, rx].nonzero()[0]
        deps = reactants[:, rx].nonzero()[0]
        if len(affected) == 0 or len(deps) == 0:
            continue
        for row in affected:
            rows.extend([int(row)] * len(deps))
            cols.extend([int(col) for col in deps])
    diag = np.arange(n_species, dtype=int)
    rows.extend(diag.tolist())
    cols.extend(diag.tolist())
    return sparse.coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_species, n_species)).tocsr()


def load_eval_time(args: argparse.Namespace) -> np.ndarray | None:
    if int(args.n_eval) <= 0 and args.pinn_grid_npz is None:
        return None
    if args.pinn_grid_npz is not None:
        z = np.load(args.pinn_grid_npz)
        t = z["time_s"].astype(float)
        t = t[(t >= float(args.start_time_s) - 1e-9) & (t <= float(args.max_time_s) + 1e-9)]
        if len(t) == 0 or abs(t[0] - float(args.start_time_s)) > 1e-9:
            t = np.concatenate([[float(args.start_time_s)], t])
        if t[-1] < float(args.max_time_s) - 1e-9:
            t = np.concatenate([t, [float(args.max_time_s)]])
        return np.unique(t)
    return np.linspace(float(args.start_time_s), float(args.max_time_s), int(args.n_eval))


def load_initial_state(args: argparse.Namespace, species_ids: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    x0 = np.maximum(np.load(args.assets / "initial_concentrations_x0_M.npy").astype(float), 0.0)
    meta: dict[str, object] = {
        "initial_state_source": str(args.assets / "initial_concentrations_x0_M.npy"),
        "initial_state_matched_species": int(len(x0)),
        "initial_state_missing_species": 0,
    }
    if args.initial_state_npz is None:
        return x0, meta
    z = np.load(args.initial_state_npz, allow_pickle=True)
    source_time = z["time_s"].astype(float)
    if "concentrations_M" in z:
        state_key = "concentrations_M"
    elif "optimized_M" in z:
        state_key = "optimized_M"
    elif "prediction_M" in z:
        state_key = "prediction_M"
    else:
        raise ValueError(f"{args.initial_state_npz} lacks concentrations_M, optimized_M, or prediction_M")
    source_x = np.maximum(z[state_key].astype(float), 0.0)
    row = int(np.argmin(np.abs(source_time - float(args.initial_state_time_s))))
    if "species_id" not in z:
        if source_x.shape[1] != len(species_ids):
            raise ValueError(
                f"{args.initial_state_npz} has no species_id and width {source_x.shape[1]} "
                f"does not match asset species count {len(species_ids)}"
            )
        meta = {
            "initial_state_source": str(args.initial_state_npz),
            "initial_state_time_s_requested": float(args.initial_state_time_s),
            "initial_state_time_s_used": float(source_time[row]),
            "initial_state_matched_species": int(len(species_ids)),
            "initial_state_missing_species": 0,
            "initial_missing_mode": "full_order_npz",
            "initial_state_key": state_key,
        }
        return source_x[row].copy(), meta
    source_species = [str(s) for s in z["species_id"].tolist()]
    source_lookup = {sid: i for i, sid in enumerate(source_species)}
    matched = 0
    missing = 0
    x_new = x0.copy() if args.initial_missing_mode == "asset" else np.zeros_like(x0)
    for i, sid in enumerate(species_ids):
        j = source_lookup.get(sid)
        if j is None:
            missing += 1
            continue
        x_new[i] = source_x[row, j]
        matched += 1
    meta = {
        "initial_state_source": str(args.initial_state_npz),
        "initial_state_time_s_requested": float(args.initial_state_time_s),
        "initial_state_time_s_used": float(source_time[row]),
        "initial_state_matched_species": int(matched),
        "initial_state_missing_species": int(missing),
        "initial_missing_mode": args.initial_missing_mode,
        "initial_state_key": state_key,
    }
    return x_new, meta


AA_RS_PATTERN = re.compile(
    r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)RS"
)


def apply_reaction_family_scales(args: argparse.Namespace, k: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    inventory_path = args.assets / "reaction_mass_action_inventory.csv"
    inv = pd.read_csv(inventory_path)
    text = (
        inv["reactants"].fillna("").astype(str)
        + " -> "
        + inv["products"].fillna("").astype(str)
        + " ; "
        + inv["kinetic_law"].fillna("").astype(str)
    )
    text_values = text.to_numpy(dtype=str)
    scaled = k.copy()
    scale_report: dict[str, object] = {"reaction_family_scale_inventory": str(inventory_path)}

    def apply_mask(name: str, mask: np.ndarray, factor: float) -> None:
        factor = float(factor)
        count = int(np.sum(mask))
        scale_report[f"{name}_scale"] = factor
        scale_report[f"{name}_count"] = count
        if count and factor != 1.0:
            scaled[mask] *= factor
            examples = inv.loc[mask, ["reaction_id", "reactants", "products"]].head(8).to_dict(orient="records")
            scale_report[f"{name}_examples"] = examples

    aars_mask = text.map(lambda s: bool(AA_RS_PATTERN.search(s))).to_numpy(dtype=bool)
    aars_amp_forward_mask = (
        aars_mask
        & inv["reactants"].fillna("").astype(str).str.contains("_ATP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("AMP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("PPi", regex=False).to_numpy(dtype=bool)
    )
    aaamp_free_pattern = re.compile(
        r"\b(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)AMP\b"
    )
    aars_aaamp_leak_mask = (
        aars_mask
        & inv["reactants"].fillna("").astype(str).str.contains("AMP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).map(lambda s: bool(AA_RS_PATTERN.search(s))).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).map(lambda s: bool(aaamp_free_pattern.search(s))).to_numpy(dtype=bool)
    )
    charged_trna_pattern = re.compile(
        r"\b(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)tRNA"
    )
    aars_trna_transfer_mask = (
        aars_mask
        & inv["reactants"].fillna("").astype(str).str.contains("AMP", regex=False).to_numpy(dtype=bool)
        & inv["reactants"].fillna("").astype(str).str.contains("tRNA", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).map(lambda s: bool(charged_trna_pattern.search(s))).to_numpy(dtype=bool)
    )
    pept_pattern = re.compile(r"Pept(\d{4})")

    def max_pept_index(value: str) -> int:
        matches = pept_pattern.findall(str(value))
        return max([int(v) for v in matches], default=-1)

    reactant_pept = inv["reactants"].fillna("").astype(str).map(max_pept_index).to_numpy(dtype=int)
    product_pept = inv["products"].fillna("").astype(str).map(max_pept_index).to_numpy(dtype=int)
    translation_productive_mask = product_pept > reactant_pept
    protein_release_mask = (
        inv["reactants"].fillna("").astype(str).str.fullmatch(r"\s*1\s+Pept0225\s*").to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.fullmatch(r"\s*1\s+deGFP_m\s*").to_numpy(dtype=bool)
    )
    transcription_mask = text.str.contains("T7RNAP", regex=False).to_numpy(dtype=bool)
    translation_mask = text.str.contains(
        r"elRS70S|termRS70S|RS30S|RS50S|EFTu|EFG|IF1|IF2|IF3|RF1|RF2|RRF",
        regex=True,
    ).to_numpy(dtype=bool)
    translation_initiation_mask = (
        text.str.contains(r"RS30S|RS50S|RS70S|IF1|IF2|IF3", regex=True).to_numpy(dtype=bool)
        & text.str.contains("mRNA", regex=False).to_numpy(dtype=bool)
        & ~text.str.contains(r"elRS70S|termRS70S", regex=True).to_numpy(dtype=bool)
    )
    reactant_text = inv["reactants"].fillna("").astype(str)
    product_text = inv["products"].fillna("").astype(str)
    init_mrna_binding_mask = (
        reactant_text.str.contains(r"\bmRNA\b", regex=True).to_numpy(dtype=bool)
        & reactant_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("_mRNA", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
    )
    init_mrna_unbinding_mask = (
        reactant_text.str.contains("_mRNA", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains(r"\bmRNA\b", regex=True).to_numpy(dtype=bool)
        & product_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
    )
    init_50s_joining_mask = (
        reactant_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("RS50S", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("_mRNA", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("RS70S", regex=False).to_numpy(dtype=bool)
    )
    init_50s_splitting_mask = (
        reactant_text.str.contains("RS70S", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("_mRNA", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("RS30S", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("RS50S", regex=False).to_numpy(dtype=bool)
    )
    init_if2_hydrolysis_mask = (
        reactant_text.str.contains("RS70S", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("IF2_GTP", regex=False).to_numpy(dtype=bool)
        & product_text.str.contains("IF2_GDP_PO4", regex=False).to_numpy(dtype=bool)
    )
    init_if_release_mask = (
        reactant_text.str.contains("RS70S", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains("_mRNA", regex=False).to_numpy(dtype=bool)
        & reactant_text.str.contains(r"IF1|IF2_GDP|IF3", regex=True).to_numpy(dtype=bool)
        & product_text.str.contains(r"\bIF1\b|\bIF2_GDP\b|\bIF3\b", regex=True).to_numpy(dtype=bool)
    )
    eftu_delivery_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("EFTu_GTP_", regex=False).to_numpy(dtype=bool)
        & inv["reactants"].fillna("").astype(str).str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("_EFTu_GTP_", regex=False).to_numpy(dtype=bool)
    )
    eftu_free_nucleotide_binding_mask = (
        (
            reactant_text.str.fullmatch(r"\s*1\s+EFTu;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+EFTu_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
        | (
            reactant_text.str.fullmatch(r"\s*1\s+EFTu_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+(?:GDP|GTP);\s*1\s+EFTu\s*|\s*1\s+EFTu;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
    )
    eftu_hydrolysis_release_mask = (
        text.str.contains("EFTu", regex=False).to_numpy(dtype=bool)
        & text.str.contains(r"EFTu_GDP|EFTu_GDP_PO4|PO4", regex=True).to_numpy(dtype=bool)
        & text.str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
    )
    eftu_delivery_reverse_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFTu_GTP_", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("EFTu_GTP_", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
        & ~inv["products"].fillna("").astype(str).str.contains("_EFTu_GTP_", regex=False).to_numpy(dtype=bool)
    )
    eftu_gtp_hydrolysis_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFTu_GTP_", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("_EFTu_GDP_PO4_", regex=False).to_numpy(dtype=bool)
    )
    eftu_po4_release_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFTu_GDP_PO4_", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("_EFTu_GDP_", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("PO4", regex=False).to_numpy(dtype=bool)
    )
    eftu_gdp_release_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFTu_GDP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("EFTu_GDP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
        & ~inv["products"].fillna("").astype(str).str.contains("_EFTu_GDP", regex=False).to_numpy(dtype=bool)
    )
    efg_translocation_mask = (
        text.str.contains("EFG", regex=False).to_numpy(dtype=bool)
        & text.str.contains(r"elRS70S|termRS70S", regex=True).to_numpy(dtype=bool)
    )
    efg_free_nucleotide_binding_mask = (
        (
            reactant_text.str.fullmatch(r"\s*1\s+EFG;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+EFG_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
        | (
            reactant_text.str.fullmatch(r"\s*1\s+EFG_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+(?:GDP|GTP);\s*1\s+EFG\s*|\s*1\s+EFG;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
    )
    efg_binding_reverse_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFG_GTP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("EFG_GTP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
        & ~inv["products"].fillna("").astype(str).str.contains("_EFG_GTP", regex=False).to_numpy(dtype=bool)
    )
    efg_gtp_hydrolysis_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFG_GTP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("_EFG_GDP_PO4", regex=False).to_numpy(dtype=bool)
    )
    efg_free_ribosome_hydrolysis_mask = (
        inv["reactants"].fillna("").astype(str).str.fullmatch(r"\s*1\s+RS(?:50S|70S)_EFG_GTP\s*").to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.fullmatch(r"\s*1\s+RS(?:50S|70S)_EFG_GDP_PO4\s*").to_numpy(dtype=bool)
    )
    efg_po4_translocation_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFG_GDP_PO4", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("_EFG_GDP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("PO4", regex=False).to_numpy(dtype=bool)
    )
    efg_gdp_release_mask = (
        inv["reactants"].fillna("").astype(str).str.contains("_EFG_GDP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("EFG_GDP", regex=False).to_numpy(dtype=bool)
        & inv["products"].fillna("").astype(str).str.contains("elRS70S", regex=False).to_numpy(dtype=bool)
        & ~inv["products"].fillna("").astype(str).str.contains("_EFG_GDP", regex=False).to_numpy(dtype=bool)
    )
    translation_termination_mask = text.str.contains(r"RF1|RF2|RRF|termRS70S", regex=True).to_numpy(dtype=bool)
    if2_free_nucleotide_binding_mask = (
        (
            reactant_text.str.fullmatch(r"\s*1\s+IF2;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+IF2_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
        | (
            reactant_text.str.fullmatch(r"\s*1\s+IF2_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+(?:GDP|GTP);\s*1\s+IF2\s*|\s*1\s+IF2;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
    )
    rf3_free_nucleotide_binding_mask = (
        (
            reactant_text.str.fullmatch(r"\s*1\s+RF3;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+RF3_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
        | (
            reactant_text.str.fullmatch(r"\s*1\s+RF3_(?:GDP|GTP)\s*").to_numpy(dtype=bool)
            & product_text.str.fullmatch(r"\s*1\s+(?:GDP|GTP);\s*1\s+RF3\s*|\s*1\s+RF3;\s*1\s+(?:GDP|GTP)\s*").to_numpy(dtype=bool)
        )
    )
    ck_mask = text.str.contains(r"\bCK\b|Creatine|creatine", regex=True).to_numpy(dtype=bool)
    mk_mask = text.str.contains(r"\bMK\b", regex=True).to_numpy(dtype=bool)
    ndk_mask = text.str.contains(r"\bNDK\b", regex=True).to_numpy(dtype=bool)
    energy_mask = ck_mask | mk_mask | ndk_mask | text.str.contains(r"\b(?:PCK|PK)\b", regex=True).to_numpy(dtype=bool)

    apply_mask("aars_family", aars_mask, args.scale_aars_family)
    apply_mask("aars_amp_forward", aars_amp_forward_mask, args.scale_aars_amp_forward)
    apply_mask("aars_aaamp_leak", aars_aaamp_leak_mask, args.scale_aars_aaamp_leak)
    apply_mask("aars_trna_transfer", aars_trna_transfer_mask, args.scale_aars_trna_transfer)
    apply_mask("transcription_family", transcription_mask, args.scale_transcription_family)
    apply_mask("translation_family", translation_mask, args.scale_translation_family)
    apply_mask("translation_initiation", translation_initiation_mask, args.scale_translation_initiation)
    apply_mask("init_mrna_binding", init_mrna_binding_mask, args.scale_init_mrna_binding)
    apply_mask("init_mrna_unbinding", init_mrna_unbinding_mask, args.scale_init_mrna_unbinding)
    apply_mask("init_50s_joining", init_50s_joining_mask, args.scale_init_50s_joining)
    apply_mask("init_50s_splitting", init_50s_splitting_mask, args.scale_init_50s_splitting)
    apply_mask("init_if2_hydrolysis", init_if2_hydrolysis_mask, args.scale_init_if2_hydrolysis)
    apply_mask("init_if_release", init_if_release_mask, args.scale_init_if_release)
    apply_mask("eftu_delivery", eftu_delivery_mask, args.scale_eftu_delivery)
    apply_mask("eftu_free_nucleotide_binding", eftu_free_nucleotide_binding_mask, args.scale_eftu_free_nucleotide_binding)
    apply_mask("eftu_hydrolysis_release", eftu_hydrolysis_release_mask, args.scale_eftu_hydrolysis_release)
    apply_mask("eftu_delivery_reverse", eftu_delivery_reverse_mask, args.scale_eftu_delivery_reverse)
    apply_mask("eftu_gtp_hydrolysis", eftu_gtp_hydrolysis_mask, args.scale_eftu_gtp_hydrolysis)
    apply_mask("eftu_po4_release", eftu_po4_release_mask, args.scale_eftu_po4_release)
    apply_mask("eftu_gdp_release", eftu_gdp_release_mask, args.scale_eftu_gdp_release)
    apply_mask("efg_translocation", efg_translocation_mask, args.scale_efg_translocation)
    apply_mask("efg_free_nucleotide_binding", efg_free_nucleotide_binding_mask, args.scale_efg_free_nucleotide_binding)
    apply_mask("efg_binding_reverse", efg_binding_reverse_mask, args.scale_efg_binding_reverse)
    apply_mask("efg_gtp_hydrolysis", efg_gtp_hydrolysis_mask, args.scale_efg_gtp_hydrolysis)
    apply_mask("efg_free_ribosome_hydrolysis", efg_free_ribosome_hydrolysis_mask, args.scale_efg_free_ribosome_hydrolysis)
    apply_mask("efg_po4_translocation", efg_po4_translocation_mask, args.scale_efg_po4_translocation)
    apply_mask("efg_gdp_release", efg_gdp_release_mask, args.scale_efg_gdp_release)
    apply_mask("translation_productive", translation_productive_mask, args.scale_translation_productive)
    apply_mask("translation_termination", translation_termination_mask, args.scale_translation_termination)
    apply_mask("if2_free_nucleotide_binding", if2_free_nucleotide_binding_mask, args.scale_if2_free_nucleotide_binding)
    apply_mask("rf3_free_nucleotide_binding", rf3_free_nucleotide_binding_mask, args.scale_rf3_free_nucleotide_binding)
    apply_mask("protein_release", protein_release_mask, args.scale_protein_release)
    apply_mask("energy_regen_family", energy_mask, args.scale_energy_regen_family)
    apply_mask("ck_family", ck_mask, args.scale_ck_family)
    apply_mask("mk_family", mk_mask, args.scale_mk_family)
    apply_mask("ndk_family", ndk_mask, args.scale_ndk_family)
    scale_report["reaction_family_scale_nondefault"] = {
        key: scale_report[key]
        for key in [
            "aars_family_scale",
            "aars_amp_forward_scale",
            "aars_aaamp_leak_scale",
            "aars_trna_transfer_scale",
            "transcription_family_scale",
            "translation_family_scale",
            "translation_initiation_scale",
            "init_mrna_binding_scale",
            "init_mrna_unbinding_scale",
            "init_50s_joining_scale",
            "init_50s_splitting_scale",
            "init_if2_hydrolysis_scale",
            "init_if_release_scale",
            "eftu_delivery_scale",
            "eftu_free_nucleotide_binding_scale",
            "eftu_hydrolysis_release_scale",
            "eftu_delivery_reverse_scale",
            "eftu_gtp_hydrolysis_scale",
            "eftu_po4_release_scale",
            "eftu_gdp_release_scale",
            "efg_translocation_scale",
            "efg_free_nucleotide_binding_scale",
            "efg_binding_reverse_scale",
            "efg_gtp_hydrolysis_scale",
            "efg_free_ribosome_hydrolysis_scale",
            "efg_po4_translocation_scale",
            "efg_gdp_release_scale",
            "translation_productive_scale",
            "translation_termination_scale",
            "if2_free_nucleotide_binding_scale",
            "rf3_free_nucleotide_binding_scale",
            "protein_release_scale",
            "energy_regen_family_scale",
            "ck_family_scale",
            "mk_family_scale",
            "ndk_family_scale",
        ]
        if scale_report[key] != 1.0
    }
    return scaled, scale_report


def build_background_buffer_plan(
    species_ids: list[str],
    continuous_xml: Path,
    state_unit: str,
    enabled: bool,
    scale: float,
    include_atp_gtp_fx: bool = False,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    if not enabled:
        return [], {"background_buffers_enabled": False, "background_buffer_scale": float(scale), "background_buffers": []}
    params = parse_parameters(continuous_xml)
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    bg_k_scale = float(params.get("bg_k_scale", 1.0))
    bg_load_clock = float(params.get("bg_load_clock", 1.0))
    plans: list[dict[str, float | int | str]] = []
    if include_atp_gtp_fx:
        po4_idx = lookup.get("PO4")
        for sid, diphosphate in [("ATP", "ADP"), ("GTP", "GDP")]:
            if sid not in lookup or diphosphate not in lookup or po4_idx is None:
                continue
            theta = float(params.get(f"bg_{sid}_theta", 0.0))
            if sid == "ATP":
                a = float(params.get("bg_ATP_a", 1.0))
                k1 = float(params.get("bg_ATP_k1", params.get("bg_ATP_k", 0.0)))
                k2 = float(params.get("bg_ATP_k2", params.get("bg_ATP_k", 0.0)))
                raw_k = a * k1 + (1.0 - a) * k2
            else:
                raw_k = float(params.get("bg_GTP_k", 0.0))
            raw_k *= bg_k_scale * bg_load_clock * float(scale)
            plans.append(
                {
                    "kind": "nucleotide_fx",
                    "species": sid,
                    "index": int(lookup[sid]),
                    "diphosphate": diphosphate,
                    "diphosphate_index": int(lookup[diphosphate]),
                    "po4_index": int(po4_idx),
                    "theta": float(theta),
                    "raw_k_s_inv": float(raw_k),
                }
            )

    for sid in ["CTP", "UTP", "PO4", "PPi"]:
        if sid not in lookup:
            continue
        total_key = f"bg_{sid}_total"
        theta_key = f"bg_{sid}_theta"
        if total_key not in params or theta_key not in params:
            continue
        theta = float(params[theta_key])
        if sid == "PPi":
            raw_k = float(params.get("bg_PPi_k", 0.0))
        else:
            a = float(params.get(f"bg_{sid}_a", 1.0))
            k1 = float(params.get(f"bg_{sid}_k1", params.get(f"bg_{sid}_k", 0.0)))
            k2 = float(params.get(f"bg_{sid}_k2", params.get(f"bg_{sid}_k", 0.0)))
            raw_k = a * k1 + (1.0 - a) * k2
        raw_k *= bg_k_scale * bg_load_clock * float(scale)
        total = float(params[total_key])
        if state_unit == "uM":
            total *= 1e6
        plans.append(
            {
                "species": sid,
                "kind": "single_buffer",
                "index": int(lookup[sid]),
                "theta": float(theta),
                "total_state_unit": float(total),
                "raw_k_s_inv": float(raw_k),
            }
        )
    return plans, {
        "background_buffers_enabled": True,
        "background_buffer_scale": float(scale),
        "atp_gtp_background_fx_enabled": bool(include_atp_gtp_fx),
        "background_buffers": plans,
    }


def main() -> None:
    args = parse_args()
    species = pd.read_csv(args.assets / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    stoich = sparse.load_npz(args.assets / "stoichiometric_matrix_S.npz").tocsr()
    reactants = sparse.load_npz(args.assets / "reactant_exponent_matrix_A.npz").tocsr()
    k = np.load(args.assets / "rate_constants_k.npy").astype(float)
    k, scale_meta = apply_reaction_family_scales(args, k)
    molecularity = np.asarray(reactants.sum(axis=0)).ravel().astype(float)
    if args.rate_unit == "uM" and args.state_unit == "M":
        k = k * np.power(1e6, np.maximum(molecularity - 1.0, 0.0))
    elif args.rate_unit == "M" and args.state_unit == "uM":
        k = k * np.power(1e-6, np.maximum(molecularity - 1.0, 0.0))
    log_k = np.log(np.clip(k, 1e-300, None))
    x0, initial_meta = load_initial_state(args, species_ids)
    if args.state_unit == "uM":
        x0 = x0 * 1e6
    exchange_plan = build_exchange_plan(species_ids, DEFAULT_CONTINUOUS_XML)
    gamma, gamma_prod, feed_idx, feed_value, product_idx = exchange_plan
    if args.exchange_mode == "batch":
        gamma = 0.0
        gamma_prod = 0.0
    gamma *= float(args.scale_exchange_feed)
    gamma_prod *= float(args.scale_product_washout)
    exchange_plan = (gamma, gamma_prod, feed_idx, feed_value, product_idx)
    feed_scale_vec = np.ones(len(feed_idx), dtype=float)
    feed_species = [species_ids[int(i)] for i in feed_idx]
    for n, sid in enumerate(feed_species):
        if sid == "ATP":
            feed_scale_vec[n] *= float(args.scale_exchange_atp)
        elif sid == "GTP":
            feed_scale_vec[n] *= float(args.scale_exchange_gtp)
    feed_value_state = feed_value.copy()
    if args.state_unit == "uM":
        feed_value_state = feed_value_state * 1e6
    buffer_plan, buffer_meta = build_background_buffer_plan(
        species_ids,
        DEFAULT_CONTINUOUS_XML,
        args.state_unit,
        args.use_background_buffers,
        args.background_buffer_scale,
        args.use_atp_gtp_background_fx,
    )
    t_eval = load_eval_time(args)
    t_span = (float(args.start_time_s), float(args.max_time_s)) if t_eval is None else (float(t_eval[0]), float(t_eval[-1]))
    jac_sparsity = None
    if args.use_jac_sparsity and args.method in {"BDF", "Radau"} and not args.use_analytic_jac:
        jac_sparsity = build_full_jac_sparsity(stoich, reactants)
    reactants_t = reactants.T.tocsr()
    exchange_jac_diag = np.zeros(stoich.shape[0], dtype=float)
    if len(feed_idx) > 0:
        exchange_jac_diag[feed_idx] -= gamma * feed_scale_vec
    if product_idx is not None:
        exchange_jac_diag[int(product_idx)] -= float(gamma_prod)
    jac_rows: list[int] = []
    jac_cols: list[int] = []
    jac_data: list[float] = []
    for plan in buffer_plan:
        kind = str(plan.get("kind", "single_buffer"))
        if kind == "single_buffer":
            exchange_jac_diag[int(plan["index"])] -= float(plan["raw_k_s_inv"])
        elif kind == "nucleotide_fx":
            nt = int(plan["index"])
            dp = int(plan["diphosphate_index"])
            po4 = int(plan["po4_index"])
            theta = float(plan["theta"])
            k_eff = float(plan["raw_k_s_inv"]) / (1.0 + theta)
            k_rev = k_eff * theta
            jac_rows.extend([nt, dp, po4, nt, dp, po4])
            jac_cols.extend([nt, nt, nt, dp, dp, dp])
            jac_data.extend([-k_eff, k_eff, k_eff, k_rev, -k_rev, -k_rev])
    exchange_jac = (
        sparse.diags(exchange_jac_diag, format="csr")
        + sparse.coo_matrix((jac_data, (jac_rows, jac_cols)), shape=(stoich.shape[0], stoich.shape[0])).tocsr()
    )

    rhs_calls = {"n": 0}
    jac_calls = {"n": 0}

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rhs_calls["n"] += 1
        x = np.maximum(y, 0.0)
        log_x = np.log(np.clip(x, 1e-30, None))
        log_monomial = reactants_t @ log_x
        flux = np.exp(np.clip(np.asarray(log_monomial).ravel() + log_k, -100.0, 100.0))
        exchange = np.zeros_like(x)
        if len(feed_idx) > 0:
            exchange[feed_idx] += gamma * feed_scale_vec * (feed_value_state - x[feed_idx])
        if product_idx is not None:
            exchange[int(product_idx)] += gamma_prod * (0.0 - x[int(product_idx)])
        for plan in buffer_plan:
            kind = str(plan.get("kind", "single_buffer"))
            theta = float(plan["theta"])
            k_eff = float(plan["raw_k_s_inv"]) / (1.0 + theta)
            if kind == "single_buffer":
                idx = int(plan["index"])
                total = float(plan["total_state_unit"])
                exchange[idx] += k_eff * (theta * (total - x[idx]) - x[idx])
            elif kind == "nucleotide_fx":
                nt = int(plan["index"])
                dp = int(plan["diphosphate_index"])
                po4 = int(plan["po4_index"])
                out_flux = k_eff * x[nt]
                in_flux = k_eff * theta * x[dp]
                exchange[nt] += -out_flux + in_flux
                exchange[dp] += out_flux - in_flux
                exchange[po4] += out_flux - in_flux
        out = stoich @ flux + exchange
        return np.asarray(out).ravel()

    def jac(_t: float, y: np.ndarray) -> sparse.csr_matrix:
        jac_calls["n"] += 1
        x = np.maximum(y, 0.0)
        log_x = np.log(np.clip(x, 1e-30, None))
        log_monomial = reactants_t @ log_x
        flux = np.exp(np.clip(np.asarray(log_monomial).ravel() + log_k, -100.0, 100.0))
        inv_x = 1.0 / np.clip(x, 1e-30, None)
        dflux_dx = reactants_t.multiply(flux[:, None]).multiply(inv_x[None, :])
        return (stoich @ dflux_dx + exchange_jac).tocsr()

    jac_arg = jac if args.use_analytic_jac and args.method in {"BDF", "Radau"} else None

    chunk_reports: list[dict[str, object]] = []
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    success = True
    message = ""
    nfev = 0
    njev = 0
    nlu = 0
    if float(args.chunk_s) > 0:
        y_left = x0.copy()
        t_left = float(t_span[0])
        chunk_counter = 0
        while t_left < float(t_span[-1]) - 1e-12:
            t_right = min(float(t_span[-1]), t_left + float(args.chunk_s))
            if t_eval is None:
                chunk_t_eval = None
            else:
                mask = (t_eval >= t_left - 1e-9) & (t_eval <= t_right + 1e-9)
                chunk_t_eval = np.unique(np.concatenate([[t_left], t_eval[mask], [t_right]]))
            try:
                sol = solve_ivp(
                    rhs,
                    (t_left, t_right),
                    y_left,
                    method=args.method,
                    t_eval=chunk_t_eval,
                    rtol=float(args.rtol),
                    atol=float(args.atol_uM) if args.state_unit == "uM" else float(args.atol_uM) * 1e-6,
                    max_step=float(args.max_step_s),
                    jac=jac_arg,
                    jac_sparsity=jac_sparsity,
                )
            except Exception as exc:  # noqa: BLE001
                success = False
                message = f"chunk {len(chunk_reports)} exception: {exc}"
                chunk_reports.append({"start_s": t_left, "end_s": t_right, "success": False, "message": str(exc)})
                break
            nfev += int(sol.nfev)
            njev += int(getattr(sol, "njev", 0))
            nlu += int(getattr(sol, "nlu", 0))
            chunk_report = {
                "start_s": t_left,
                "end_s": t_right,
                "success": bool(sol.success),
                "message": str(sol.message),
                "nfev": int(sol.nfev),
                "njev": int(getattr(sol, "njev", 0)),
                "nlu": int(getattr(sol, "nlu", 0)),
                "time_points": int(len(sol.t)),
            }
            if (
                not sol.success
                or int(args.chunk_report_stride) <= 1
                or chunk_counter % int(args.chunk_report_stride) == 0
            ):
                chunk_reports.append(chunk_report)
            if len(sol.t) > 0:
                if all_t and abs(float(sol.t[0]) - float(all_t[-1][-1])) < 1e-9:
                    all_t.append(sol.t[1:].astype(float))
                    all_y.append(sol.y[:, 1:].T.astype(float))
                else:
                    all_t.append(sol.t.astype(float))
                    all_y.append(sol.y.T.astype(float))
                y_left = np.maximum(sol.y[:, -1], 0.0)
            if not sol.success:
                success = False
                message = str(sol.message)
                break
            t_left = t_right
            chunk_counter += 1
        if success:
            message = "All chunks reached their endpoints."
        sol_t = np.concatenate(all_t) if all_t else np.array([], dtype=float)
        x = np.maximum(np.vstack(all_y), 0.0) if all_y else np.empty((0, len(x0)), dtype=float)
    else:
        sol = solve_ivp(
            rhs,
            t_span,
            x0,
            method=args.method,
            t_eval=t_eval,
            rtol=float(args.rtol),
            atol=float(args.atol_uM) if args.state_unit == "uM" else float(args.atol_uM) * 1e-6,
            max_step=float(args.max_step_s),
            jac=jac_arg,
            jac_sparsity=jac_sparsity,
        )
        success = bool(sol.success)
        message = str(sol.message)
        nfev = int(sol.nfev)
        njev = int(getattr(sol, "njev", 0))
        nlu = int(getattr(sol, "nlu", 0))
        sol_t = sol.t.astype(float)
        x = np.maximum(sol.y.T, 0.0) if sol.y.size else np.empty((0, len(x0)), dtype=float)

    if len(x):
        rhs_eval = np.vstack([rhs(float(t), row) for t, row in zip(sol_t, x)])
    else:
        rhs_eval = np.empty_like(x)
    if args.state_unit == "uM":
        x_out = x * 1e-6
        rhs_out = rhs_eval * 1e-6
        x0_out = x0 * 1e-6
    else:
        x_out = x
        rhs_out = rhs_eval
        x0_out = x0
    if args.endpoint_only and len(x_out):
        sol_t = sol_t[-1:].copy()
        x_out = x_out[-1:].copy()
        rhs_out = rhs_out[-1:].copy()

    all_species_idx = np.arange(stoich.shape[0], dtype=int)

    outdir = args.assets
    prefix = str(args.output_prefix)
    np.savez_compressed(
        outdir / f"{prefix}_direct_full_elementary_solution.npz",
        time_s=sol_t.astype(float),
        concentrations_M=x_out,
        rhs_M_per_s=rhs_out,
    )
    end = x_out[-1] if len(x_out) else x0_out
    summary = {
        "scope": "direct full-elementary mass-action ODE solve with optional continuous exchange; no teacher trajectory constraint",
        "output_npz": str(outdir / f"{prefix}_direct_full_elementary_solution.npz"),
        "method": args.method,
        "rtol": float(args.rtol),
        "atol_uM": float(args.atol_uM),
        "max_step_s": float(args.max_step_s),
        "exchange_mode": args.exchange_mode,
        "max_time_s": float(t_span[-1]),
        "success": bool(success),
        "message": str(message),
        "nfev": int(nfev),
        "njev": int(njev),
        "nlu": int(nlu),
        "rhs_calls_including_eval": int(rhs_calls["n"]),
        "jac_calls": int(jac_calls["n"]),
        "time_points_returned": int(len(sol_t)),
        "species_count": int(stoich.shape[0]),
        "reaction_count": int(stoich.shape[1]),
        "use_jac_sparsity": bool(args.use_jac_sparsity),
        "use_analytic_jac": bool(args.use_analytic_jac),
        "rate_unit": args.rate_unit,
        "state_unit": args.state_unit,
        "exchange_feed_scale": float(args.scale_exchange_feed),
        "exchange_atp_scale": float(args.scale_exchange_atp),
        "exchange_gtp_scale": float(args.scale_exchange_gtp),
        "product_washout_scale": float(args.scale_product_washout),
        "exchange_feed_species_count": int(len(feed_idx)),
        "exchange_feed_species": feed_species,
        "exchange_gamma_h_inv": float(gamma * 3600.0),
        "exchange_gamma_prod_h_inv": float(gamma_prod * 3600.0),
        **scale_meta,
        **buffer_meta,
        "chunk_s": float(args.chunk_s),
        "chunk_reports": chunk_reports,
        "pinn_grid_npz": str(args.pinn_grid_npz) if args.pinn_grid_npz is not None else None,
        "endpoint_only": bool(args.endpoint_only),
        **initial_meta,
        "end_total_concentration_uM": float(np.sum(end) * 1e6),
        "end_max_concentration_uM": float(np.max(end) * 1e6),
        "end_nonzero_gt_1e-12_M": int(np.sum(end > 1e-12)),
    }
    for sid in ["deGFP_m", "mRNA_t", "ATP", "GTP", "CTP", "UTP", "AMP", "PPi", "PO4", "GDP"]:
        if sid in species_ids:
            summary[f"end_{sid}_uM"] = float(end[species_ids.index(sid)] * 1e6)
    if len(sol_t) >= 2:
        summary.update(
            hidden_integral_metrics_np(
                x_out,
                rhs_out,
                sol_t.astype(float),
                all_species_idx,
                label="all_species",
            )
        )
    (outdir / f"{prefix}_direct_full_elementary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
