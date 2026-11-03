#!/usr/bin/env python3
"""
Route-1 high-granularity two-gene model:
explicitly duplicate the transcription chain (r0..r343) per gene.

This version is intended to answer "independent DNA/mRNA line with explicit
intermediate states" rather than coarse-grained proxy channels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import libsbml
import numpy as np
import pandas as pd
import tellurium as te

import simulate_route1_multigene_background as route1


DEFAULT_OUTDIR = Path("output_route1_twogene_explicit_tx")
DEFAULT_TX_NT_STEPS = 339
DEFAULT_TX_COMPRESS_REF_CDS_NT = 1650.0
DEFAULT_TX_COMPRESS_SCALE = DEFAULT_TX_NT_STEPS / DEFAULT_TX_COMPRESS_REF_CDS_NT
DEFAULT_MRNA_DECAY_BASE = 2.0e-4
DEFAULT_MRNA_BOUND_DECAY_RATIO = 0.2

# Mechanistic/QSSA defaults aligned with the 2026-04-09 handover baseline.
# These are used as robust fallbacks when a summary file omits some IDs.
DEFAULT_MECH_QSSA_PARAMS: Dict[str, float] = {
    "k_ribo_bind_f": 195464.87719773027,
    "k_ribo_bind_r": 0.34584637957254144,
    "k_tl_init": 0.026998731015091766,
    "k_tl_elong": 0.009476441238810956,
    "k_tl_term": 0.055568416391920346,
    "K_tl_AA": 3.2535684250485034e-05,
    "K_tl_ATP": 6.11033863918126e-05,
    "K_tl_GTP": 3.113087089656521e-05,
    "K_tl_eps": 1.0e-18,
}


def _clean_dna(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGT")


# MG1655 CDS from NC_000913.3
# pgi: 4233758..4235407 (1650 nt, includes stop codon)
_PGI_CDS_DNA = _clean_dna(
    """
ATGAAAAACATCAATCCAACGCAGACCGCTGCCTGGCAGGCACTACAGAAACACTTCGATGAAATGAAAGACGTTACGATCGCCGATCTTTTTGCTAAAGACGGCGATCGTTTTTCTAAGTTCTCCGCAACCTTCGACGATCAGATGCTGGTGGATTACTCCAAAAACCGCATCACTGAAGAGACGCTGGCGAAATTACAGGATCTGGCGAAAGAGTGCGATCTGGCGGGCGCGATTAAGTCGATGTTCTCTGGCGAGAAGATCAACCGCACTGAAAACCGCGCCGTGCTGCACGTAGCGCTGCGTAACCGTAGCAATACCCCGATTTTGGTTGATGGCAAAGACGTAATGCCGGAAGTCAACGCGGTGCTGGAGAAGATGAAAACCTTCTCAGAAGCGATTATTTCCGGTGAGTGGAAAGGTTATACCGGCAAAGCAATCACTGACGTAGTGAACATCGGGATCGGCGGTTCTGACCTCGGCCCATACATGGTGACCGAAGCTCTGCGTCCGTACAAAAACCACCTGAACATGCACTTTGTTTCTAACGTCGATGGGACTCACATCGCGGAAGTGCTGAAAAAAGTAAACCCGGAAACCACGCTGTTCTTGGTAGCATCTAAAACCTTCACCACTCAGGAAACTATGACCAACGCCCATAGCGCGCGTGACTGGTTCCTGAAAGCGGCAGGTGATGAAAAACACGTTGCAAAACACTTTGCGGCGCTTTCCACCAATGCCAAAGCCGTTGGCGAGTTTGGTATTGATACTGCCAACATGTTCGAGTTCTGGGACTGGGTTGGCGGCCGTTACTCTTTGTGGTCAGCGATTGGCCTGTCGATTGTTCTCTCCATCGGCTTTGATAACTTCGTTGAACTGCTTTCCGGCGCACACGCGATGGACAAGCATTTCTCCACCACGCCTGCCGAGAAAAACCTGCCTGTACTGCTGGCGCTGATTGGCATCTGGTACAACAATTTCTTTGGTGCGGAAACTGAAGCGATTCTGCCGTATGACCAGTATATGCACCGTTTCGCGGCGTACTTCCAGCAGGGCAATATGGAGTCCAACGGTAAGTATGTTGACCGTAACGGTAACGTTGTGGATTACCAGACTGGCCCGATTATCTGGGGTGAACCAGGCACTAACGGTCAGCACGCGTTCTACCAGCTGATCCACCAGGGAACCAAAATGGTACCGTGCGATTTCATCGCTCCGGCTATCACCCATAACCCGCTCTCTGATCATCACCAGAAACTGCTGTCTAACTTCTTCGCCCAGACCGAAGCGCTGGCGTTTGGTAAATCCCGCGAAGTGGTTGAGCAGGAATATCGTGATCAGGGTAAAGATCCGGCAACGCTTGACTACGTGGTGCCGTTCAAAGTATTCGAAGGTAACCGCCCGACCAACTCCATCCTGCTGCGTGAAATCACTCCGTTCAGCCTGGGTGCGTTGATTGCGCTGTATGAGCACAAAATCTTTACTCAGGGCGTGATCCTGAACATCTTCACCTTCGACCAGTGGGGCGTGGAACTGGGTAAACAGCTGGCGAACCGTATTCTGCCAGAGCTGAAAGATGATAAAGAAATCAGCAGCCACGATAGCTCGACCAATGGTCTGATTAACCGCTATAAAGCGTGGCGCGGTTAA
"""
)

# pfkA: 4107552..4108514 (963 nt, includes stop codon)
_PFKA_CDS_DNA = _clean_dna(
    """
ATGATTAAGAAAATCGGTGTGTTGACAAGCGGCGGTGATGCGCCAGGCATGAACGCCGCAATTCGCGGGGTTGTTCGTTCTGCGCTGACAGAAGGTCTGGAAGTAATGGGTATTTATGACGGCTATCTGGGTCTGTATGAAGACCGTATGGTACAGCTAGACCGTTACAGCGTGTCTGACATGATCAACCGTGGCGGTACGTTCCTCGGTTCTGCGCGTTTCCCGGAATTCCGCGACGAGAACATCCGCGCCGTGGCTATCGAAAACCTGAAAAAACGTGGTATCGACGCGCTGGTGGTTATCGGCGGTGACGGTTCCTACATGGGTGCAATGCGTCTGACCGAAATGGGCTTCCCGTGCATCGGTCTGCCGGGCACTATCGACAACGACATCAAAGGCACTGACTACACTATCGGTTTCTTCACTGCGCTGAGCACCGTTGTAGAAGCGATCGACCGTCTGCGTGACACCTCTTCTTCTCACCAGCGTATTTCCGTGGTGGAAGTGATGGGCCGTTATTGTGGAGATCTGACGTTGGCTGCGGCCATTGCCGGTGGCTGTGAATTCGTTGTGGTTCCGGAAGTTGAATTCAGCCGTGAAGACCTGGTAAACGAAATCAAAGCGGGTATCGCGAAAGGTAAAAAACACGCGATCGTGGCGATTACCGAACATATGTGTGATGTTGACGAACTGGCGCATTTCATCGAGAAAGAAACCGGTCGTGAAACCCGCGCAACTGTGCTGGGCCACATCCAGCGCGGTGGTTCTCCGGTGCCTTACGACCGTATTCTGGCTTCCCGTATGGGCGCTTACGCTATCGATCTGCTGCTGGCAGGTTACGGCGGTCGTTGTGTAGGTATCCAGAACGAACAGCTGGTTCACCACGACATCATCGACGCTATCGAAAACATGAAGCGTCCGTTCAAAGGTGACTGGCTGGACTGCGCGAAAAAACTGTATTAA
"""
)

# tpiA: complement(4110740..4111507) (768 nt, includes stop codon)
# Source: NCBI RefSeq protein NP_418354.1 -> fasta_cds_na on NC_000913.3
_TPIA_CDS_DNA = _clean_dna(
    """
ATGCGACATCCTTTAGTGATGGGTAACTGGAAACTGAACGGCAGCCGCCACATGGTTCACGAGCTGGTTTCTAACCTGCGTAAAGAGCTGGCAGGTGTTGCTGGCTGTGCGGTTGCAATCGCACCACCGGAAATGTATATCGATATGGCGAAGCGCGAAGCTGAAGGCAGCCACATCATGCTGGGTGCGCAAAACGTGGACCTGAACCTGTCCGGCGCATTCACCGGTGAAACCTCTGCTGCTATGCTGAAAGACATCGGCGCACAGTACATCATCATCGGTCACTCTGAACGTCGTACTTACCACAAAGAATCTGACGAACTGATCGCGAAAAAATTCGCGGTGCTGAAAGAGCAGGGCCTGACTCCGGTTCTGTGCATCGGTGAAACCGAAGCTGAAAATGAAGCGGGCAAAACTGAAGAAGTTTGCGCACGTCAGATCGACGCGGTACTGAAAACTCAGGGTGCTGCGGCATTCGAAGGTGCGGTTATCGCTTACGAACCTGTATGGGCAATCGGTACTGGCAAATCTGCAACTCCGGCTCAGGCACAGGCTGTTCACAAATTCATCCGTGACCACATCGCTAAAGTTGACGCTAACATCGCTGAACAAGTGATCATTCAGTACGGCGGCTCTGTAAACGCGTCTAACGCTGCAGAACTGTTTGCTCAGCCGGATATCGACGGCGCGCTGGTTGGTGGTGCTTCTCTGAAAGCTGACGCCTTCGCAGTAATCGTTAAAGCTGCAGAAGCGGCTAAACAGGCTTAA
"""
)

# gapA: 1862771..1863766 (996 nt, includes stop codon)
# Source: NCBI RefSeq protein NP_416293.1 -> fasta_cds_na on NC_000913.3
_GAPA_CDS_DNA = _clean_dna(
    """
ATGACTATCAAAGTAGGTATCAACGGTTTTGGCCGTATCGGTCGCATTGTTTTCCGTGCTGCTCAGAAACGTTCTGACATCGAGATCGTTGCAATCAACGACCTGTTAGACGCTGATTACATGGCATACATGCTGAAATATGACTCCACTCACGGCCGTTTCGACGGTACCGTTGAAGTGAAAGACGGTCATCTGATCGTTAACGGTAAAAAAATCCGTGTTACCGCTGAACGTGATCCGGCTAACCTGAAATGGGACGAAGTTGGTGTTGACGTTGTCGCTGAAGCAACTGGTCTGTTCCTGACTGACGAAACTGCTCGTAAACACATCACCGCTGGTGCGAAGAAAGTGGTTATGACTGGTCCGTCTAAAGACAACACTCCGATGTTCGTTAAAGGCGCTAACTTCGACAAATATGCTGGCCAGGACATCGTTTCCAACGCTTCCTGCACCACCAACTGCCTGGCTCCGCTGGCTAAAGTTATCAACGATAACTTCGGCATCATCGAAGGTCTGATGACCACCGTTCACGCTACTACCGCTACTCAGAAAACCGTTGATGGCCCGTCTCACAAAGACTGGCGCGGCGGCCGCGGCGCTTCCCAGAACATCATCCCGTCCTCTACCGGTGCTGCTAAAGCTGTAGGTAAAGTACTGCCAGAACTGAATGGCAAACTGACTGGTATGGCGTTCCGCGTTCCGACCCCGAACGTATCTGTAGTTGACCTGACCGTTCGTCTGGAAAAAGCTGCAACTTACGAGCAGATCAAAGCTGCCGTTAAAGCTGCTGCTGAAGGCGAAATGAAAGGCGTTCTGGGCTACACCGAAGATGACGTAGTATCTACCGATTTCAACGGCGAAGTTTGCACTTCCGTGTTCGATGCTAAAGCTGGTATCGCTCTGAACGACAACTTCGTGAAACTGGTATCCTGGTACGACAACGAAACCGGTTACTCCAACAAAGTTCTGGACCTGATCGCTCACATCTCCAAATAA
"""
)

# pgk: complement(3071459..3072622) (1164 nt, includes stop codon)
# Source: NCBI RefSeq protein NP_417401.1 -> fasta_cds_na on NC_000913.3
_PGK_CDS_DNA = _clean_dna(
    """
ATGTCTGTAATTAAGATGACCGATCTGGATCTTGCTGGGAAACGTGTATTTATCCGTGCGGATCTGAACGTACCAGTAAAAGACGGGAAAGTAACCAGCGACGCGCGTATCCGTGCTTCTCTGCCGACCATCGAACTGGCCCTGAAACAAGGCGCAAAAGTGATGGTAACTTCCCACCTGGGTCGTCCTACCGAAGGCGAGTACAACGAAGAATTCTCTCTGCTGCCGGTTGTTAACTACCTGAAAGACAAACTGTCTAACCCGGTTCGTCTGGTTAAAGATTACCTCGACGGCGTTGACGTTGCTGAAGGTGAACTGGTTGTTCTGGAAAACGTTCGCTTCAACAAAGGCGAGAAGAAAGACGACGAAACCCTGTCCAAAAAATACGCTGCACTGTGTGACGTGTTCGTAATGGACGCATTCGGTACTGCTCACCGCGCGCAGGCTTCTACTCACGGTATCGGTAAATTCGCTGACGTTGCGTGCGCAGGCCCGCTGCTGGCAGCTGAACTGGACGCGCTGGGTAAAGCACTGAAAGAACCTGCTCGCCCGATGGTGGCTATCGTTGGTGGTTCTAAAGTATCTACCAAACTGACCGTTCTGGACTCCCTGTCTAAAATCGCTGACCAGCTGATTGTTGGTGGTGGTATCGCTAACACCTTTATCGCGGCACAAGGCCACGATGTGGGTAAATCCCTGTACGAAGCTGACCTGGTTGACGAAGCTAAACGTCTGCTGACCACCTGCAACATCCCGGTTCCGTCTGATGTTCGCGTAGCAACCGAGTTCTCTGAAACTGCACCGGCTACCCTGAAATCTGTTAACGATGTGAAAGCTGACGAGCAGATCCTGGATATCGGTGATGCTTCCGCTCAGGAACTGGCTGAAATCCTGAAGAATGCGAAAACCATTCTGTGGAACGGTCCGGTTGGCGTGTTCGAATTCCCGAACTTCCGCAAAGGTACTGAAATCGTGGCTAACGCTATCGCAGACAGCGAAGCGTTCTCCATCGCTGGCGGCGGCGACACTCTGGCAGCAATCGACCTGTTCGGCATTGCTGACAAAATCTCCTACATCTCCACTGGCGGCGGCGCATTCCTCGAATTCGTGGAAGGTAAAGTACTGCCTGCAGTAGCGATGCTCGAAGAGCGCGCTAAGAAGTAA
"""
)

# eno: complement(2906643..2907941) (1299 nt, includes stop codon)
# Source: NCBI RefSeq protein NP_417259.1 -> fasta_cds_na on NC_000913.3
_ENO_CDS_DNA = _clean_dna(
    """
ATGTCCAAAATCGTAAAAATCATCGGTCGTGAAATCATCGACTCCCGTGGTAACCCGACTGTTGAAGCCGAAGTACATCTGGAGGGTGGTTTCGTCGGTATGGCAGCTGCTCCGTCAGGTGCTTCTACTGGTTCCCGTGAAGCTCTGGAACTGCGCGATGGCGACAAATCCCGTTTCCTGGGTAAAGGCGTAACCAAAGCTGTTGCTGCGGTAAACGGCCCGATCGCTCAGGCGCTGATTGGCAAAGATGCTAAAGATCAGGCTGGCATTGACAAGATCATGATCGACCTGGACGGCACCGAAAACAAATCCAAATTCGGCGCGAACGCAATCCTGGCTGTATCTCTGGCTAACGCCAAAGCTGCTGCAGCTGCTAAAGGTATGCCGCTGTACGAGCACATCGCTGAACTGAACGGTACTCCGGGCAAATACTCTATGCCGGTTCCGATGATGAACATCATCAACGGTGGTGAGCACGCTGACAACAACGTTGATATCCAGGAATTCATGATTCAGCCGGTTGGCGCGAAAACTGTGAAAGAAGCCATCCGCATGGGTTCTGAAGTTTTCCATCACCTGGCAAAAGTTCTGAAAGCGAAAGGCATGAACACTGCTGTTGGTGACGAAGGTGGCTATGCGCCGAACCTGGGTTCCAACGCTGAAGCTCTGGCTGTTATCGCTGAAGCTGTTAAAGCTGCTGGTTATGAACTGGGCAAAGACATCACTTTGGCGATGGACTGCGCAGCTTCTGAATTCTACAAAGATGGTAAATACGTTCTGGCTGGCGAAGGCAACAAAGCGTTCACCTCTGAAGAATTCACTCACTTCCTGGAAGAACTGACCAAACAGTACCCGATCGTTTCTATCGAAGACGGTCTGGACGAATCTGACTGGGACGGTTTCGCATACCAGACCAAAGTTCTGGGCGACAAAATCCAGCTGGTTGGTGACGACCTGTTCGTAACCAACACCAAGATCCTGAAAGAAGGTATCGAAAAAGGTATCGCTAACTCCATCCTGATCAAATTCAACCAGATCGGTTCTCTGACCGAAACTCTGGCTGCAATCAAGATGGCGAAAGATGCTGGCTACACTGCAGTTATCTCTCACCGTTCTGGCGAAACTGAAGACGCTACCATCGCTGACCTGGCTGTTGGTACTGCTGCAGGCCAGATCAAAACTGGTTCTATGAGCCGTTCTGACCGTGTTGCTAAATACAACCAGCTGATTCGTATCGAAGAAGCTCTGGGCGAAAAAGCACCGTACAACGGTCGTAAAGAGATCAAAGGCCAGGCATAA
"""
)

KNOWN_CDS_DNA_BY_GENE: Dict[str, str] = {
    "pgi": _PGI_CDS_DNA,
    "pfka": _PFKA_CDS_DNA,
    "tpia": _TPIA_CDS_DNA,
    "gapa": _GAPA_CDS_DNA,
    "gapdh": _GAPA_CDS_DNA,  # user-facing alias used in this project
    "pgk": _PGK_CDS_DNA,
    "eno": _ENO_CDS_DNA,
}


def get_known_cds_dna(gene_name: str) -> Optional[str]:
    k = str(gene_name or "").strip().lower()
    if not k:
        return None
    return KNOWN_CDS_DNA_BY_GENE.get(k)


@dataclass(frozen=True)
class GeneConfig:
    name: str
    aa_length: int
    tl_scale: float = 1.0
    tx_scale: float = 1.0
    cds_dna: Optional[str] = None
    tx_nt_steps: Optional[int] = None


TWO_GENES: List[GeneConfig] = [
    # Keep CDS-driven transcription template and use auto tx-step compression
    # to preserve per-gene transcript-length differences.
    GeneConfig(name="Pgi", aa_length=549, tl_scale=1.0, tx_scale=1.0, cds_dna=_PGI_CDS_DNA, tx_nt_steps=None),
    GeneConfig(name="PfkA", aa_length=320, tl_scale=1.0, tx_scale=1.0, cds_dna=_PFKA_CDS_DNA, tx_nt_steps=None),
]


TX_TEMPLATE_RIDS = [f"r{i}" for i in range(344)]  # r0..r343

AA20: List[str] = [
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

AA3_FROM_AA1 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
}

# Standard bacterial genetic code (DNA codon -> one-letter AA; "*" is stop).
CODON_TO_AA1 = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def _ensure_species(
    model: "libsbml.Model",
    sid: str,
    compartment_id: str,
    init_conc: float = 0.0,
    boundary: bool = False,
    constant: bool = False,
    has_only_substance_units: bool = False,
) -> "libsbml.Species":
    sp = model.getSpecies(sid)
    if sp is None:
        sp = model.createSpecies()
        sp.setId(sid)
        sp.setCompartment(compartment_id)
    sp.setInitialConcentration(float(init_conc))
    if sp.isSetInitialAmount():
        sp.unsetInitialAmount()
    sp.setBoundaryCondition(bool(boundary))
    sp.setConstant(bool(constant))
    sp.setHasOnlySubstanceUnits(bool(has_only_substance_units))
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
    route1.wf.remove_reaction_if_exists(model, rid)
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
    out = {str(k): float(v) for k, v in data.get("batch_best_params", {}).items()}
    for pid, default_v in DEFAULT_MECH_QSSA_PARAMS.items():
        out.setdefault(pid, float(default_v))
    return out


def _apply_params_if_present(rr, params: Dict[str, float]) -> None:
    pids = set(rr.getGlobalParameterIds())
    for pid, val in params.items():
        if pid in pids:
            rr[pid] = float(val)


def _tx_species_map(template_sid: str, gid: str) -> str:
    if template_sid == "DNA":
        return f"DNA_{gid}"
    if template_sid == "mRNA_t":
        return f"mRNA_t_{gid}"
    if template_sid == "mRNA_i":
        return f"mRNA_i_{gid}"
    if template_sid == "MGapt":
        return f"MGapt_{gid}"
    if template_sid == "T7RNAP_bound_GTP":
        return f"T7RNAP_bound_GTP_{gid}"
    if template_sid == "T7RNAP_bound_GDP_PO4":
        return f"T7RNAP_bound_GDP_PO4_{gid}"
    if template_sid == "T7RNAP_bound":
        return f"T7RNAP_bound_{gid}"
    if template_sid.startswith("T7RNAP_bound_mRNA"):
        # e.g. T7RNAP_bound_mRNA0113_PPi -> T7RNAP_bound_<gid>_mRNA0113_PPi
        return template_sid.replace("T7RNAP_bound_mRNA", f"T7RNAP_bound_{gid}_mRNA", 1)
    return template_sid


def _replace_symbols(formula: str, mapping: Dict[str, str]) -> str:
    out = formula
    # Longest-first avoids partial replacement overlap.
    for old in sorted(mapping.keys(), key=len, reverse=True):
        new = mapping[old]
        if old == new:
            continue
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, out)
    return out


def _dna_to_rna(seq_dna: str) -> str:
    return _clean_dna(seq_dna).replace("T", "U")


def _compress_rna_sequence(seq_rna: str, n_steps: int) -> str:
    seq = "".join(ch for ch in str(seq_rna).upper() if ch in "ACGU")
    if not seq:
        raise RuntimeError("Empty RNA sequence")
    n = int(max(1, n_steps))
    if n >= len(seq):
        return seq
    out: List[str] = []
    L = len(seq)
    for i in range(n):
        a = int((i * L) // n)
        b = int(((i + 1) * L) // n)
        if b <= a:
            b = min(L, a + 1)
        chunk = seq[a:b]
        out.append(chunk[len(chunk) // 2])
    return "".join(out)


def _template_rna_sequence(model: "libsbml.Model") -> str:
    nts: List[str] = []
    for rid in TX_TEMPLATE_RIDS:
        rxn = model.getReaction(rid)
        if rxn is None:
            continue
        for j in range(rxn.getNumReactants()):
            sid = rxn.getReactant(j).getSpecies()
            if sid in ("ATP", "CTP", "GTP", "UTP"):
                nts.append(sid[0])  # ATP -> A, ...
                break
    return "".join(nts)


def _build_gene_tx_rna_sequence(model: "libsbml.Model", g: GeneConfig) -> Tuple[str, Dict[str, object]]:
    if g.cds_dna:
        rna_full = _dna_to_rna(g.cds_dna)
        nt_source = "mg1655_cds"
    else:
        rna_full = _template_rna_sequence(model)
        nt_source = "template_r0_r343"

    if not rna_full:
        raise RuntimeError(f"No transcription sequence available for gene {g.name}")

    if g.tx_nt_steps is not None:
        tx_steps_req = int(g.tx_nt_steps)
        tx_steps_mode = "manual"
    elif g.cds_dna:
        cds_nt = max(1, len(_clean_dna(g.cds_dna)))
        tx_steps_req = int(round(float(cds_nt) * float(DEFAULT_TX_COMPRESS_SCALE)))
        tx_steps_mode = "auto_cds_scaled"
    else:
        tx_steps_req = len(rna_full)
        tx_steps_mode = "full_template"
    tx_steps = max(1, min(int(tx_steps_req), len(rna_full)))
    rna_used = _compress_rna_sequence(rna_full, tx_steps) if tx_steps < len(rna_full) else rna_full

    meta = {
        "gene": g.name,
        "nt_source": nt_source,
        "cds_nt_full_len": int(len(rna_full)),
        "tx_nt_steps_used": int(len(rna_used)),
        "tx_nt_steps_request": int(tx_steps_req),
        "tx_nt_steps_mode": tx_steps_mode,
        "tx_nt_compressed": bool(len(rna_used) < len(rna_full)),
    }
    return rna_used, meta


def _build_gene_aa20_profile(g: GeneConfig) -> Tuple[Dict[str, int], Dict[str, object]]:
    """
    Build amino-acid demand profile for translation.
    Priority:
    1) Use CDS codon composition when available.
    2) Fallback to uniform AA20 split by configured aa_length.
    """
    counts = {aa: 0 for aa in AA20}
    cds = _clean_dna(g.cds_dna or "")
    invalid_codon_count = 0
    stop_seen = False
    source = "uniform_length_fallback"

    if cds:
        source = "cds_codon_profile"
        n_codons = len(cds) // 3
        for i in range(n_codons):
            codon = cds[3 * i : 3 * i + 3]
            aa1 = CODON_TO_AA1.get(codon)
            if aa1 is None:
                invalid_codon_count += 1
                continue
            if aa1 == "*":
                stop_seen = True
                break
            aa3 = AA3_FROM_AA1.get(aa1)
            if aa3 is None:
                invalid_codon_count += 1
                continue
            counts[aa3] += 1

    aa_total = int(sum(counts.values()))
    if aa_total <= 0:
        # Robust fallback for custom genes with missing CDS.
        aa_target = max(1, int(g.aa_length))
        base = aa_target // len(AA20)
        rem = aa_target % len(AA20)
        for idx, aa in enumerate(AA20):
            counts[aa] = int(base + (1 if idx < rem else 0))
        aa_total = int(sum(counts.values()))
        source = "uniform_length_fallback"

    nonzero = {aa: int(v) for aa, v in counts.items() if int(v) > 0}
    meta = {
        "gene": g.name,
        "aa_source": source,
        "aa_total": int(aa_total),
        "aa_nonzero_n": int(len(nonzero)),
        "aa_nonzero_counts": nonzero,
        "cds_nt_len": int(len(cds)),
        "cds_codon_len": int(len(cds) // 3),
        "cds_invalid_codons": int(invalid_codon_count),
        "cds_stop_seen": bool(stop_seen),
        "aa_length_config": int(g.aa_length),
        "aa_length_mismatch_vs_config": int(aa_total - int(g.aa_length)),
    }
    return counts, meta


def _aa20_drive_expr(aa_counts: Dict[str, int], half_sat_pid: str = "K_tl_AA") -> str:
    """
    Composition-aware availability term.
    Weighted geometric mean keeps 0..1 scaling and makes each required AA limiting.
    """
    def _sat_nonneg(sid: str) -> str:
        return f"(max({sid}, 0) / ({half_sat_pid} + max({sid}, 0)))"

    total = float(sum(max(0, int(v)) for v in aa_counts.values()))
    if total <= 0.0:
        return _sat_nonneg("AA")

    terms: List[str] = []
    for aa in AA20:
        cnt = int(aa_counts.get(aa, 0))
        if cnt <= 0:
            continue
        w = float(cnt / total)
        terms.append(f"({_sat_nonneg(aa)} ^ {w:.12g})")
    return "(" + " * ".join(terms) + ")" if terms else _sat_nonneg("AA")


def _ensure_mechanistic_qssa_params(model: "libsbml.Model") -> None:
    for pid, val in DEFAULT_MECH_QSSA_PARAMS.items():
        route1.wf.ensure_parameter(model, pid, float(val))


def _qssa_tl_rate_expr(
    sid_ribo: str,
    tl_scale_pid: str,
    len_scale_pid: str,
    aa_drive_expr: str,
    keff_factor: str = "",
) -> str:
    gtp_sat = "(max(GTP, 0) / (K_tl_GTP + max(GTP, 0)))"
    atp_sat = "(max(ATP, 0) / (K_tl_ATP + max(ATP, 0)))"
    k_init = f"(default * k_tl_init * {gtp_sat})"
    k_elong = f"(default * k_tl_elong * {aa_drive_expr} * {atp_sat} * {gtp_sat})"
    k_term = f"(default * k_tl_term * {gtp_sat})"
    v_eff = (
        f"(({k_init}) * ({k_elong}) * ({k_term})) / "
        f"(K_tl_eps + ({k_init}) * ({k_elong}) + ({k_init}) * ({k_term}) + ({k_elong}) * ({k_term}))"
    )
    return f"{tl_scale_pid} * {len_scale_pid} * {sid_ribo} * ({v_eff}){keff_factor}"


def _ensure_species_like(model: "libsbml.Model", src_sid: str, new_sid: str) -> bool:
    src = model.getSpecies(src_sid)
    if src is None:
        raise RuntimeError(f"Missing template species: {src_sid}")
    existed = model.getSpecies(new_sid) is not None
    _ensure_species(
        model=model,
        sid=new_sid,
        compartment_id=src.getCompartment(),
        init_conc=0.0,
        boundary=False,  # gene-specific clones must always be floating ODE variables
        constant=src.getConstant(),
        has_only_substance_units=src.getHasOnlySubstanceUnits(),
    )
    return existed


def _add_explicit_tx_chain_from_rna(
    model: "libsbml.Model",
    gid: str,
    rna_seq: str,
    tx_scale_pid: Optional[str] = None,
) -> Tuple[Dict[str, str], int, int]:
    nts = "".join(ch for ch in str(rna_seq).upper() if ch in "ACGU")
    if not nts:
        raise RuntimeError(f"Empty RNA sequence for {gid}")

    species_added = 0
    reactions_added = 0

    mapping = {
        "DNA": f"DNA_{gid}",
        "mRNA_t": f"mRNA_t_{gid}",
        "T7RNAP_bound_GTP": f"T7RNAP_bound_GTP_{gid}",
        "T7RNAP_bound_GDP_PO4": f"T7RNAP_bound_GDP_PO4_{gid}",
        "T7RNAP_bound": f"T7RNAP_bound_{gid}",
    }

    for src, dst in (
        ("DNA", mapping["DNA"]),
        ("mRNA_t", mapping["mRNA_t"]),
        ("T7RNAP_bound_GTP", mapping["T7RNAP_bound_GTP"]),
        ("T7RNAP_bound_GDP_PO4", mapping["T7RNAP_bound_GDP_PO4"]),
        ("T7RNAP_bound", mapping["T7RNAP_bound"]),
        ("T7RNAP_bound_mRNA0000", f"T7RNAP_bound_{gid}_mRNA0000"),
    ):
        existed = _ensure_species_like(model, src, dst)
        if not existed:
            species_added += 1

    # Optional numerical/model-reduction mode for large 1nt multi-gene panels.
    #
    # The full template represents each nucleotide addition as three very fast
    # microstates (NTP-bound, polymerized-PPi, next position).  In 6-gene 1nt
    # co-expression those intermediates sit around 1e-20--1e-18 M, below the
    # practical absolute-error scale of CVODE.  This QSSA mode keeps the 1nt
    # position chain and the same fitted kinetic constants, but collapses the
    # sub-nt microstates into one positive effective step per nucleotide.
    micro_qssa = str(os.getenv("PURE_TX_MICRO_QSSA", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "qssa",
    }

    route1.wf.ensure_parameter(model, "gene_tx_qssa_eps", 1.0e-30)

    # initiation
    tx_scale_mul = f" * {tx_scale_pid}" if tx_scale_pid else ""
    tx_scale_expr = f"{tx_scale_pid}" if tx_scale_pid else "1"
    if micro_qssa:
        # r0 remains the fitted promoter/RNAP/NTP association step.  The
        # subsequent r1/r2/r3 initiation microstates are fast relative to the
        # transcript-position chain, so they are collapsed directly into the
        # mRNA0000-loaded polymerase state.  DNA and RNAP are still held until
        # transcript release, preserving template/polymerase occupancy.
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_init_r0",
            reactants={"T7RNAP": 1.0, mapping["DNA"]: 1.0, "GTP": 1.0},
            products={
                f"T7RNAP_bound_{gid}_mRNA0000": 1.0,
                "GDP": 1.0,
                "PO4": 1.0,
            },
            formula=f"k_forward * T7RNAP * {mapping['DNA']} * GTP{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
    else:
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_init_r0",
            reactants={"T7RNAP": 1.0, mapping["DNA"]: 1.0, "GTP": 1.0},
            products={mapping["T7RNAP_bound_GTP"]: 1.0},
            formula=f"k_forward * T7RNAP * {mapping['DNA']} * GTP{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_init_r1",
            reactants={mapping["T7RNAP_bound_GTP"]: 1.0},
            products={mapping["T7RNAP_bound_GDP_PO4"]: 1.0},
            formula=f"k_forward1 * {mapping['T7RNAP_bound_GTP']}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_init_r2",
            reactants={mapping["T7RNAP_bound_GDP_PO4"]: 1.0},
            products={mapping["T7RNAP_bound"]: 1.0, "GDP": 1.0, "PO4": 1.0},
            formula=f"k_forward2 * {mapping['T7RNAP_bound_GDP_PO4']}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_init_r3",
            reactants={mapping["T7RNAP_bound"]: 1.0},
            products={f"T7RNAP_bound_{gid}_mRNA0000": 1.0},
            formula=f"k_forward3 * {mapping['T7RNAP_bound']}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1

    ntp_sid = {"A": "ATP", "C": "CTP", "G": "GTP", "U": "UTP"}
    for i, nt in enumerate(nts):
        s_cur = f"T7RNAP_bound_{gid}_mRNA{i:04d}"
        s_bound = f"{s_cur}_{ntp_sid[nt]}"
        s_ppi = f"T7RNAP_bound_{gid}_mRNA{i + 1:04d}_PPi"
        s_next = f"T7RNAP_bound_{gid}_mRNA{i + 1:04d}"

        if micro_qssa:
            existed = _ensure_species_like(model, "T7RNAP_bound_mRNA0001", s_next)
            if not existed:
                species_added += 1

            ntp = ntp_sid[nt]
            eff_rate = (
                f"(1 / ("
                f"(1 / (k_forward4 * {ntp} * {tx_scale_expr} + gene_tx_qssa_eps))"
                f" + (1 / (k_forward5 * {tx_scale_expr} + gene_tx_qssa_eps))"
                f" + (1 / (k_forward6 * {tx_scale_expr} + gene_tx_qssa_eps))"
                f"))"
            )
            _create_reaction(
                model=model,
                rid=f"rxn_tx_{gid}_elong_{i:04d}_bind",
                reactants={s_cur: 1.0, ntp: 1.0},
                products={s_next: 1.0, "PPi": 1.0},
                formula=f"{s_cur} * {eff_rate}",
                reversible=False,
            )
            reactions_added += 1
            continue

        for src_sid, new_sid in (
            ("T7RNAP_bound_mRNA0000_GTP", s_bound),
            ("T7RNAP_bound_mRNA0001_PPi", s_ppi),
            ("T7RNAP_bound_mRNA0001", s_next),
        ):
            existed = _ensure_species_like(model, src_sid, new_sid)
            if not existed:
                species_added += 1

        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_elong_{i:04d}_bind",
            reactants={s_cur: 1.0, ntp_sid[nt]: 1.0},
            products={s_bound: 1.0},
            formula=f"k_forward4 * {s_cur} * {ntp_sid[nt]}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_elong_{i:04d}_poly",
            reactants={s_bound: 1.0},
            products={s_ppi: 1.0},
            formula=f"k_forward5 * {s_bound}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_elong_{i:04d}_ppi",
            reactants={s_ppi: 1.0},
            products={s_next: 1.0, "PPi": 1.0},
            formula=f"k_forward6 * {s_ppi}{tx_scale_mul}",
            reversible=False,
        )
        reactions_added += 1

    mapping["tx_release_state"] = f"T7RNAP_bound_{gid}_mRNA{len(nts):04d}"
    return mapping, species_added, reactions_added


def _collect_tx_template_species(model: "libsbml.Model") -> Set[str]:
    sids: Set[str] = set()
    for rid in TX_TEMPLATE_RIDS:
        rxn = model.getReaction(rid)
        if rxn is None:
            raise RuntimeError(f"Missing transcription template reaction: {rid}")
        for j in range(rxn.getNumReactants()):
            sids.add(rxn.getReactant(j).getSpecies())
        for j in range(rxn.getNumProducts()):
            sids.add(rxn.getProduct(j).getSpecies())
        for j in range(rxn.getNumModifiers()):
            sids.add(rxn.getModifier(j).getSpecies())
    return sids


def _clone_explicit_tx_chain(
    model: "libsbml.Model",
    gid: str,
) -> Tuple[Dict[str, str], int, int]:
    template_sids = _collect_tx_template_species(model)
    mapping = {sid: _tx_species_map(sid, gid) for sid in template_sids}

    species_added = 0
    for sid in sorted(template_sids):
        mapped = mapping[sid]
        if mapped == sid:
            continue  # shared global pool species
        src = model.getSpecies(sid)
        if src is None:
            raise RuntimeError(f"Missing template species: {sid}")
        already = model.getSpecies(mapped) is not None
        _ensure_species(
            model=model,
            sid=mapped,
            compartment_id=src.getCompartment(),
            init_conc=0.0,
            boundary=src.getBoundaryCondition(),
            constant=src.getConstant(),
            has_only_substance_units=src.getHasOnlySubstanceUnits(),
        )
        if not already:
            species_added += 1

    reactions_added = 0
    for rid in TX_TEMPLATE_RIDS:
        src_rxn = model.getReaction(rid)
        if src_rxn is None:
            raise RuntimeError(f"Missing transcription template reaction: {rid}")
        new_rid = f"rxn_tx_{gid}_{rid}"
        if model.getReaction(new_rid) is not None:
            continue

        reactants: Dict[str, float] = {}
        products: Dict[str, float] = {}
        for j in range(src_rxn.getNumReactants()):
            ref = src_rxn.getReactant(j)
            reactants[mapping.get(ref.getSpecies(), ref.getSpecies())] = float(ref.getStoichiometry())
        for j in range(src_rxn.getNumProducts()):
            ref = src_rxn.getProduct(j)
            products[mapping.get(ref.getSpecies(), ref.getSpecies())] = float(ref.getStoichiometry())

        kl = src_rxn.getKineticLaw()
        formula_src = libsbml.formulaToL3String(kl.getMath()) if kl is not None else "0"
        formula_new = _replace_symbols(formula_src, mapping)

        _create_reaction(
            model=model,
            rid=new_rid,
            reactants=reactants,
            products=products,
            formula=formula_new,
            reversible=src_rxn.getReversible(),
        )
        reactions_added += 1

    return mapping, species_added, reactions_added


def _extend_with_explicit_two_lines(
    sbml_text: str,
    genes: List[GeneConfig],
    native_degfp_share: float,
    mrna_decay_base: float = DEFAULT_MRNA_DECAY_BASE,
    mrna_bound_decay_ratio: float = DEFAULT_MRNA_BOUND_DECAY_RATIO,
    tx_fast_scale: float = 1.0,
    ribo_bind_scale: float = 1.0,
) -> Tuple[str, Dict[str, str], Dict[str, str], Dict[str, int]]:
    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Failed to parse SBML")

    counts_before = {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
    }

    # Optionally keep native channel by scaling its final release.
    route1.wf.ensure_parameter(model, "gene_share_degfp", float(native_degfp_share))
    r343 = model.getReaction("r343")
    if r343 is None:
        raise RuntimeError("Base model missing r343")
    _set_reaction_formula(r343, "k_forward7 * T7RNAP_bound_mRNA0113 * gene_share_degfp")

    has_bg_keff = model.getParameter("bg_keff_scale") is not None and model.getParameter("bg_keff") is not None
    keff_factor = " * (bg_keff_scale * bg_keff)" if has_bg_keff else ""
    route1.wf.ensure_parameter(model, "gene_mrna_decay_base", float(max(0.0, mrna_decay_base)))
    route1.wf.ensure_parameter(model, "gene_mrna_bound_decay_ratio", float(max(0.0, mrna_bound_decay_ratio)))
    route1.wf.ensure_parameter(model, "gene_tx_fast_scale", float(max(0.0, tx_fast_scale)))
    route1.wf.ensure_parameter(model, "gene_ribo_bind_scale", float(max(0.0, ribo_bind_scale)))
    _ensure_mechanistic_qssa_params(model)
    # Keep TL pathway mode aligned with the finest handover:
    # step_chain + g=1 can be enabled globally via env.
    tl_pathway_mode = os.getenv("PURE_TL_PATHWAY_MODE", "core_qssa").strip().lower()
    use_tl_chain = tl_pathway_mode in {"step_chain", "chain", "net_step"}

    protein_by_gene: Dict[str, str] = {}
    dna_by_gene: Dict[str, str] = {}
    explicit_added_species = 0
    explicit_added_reactions = 0

    for g in genes:
        gid = route1.wf.sanitize_id(g.name)
        aa_counts, aa_meta = _build_gene_aa20_profile(g)
        aa_need = max(1, int(aa_meta["aa_total"]))
        route1.wf.ensure_parameter(model, f"gene_tl_scale_{gid}", float(g.tl_scale))
        route1.wf.ensure_parameter(model, f"gene_len_scale_{gid}", 226.0 / float(max(1, aa_need)))
        route1.wf.ensure_parameter(model, f"gene_tx_scale_{gid}", float(g.tx_scale))
        route1.wf.ensure_parameter(model, f"gene_tx_chain_scale_{gid}", float(max(0.0, g.tx_scale) * max(0.0, tx_fast_scale)))
        route1.wf.ensure_parameter(model, f"gene_mrna_decay_scale_{gid}", 1.0)

        rna_seq_gene, _seq_meta = _build_gene_tx_rna_sequence(model, g)
        mapping, s_add, r_add = _add_explicit_tx_chain_from_rna(
            model=model,
            gid=gid,
            rna_seq=rna_seq_gene,
            tx_scale_pid=f"gene_tx_chain_scale_{gid}",
        )
        explicit_added_species += s_add
        explicit_added_reactions += r_add

        sid_dna = mapping["DNA"]
        sid_mrna = mapping["mRNA_t"]
        sid_ribo = f"Ribosome_mRNA_{gid}"
        sid_pept = f"Pept_{gid}"
        sid_prot = f"{gid}_m"
        protein_by_gene[g.name] = sid_prot
        dna_by_gene[g.name] = sid_dna

        mrna_sp = model.getSpecies(sid_mrna)
        if mrna_sp is None:
            raise RuntimeError(f"Missing mapped mRNA species: {sid_mrna}")
        compartment_id = mrna_sp.getCompartment()

        # Gene-specific release from the gene-specific final transcription state.
        _create_reaction(
            model=model,
            rid=f"rxn_tx_{gid}_release",
            reactants={mapping["tx_release_state"]: 1.0},
            # DNA acts as a template and must be regenerated at transcript release.
            products={"T7RNAP": 1.0, mapping["DNA"]: 1.0, sid_mrna: 1.0},
            formula=f"(k_forward7 * gene_tx_chain_scale_{gid}) * {mapping['tx_release_state']}",
            reversible=False,
        )
        explicit_added_reactions += 1

        _ensure_species(model, sid_ribo, compartment_id, init_conc=0.0, boundary=False, constant=False)
        _ensure_species(model, sid_pept, compartment_id, init_conc=0.0, boundary=False, constant=False)
        _ensure_species(model, sid_prot, compartment_id, init_conc=0.0, boundary=False, constant=False)

        _create_reaction(
            model=model,
            rid=f"rxn_ribosome_binding_{gid}",
            reactants={sid_mrna: 1.0, "Ribosome": 1.0},
            products={sid_ribo: 1.0},
            formula=(
                f"gene_ribo_bind_scale * ("
                f"default * k_ribo_bind_f * {sid_mrna} * Ribosome"
                f" - default * k_ribo_bind_r * {sid_ribo}"
                f")"
            ),
            reversible=True,
        )
        gtp_need = 2 * aa_need
        aa_reactants: Dict[str, float] = {}
        aa_missing = [aa for aa, cnt in aa_counts.items() if int(cnt) > 0 and model.getSpecies(aa) is None]
        if aa_missing:
            aa_reactants["AA"] = float(aa_need)
            aa_drive_expr = "(max(AA, 0) / (K_tl_AA + max(AA, 0)))"
        else:
            for aa in AA20:
                cnt = int(aa_counts.get(aa, 0))
                if cnt > 0:
                    aa_reactants[aa] = float(cnt)
            aa_drive_expr = _aa20_drive_expr(aa_counts, half_sat_pid="K_tl_AA")

        # Build factorized translation gates aligned with the single-protein finest profile.
        def _sid_or_zero(sid: str) -> str:
            return sid if model.getSpecies(sid) is not None else "0"

        def _soft_sat(expr: str, km_pid: str, floor_pid: str) -> str:
            return f"({floor_pid} + (1 - {floor_pid}) * (({expr})/({km_pid} + ({expr}))))"

        route1.wf.ensure_parameter(model, "K_tl_IF2_tot", 1.0e-8)
        route1.wf.ensure_parameter(model, "K_tl_EFTu_tot", 5.0e-8)
        route1.wf.ensure_parameter(model, "K_tl_EFG_tot", 5.0e-8)
        route1.wf.ensure_parameter(model, "phi_tl_init_floor", 0.25)
        route1.wf.ensure_parameter(model, "phi_tl_elong_floor", 0.25)
        route1.wf.ensure_parameter(model, "phi_tl_term_floor", 0.25)

        gtp_sat = "(max(GTP, 0) / (K_tl_GTP + max(GTP, 0)))"
        atp_sat = "(max(ATP, 0) / (K_tl_ATP + max(ATP, 0)))"
        tl_qssa_mode = os.getenv("PURE_TL_QSSA_MODE", "classic").strip().lower()
        if tl_qssa_mode == "factorized":
            if2_tot = f"({_sid_or_zero('IF2')} + {_sid_or_zero('IF2_GTP')})"
            eftu_tot = f"({_sid_or_zero('EFTu')} + {_sid_or_zero('EFTu_GTP')})"
            efg_tot = f"({_sid_or_zero('EFG')} + {_sid_or_zero('EFG_GTP')})"
            init_gate = _soft_sat(if2_tot, "K_tl_IF2_tot", "phi_tl_init_floor")
            elong_gate = (
                f"({_soft_sat(eftu_tot, 'K_tl_EFTu_tot', 'phi_tl_elong_floor')} * "
                f"{_soft_sat(efg_tot, 'K_tl_EFG_tot', 'phi_tl_elong_floor')})"
            )
            term_gate = _soft_sat(efg_tot, "K_tl_EFG_tot", "phi_tl_term_floor")
        else:
            init_gate = "1"
            elong_gate = "1"
            term_gate = "1"

        k_init_expr = f"(default * k_tl_init * {gtp_sat} * {init_gate})"
        k_elong_expr = f"(default * k_tl_elong * {aa_drive_expr} * {atp_sat} * {gtp_sat} * {elong_gate})"
        k_term_expr = f"(default * k_tl_term * {gtp_sat} * {term_gate})"

        if use_tl_chain:
            # Per-gene translation chain (step_chain). With PURE_TL_CHAIN_GROUP_AA=1.0,
            # this becomes one net step per amino acid.
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
                    n_steps = max(1, int(np.ceil(float(aa_need) / max(1.0, group_aa))))
            else:
                n_steps = max(1, int(np.ceil(float(aa_need) / max(1.0, group_aa))))

            chain_scale_env = os.getenv("PURE_TL_CHAIN_SCALE", "").strip()
            chain_scale = 1.0
            if chain_scale_env:
                try:
                    chain_scale = max(1e-6, float(chain_scale_env))
                except Exception:
                    chain_scale = 1.0
            route1.wf.ensure_parameter(model, f"gene_tl_chain_scale_{gid}", float(chain_scale))
            route1.wf.ensure_parameter(model, f"gene_tl_chain_steps_{gid}", float(n_steps))

            # Resource stoichiometry per chain step (preserves AA20 composition).
            atp_per = float(aa_need) / float(n_steps)
            gtp_per = float(gtp_need) / float(n_steps)
            aa_step_reactants: Dict[str, float] = {
                sid: float(sto) / float(n_steps) for sid, sto in aa_reactants.items() if float(sto) > 0.0
            }

            chain_sids: List[str] = []
            for i in range(n_steps + 1):
                sid_chain = f"TLchain_{gid}_pos_{i:04d}"
                existed = model.getSpecies(sid_chain) is not None
                _ensure_species(model, sid_chain, compartment_id, init_conc=0.0, boundary=False, constant=False)
                if not existed:
                    explicit_added_species += 1
                chain_sids.append(sid_chain)

            _create_reaction(
                model=model,
                rid=f"rxn_tl_chain_load_{gid}",
                reactants={sid_ribo: 1.0},
                products={chain_sids[0]: 1.0},
                formula=f"{sid_ribo} * (gene_tl_scale_{gid} * {k_init_expr}){keff_factor}",
                reversible=False,
            )
            explicit_added_reactions += 1

            k_step_expr = (
                f"(gene_tl_scale_{gid} * gene_tl_chain_scale_{gid} * "
                f"gene_tl_chain_steps_{gid} * {k_elong_expr})"
            )
            for i in range(n_steps):
                reactants_step: Dict[str, float] = {chain_sids[i]: 1.0}
                if model.getSpecies("ATP") is not None and atp_per > 0.0:
                    reactants_step["ATP"] = float(atp_per)
                if model.getSpecies("GTP") is not None and gtp_per > 0.0:
                    reactants_step["GTP"] = float(gtp_per)
                for sid_aa, sto in aa_step_reactants.items():
                    if model.getSpecies(sid_aa) is not None and float(sto) > 0.0:
                        reactants_step[sid_aa] = float(sto)

                _create_reaction(
                    model=model,
                    rid=f"rxn_tl_chain_step_{gid}_{i:04d}",
                    reactants=reactants_step,
                    products={chain_sids[i + 1]: 1.0},
                    formula=f"{chain_sids[i]} * {k_step_expr}{keff_factor}",
                    reversible=False,
                )
                explicit_added_reactions += 1

            _create_reaction(
                model=model,
                rid=f"rxn_tl_chain_term_{gid}",
                reactants={chain_sids[-1]: 1.0},
                products={"Ribosome": 1.0, sid_mrna: 1.0, sid_pept: 1.0},
                formula=f"{chain_sids[-1]} * (gene_tl_scale_{gid} * {k_term_expr}){keff_factor}",
                reversible=False,
            )
            explicit_added_reactions += 1
        else:
            tl_rate_expr = _qssa_tl_rate_expr(
                sid_ribo=sid_ribo,
                tl_scale_pid=f"gene_tl_scale_{gid}",
                len_scale_pid=f"gene_len_scale_{gid}",
                aa_drive_expr=aa_drive_expr,
                keff_factor=keff_factor,
            )
            _create_reaction(
                model=model,
                rid=f"rxn_translation_elongation_{gid}",
                reactants={sid_ribo: 1.0, "ATP": float(aa_need), "GTP": float(gtp_need), **aa_reactants},
                products={"Ribosome": 1.0, sid_mrna: 1.0, sid_pept: 1.0},
                formula=tl_rate_expr,
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
        _create_reaction(
            model=model,
            rid=f"rxn_mrna_decay_{gid}",
            reactants={sid_mrna: 1.0},
            products={},
            formula=f"gene_mrna_decay_base * gene_mrna_decay_scale_{gid} * {sid_mrna}",
            reversible=False,
        )
        _create_reaction(
            model=model,
            rid=f"rxn_mrna_bound_decay_{gid}",
            reactants={sid_ribo: 1.0},
            products={"Ribosome": 1.0},
            formula=(
                f"gene_mrna_decay_base * gene_mrna_bound_decay_ratio * "
                f"gene_mrna_decay_scale_{gid} * {sid_ribo}"
            ),
            reversible=False,
        )
        explicit_added_reactions += 2

    counts_after = {
        "species_total": model.getNumSpecies(),
        "reactions_total": model.getNumReactions(),
    }
    delta = {
        "species_added": counts_after["species_total"] - counts_before["species_total"],
        "reactions_added": counts_after["reactions_total"] - counts_before["reactions_total"],
        "explicit_tx_species_added": int(explicit_added_species),
        "explicit_tx_reactions_added": int(explicit_added_reactions),
    }
    return libsbml.writeSBMLToString(doc), protein_by_gene, dna_by_gene, delta


def _set_dna_initials(sbml_text: str, dna_init_M: Dict[str, float]) -> str:
    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        raise RuntimeError("Invalid SBML")
    for sid, v in dna_init_M.items():
        sp = model.getSpecies(sid)
        if sp is None:
            continue
        sp.setInitialConcentration(float(v))
        if sp.isSetInitialAmount():
            sp.unsetInitialAmount()
    return libsbml.writeSBMLToString(doc)


def _build_timecourse_selections(protein_species: Dict[str, str], dna_species: Dict[str, str]) -> List[str]:
    sel = ["time", "[ATP]", "[GTP]", "[PPi]", "[PO4]", "[Ribosome]", "[T7RNAP]", "[DNA]", "[deGFP_m]"]
    for _, sid_dna in dna_species.items():
        sel.append(f"[{sid_dna}]")
    for _, sid in protein_species.items():
        sel.append(f"[{sid}]")
    # Track mRNA explicitly for both genes
    for gname, sid in dna_species.items():
        gid = route1.wf.sanitize_id(gname)
        sel.append(f"[mRNA_t_{gid}]")
    return list(dict.fromkeys(sel))


def _simulate(rr, protein_species: Dict[str, str], dna_species: Dict[str, str], t_end_h: float, points: int) -> pd.DataFrame:
    sel = _build_timecourse_selections(protein_species=protein_species, dna_species=dna_species)

    t_end_s = float(t_end_h) * 3600.0
    n_points = int(points)
    try:
        res = rr.simulate(0.0, t_end_s, n_points, sel)
        return pd.DataFrame(res, columns=res.colnames)
    except RuntimeError:
        integ = rr.getIntegrator()
        atol = float(os.environ.get("PURE_RR_ATOL_FALLBACK", os.environ.get("PURE_RR_ATOL", "1e-9")))
        rtol = float(os.environ.get("PURE_RR_RTOL_FALLBACK", os.environ.get("PURE_RR_RTOL", "3e-5")))
        max_steps = int(float(os.environ.get("PURE_RR_MAX_STEPS_FALLBACK", os.environ.get("PURE_RR_MAX_STEPS", "2000000"))))
        for key, value in (("absolute_tolerance", atol), ("relative_tolerance", rtol), ("maximum_num_steps", max_steps)):
            try:
                integ.setValue(key, value)
            except Exception:
                pass
        rr.resetToOrigin()
        res = rr.simulate(0.0, t_end_s, n_points, sel)
        return pd.DataFrame(res, columns=res.colnames)


def _write_status_json(path: Optional[Path], payload: Dict[str, object]) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["wall_time_s"] = float(time.time())
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(p)


def _load_rr_with_compile_heartbeat(
    sbml_text: str,
    status_json: Optional[Path],
    progress_label: str,
    heartbeat_s: float = 2.0,
):
    if status_json is None:
        return te.loadSBMLModel(sbml_text)

    t0 = time.time()
    stop = threading.Event()

    def _heartbeat() -> None:
        tick = 0
        dt = float(max(0.2, heartbeat_s))
        while not stop.is_set():
            _write_status_json(
                status_json,
                {
                    "stage": "simulate_prepare",
                    "label": str(progress_label),
                    "message": "loading SBML into roadrunner",
                    "compile_elapsed_s": float(time.time() - t0),
                    "compile_heartbeat_idx": int(tick),
                },
            )
            tick += 1
            stop.wait(dt)

    th = threading.Thread(target=_heartbeat, daemon=True)
    th.start()
    try:
        rr = te.loadSBMLModel(sbml_text)
    finally:
        stop.set()
        try:
            th.join(timeout=0.5)
        except Exception:
            pass

    _write_status_json(
        status_json,
        {
            "stage": "simulate_prepare",
            "label": str(progress_label),
            "message": "roadrunner loaded; configuring integrator",
            "compile_elapsed_s": float(time.time() - t0),
        },
    )
    return rr


def _simulate_progressive(
    rr,
    protein_species: Dict[str, str],
    dna_species: Dict[str, str],
    t_end_h: float,
    points: int,
    chunk_h: float,
    points_per_chunk: int,
    status_json: Optional[Path],
    live_csv: Optional[Path],
    checkpoint_csv: Optional[Path],
    progress_label: str,
) -> pd.DataFrame:
    sel = _build_timecourse_selections(protein_species=protein_species, dna_species=dna_species)
    t_end_s = float(max(0.0, t_end_h)) * 3600.0
    if t_end_s <= 0.0:
        return pd.DataFrame(columns=sel)

    chunk_s = float(max(1e-6, chunk_h)) * 3600.0
    n_chunks = int(max(1, np.ceil(t_end_s / chunk_s)))
    if int(points_per_chunk) > 1:
        npt_chunk = int(points_per_chunk)
    else:
        npt_chunk = int(max(2, np.ceil(float(max(2, int(points))) / float(n_chunks))))

    rows_ckpt: List[Dict[str, object]] = []
    all_chunks: List[pd.DataFrame] = []
    t_cur_s = 0.0
    wall_t0 = time.time()
    avoid_hour_eps_s = float(max(0.0, float(os.environ.get("PURE_PROGRESSIVE_AVOID_HOUR_EPS_S", "1e-6"))))

    def _snapshot_row_at_time(t_s: float) -> Dict[str, float]:
        row: Dict[str, float] = {"time": float(t_s)}
        for sid in sel:
            if sid == "time":
                continue
            try:
                row[str(sid)] = float(rr[str(sid)])
            except Exception:
                row[str(sid)] = 0.0
        return row

    def _simulate_chunk_bisect(
        t0_s: float,
        t1_s: float,
        npt: int,
        depth: int,
        min_window_s: float,
    ) -> pd.DataFrame:
        """
        Robust fallback for stiff windows:
        recursively bisect one chunk until CVODE converges.
        """
        npt_local = int(max(2, npt))
        try:
            res_local = rr.simulate(float(t0_s), float(t1_s), npt_local, sel)
            return pd.DataFrame(res_local, columns=res_local.colnames)
        except RuntimeError:
            if depth <= 0 or (t1_s - t0_s) <= float(min_window_s):
                raise
            tm = 0.5 * (float(t0_s) + float(t1_s))
            left = _simulate_chunk_bisect(float(t0_s), float(tm), npt_local, depth - 1, min_window_s)
            right = _simulate_chunk_bisect(float(tm), float(t1_s), npt_local, depth - 1, min_window_s)
            if len(right) > 0:
                right = right.iloc[1:].reset_index(drop=True)
            if len(left) <= 0:
                return right
            if len(right) <= 0:
                return left
            return pd.concat([left, right], ignore_index=True)

    _write_status_json(
        status_json,
        {
            "stage": "simulate",
            "label": str(progress_label),
            "message": "simulation started",
            "t_end_h": float(t_end_h),
            "chunk_h": float(chunk_h),
            "n_chunks": int(n_chunks),
            "n_points_per_chunk": int(npt_chunk),
            "progress": {"done_chunks": 0, "total_chunks": int(n_chunks), "time_h": 0.0},
        },
    )

    for i in range(1, n_chunks + 1):
        t_next_s = min(t_end_s, t_cur_s + chunk_s)
        try:
            res = rr.simulate(float(t_cur_s), float(t_next_s), int(npt_chunk), sel)
        except RuntimeError:
            integ = rr.getIntegrator()
            atol = float(os.environ.get("PURE_RR_ATOL_FALLBACK", os.environ.get("PURE_RR_ATOL", "1e-9")))
            rtol = float(os.environ.get("PURE_RR_RTOL_FALLBACK", os.environ.get("PURE_RR_RTOL", "3e-5")))
            max_steps = int(
                float(os.environ.get("PURE_RR_MAX_STEPS_FALLBACK", os.environ.get("PURE_RR_MAX_STEPS", "2000000")))
            )
            for key, value in (
                ("absolute_tolerance", atol),
                ("relative_tolerance", rtol),
                ("maximum_num_steps", max_steps),
            ):
                try:
                    integ.setValue(key, value)
                except Exception:
                    pass
            try:
                res = rr.simulate(float(t_cur_s), float(t_next_s), int(npt_chunk), sel)
                df_chunk = pd.DataFrame(res, columns=res.colnames)
            except RuntimeError:
                max_depth = int(float(os.environ.get("PURE_PROGRESSIVE_SPLIT_MAX_DEPTH", "6")))
                min_win_s = float(max(1e-6, float(os.environ.get("PURE_PROGRESSIVE_MIN_WINDOW_S", "1.0"))))
                # Boundary-stiffness guard:
                # if solver repeatedly fails exactly at the chunk start, nudge by epsilon
                # and continue from a slightly advanced time.
                eps_s = float(max(1e-9, float(os.environ.get("PURE_PROGRESSIVE_START_NUDGE_S", "1e-6"))))
                eps_s = float(min(eps_s, max(1e-9, 0.01 * (float(t_next_s) - float(t_cur_s)))))
                row0 = _snapshot_row_at_time(float(t_cur_s))
                try:
                    df_nudge = _simulate_chunk_bisect(
                        float(t_cur_s + eps_s),
                        float(t_next_s),
                        int(npt_chunk),
                        depth=max_depth,
                        min_window_s=min_win_s,
                    )
                    df0 = pd.DataFrame([row0])
                    if len(df_nudge) > 0:
                        df_chunk = pd.concat([df0, df_nudge], ignore_index=True)
                    else:
                        df_chunk = df0
                except RuntimeError:
                    # Final fallback: try without nudge using recursive split.
                    df_chunk = _simulate_chunk_bisect(
                        float(t_cur_s),
                        float(t_next_s),
                        int(npt_chunk),
                        depth=max_depth,
                        min_window_s=min_win_s,
                    )
        else:
            df_chunk = pd.DataFrame(res, columns=res.colnames)
        if len(all_chunks) > 0 and len(df_chunk) > 0:
            df_chunk = df_chunk.iloc[1:].reset_index(drop=True)
        all_chunks.append(df_chunk)
        df_all = pd.concat(all_chunks, ignore_index=True) if len(all_chunks) > 1 else all_chunks[0].copy()

        if live_csv is not None:
            p_live = Path(live_csv)
            p_live.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(p_live, index=False)

        last_row = df_all.iloc[-1] if len(df_all) > 0 else None
        t_last_h = float(last_row["time"] / 3600.0) if last_row is not None else float(t_next_s / 3600.0)
        ck = {
            "chunk": int(i),
            "chunks_total": int(n_chunks),
            "time_h": float(t_last_h),
            "wall_elapsed_s": float(time.time() - wall_t0),
        }
        if last_row is not None:
            for g, sid in protein_species.items():
                col = f"[{sid}]"
                ck[f"{g}_uM"] = float(max(0.0, float(last_row[col])) * 1e6) if col in df_all.columns else 0.0
        rows_ckpt.append(ck)
        if checkpoint_csv is not None:
            p_ck = Path(checkpoint_csv)
            p_ck.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows_ckpt).to_csv(p_ck, index=False)

        _write_status_json(
            status_json,
            {
                "stage": "simulate",
                "label": str(progress_label),
                "message": f"chunk {i}/{n_chunks} done",
                "progress": {
                    "done_chunks": int(i),
                    "total_chunks": int(n_chunks),
                    "time_h": float(t_last_h),
                    "fraction": float(i / max(1, n_chunks)),
                },
                "chunk_time_window_h": [float(t_cur_s / 3600.0), float(t_next_s / 3600.0)],
                "wall_elapsed_s": float(time.time() - wall_t0),
                "live_csv": (None if live_csv is None else str(live_csv)),
                "checkpoint_csv": (None if checkpoint_csv is None else str(checkpoint_csv)),
            },
        )
        print(
            f"[progress:{progress_label}] chunk {i}/{n_chunks} | sim_t={t_last_h:.3f} h | wall={time.time() - wall_t0:.1f} s",
            flush=True,
        )
        t_cur_s = float(t_next_s)
        # Some model variants are numerically sensitive when the next chunk starts
        # exactly at integer-hour boundaries (e.g., 3600 s). Nudge forward by an
        # epsilon to avoid restarting CVODE on a pathological time knot.
        if avoid_hour_eps_s > 0.0 and t_cur_s < t_end_s:
            h_unit = float(t_cur_s / 3600.0)
            if abs(h_unit - round(h_unit)) <= 1e-14:
                t_cur_s = float(min(t_end_s, t_cur_s + avoid_hour_eps_s))

    return pd.concat(all_chunks, ignore_index=True) if len(all_chunks) > 1 else all_chunks[0]


def _protein_metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    t_h = df["time"].to_numpy(dtype=float) / 3600.0
    y_uM = np.maximum(df[col].to_numpy(dtype=float), 0.0) * 1e6
    auc = float(np.trapezoid(y_uM, t_h)) if (hasattr(np, "trapezoid") and t_h.size > 1) else (
        float(np.trapz(y_uM, t_h)) if t_h.size > 1 else 0.0
    )
    return {
        "protein_2h_uM": float(np.interp(2.0, t_h, y_uM)),
        "protein_4h_uM": float(np.interp(4.0, t_h, y_uM)),
        "protein_6h_uM": float(np.interp(6.0, t_h, y_uM)),
        "protein_8h_uM": float(np.interp(8.0, t_h, y_uM)),
        "protein_peak_uM": float(np.max(y_uM)) if y_uM.size else 0.0,
        "protein_auc_uM_h": auc,
    }


def _run_case(
    sbml_text: str,
    protein_by_gene: Dict[str, str],
    dna_by_gene: Dict[str, str],
    batch_params: Dict[str, float],
    t_end_h: float,
    points: int,
    init_overrides: Optional[Dict[str, float]] = None,
    progress_chunk_h: float = 0.0,
    progress_points_per_chunk: int = 0,
    progress_status_json: Optional[Path] = None,
    progress_live_csv: Optional[Path] = None,
    progress_checkpoint_csv: Optional[Path] = None,
    progress_label: str = "run",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float]]:
    _write_status_json(
        progress_status_json,
        {
            "stage": "simulate_prepare",
            "label": str(progress_label),
            "message": "loading SBML into roadrunner",
        },
    )
    rr = _load_rr_with_compile_heartbeat(
        sbml_text=sbml_text,
        status_json=progress_status_json,
        progress_label=str(progress_label),
    )
    route1._configure_rr(rr)
    return _run_case_with_rr(
        rr=rr,
        protein_by_gene=protein_by_gene,
        dna_by_gene=dna_by_gene,
        batch_params=batch_params,
        t_end_h=t_end_h,
        points=points,
        init_overrides=init_overrides,
        progress_chunk_h=progress_chunk_h,
        progress_points_per_chunk=progress_points_per_chunk,
        progress_status_json=progress_status_json,
        progress_live_csv=progress_live_csv,
        progress_checkpoint_csv=progress_checkpoint_csv,
        progress_label=progress_label,
    )


def _run_case_with_rr(
    rr,
    protein_by_gene: Dict[str, str],
    dna_by_gene: Dict[str, str],
    batch_params: Dict[str, float],
    t_end_h: float,
    points: int,
    init_overrides: Optional[Dict[str, float]] = None,
    progress_chunk_h: float = 0.0,
    progress_points_per_chunk: int = 0,
    progress_status_json: Optional[Path] = None,
    progress_live_csv: Optional[Path] = None,
    progress_checkpoint_csv: Optional[Path] = None,
    progress_label: str = "run",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, float]]:
    try:
        rr.resetToOrigin()
    except Exception:
        pass
    if init_overrides is not None:
        for sid, val in dict(init_overrides).items():
            try:
                rr[str(sid)] = float(val)
                continue
            except Exception:
                pass
            try:
                rr.setValue(str(sid), float(val))
            except Exception:
                pass
    _apply_params_if_present(rr, batch_params)
    _write_status_json(
        progress_status_json,
        {
            "stage": "simulate_prepare",
            "label": str(progress_label),
            "message": "roadrunner configured; starting integration",
            "t_end_h": float(t_end_h),
            "points": int(points),
        },
    )
    if float(progress_chunk_h) > 0.0:
        df = _simulate_progressive(
            rr=rr,
            protein_species=protein_by_gene,
            dna_species=dna_by_gene,
            t_end_h=float(t_end_h),
            points=int(points),
            chunk_h=float(progress_chunk_h),
            points_per_chunk=int(progress_points_per_chunk),
            status_json=progress_status_json,
            live_csv=progress_live_csv,
            checkpoint_csv=progress_checkpoint_csv,
            progress_label=str(progress_label),
        )
    else:
        df = _simulate(rr, protein_by_gene, dna_by_gene, t_end_h=t_end_h, points=points)

    metrics = {g: _protein_metrics(df, f"[{sid}]") for g, sid in protein_by_gene.items()}
    resources = {}
    for sid in ("ATP", "GTP", "PO4", "PPi", "Ribosome", "T7RNAP"):
        col = f"[{sid}]"
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            resources[f"{sid}_min_uM"] = float(np.min(y) * 1e6)
            resources[f"{sid}_max_uM"] = float(np.max(y) * 1e6)
    return df, metrics, resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-gene explicit transcription-chain model.")
    parser.add_argument("--summary-json", type=Path, default=route1.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gamma-h", type=float, default=1.0)
    parser.add_argument("--fit-points", type=int, default=100)
    parser.add_argument("--t-end-h", type=float, default=8.0)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument("--native-degfp-share", type=float, default=0.0, help="share on native r343 line")
    parser.add_argument("--dna-total-M", type=float, default=5.0e-9, help="total DNA allocated across active lines")
    parser.add_argument("--mrna-decay-base", type=float, default=DEFAULT_MRNA_DECAY_BASE, help="first-order decay rate (s^-1) for free mRNA_t_*")
    parser.add_argument(
        "--mrna-bound-decay-ratio",
        type=float,
        default=DEFAULT_MRNA_BOUND_DECAY_RATIO,
        help="relative decay factor for ribosome-bound mRNA complex",
    )
    parser.add_argument(
        "--tx-fast-scale",
        type=float,
        default=1.0,
        help="uniform scale on explicit transcription-chain fast steps (init/elong/release)",
    )
    parser.add_argument(
        "--ribo-bind-scale",
        type=float,
        default=1.0,
        help="uniform scale on ribosome binding/unbinding pair; lower values reduce stiffness while preserving Kd",
    )
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    genes = list(TWO_GENES)
    sbml_bg, counts_bg, hf_meta = route1._rebuild_background_batch_model(gamma_h=float(args.gamma_h), fit_points=int(args.fit_points))
    batch_params = _load_formal_batch_params(args.summary_json)
    bg_doc = libsbml.readSBMLFromString(sbml_bg)
    bg_model = bg_doc.getModel()
    if bg_model is None:
        raise RuntimeError("Invalid background SBML")
    tx_seq_meta = []
    aa_profile_meta = []
    for g in genes:
        _, meta = _build_gene_tx_rna_sequence(bg_model, g)
        tx_seq_meta.append(meta)
        _, aa_meta = _build_gene_aa20_profile(g)
        aa_profile_meta.append(aa_meta)

    sbml_ext, protein_by_gene, dna_by_gene, delta = _extend_with_explicit_two_lines(
        sbml_text=sbml_bg,
        genes=genes,
        native_degfp_share=float(args.native_degfp_share),
        mrna_decay_base=float(args.mrna_decay_base),
        mrna_bound_decay_ratio=float(args.mrna_bound_decay_ratio),
        tx_fast_scale=float(args.tx_fast_scale),
        ribo_bind_scale=float(args.ribo_bind_scale),
    )

    dna_total = float(args.dna_total_M)
    native_dna = dna_total * max(0.0, float(args.native_degfp_share))
    rem = max(0.0, dna_total - native_dna)
    each = rem / float(len(genes))

    # Multiplex
    dna_multiplex = {"DNA": native_dna}
    for g in genes:
        dna_multiplex[dna_by_gene[g.name]] = each
    sbml_multi = _set_dna_initials(sbml_ext, dna_multiplex)
    df_multi, m_multi, r_multi = _run_case(
        sbml_text=sbml_multi,
        protein_by_gene=protein_by_gene,
        dna_by_gene=dna_by_gene,
        batch_params=batch_params,
        t_end_h=float(args.t_end_h),
        points=int(args.points),
    )
    df_multi.to_csv(outdir / "twogene_explicit_tx_multiplex.csv", index=False)

    # Single controls
    single_metrics: Dict[str, Dict[str, float]] = {}
    for g in genes:
        dna_single = {"DNA": native_dna}
        for x in genes:
            dna_single[dna_by_gene[x.name]] = rem if x.name == g.name else 0.0
        sbml_single = _set_dna_initials(sbml_ext, dna_single)
        df_single, m_single, _ = _run_case(
            sbml_text=sbml_single,
            protein_by_gene=protein_by_gene,
            dna_by_gene=dna_by_gene,
            batch_params=batch_params,
            t_end_h=float(args.t_end_h),
            points=int(args.points),
        )
        df_single.to_csv(outdir / f"twogene_explicit_tx_single_{route1.wf.sanitize_id(g.name)}.csv", index=False)
        single_metrics[g.name] = m_single[g.name]

    rows = []
    for g in genes:
        multi_p6 = float(m_multi[g.name]["protein_6h_uM"])
        single_p6 = float(single_metrics[g.name]["protein_6h_uM"])
        ratio = multi_p6 / max(single_p6, 1e-12)
        rows.append(
            {
                "gene": g.name,
                "aa_length": g.aa_length,
                "multiplex_p6_uM": multi_p6,
                "single_p6_uM": single_p6,
                "multiplex_to_single_ratio": ratio,
                "suppression_fraction": 1.0 - ratio,
            }
        )
    df_ct = pd.DataFrame(rows).sort_values("suppression_fraction", ascending=False)
    df_ct.to_csv(outdir / "twogene_explicit_tx_crosstalk.csv", index=False)

    summary = {
        "workflow": "route1_twogene_explicit_tx_lines",
        "assumptions": {
            "genes": [
                {
                    "name": g.name,
                    "aa_length": int(g.aa_length),
                    "tl_scale": float(g.tl_scale),
                    "tx_scale": float(g.tx_scale),
                    "has_cds_dna": bool(g.cds_dna),
                    "cds_nt_len": int(len(_clean_dna(g.cds_dna or ""))) if g.cds_dna else 0,
                    "tx_nt_steps": int(g.tx_nt_steps) if g.tx_nt_steps is not None else None,
                    "aa_profile_source": str(next((m.get("aa_source") for m in aa_profile_meta if m.get("gene") == g.name), "")),
                }
                for g in genes
            ],
            "explicit_transcription_chain": "generated per gene from gene-specific CDS (optionally compressed in nt steps)",
            "transcription_sequence_meta": tx_seq_meta,
            "gene_tx_scale_effective_step": "all_tx_chain_steps_init_elong_release",
            "translation_aa_profile_meta": aa_profile_meta,
            "independent_dna_mrna_lines": True,
            "shared_translation_resources": ["Ribosome", "ATP", "GTP", "AA20"],
            "native_degfp_share": float(args.native_degfp_share),
            "dna_total_M": dna_total,
            "dna_multiplex_each_M": each,
            "mrna_decay_base_s-1": float(args.mrna_decay_base),
            "mrna_bound_decay_ratio": float(args.mrna_bound_decay_ratio),
            "tx_fast_scale": float(args.tx_fast_scale),
            "ribo_bind_scale": float(args.ribo_bind_scale),
        },
        "counts_background_batch": counts_bg,
        "counts_delta": delta,
        "batch_params_source": str(args.summary_json),
        "batch_params_used": batch_params,
        "hf_meta": hf_meta,
        "metrics": {
            "multiplex_protein_metrics": m_multi,
            "single_protein_metrics": single_metrics,
            "multiplex_resource_metrics": r_multi,
        },
        "files": {
            "multiplex_csv": str(outdir / "twogene_explicit_tx_multiplex.csv"),
            "crosstalk_csv": str(outdir / "twogene_explicit_tx_crosstalk.csv"),
        },
    }
    (outdir / "twogene_explicit_tx_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("Saved:")
    print(outdir / "twogene_explicit_tx_summary.json")
    print(outdir / "twogene_explicit_tx_multiplex.csv")
    print(outdir / "twogene_explicit_tx_crosstalk.csv")
    for g in genes:
        print(outdir / f"twogene_explicit_tx_single_{route1.wf.sanitize_id(g.name)}.csv")


if __name__ == "__main__":
    main()
