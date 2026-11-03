#!/usr/bin/env python3
"""Prepare literature-audited inputs for the six-protein batch validation.

Figure 5 and its SDS-PAGE support an msfGFP-enzyme product.  Methods names the
source construct pSUMO-msfGFP, so a second SUMO-retained panel is generated as
a sensitivity case.  Exact in-house tag DNA was not supplied; tag amino-acid
identity is preserved and deterministic E. coli-preferred codons are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import simulate_route1_twogene_explicit_tx_lines as etx


GENES = ["Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno"]

# Saccharomyces cerevisiae Smt3/SUMO tag ending at the Ulp1 cleavage motif.
SUMO_AA = (
    "MSDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRLMEAFAKRQGKEMDSLRFLYDGIRIQADQTPEDLDMEDNDIIEAHREQIGG"
)

# Monomeric superfolder GFP. Residue 206 is K (the V206K monomerizing change).
MSFGFP_AA = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTFGYGLMCFARYPDHMKQHDFFKSAMPEGYVQERTI"
    "FFKDDGNYKTRAEVKFEGDTLVNRIELKGTDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLP"
    "DNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)

# One deterministic preferred codon per amino acid.  This is a documented
# reconstruction for resource accounting, not a claim about the undisclosed
# nucleotide sequence of the authors' in-house pSUMO-msfGFP plasmid.
PREFERRED_CODON = {
    "A": "GCG", "R": "CGT", "N": "AAC", "D": "GAT", "C": "TGC",
    "Q": "CAG", "E": "GAA", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}


def reverse_translate(aa: str) -> str:
    return "".join(PREFERRED_CODON[x] for x in aa)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-panel",
        type=Path,
        default=Path(
            "output_sixgene_single_gene_txfit_then_all_bgtx2p25_muxTotal2p8_20260425/"
            "sixgene_panel_single_gene_txfit_absolutechains.json"
        ),
    )
    p.add_argument(
        "--proxy-csv",
        type=Path,
        default=Path(
            "output_sixgene_single_gene_proxy_paperanchored_20260424/"
            "selected_sixgene_single_gene_vs_lit_paperanchored.csv"
        ),
    )
    p.add_argument("--outdir", type=Path, default=Path("inputs/fusion_batch_validation"))
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_panel.read_text(encoding="utf-8"))
    base_by_name = {str(g["name"]): dict(g) for g in base["genes"]}

    def build_panel(tag_label: str, tag_aa: str, interpretation: str) -> tuple[dict, list[dict]]:
        tag_dna = reverse_translate(tag_aa)
        genes = []
        rows = []
        for name in GENES:
            native_cds = etx.get_known_cds_dna(name)
            if not native_cds:
                raise RuntimeError(f"Missing native CDS for {name}")
            native_cds = etx._clean_dna(native_cds)
            native_counts, native_meta = etx._build_gene_aa20_profile(
                etx.GeneConfig(name=name, aa_length=1, cds_dna=native_cds)
            )
            del native_counts
            native_aa = int(native_meta["aa_total"])
            fusion_cds = tag_dna + native_cds
            fusion_aa = len(tag_aa) + native_aa

            g = dict(base_by_name[name])
            g["construct"] = f"{tag_label}-{name}"
            g["aa_length"] = fusion_aa
            g["cds_dna"] = fusion_cds
            g["tx_nt_steps"] = len(fusion_cds)
            g["native_enzyme_aa"] = native_aa
            g["sumo_aa"] = len(SUMO_AA) if tag_label.startswith("SUMO-") else 0
            g["msfgfp_aa"] = len(MSFGFP_AA)
            g["tag_sequence_status"] = "amino_acid identity preserved; tag DNA reverse-translated with preferred E. coli codons"
            genes.append(g)
            rows.append(
                {
                    "gene": name,
                    "construct": g["construct"],
                    "native_enzyme_aa": native_aa,
                    "SUMO_aa": g["sumo_aa"],
                    "msfGFP_aa": len(MSFGFP_AA),
                    "fusion_total_aa": fusion_aa,
                    "fusion_cds_nt_including_stop": len(fusion_cds),
                    "interpretation": interpretation,
                    "native_CDS_source": "E. coli MG1655 NC_000913.3",
                    "tag_nucleotide_source": "deterministic reverse translation; exact in-house plasmid sequence unavailable",
                }
            )
        payload = {
            "dna_total_M": 2.8e-9,
            "dna_concentration_note": (
                "Single-gene reactions use 2.8 nM. For All G.E., the paper does not state unambiguously whether "
                "2.8 nM is total or per template; both hypotheses must be tested."
            ),
            "construct_source": {
                "paper": "Sato et al., Molecules 2024, 29, 2956, Figure 5 and Methods 4.1/4.5",
                "modeled_construct": tag_label + "-glycolytic enzyme",
                "interpretation": interpretation,
                "SUMO_aa_length": len(SUMO_AA) if tag_label.startswith("SUMO-") else 0,
                "msfGFP_aa_length": len(MSFGFP_AA),
                "exact_tag_DNA_available": False,
                "reconstruction": "published amino-acid tag identity plus deterministic E. coli-preferred codons",
            },
            "genes": genes,
        }
        return payload, rows

    primary_payload, primary_rows = build_panel(
        "msfGFP",
        MSFGFP_AA,
        "primary: Figure 5 caption and apparent full-length SDS-PAGE molecular weights",
    )
    sensitivity_payload, sensitivity_rows = build_panel(
        "SUMO-msfGFP",
        SUMO_AA + MSFGFP_AA,
        "sensitivity: literal retention of SUMO from the Methods construct name",
    )
    panel_path = args.outdir / "sixgene_panel_msfgfp_fusions.json"
    panel_path.write_text(json.dumps(primary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.outdir / "sixgene_panel_sumo_msfgfp_sensitivity.json").write_text(
        json.dumps(sensitivity_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(primary_rows + sensitivity_rows).to_csv(args.outdir / "fusion_construct_summary.csv", index=False)

    targets = pd.read_csv(args.proxy_csv)
    targets["direct_observation"] = "Figure 5A normalized total GFP fluorescence at 3 h and 6 h"
    targets["absolute_conversion"] = (
        "relative fluorescence multiplied by 36.537753 uM; scale inferred from 16.0 uM SG-PURE sfGFP "
        "and the digitized sfGFP relative signal"
    )
    targets["measurement_scope"] = targets["gene"].map(lambda x: f"one fluorescent {x} fusion product per completed chain")
    targets["all_GE_scope"] = "aggregate GFP fluorescence of six templates; no protein-resolved coexpression trajectory"
    targets.to_csv(args.outdir / "experimental_targets_with_provenance.csv", index=False)

    audit = f"""# Six-protein batch-validation source audit

## Directly supported by the paper

- Figure 5A reports normalized total GFP fluorescence for six single-gene reactions and the mixed `All G.E.` reaction.
- Results and Figure 5 identify all six measured products as N-terminal msfGFP-tagged glycolytic enzymes.
- Methods 4.1 calls the source templates `SUMO-msfGFP` fusions, but the apparent molecular weights of the Figure S5 full-length bands are consistent with msfGFP-enzyme products rather than an additional retained SUMO tag.
- The reaction template concentration is reported as 2.8 nM, and the reaction time course includes 3 h and 6 h.
- SG-PURE 2.0 produces 16.0 uM sfGFP at 6 h under the optimized condition.

## Reconstruction used by the model

- Native enzyme CDS: MG1655 `NC_000913.3`, unchanged.
- Primary panel: msfGFP tag, {len(MSFGFP_AA)} aa, followed by the native enzyme.
- Sensitivity panel: SUMO ({len(SUMO_AA)} aa) + msfGFP ({len(MSFGFP_AA)} aa) + native enzyme.
- The exact nucleotide sequence of the authors' in-house construct is not supplied. Tag DNA is therefore reverse-translated deterministically with E. coli-preferred codons. This preserves protein length and amino-acid demand, while nucleotide composition remains an explicit reconstruction assumption.
- No unreported linker is inserted.

## Quantitative limitations that must stay visible

- Figure 5 provides normalized fluorescence rather than a raw molar concentration table. The uM targets are paper-anchored conversions, not directly tabulated measurements.
- `All G.E.` is an aggregate fluorescence signal. It cannot validate six protein-resolved concentrations.
- The wording does not unambiguously state whether mixed-template reactions contain 2.8 nM total DNA or 2.8 nM of each template. Both `{2.8:.1f} nM total` and `{2.8:.1f} nM per template` are evaluated; the selected interpretation must be reported as an assumption.
- The figure shows means and standard deviations, but the source supplies no machine-readable raw replicate table. Numerical digitization uncertainty is therefore not zero.
"""
    (args.outdir / "source_and_assumption_audit.md").write_text(audit, encoding="utf-8")
    print(panel_path)


if __name__ == "__main__":
    main()
