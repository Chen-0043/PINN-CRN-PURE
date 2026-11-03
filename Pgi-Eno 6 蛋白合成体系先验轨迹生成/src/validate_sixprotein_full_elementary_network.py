#!/usr/bin/env python3
"""Audit a merged six-protein full-elementary mass-action operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")


def _mapping(value: str) -> dict[str, float]:
    parsed = json.loads(value)
    return {str(key): float(coefficient) for key, coefficient in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_dir", type=Path)
    parser.add_argument("--expected-dna-each-nM", type=float, default=2.8)
    args = parser.parse_args()

    root = args.network_dir.resolve()
    species = pd.read_csv(root / "species_index.csv")
    reactions = pd.read_csv(root / "reaction_mass_action_inventory.csv")
    stoich = sparse.load_npz(root / "stoichiometric_matrix_S.npz").tocsc()
    reactant_matrix = sparse.load_npz(root / "reactant_exponent_matrix_A.npz").tocsc()
    rates = np.load(root / "rate_constants.npy")
    initial = np.load(root / "initial_concentrations_M.npy")
    summary = json.loads((root / "sixprotein_full_elementary_summary.json").read_text(encoding="utf-8"))

    issues: list[dict[str, object]] = []

    def check(condition: bool, code: str, detail: str) -> None:
        if not condition:
            issues.append({"code": code, "detail": detail})

    n_species = len(species)
    n_reactions = len(reactions)
    check(stoich.shape == (n_species, n_reactions), "S_SHAPE", f"S={stoich.shape}")
    check(reactant_matrix.shape == (n_species, n_reactions), "A_SHAPE", f"A={reactant_matrix.shape}")
    check(len(rates) == n_reactions, "K_LENGTH", f"rates={len(rates)}")
    check(len(initial) == n_species, "X0_LENGTH", f"initial={len(initial)}")
    check(species["species_index"].tolist() == list(range(n_species)), "SPECIES_INDEX", "not contiguous")
    check(reactions["reaction_index"].tolist() == list(range(n_reactions)), "REACTION_INDEX", "not contiguous")
    check(species["species_id"].is_unique, "SPECIES_DUPLICATE", "species IDs are not unique")
    check(reactions["reaction_id"].is_unique, "REACTION_DUPLICATE", "reaction IDs are not unique")
    check(np.all(np.isfinite(initial)) and np.all(initial >= 0.0), "INITIAL_VALUES", "non-finite or negative")
    check(np.all(np.isfinite(rates)) and np.all(rates > 0.0), "RATE_VALUES", "non-finite or non-positive")
    check(int(summary["species"]) == n_species, "SUMMARY_SPECIES", str(summary.get("species")))
    check(int(summary["reactions"]) == n_reactions, "SUMMARY_REACTIONS", str(summary.get("reactions")))

    species_ids = species["species_id"].astype(str).tolist()
    species_index = {sid: index for index, sid in enumerate(species_ids)}
    expected_dna = float(args.expected_dna_each_nM) * 1e-9
    modeled_genes = tuple(str(gene) for gene in summary.get("genes", GENES))
    for gene in modeled_genes:
        dna_id = f"DNA__{gene}"
        product_id = f"deGFP_m__{gene}"
        check(dna_id in species_index, "DNA_MISSING", dna_id)
        check(product_id in species_index, "PRODUCT_MISSING", product_id)
        if dna_id in species_index:
            value = float(initial[species_index[dna_id]])
            check(np.isclose(value, expected_dna, rtol=0.0, atol=1e-18), "DNA_INITIAL", f"{dna_id}={value}")

    reconstructed_s_rows: list[int] = []
    reconstructed_s_cols: list[int] = []
    reconstructed_s_data: list[float] = []
    reconstructed_a_rows: list[int] = []
    reconstructed_a_cols: list[int] = []
    reconstructed_a_data: list[float] = []
    shared_signatures: set[tuple[object, ...]] = set()

    for column, row in reactions.iterrows():
        reactants = _mapping(row["reactants_json"])
        products = _mapping(row["products_json"])
        scope = str(row["scope"])
        participating = set(reactants) | set(products)
        missing = sorted(participating - species_index.keys())
        check(not missing, "REACTION_UNKNOWN_SPECIES", f"{row['reaction_id']}: {missing[:5]}")

        tagged_genes = {
            sid.rsplit("__", 1)[1]
            for sid in participating
            if "__" in sid and sid.rsplit("__", 1)[1] in modeled_genes
        }
        if scope == "shared":
            check(not tagged_genes, "SHARED_HAS_GENE_SPECIES", str(row["reaction_id"]))
            signature = (
                tuple(sorted(reactants.items())),
                tuple(sorted(products.items())),
                round(float(row["rate_constant"]), 15),
            )
            check(signature not in shared_signatures, "SHARED_DUPLICATE", str(row["reaction_id"]))
            shared_signatures.add(signature)
        else:
            check(scope in modeled_genes, "UNKNOWN_SCOPE", scope)
            check(tagged_genes == {scope}, "CROSS_GENE_REACTION", f"{row['reaction_id']}: {sorted(tagged_genes)}")

        net: dict[str, float] = {}
        for sid, coefficient in reactants.items():
            if sid not in species_index:
                continue
            reconstructed_a_rows.append(species_index[sid])
            reconstructed_a_cols.append(column)
            reconstructed_a_data.append(coefficient)
            net[sid] = net.get(sid, 0.0) - coefficient
        for sid, coefficient in products.items():
            if sid not in species_index:
                continue
            net[sid] = net.get(sid, 0.0) + coefficient
        for sid, coefficient in net.items():
            if coefficient != 0.0:
                reconstructed_s_rows.append(species_index[sid])
                reconstructed_s_cols.append(column)
                reconstructed_s_data.append(coefficient)

    rebuilt_s = sparse.coo_matrix(
        (reconstructed_s_data, (reconstructed_s_rows, reconstructed_s_cols)),
        shape=stoich.shape,
    ).tocsc()
    rebuilt_a = sparse.coo_matrix(
        (reconstructed_a_data, (reconstructed_a_rows, reconstructed_a_cols)),
        shape=reactant_matrix.shape,
    ).tocsc()
    s_difference = stoich - rebuilt_s
    a_difference = reactant_matrix - rebuilt_a
    check(s_difference.nnz == 0, "S_CONTENT", f"different entries={s_difference.nnz}")
    check(a_difference.nnz == 0, "A_CONTENT", f"different entries={a_difference.nnz}")
    check(
        np.allclose(rates, reactions["rate_constant"].to_numpy(dtype=float), rtol=1e-14, atol=0.0),
        "K_CONTENT",
        "array differs from inventory",
    )
    check(
        np.allclose(initial, species["initial_concentration_M"].to_numpy(dtype=float), rtol=1e-14, atol=1e-30),
        "X0_CONTENT",
        "array differs from inventory",
    )

    scope_counts = reactions["scope"].value_counts().to_dict()
    report = {
        "network_dir": str(root),
        "passed": not issues,
        "species": n_species,
        "reactions": n_reactions,
        "shared_reactions": int(scope_counts.get("shared", 0)),
        "gene_specific_reactions": {gene: int(scope_counts.get(gene, 0)) for gene in modeled_genes},
        "stoichiometric_matrix_nonzeros": int(stoich.nnz),
        "reactant_matrix_nonzeros": int(reactant_matrix.nnz),
        "issues": issues,
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(issues, columns=["code", "detail"]).to_csv(root / "validation_issues.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
