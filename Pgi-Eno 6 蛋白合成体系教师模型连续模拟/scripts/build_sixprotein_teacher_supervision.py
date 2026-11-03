#!/usr/bin/env python3
"""Map the reduced six-protein teacher trajectory to full-CRN constraints.

Every independent teacher metabolite/state is retained.  Directly comparable
pools and gene states are mapped one-to-one.  Reduced
transcription and translation occupancies are represented as sums over the
corresponding full-elementary microstate families.  This keeps supervision at
the resolution actually available from the teacher instead of assigning an
aggregate concentration to an arbitrary hidden elementary state.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "outputs" / "sixprotein_full_elementary"
DEFAULT_TEACHER = (
    ROOT
    / "outputs"
    / "sixprotein_teacher_detailed_6h_converged"
    / "sixprotein_continuous_timeseries.csv"
)
DEFAULT_OUTPUT = ASSETS / "sixprotein_teacher_supervision_positional_6h_converged.npz"
GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")
HYBRID_EXTRA_STATES = ("Glucose", "NAD", "NADH", "TPP", "Mg", "AA")


def _require(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Teacher trajectory is missing required column: {column}")
    return frame[column].to_numpy(dtype=float)


def _position_columns(frame: pd.DataFrame, gene: str, chain: str) -> list[tuple[int, str]]:
    pattern = re.compile(rf"^teacher_gene::{re.escape(gene)}::{chain}_pos::(\d+)_M$")
    positions: list[tuple[int, str]] = []
    for column in frame.columns:
        match = pattern.match(column)
        if match is not None:
            positions.append((int(match.group(1)), column))
    positions.sort()
    expected = list(range(len(positions)))
    observed = [position for position, _column in positions]
    if observed != expected:
        raise ValueError(f"Teacher {gene} {chain} positions are not contiguous: {observed[:8]}...")
    return positions


def _transcription_position(species_id: str, gene: str) -> int | None:
    if not species_id.endswith(f"__{gene}") or not species_id.startswith("T7RNAP_bound_mRNA"):
        return None
    match = re.match(r"^T7RNAP_bound_mRNA(\d{4})(?:_|__)", species_id)
    return int(match.group(1)) if match is not None else None


def _translation_stage(species_id: str, gene: str) -> tuple[str, int] | None:
    if not species_id.endswith(f"__{gene}"):
        return None
    elongation = re.match(r"^elRS70S[A-C][A-Z]{3}(\d{4})", species_id)
    if elongation is not None:
        return "elongation", int(elongation.group(1))
    termination = re.match(r"^termRS70S[A-Z]{3}(\d{4})", species_id)
    if termination is not None:
        return "termination", int(termination.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    teacher = pd.read_csv(args.teacher_csv)
    species_frame = pd.read_csv(args.assets / "species_index.csv").sort_values("species_index")
    species_ids = species_frame["species_id"].astype(str).tolist()
    species_ids.extend(sid for sid in HYBRID_EXTRA_STATES if sid not in species_ids)
    lookup = {sid: index for index, sid in enumerate(species_ids)}
    time_s = _require(teacher, "time_s")

    direct_names: list[str] = []
    direct_indices: list[int] = []
    direct_targets: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []

    def add_direct(
        name: str,
        species_id: str,
        values_M: np.ndarray,
        source: str,
        **audit: object,
    ) -> None:
        if species_id not in lookup:
            raise ValueError(
                f"Teacher constraint {name} cannot be mapped to full/hybrid state {species_id}"
            )
        direct_names.append(name)
        direct_indices.append(lookup[species_id])
        direct_targets.append(np.asarray(values_M, dtype=float))
        row: dict[str, object] = {
                "teacher_constraint": name,
                "mapping_type": "direct",
                "full_species_members": 1,
                "full_species_ids": species_id,
                "teacher_source": source,
            }
        row.update(audit)
        audit_rows.append(row)

    for column in teacher.columns:
        prefix = "teacher_pool::"
        if not column.startswith(prefix) or not column.endswith("_M"):
            continue
        sid = column[len(prefix) : -2]
        if sid == "Ribosome":
            add_direct(
                "pool::Ribosome",
                "RS70S",
                teacher[column].to_numpy(float),
                column,
                biological_stage="free_intact_ribosome_pool",
                alignment_rule="teacher_free_Ribosome=full_free_RS70S",
            )
        else:
            add_direct(f"pool::{sid}", sid, teacher[column].to_numpy(float), column)

    for gene in GENES:
        prefix = f"teacher_gene::{gene}"
        add_direct(
            f"gene::{gene}::dna_free",
            f"DNA__{gene}",
            _require(teacher, f"{prefix}::dna_free_M"),
            f"{prefix}::dna_free_M",
        )
        add_direct(
            f"gene::{gene}::mrna_product",
            f"mRNA_t__{gene}",
            _require(teacher, f"{prefix}::mrna_M"),
            f"{prefix}::mrna_M",
        )
        terminal_peptides = [
            sid
            for sid in species_ids
            if sid.startswith("Pept") and sid.endswith(f"__{gene}") and sid[4:8].isdigit()
        ]
        if not terminal_peptides:
            raise ValueError(f"No released peptide state found for {gene}")
        terminal_peptide = max(terminal_peptides, key=lambda sid: int(sid[4:8]))
        add_direct(
            f"gene::{gene}::released_peptide",
            terminal_peptide,
            _require(teacher, f"{prefix}::pept_M"),
            f"{prefix}::pept_M",
        )
        add_direct(
            f"gene::{gene}::protein",
            f"deGFP_m__{gene}",
            _require(teacher, f"{prefix}::protein_M"),
            f"{prefix}::protein_M",
        )

    construct = json.loads((args.assets / "reference_construct_split.json").read_text(encoding="utf-8"))
    reference_prefix_nt = int(construct["reference_prefix_nt"])

    group_names: list[str] = []
    group_member_lists: list[list[int]] = []
    group_targets: list[np.ndarray] = []

    def add_group(
        name: str,
        members: list[str],
        values_M: np.ndarray,
        source: str,
        **audit: object,
    ) -> None:
        indices = sorted({lookup[sid] for sid in members if sid in lookup})
        if not indices:
            raise ValueError(f"Aggregate teacher constraint {name} has no full-CRN members")
        group_names.append(name)
        group_member_lists.append(indices)
        group_targets.append(np.asarray(values_M, dtype=float))
        row: dict[str, object] = {
                "teacher_constraint": name,
                "mapping_type": "sum",
                "full_species_members": len(indices),
                "full_species_ids": ";".join(species_ids[index] for index in indices),
                "teacher_source": source,
            }
        row.update(audit)
        audit_rows.append(row)

    for gene in GENES:
        suffix = f"__{gene}"
        tx_by_position: dict[int, list[str]] = {}
        tl_by_stage: dict[tuple[str, int], list[str]] = {}
        for sid in species_ids:
            tx_position = _transcription_position(sid, gene)
            if tx_position is not None:
                tx_by_position.setdefault(tx_position, []).append(sid)
            tl_stage = _translation_stage(sid, gene)
            if tl_stage is not None:
                tl_by_stage.setdefault(tl_stage, []).append(sid)

        initiation_members = [
            sid
            for sid in species_ids
            if sid.endswith(suffix)
            and "mRNA" in sid
            and (sid.startswith("RS30S") or sid.startswith("RS70S"))
        ]
        teacher_tx_positions = _position_columns(teacher, gene, "tx")
        for teacher_position, column in teacher_tx_positions:
            full_position = reference_prefix_nt + teacher_position
            members = tx_by_position.get(full_position, [])
            if not members:
                raise ValueError(
                    f"No full transcription microstates for {gene} teacher position "
                    f"{teacher_position} -> full position {full_position}"
                )
            add_group(
                f"gene::{gene}::tx_pos::{teacher_position:04d}",
                members,
                _require(teacher, column),
                column,
                biological_stage="transcription_CDS_position",
                teacher_position=teacher_position,
                full_position=full_position,
                alignment_rule=f"full_position={reference_prefix_nt}+teacher_position",
            )

        teacher_tl_positions = _position_columns(teacher, gene, "tl")
        amino_acid_length = len(teacher_tl_positions) - 1
        stop_codon_position = amino_acid_length + 1
        for teacher_position, column in teacher_tl_positions:
            if teacher_position < amino_acid_length:
                stage = "elongation"
                full_position = teacher_position + 2
            else:
                stage = "termination"
                full_position = stop_codon_position
            members = tl_by_stage.get((stage, full_position), [])
            if not members:
                raise ValueError(
                    f"No full translation microstates for {gene} teacher position "
                    f"{teacher_position} -> {stage} position {full_position}"
                )
            add_group(
                f"gene::{gene}::tl_pos::{teacher_position:04d}",
                members,
                _require(teacher, column),
                column,
                biological_stage=stage,
                teacher_position=teacher_position,
                full_position=full_position,
                alignment_rule=(
                    "full_elongation_position=teacher_position+2"
                    if stage == "elongation"
                    else "last_teacher_position=full_termination_microstates"
                ),
            )
        add_group(
            f"gene::{gene}::ribosome_mrna_assembly",
            initiation_members,
            _require(teacher, f"teacher_gene::{gene}::ribo_mrna_M"),
            f"teacher_gene::{gene}::ribo_mrna_M",
            biological_stage="translation_initiation_assembly",
            teacher_position="",
            full_position="",
            alignment_rule="sum_of_gene_specific_RS30S_RS70S_mRNA_assembly_states",
        )

    direct_target = np.stack(direct_targets, axis=1)
    group_target = np.stack(group_targets, axis=1)
    offsets = [0]
    flattened: list[int] = []
    for members in group_member_lists:
        flattened.extend(members)
        offsets.append(len(flattened))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time_s=time_s,
        direct_names=np.asarray(direct_names, dtype=str),
        direct_indices=np.asarray(direct_indices, dtype=np.int64),
        direct_target_M=direct_target,
        group_names=np.asarray(group_names, dtype=str),
        group_member_indices=np.asarray(flattened, dtype=np.int64),
        group_member_offsets=np.asarray(offsets, dtype=np.int64),
        group_target_M=group_target,
    )
    audit_path = args.output.with_name(args.output.stem + "_mapping.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    membership_counts: dict[int, int] = {}
    for members in group_member_lists:
        for index in members:
            membership_counts[index] = membership_counts.get(index, 0) + 1
    duplicate_group_members = [
        species_ids[index] for index, count in membership_counts.items() if count > 1
    ]
    if duplicate_group_members:
        raise ValueError(
            "Full species assigned to multiple aggregate constraints: "
            + ", ".join(duplicate_group_members[:10])
        )

    expected_teacher_sources: list[str] = [
        column
        for column in teacher.columns
        if column.startswith("teacher_pool::") and column.endswith("_M")
    ]
    for gene in GENES:
        prefix = f"teacher_gene::{gene}"
        expected_teacher_sources.extend(
            (
                f"{prefix}::dna_free_M",
                f"{prefix}::mrna_M",
                f"{prefix}::ribo_mrna_M",
                f"{prefix}::pept_M",
                f"{prefix}::protein_M",
            )
        )
        expected_teacher_sources.extend(column for _position, column in _position_columns(teacher, gene, "tx"))
        expected_teacher_sources.extend(column for _position, column in _position_columns(teacher, gene, "tl"))
    mapped_teacher_sources = [str(row["teacher_source"]) for row in audit_rows]
    missing_teacher_sources = sorted(set(expected_teacher_sources) - set(mapped_teacher_sources))
    duplicate_teacher_sources = sorted(
        source for source in set(mapped_teacher_sources) if mapped_teacher_sources.count(source) > 1
    )
    unexpected_teacher_sources = sorted(set(mapped_teacher_sources) - set(expected_teacher_sources))
    if missing_teacher_sources or duplicate_teacher_sources or unexpected_teacher_sources:
        raise ValueError(
            "Teacher-state mapping is not one-to-one and complete: "
            f"missing={missing_teacher_sources[:10]}, "
            f"duplicates={duplicate_teacher_sources[:10]}, "
            f"unexpected={unexpected_teacher_sources[:10]}"
        )

    mapped_tx = sum(name.count("::tx_pos::") for name in group_names)
    mapped_tl = sum(name.count("::tl_pos::") for name in group_names)
    expected_tx = sum(len(_position_columns(teacher, gene, "tx")) for gene in GENES)
    expected_tl = sum(len(_position_columns(teacher, gene, "tl")) for gene in GENES)
    summary = {
        "teacher_csv": str(args.teacher_csv),
        "time_points": len(time_s),
        "time_end_h": float(time_s[-1] / 3600.0),
        "direct_constraints": len(direct_names),
        "aggregate_constraints": len(group_names),
        "total_teacher_comparable_states": len(direct_names) + len(group_names),
        "expected_independent_teacher_states": len(expected_teacher_sources),
        "all_independent_teacher_states_mapped": True,
        "missing_independent_teacher_states": [],
        "aggregate_full_state_memberships": len(flattened),
        "mapped_teacher_tx_positions": mapped_tx,
        "expected_teacher_tx_positions": expected_tx,
        "mapped_teacher_tl_positions": mapped_tl,
        "expected_teacher_tl_positions": expected_tl,
        "reference_prefix_nt": reference_prefix_nt,
        "duplicate_aggregate_memberships": 0,
        "mapping_rules": {
            "transcription": (
                "Each CDS-only teacher tx_pos[i] supervises the sum of full-elementary "
                f"T7RNAP microstates at construct position {reference_prefix_nt}+i."
            ),
            "translation": (
                "Teacher tl_pos[0..L-1] supervise consecutive full elongation stages "
                "2..L+1; teacher tl_pos[L] supervises termination/recycling microstates."
            ),
            "initiation": (
                "Teacher ribosome-mRNA occupancy supervises the sum of gene-specific "
                "RS30S/RS70S initiation-assembly microstates containing mRNA."
            ),
            "ribosome_pool": (
                "The teacher free intact Ribosome pool maps to the full-elementary "
                "free RS70S state."
            ),
            "legacy_AA_pool": (
                "The teacher AA bookkeeping pool is retained as an auxiliary directly "
                "comparable state; the 20 dynamic amino-acid pools are also mapped separately."
            ),
        },
        "teacher_hidden_microstates_fabricated": False,
        "output": str(args.output),
        "mapping_audit": str(audit_path),
    }
    args.output.with_name(args.output.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
