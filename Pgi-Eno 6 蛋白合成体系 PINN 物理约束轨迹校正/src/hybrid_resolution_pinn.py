#!/usr/bin/env python3
"""Hybrid-resolution PINN operator for the six-protein SG-PURE model.

The transcription/translation block uses the archived 49,143-state,
57,368-reaction mass-action CRN.  The missing SG-PURE glucose-driven ATP
regeneration step is evaluated at every PINN collocation point with the same
effective equation used by the validated teacher workflow.  Shared nucleotide
pools are represented exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.func import jvp


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ROOT / "outputs" / "sixprotein_full_elementary"
DEFAULT_POOL = ROOT / "input_pool_override_sg_pure2_sixprotein_aa0p5mM.json"
EXTRA_TEACHER_STATES = ("Glucose", "NAD", "NADH", "TPP", "Mg")
# ``AA`` is a legacy total-pool state retained by the six-protein teacher.  It
# stays constant while the 20 explicit amino-acid pools are consumed, so it is
# not mathematically equivalent to their sum.  Include it only for runs that
# require every teacher metabolite to be directly comparable.
OPTIONAL_COMPARABLE_TEACHER_STATES = ("AA",)
AA20 = (
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
)


def scipy_to_torch(matrix: sparse.spmatrix, dtype: torch.dtype) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, dtype=dtype, check_invariants=False
    ).coalesce()


def _sat(value: torch.Tensor, half_saturation: float) -> torch.Tensor:
    value = torch.clamp(value, min=0.0)
    return value / (value + float(half_saturation) + 1e-30)


class HybridResolutionOperator(nn.Module):
    """Full-elementary TX/TL plus effective teacher energy dynamics."""

    def __init__(
        self,
        assets: Path = DEFAULT_ASSETS,
        pool_json: Path = DEFAULT_POOL,
        *,
        energy_scale: float = 0.8,
        gamma_h: float = 0.5,
        include_all_teacher_metabolites: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        assets = Path(assets)
        stoich = sparse.load_npz(assets / "stoichiometric_matrix_S.npz")
        reactants = sparse.load_npz(assets / "reactant_exponent_matrix_A.npz")
        rate_constants = np.load(assets / "rate_constants_k.npy")
        species = pd.read_csv(assets / "species_index.csv")["species_id"].astype(str).tolist()
        reaction_inventory = pd.read_csv(assets / "reaction_mass_action_inventory.csv")
        initial_base = np.load(assets / "initial_concentrations_x0_M.npy")

        if len(species) != stoich.shape[0] or len(initial_base) != len(species):
            raise ValueError("Full-elementary assets have inconsistent species dimensions")
        if len(species) != len(set(species)):
            raise ValueError("Full-elementary species IDs are not unique")

        self.base_species = tuple(species)
        self.base_species_count = len(species)
        requested_extras = list(EXTRA_TEACHER_STATES)
        if include_all_teacher_metabolites:
            requested_extras.extend(OPTIONAL_COMPARABLE_TEACHER_STATES)
        appended = [sid for sid in requested_extras if sid not in species]
        self.species_ids = tuple(species + appended)
        self.species_index = {sid: i for i, sid in enumerate(self.species_ids)}
        self.n_species = len(self.species_ids)
        self.n_elementary_reactions = stoich.shape[1]
        self.n_teacher_reactions = 1
        ndk_reaction_mask = reaction_inventory[["reactants_json", "products_json"]].astype(str).apply(
            lambda column: column.str.contains(r'"NDK', regex=True)
        ).any(axis=1)
        self.elementary_ndk_reaction_count = int(ndk_reaction_mask.sum())
        if self.elementary_ndk_reaction_count == 0:
            raise ValueError("Base full-elementary CRN has no NDK mechanism; teacher NDK cannot be omitted")

        required_shared = ("ATP", "ADP", "GTP", "GDP")
        missing_shared = [sid for sid in required_shared if sid not in self.species_index]
        if missing_shared:
            raise ValueError(f"Missing shared full-CRN states: {missing_shared}")

        pool_payload = json.loads(Path(pool_json).read_text(encoding="utf-8"))
        pool = {str(k): float(v) for k, v in pool_payload["pool_overrides_M"].items()}
        initial = np.zeros(self.n_species, dtype=float)
        initial[: self.base_species_count] = initial_base
        for sid in appended:
            initial[self.species_index[sid]] = float(pool.get(sid, 0.0))
        # The hybrid model must start from the batch-validated six-protein pool,
        # including its explicit 0.5 mM-per-amino-acid reconstruction.
        for sid, value in pool.items():
            if sid in self.species_index:
                initial[self.species_index[sid]] = float(value)
        if "AA" in self.species_index and "AA" not in pool:
            # The teacher initializes its legacy AA bookkeeping pool from the
            # sum of the 20 explicit amino-acid pools, then leaves it constant.
            initial[self.species_index["AA"]] = float(
                sum(initial[self.species_index[sid]] for sid in AA20)
            )
        # SG-PURE replaces the reference PURE phosphocreatine energy source
        # with the glucose pathway.  Keep the CK mechanism in the elementary
        # inventory, but disable its absent substrate instead of deleting CRN
        # reactions.  Elementary MK and NDK remain active.
        self.reference_cp_original_initial_M = (
            float(initial[self.species_index["CP"]]) if "CP" in self.species_index else 0.0
        )
        if "CP" in self.species_index:
            initial[self.species_index["CP"]] = 0.0

        self.register_buffer("S_elementary", scipy_to_torch(stoich, dtype))
        self.register_buffer("A_transpose", scipy_to_torch(reactants.T, dtype))
        self.register_buffer("log_k", torch.log(torch.tensor(rate_constants, dtype=dtype)))
        self.register_buffer("initial_state_M", torch.tensor(initial, dtype=dtype))

        # One missing effective reaction: glucose-driven ATP regeneration.
        # NDK is already represented by 17 elementary reactions in the base CRN.
        teacher_rows: list[int] = []
        teacher_cols: list[int] = []
        teacher_values: list[float] = []
        teacher_stoich = {
            "ATP": (1.0,),
            "ADP": (-1.0,),
            "Glucose": (-0.5,),
        }
        for sid, coefficients in teacher_stoich.items():
            for column, coefficient in enumerate(coefficients):
                if coefficient:
                    teacher_rows.append(self.species_index[sid])
                    teacher_cols.append(column)
                    teacher_values.append(coefficient)
        teacher_matrix = sparse.coo_matrix(
            (teacher_values, (teacher_rows, teacher_cols)),
            shape=(self.n_species, self.n_teacher_reactions),
        )
        self.register_buffer("S_teacher", scipy_to_torch(teacher_matrix, dtype))

        self.k_atp_reg_M_s = 2e-6 * float(energy_scale)
        self.K_adp_reg_M = 4e-4
        self.K_glucose_M = 5e-3
        self.K_nad_M = 2e-4
        self.K_tpp_M = 2e-5
        self.K_mg_M = 4e-3
        self.gamma_per_s = float(gamma_h) / 3600.0

        feed_ids = ("ATP", "GTP", "CTP", "UTP", "Glucose", "NAD", "TPP", "Mg", *AA20)
        washout_ids = ("ADP", "GDP", "PPi", "PO4", "NADH")
        product_ids = tuple(sid for sid in self.species_ids if sid.startswith("deGFP_m__"))
        feed_target = np.zeros(self.n_species, dtype=float)
        feed_mask = np.zeros(self.n_species, dtype=float)
        washout_mask = np.zeros(self.n_species, dtype=float)
        for sid in feed_ids:
            if sid in self.species_index:
                feed_mask[self.species_index[sid]] = 1.0
                feed_target[self.species_index[sid]] = float(
                    pool.get(sid, initial[self.species_index[sid]])
                )
        for sid in (*washout_ids, *product_ids):
            if sid in self.species_index:
                washout_mask[self.species_index[sid]] = 1.0
        if np.any((feed_mask > 0) & (washout_mask > 0)):
            raise ValueError("A state cannot be both continuously fed and washed out")
        self.register_buffer("feed_target_M", torch.tensor(feed_target, dtype=dtype))
        self.register_buffer("feed_mask", torch.tensor(feed_mask, dtype=dtype))
        self.register_buffer("washout_mask", torch.tensor(washout_mask, dtype=dtype))

    def elementary_flux(self, concentration_M: torch.Tensor) -> torch.Tensor:
        base = concentration_M[:, : self.base_species_count]
        log_x = torch.log(torch.clamp(base, min=1e-30))
        log_monomial = torch.sparse.mm(self.A_transpose, log_x.T).T
        return torch.exp(torch.clamp(log_monomial + self.log_k, min=-100.0, max=100.0))

    def teacher_flux(self, concentration_M: torch.Tensor) -> torch.Tensor:
        x = lambda sid: concentration_M[:, self.species_index[sid]]
        gate_reg = (
            _sat(x("Glucose"), self.K_glucose_M)
            * _sat(x("NAD"), self.K_nad_M)
            * _sat(x("TPP"), self.K_tpp_M)
            * _sat(x("Mg"), self.K_mg_M)
            * _sat(x("ADP"), self.K_adp_reg_M)
        )
        return (self.k_atp_reg_M_s * gate_reg).unsqueeze(1)

    def exchange_rhs(self, concentration_M: torch.Tensor) -> torch.Tensor:
        return self.gamma_per_s * (
            self.feed_mask * (self.feed_target_M - concentration_M)
            - self.washout_mask * concentration_M
        )

    def forward(
        self,
        concentration_M: torch.Tensor,
        *,
        include_exchange: bool = True,
    ) -> torch.Tensor:
        if concentration_M.ndim != 2 or concentration_M.shape[1] != self.n_species:
            raise ValueError(f"Expected [batch,{self.n_species}] state tensor")
        elementary_flux = self.elementary_flux(concentration_M)
        elementary_rhs = torch.sparse.mm(self.S_elementary, elementary_flux.T).T
        if self.n_species > self.base_species_count:
            padding = concentration_M.new_zeros(
                (concentration_M.shape[0], self.n_species - self.base_species_count)
            )
            elementary_rhs = torch.cat((elementary_rhs, padding), dim=1)
        teacher_rhs = torch.sparse.mm(self.S_teacher, self.teacher_flux(concentration_M).T).T
        rhs = elementary_rhs + teacher_rhs
        if include_exchange:
            rhs = rhs + self.exchange_rhs(concentration_M)
        return rhs


class HybridStatePINN(nn.Module):
    def __init__(
        self,
        concentration_scale_M: torch.Tensor,
        width: int = 16,
        depth: int = 2,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        n_species = int(concentration_scale_M.numel())
        layers: list[nn.Module] = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, n_species))
        self.network = nn.Sequential(*layers).to(dtype=dtype)
        self.register_buffer("concentration_scale_M", concentration_scale_M.to(dtype=dtype))

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        normalized = time_s / 3600.0
        return torch.nn.functional.softplus(self.network(normalized)) * self.concentration_scale_M


def time_derivative(model: HybridStatePINN, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = torch.ones_like(time_s)
    return jvp(model, (time_s,), (tangent,))


def physics_loss(
    model: HybridStatePINN,
    operator: HybridResolutionOperator,
    time_s: torch.Tensor,
    derivative_scale_M_per_s: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, dxdt_autograd = time_derivative(model, time_s)
    dxdt_physics = operator(state)
    scaled_residual = (dxdt_autograd - dxdt_physics) / derivative_scale_M_per_s
    loss = torch.mean(scaled_residual.square())
    return loss, {
        "physics_loss": float(loss.detach()),
        "max_abs_scaled_residual": float(torch.max(torch.abs(scaled_residual)).detach()),
        "negative_state_fraction": float(torch.mean((state < 0).to(torch.float64)).detach()),
    }


def smoke_test(assets: Path, pool_json: Path) -> dict[str, object]:
    dtype = torch.float64
    operator = HybridResolutionOperator(assets, pool_json, dtype=dtype)
    initial = operator.initial_state_M
    initial_batch = initial.unsqueeze(0)
    rhs = operator(initial_batch)
    teacher_flux = operator.teacher_flux(initial_batch)
    activated = initial_batch.clone()
    activated[:, operator.species_index["ADP"]] = 4e-4
    activated_teacher_flux = operator.teacher_flux(activated)

    scale = torch.clamp(initial, min=1e-8)
    model = HybridStatePINN(scale, width=8, depth=2, dtype=dtype)
    times = torch.tensor([[0.0], [1.0]], dtype=dtype)
    derivative_scale = torch.clamp(scale / 3600.0, min=1e-12)
    loss, diagnostics = physics_loss(model, operator, times, derivative_scale)
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    shared_ids = ("ATP", "ADP", "GTP", "GDP")
    return {
        "model_class": "hybrid-resolution PINN",
        "elementary_block": "six sequence-specific transcription-translation CRNs",
        "teacher_block": "effective SG-PURE glucose-driven ATP-regeneration dynamics",
        "base_elementary_species": operator.base_species_count,
        "teacher_only_states_added": operator.n_species - operator.base_species_count,
        "total_states": operator.n_species,
        "elementary_reactions": operator.n_elementary_reactions,
        "teacher_effective_reactions": operator.n_teacher_reactions,
        "shared_state_copy_count": {
            sid: int(operator.species_ids.count(sid)) for sid in shared_ids
        },
        "teacher_flux_at_initial_M_per_s": {
            "ATP_regeneration": float(teacher_flux[0, 0]),
        },
        "teacher_flux_at_resource_debt_probe_M_per_s": {
            "probe_ADP_M": 4e-4,
            "ATP_regeneration": float(activated_teacher_flux[0, 0]),
        },
        "finite_initial_rhs": bool(torch.isfinite(rhs).all()),
        "finite_initial_teacher_flux": bool(torch.isfinite(teacher_flux).all()),
        "all_parameter_gradients_finite": finite_gradients,
        "continuous_exchange_is_state_dependent": True,
        "teacher_trajectory_used_as_fixed_forcing": False,
        "cofactor_scope_note": (
            "NAD, NADH, TPP and Mg follow the existing teacher closure: they gate the effective "
            "ATP-regeneration flux but are not expanded into catalytic-complex stoichiometry."
        ),
        "energy_overlap_controls": {
            "reference_CP_initial_M": 0.0,
            "elementary_NDK_retained": True,
            "elementary_NDK_reaction_count": operator.elementary_ndk_reaction_count,
            "teacher_NDK_added": False,
        },
        **diagnostics,
    }


def export_hybrid_assets(operator: HybridResolutionOperator, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = operator.initial_state_M.detach().cpu().numpy()
    np.save(output_dir / "hybrid_initial_concentrations_M.npy", initial)
    pd.DataFrame(
        {
            "species_index": np.arange(operator.n_species),
            "species_id": operator.species_ids,
            "initial_concentration_M": initial,
            "resolution": [
                "full_elementary" if i < operator.base_species_count else "teacher_effective"
                for i in range(operator.n_species)
            ],
        }
    ).to_csv(output_dir / "hybrid_species_index.csv", index=False)
    manifest = {
        "model_class": "transcription-translation full-elementary / glycolysis effective hybrid PINN",
        "base_elementary_species": operator.base_species_count,
        "teacher_only_states_added": operator.n_species - operator.base_species_count,
        "total_states": operator.n_species,
        "elementary_reactions": operator.n_elementary_reactions,
        "teacher_effective_reactions": operator.n_teacher_reactions,
        "teacher_only_state_ids": [
            sid for sid in operator.species_ids if sid not in operator.base_species
        ],
        "shared_states_single_copy": ["ATP", "ADP", "GTP", "GDP"],
        "continuous_exchange_gamma_h_inv": operator.gamma_per_s * 3600.0,
        "teacher_closure": {
            "ATP_regeneration": "Glucose + 2 ADP -> 2 ATP (effective, cofactor-gated)",
            "NDK": "retained at full-elementary resolution from the base CRN",
            "cofactor_gate_only": ["NAD", "TPP", "Mg"],
            "washed_cofactor_state": ["NADH"],
        },
        "disabled_reference_energy_substrates": {"CP_initial_M": 0.0},
        "reference_CP_before_hybrid_override_M": operator.reference_cp_original_initial_M,
        "elementary_NDK_reaction_count": operator.elementary_ndk_reaction_count,
    }
    (output_dir / "hybrid_operator_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ASSETS / "hybrid_resolution_pinn_smoke_test.json",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test:
        raise SystemExit("Use --smoke-test; full training is a separate long-running stage.")
    result = smoke_test(args.assets, args.pool_json)
    operator = HybridResolutionOperator(args.assets, args.pool_json)
    export_hybrid_assets(operator, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
