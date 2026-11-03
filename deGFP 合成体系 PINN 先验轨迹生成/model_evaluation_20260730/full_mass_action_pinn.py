#!/usr/bin/env python3
"""Sparse full-CRN PINN kernel for the 5772-species, 7061-reaction PURE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.func import jvp


DEFAULT_ASSETS = Path(__file__).resolve().parent / "article_gapfill_20260730" / "full_elementary_pinn"


def scipy_to_torch(matrix: sparse.spmatrix, dtype: torch.dtype) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, coo.shape, dtype=dtype).coalesce()


class FullMassActionOperator(nn.Module):
    """Hard physical layer: v=k*prod(x**A), dx/dt=S*v+q_exchange."""

    def __init__(self, assets: Path, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        stoich = sparse.load_npz(assets / "stoichiometric_matrix_S.npz")
        reactants = sparse.load_npz(assets / "reactant_exponent_matrix_A.npz")
        rate_constants = np.load(assets / "rate_constants_k.npy")
        self.register_buffer("S", scipy_to_torch(stoich, dtype))
        self.register_buffer("A_transpose", scipy_to_torch(reactants.T, dtype))
        self.register_buffer("log_k", torch.log(torch.tensor(rate_constants, dtype=dtype)))
        self.n_species, self.n_reactions = stoich.shape

    def flux(self, concentration_M: torch.Tensor) -> torch.Tensor:
        # PINN states are strictly positive through softplus, so log-space mass
        # action is stable even across many concentration decades.
        log_x = torch.log(torch.clamp(concentration_M, min=1e-30))
        log_monomial = torch.sparse.mm(self.A_transpose, log_x.T).T
        return torch.exp(torch.clamp(log_monomial + self.log_k, min=-100.0, max=100.0))

    def forward(
        self, concentration_M: torch.Tensor, exchange_M_per_s: torch.Tensor | None = None
    ) -> torch.Tensor:
        flux = self.flux(concentration_M)
        rhs = torch.sparse.mm(self.S, flux.T).T
        if exchange_M_per_s is not None:
            rhs = rhs + exchange_M_per_s
        return rhs


class FullStatePINN(nn.Module):
    def __init__(
        self,
        n_species: int,
        concentration_scale_M: torch.Tensor,
        n_conditions: int = 0,
        width: int = 128,
        depth: int = 4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1 + n_conditions, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, n_species))
        self.network = nn.Sequential(*layers).to(dtype=dtype)
        self.register_buffer("concentration_scale_M", concentration_scale_M.to(dtype=dtype))

    def forward(self, time_and_conditions: torch.Tensor) -> torch.Tensor:
        normalized = time_and_conditions.clone()
        normalized[:, 0] = normalized[:, 0] / 3600.0
        return torch.nn.functional.softplus(self.network(normalized)) * self.concentration_scale_M


def time_derivative(model: FullStatePINN, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = torch.zeros_like(inputs)
    tangent[:, 0] = 1.0
    state, derivative = jvp(model, (inputs,), (tangent,))
    return state, derivative


def physics_loss(
    model: FullStatePINN,
    operator: FullMassActionOperator,
    collocation_inputs: torch.Tensor,
    derivative_scale_M_per_s: torch.Tensor,
    exchange_M_per_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, dxdt_autograd = time_derivative(model, collocation_inputs)
    dxdt_crn = operator(state, exchange_M_per_s)
    scaled_residual = (dxdt_autograd - dxdt_crn) / derivative_scale_M_per_s
    loss = torch.mean(scaled_residual * scaled_residual)
    return loss, {
        "physics_loss": float(loss.detach()),
        "max_abs_scaled_residual": float(torch.max(torch.abs(scaled_residual)).detach()),
        "negative_state_fraction": float(torch.mean((state < 0).to(torch.float64)).detach()),
    }


def teacher_loss(
    predicted_full_state: torch.Tensor,
    teacher_state: torch.Tensor,
    full_indices: torch.Tensor,
    teacher_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply reduced-model supervision only to mapped common species."""
    selected = predicted_full_state.index_select(1, full_indices)
    return torch.mean(((selected - teacher_state) / teacher_scale) ** 2)


def initial_condition_loss(
    predicted_initial_state: torch.Tensor,
    initial_state: torch.Tensor,
    concentration_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(((predicted_initial_state - initial_state) / concentration_scale) ** 2)


def smoke_test(assets: Path) -> dict[str, object]:
    dtype = torch.float64
    operator = FullMassActionOperator(assets, dtype=dtype)
    initial = torch.tensor(np.load(assets / "initial_concentrations_x0_M.npy"), dtype=dtype)
    scale = torch.clamp(initial, min=1e-8)
    model = FullStatePINN(
        n_species=operator.n_species,
        concentration_scale_M=scale,
        n_conditions=0,
        width=16,
        depth=2,
        dtype=dtype,
    )
    times = torch.tensor([[0.0], [1.0]], dtype=dtype)
    derivative_scale = torch.clamp(scale / 3600.0, min=1e-12)
    loss, diagnostics = physics_loss(model, operator, times, derivative_scale)
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    result = {
        "species": operator.n_species,
        "reactions": operator.n_reactions,
        "hard_constraints": [
            "nonnegative concentrations by softplus",
            "all reaction fluxes from fixed mass-action A and k",
            "state derivatives from fixed stoichiometric matrix S",
            "continuous exchange enters only as q_exchange",
        ],
        "physics_loss_smoke_value": float(loss.detach()),
        "all_parameter_gradients_finite": finite_gradients,
        **diagnostics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test:
        raise SystemExit("Use --smoke-test; training requires a selected teacher/experimental dataset.")
    result = smoke_test(args.assets)
    output = args.assets / "full_mass_action_pinn_smoke_test.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
