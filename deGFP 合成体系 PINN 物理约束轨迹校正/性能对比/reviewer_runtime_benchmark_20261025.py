#!/usr/bin/env python3
"""Benchmark a stiff energy-regeneration subnetwork and trained PINN inference."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.integrate import solve_ivp


MODULE_PATTERN = re.compile(r"^(?:CK|MK|NDK|PPiase)(?:_|$)")


def hardware() -> dict[str, object]:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        text = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, timeout=30
        )
        for label in ("Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"):
            match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
            if match:
                result[label.lower().replace(" ", "_")] = match.group(1).strip()
    except Exception:
        pass
    return result


def load_trained_pinn(checkpoint: Path):
    import sys

    model_dir = checkpoint.parent.parent.parent
    sys.path.insert(0, str(model_dir))
    from train_continuous_full_crn_pinn_key import HardICPositivePINN

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload["args"]
    state = payload["model_state_dict"]
    model = HardICPositivePINN(
        x0=state["x0"],
        scale=state["scale"],
        t_end_s=float(args["time_end_s"]),
        width=int(args["width"]),
        depth=int(args["depth"]),
        dtype=torch.float64,
    )
    model.load_state_dict(state)
    model.eval()
    return model, payload


def benchmark_pinn(checkpoint: Path, time_s: np.ndarray, repeats: int) -> dict[str, object]:
    model, payload = load_trained_pinn(checkpoint)
    time_tensor = torch.tensor(time_s[:, None], dtype=torch.float64)
    with torch.inference_mode():
        model(time_tensor)
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            prediction = model(time_tensor)
            elapsed.append(time.perf_counter() - start)
    return {
        "checkpoint": str(checkpoint),
        "state_count": int(prediction.shape[1]),
        "output_time_points": int(prediction.shape[0]),
        "repeats": int(repeats),
        "median_s": float(np.median(elapsed)),
        "mean_s": float(np.mean(elapsed)),
        "min_s": float(np.min(elapsed)),
        "p95_s": float(np.quantile(elapsed, 0.95)),
        "per_time_point_ms_median": float(1000.0 * np.median(elapsed) / len(time_s)),
        "finite_fraction": float(torch.isfinite(prediction).double().mean()),
        "training_steps_in_checkpoint": int(payload["args"]["steps"]),
    }


def benchmark_bdf(
    assets: Path,
    source_npz: Path,
    output_time_s: np.ndarray,
    rtol: float,
    atol_m: float,
    preequilibrate_s: float,
) -> dict[str, object]:
    species = pd.read_csv(assets / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    module_indices = np.asarray(
        [i for i, sid in enumerate(species_ids) if MODULE_PATTERN.match(sid)], dtype=np.int64
    )
    source_data = np.load(source_npz, allow_pickle=True)
    source_time_s = source_data["time_s"].astype(float)
    source_key = "prediction_M" if "prediction_M" in source_data else "optimized_M"
    source = np.maximum(source_data[source_key].astype(float), 0.0)[:, : len(species_ids)]

    stoich = sparse.load_npz(assets / "stoichiometric_matrix_S.npz").tocsc()
    reactants = sparse.load_npz(assets / "reactant_exponent_matrix_A.npz").tocsc()
    rates = np.load(assets / "rate_constants_k.npy").astype(float)
    reaction_indices = np.unique(stoich[module_indices, :].nonzero()[1]).astype(np.int64)
    local_stoich = stoich[module_indices, :][:, reaction_indices].tocsr()
    local_reactants = reactants[:, reaction_indices].tocsr()
    dependency_indices = np.union1d(
        np.unique(local_reactants.nonzero()[0]).astype(np.int64), module_indices
    ).astype(np.int64)
    dependency_lookup = {int(index): position for position, index in enumerate(dependency_indices)}
    local_positions = np.asarray(
        [dependency_lookup[int(index)] for index in module_indices], dtype=np.int64
    )
    dependency_reactants = local_reactants[dependency_indices, :].tocsr()
    log_rates = np.log(np.maximum(rates[reaction_indices], 1e-300))

    def context_at(t: float) -> np.ndarray:
        values = np.empty(len(dependency_indices), dtype=float)
        for pos, index in enumerate(dependency_indices):
            values[pos] = np.interp(t, source_time_s, source[:, index])
        return values

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        state = context_at(t)
        state[local_positions] = np.maximum(y, 0.0)
        log_flux = log_rates + dependency_reactants.T.dot(np.log(np.maximum(state, 1e-30)))
        flux = np.exp(np.clip(np.asarray(log_flux).reshape(-1), -100.0, 100.0))
        return np.asarray(local_stoich.dot(flux)).reshape(-1)

    y0 = source[0, module_indices].copy()
    pre_stats = None
    start = time.perf_counter()
    if preequilibrate_s > 0:
        pre = solve_ivp(
            lambda _t, y: rhs(float(source_time_s[0]), y),
            (0.0, float(preequilibrate_s)),
            y0,
            method="BDF",
            rtol=rtol,
            atol=atol_m,
        )
        if not pre.success:
            raise RuntimeError(pre.message)
        y0 = np.maximum(pre.y[:, -1], 0.0)
        pre_stats = {"nfev": int(pre.nfev), "njev": int(pre.njev), "nlu": int(pre.nlu)}
    solve_start = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (float(output_time_s[0]), float(output_time_s[-1])),
        y0,
        t_eval=output_time_s,
        method="BDF",
        rtol=rtol,
        atol=atol_m,
    )
    solve_elapsed = time.perf_counter() - solve_start
    total_elapsed = time.perf_counter() - start
    if not solution.success:
        raise RuntimeError(solution.message)
    return {
        "method": "SciPy solve_ivp BDF",
        "rtol": float(rtol),
        "atol_M": float(atol_m),
        "simulated_time_h": float((output_time_s[-1] - output_time_s[0]) / 3600.0),
        "output_time_points": int(len(output_time_s)),
        "module": "CK/MK/NDK/PPiase energy-regeneration module",
        "module_state_count": int(len(module_indices)),
        "linked_elementary_reaction_count": int(len(reaction_indices)),
        "external_context_species_count": int(len(dependency_indices) - len(module_indices)),
        "module_species": [species_ids[i] for i in module_indices],
        "preequilibrate_s": float(preequilibrate_s),
        "preequilibration_stats": pre_stats,
        "solve_runtime_s": float(solve_elapsed),
        "total_runtime_s_including_preequilibration": float(total_elapsed),
        "success": bool(solution.success),
        "message": str(solution.message),
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
        "finite_fraction": float(np.isfinite(solution.y).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol-M", type=float, default=1e-18)
    parser.add_argument("--preequilibrate-s", type=float, default=600.0)
    parser.add_argument("--time-points", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    time_s = np.linspace(0.0, 21600.0, args.time_points)
    report = {
        "hardware": hardware(),
        "benchmark_scope": (
            "Same 0-6 h time span. BDF integrates the extracted stiff energy module "
            "with the trained PINN trajectory supplying external context; PINN timing "
            "is a full 5772-state forward prediction on the same output grid."
        ),
        "bdf": benchmark_bdf(
            args.assets,
            args.source,
            time_s,
            args.rtol,
            args.atol_M,
            args.preequilibrate_s,
        ),
        "pinn_inference": benchmark_pinn(args.checkpoint, time_s, args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
