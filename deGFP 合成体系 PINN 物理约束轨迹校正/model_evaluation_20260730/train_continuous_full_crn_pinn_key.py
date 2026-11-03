#!/usr/bin/env python3
"""Teacher/key-supervised continuous full-CRN PINN.

The neural net outputs all 5772 species. The physics layer uses all 7061
internal elementary mass-action reactions and adds only continuous exchange:

    dx/dt = S v(x) + q_exchange,  v_j = k_j prod_i x_i ** A_ij

The available full-elementary continuous RoadRunner trajectory contains key
species only. Optionally, a reduced-teacher 6 h full-species snapshot can be
added as shared-species supervision through the teacher-to-full mapping.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import jvp

from full_mass_action_pinn import FullMassActionOperator


DEFAULT_ASSETS = Path(__file__).resolve().parent / "article_gapfill_20260730" / "full_elementary_pinn"
DEFAULT_CONTINUOUS_XML = (
    Path(__file__).resolve().parent / "article_gapfill_20260730" / "crn_constraint_map" / "formal_model.xml"
)


class HardICPositivePINN(nn.Module):
    def __init__(
        self,
        x0: torch.Tensor,
        scale: torch.Tensor,
        t_end_s: float,
        width: int,
        depth: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer("x0", x0.to(dtype=dtype))
        self.register_buffer("scale", scale.to(dtype=dtype))
        self.register_buffer("positive_mask", (x0 > 0).to(dtype=dtype))
        self.t_end_s = float(t_end_s)
        layers: list[nn.Module] = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, x0.numel()))
        self.net = nn.Sequential(*layers).to(dtype=dtype)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                final.weight.mul_(0.03)
                final.bias.zero_()

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(time_s[:, :1] / self.t_end_s, min=0.0, max=1.0)
        features = torch.cat(
            [
                2.0 * tau - 1.0,
                2.0 * torch.log1p(time_s[:, :1]) / np.log1p(self.t_end_s) - 1.0,
                torch.exp(-time_s[:, :1] / 60.0),
                torch.exp(-time_s[:, :1] / 600.0),
            ],
            dim=1,
        )
        z = torch.clamp(self.net(features), min=-25.0, max=25.0)
        positive_part = self.x0 * torch.exp(tau * z)
        zero_part = tau * self.scale * torch.nn.functional.softplus(z)
        return self.positive_mask * positive_part + (1.0 - self.positive_mask) * zero_part


def time_derivative(model: nn.Module, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = torch.ones_like(time_s)
    return jvp(model, (time_s,), (tangent,))


def parse_parameters(xml_path: Path) -> dict[str, float]:
    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for match in re.finditer(r'<parameter id="([^"]+)" value="([^"]+)"', text):
        try:
            out[match.group(1)] = float(match.group(2))
        except ValueError:
            pass
    return out


def build_exchange(
    time: torch.Tensor,
    pred: torch.Tensor,
    species_ids: list[str],
    continuous_xml: Path,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, object]]:
    params = parse_parameters(continuous_xml)
    gamma = float(params.get("bg_gamma", 0.5 / 3600.0))
    gamma_prod = float(params.get("bg_gamma_prod", gamma))
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    exchange = torch.zeros_like(pred)
    present = []
    missing = []
    for key, value in params.items():
        if not key.startswith("bg_feed_") or key == "bg_feed_deGFP_m":
            continue
        sid = key.removeprefix("bg_feed_")
        if sid == "AA":
            continue
        if sid not in lookup:
            missing.append(sid)
            continue
        idx = lookup[sid]
        feed = torch.tensor(float(value), dtype=dtype, device=pred.device)
        exchange[:, idx] += gamma * (feed - pred[:, idx])
        present.append(sid)
    if "deGFP_m" in lookup:
        idx = lookup["deGFP_m"]
        exchange[:, idx] += gamma_prod * (0.0 - pred[:, idx])
        present.append("deGFP_m_product_washout")
    meta = {
        "gamma_h_inv": gamma * 3600.0,
        "gamma_prod_h_inv": gamma_prod * 3600.0,
        "exchange_species_present": sorted(present),
        "exchange_species_missing": sorted(missing),
    }
    return exchange, meta


PERTURBATION_KEY_SPECIES = [
    "GTP",
    "GDP",
    "ATP",
    "ADP",
    "AMP",
    "CP",
    "PPi",
    "PO4",
    "mRNA_t",
    "mRNA",
    "Ribosome",
    "Ribosome_mRNA",
    "RS30S",
    "RS50S",
    "RS70S",
    "EFTu",
    "EFTu_GTP",
    "EFTu_GDP",
    "EFTu_EFTs",
    "EFG_GTP",
    "IF2_GTP",
    "MetRS",
    "MetRS_ATP",
    "deGFP_m",
]

CRITICAL_BOTTLENECK_WEIGHTS = {
    "GTP": 20.0,
    "GDP": 8.0,
    "ATP": 8.0,
    "ADP": 4.0,
    "AMP": 4.0,
    "PPi": 20.0,
    "PO4": 8.0,
    "mRNA_t": 20.0,
    "mRNA": 8.0,
    "Ribosome_mRNA": 30.0,
    "Ribosome": 8.0,
    "RS70S": 2.0,
    "EFTu_GTP": 12.0,
    "EFTu_GDP": 12.0,
    "EFTu_EFTs": 6.0,
    "EFG_GTP": 4.0,
    "IF2_GTP": 4.0,
    "deGFP_m": 40.0,
}


def build_reaction_modules(reactions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Perturbation-derived full-CRN flux modules.

    These masks intentionally stay on the 7061-reaction elementary network. They
    only aggregate elementary fluxes into biologically interpretable indicators
    used by the perturbation analysis: GTP partitioning, ATP pressure, mRNA and
    ribosome-mRNA loading, elongation and termination.
    """

    def text(row: pd.Series) -> str:
        return f"{row.get('reactants', '')} || {row.get('products', '')}"

    modules: dict[str, list[int]] = {
        "gtp_total_reactant_consumption": [],
        "gtp_tx_consumption": [],
        "gtp_non_tx_non_progression": [],
        "gtp_eftu_delivery": [],
        "gtp_factor_gtpase": [],
        "atp_tx_consumption": [],
        "atp_aars_activation": [],
        "mrna_production": [],
        "ribosome_mrna_assembly": [],
        "translation_progression": [],
        "termination_product": [],
        "ppi_po4_generation": [],
    }
    for row in reactions.itertuples(index=False):
        idx = int(row.reaction_index)
        reactants = str(row.reactants)
        products = str(row.products)
        combined = f"{reactants} || {products}"
        gtp_reactant = "GTP" in reactants
        is_tx_gtp = "1 GTP" in reactants and "T7RNAP_bound" in combined and "mRNA" in combined
        is_translation_progression = "elRS70S" in combined and "Pept" in combined
        if gtp_reactant:
            modules["gtp_total_reactant_consumption"].append(idx)
        if is_tx_gtp:
            modules["gtp_tx_consumption"].append(idx)
        if "EFTu_GTP" in combined and ("tRNA" in combined or "elRS70S" in combined):
            modules["gtp_eftu_delivery"].append(idx)
        if any(token in combined for token in ["EFG_GTP", "IF2_GTP", "RF3_GTP"]) and (
            "GDP" in products or "PO4" in products or "GDP_PO4" in products
        ):
            modules["gtp_factor_gtpase"].append(idx)
        if "1 ATP" in reactants and "T7RNAP_bound" in combined and "mRNA" in combined:
            modules["atp_tx_consumption"].append(idx)
        if "1 ATP" in reactants and "RS" in combined and (
            "AMP" in products or "PPi" in products or "_ATP" in products
        ):
            modules["atp_aars_activation"].append(idx)
        if "mRNA_t" in products:
            modules["mrna_production"].append(idx)
        if "mRNA" in reactants and any(token in combined for token in ["RS30S", "RS70S", "IF2", "IF3"]):
            modules["ribosome_mrna_assembly"].append(idx)
        if is_translation_progression:
            modules["translation_progression"].append(idx)
        if gtp_reactant and not is_tx_gtp and not is_translation_progression:
            modules["gtp_non_tx_non_progression"].append(idx)
        if "deGFP_m" in products or ("termRS70S" in combined and "Pept" in combined):
            modules["termination_product"].append(idx)
        if "PPi" in products or "PO4" in products:
            modules["ppi_po4_generation"].append(idx)
    return {name: np.array(indices, dtype=int) for name, indices in modules.items()}


def module_fluxes(flux: torch.Tensor, modules: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    values = []
    names = []
    for name, idx in modules.items():
        if idx.numel() == 0:
            continue
        values.append(torch.sum(flux.index_select(1, idx), dim=1))
        names.append(name)
    if not values:
        return torch.empty((flux.shape[0], 0), dtype=flux.dtype, device=flux.device), []
    return torch.stack(values, dim=1), names


GTP_MECHANISM_TO_FULL_MODULE = {
    "tx_gtp_entry": "gtp_tx_consumption",
    # The perturbation-analysis teacher defines tl_gtp_entry as
    # total_GTP_consumption - tx_gtp_entry - tl_gtp_progression, i.e. a
    # residual/non-transcription/non-chain-progression GTP-consuming bucket.
    # Keep EFTu-GTP delivery as a separate biological indicator, not as the
    # direct proxy for this residual teacher metric.
    "tl_gtp_entry": "gtp_non_tx_non_progression",
    "tl_gtp_progression": "translation_progression",
}


def normalized_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare timecourse shapes when teacher and full-CRN proxies have different units.

    The perturbation-analysis GTP metrics are mechanism indicators, not raw
    full-elementary mass-action fluxes in M/s. We therefore use them to constrain
    relative timing / bottleneck switching rather than absolute magnitude.
    """
    pred_scale = torch.clamp(torch.max(torch.abs(pred.detach()), dim=0).values, min=1e-30)
    target_scale = torch.clamp(torch.max(torch.abs(target), dim=0).values, min=1e-30)
    pred_norm = pred / pred_scale
    target_norm = target / target_scale
    sq = (pred_norm - target_norm) ** 2
    if weights is not None and weights.numel() == sq.shape[1]:
        w = weights / torch.clamp(torch.mean(weights), min=1e-30)
        sq = sq * w[None, :]
    return torch.mean(sq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--continuous-xml", type=Path, default=DEFAULT_CONTINUOUS_XML)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--time-end-s", type=float, default=21600.0)
    parser.add_argument("--time-points", type=int, default=25)
    parser.add_argument("--output-prefix", type=str, default="continuous_key_pinn")
    parser.add_argument("--exchange-mode", choices=["continuous", "batch"], default="continuous")
    parser.add_argument("--teacher-snapshot", type=Path, default=None)
    parser.add_argument("--teacher-trajectory", type=Path, default=None)
    parser.add_argument("--teacher-map", type=Path, default=None)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-loss-mode",
        choices=["scaled_linear", "log10"],
        default="scaled_linear",
        help="Loss used for mapped full-species teacher supervision.",
    )
    parser.add_argument(
        "--teacher-log-eps-M",
        type=float,
        default=1e-12,
        help="Epsilon for log10 full-species teacher supervision.",
    )
    parser.add_argument(
        "--teacher-train-max-s",
        type=float,
        default=None,
        help="If set, mapped full-species teacher trajectory loss is applied only at time_s <= this value.",
    )
    parser.add_argument(
        "--teacher-boost-regex",
        type=str,
        default=None,
        help="Regex for mapped teacher species receiving extra weight in the full-species teacher loss.",
    )
    parser.add_argument(
        "--teacher-boost-weight",
        type=float,
        default=1.0,
        help="Multiplicative weight for --teacher-boost-regex matched species.",
    )
    parser.add_argument("--key-weight", type=float, default=1.0)
    parser.add_argument("--key-terminal-weight", type=float, default=0.0)
    parser.add_argument("--key-terminal-scale-floor-uM", type=float, default=0.01)
    parser.add_argument("--teacher-rhs-weight", type=float, default=0.0)
    parser.add_argument("--teacher-rhs-scale-floor-uM-per-h", type=float, default=0.001)
    parser.add_argument("--teacher-flux-weight", type=float, default=0.0)
    parser.add_argument("--module-flux-weight", type=float, default=0.0)
    parser.add_argument("--mechanism-csv", type=Path, default=None)
    parser.add_argument("--gtp-mechanism-weight", type=float, default=0.0)
    parser.add_argument(
        "--gtp-mechanism-metric-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated weights for tx_gtp_entry,tl_gtp_entry,tl_gtp_progression; "
            "e.g. 0.5,4,0.5 to emphasize EFTu-GTP/tRNA delivery."
        ),
    )
    parser.add_argument("--teacher-flux-min-integrated-M", type=float, default=0.0)
    parser.add_argument("--module-flux-min-integrated-M", type=float, default=0.0)
    parser.add_argument("--bottleneck-state-weight", type=float, default=0.0)
    parser.add_argument("--critical-bottleneck-weight", type=float, default=0.0)
    parser.add_argument("--critical-terminal-weight", type=float, default=0.0)
    parser.add_argument("--plateau-derivative-weight", type=float, default=0.0)
    parser.add_argument("--plateau-start-h", type=float, default=4.0)
    parser.add_argument("--bottleneck-scale-floor-uM", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--physics-weight", type=float, default=1e-8)
    parser.add_argument(
        "--shared-physics-weight",
        type=float,
        default=0.0,
        help="Extra mass-action residual loss on teacher-comparable/shared species only, normalized by RHS RMS.",
    )
    parser.add_argument(
        "--shared-key-physics-weight",
        type=float,
        default=0.0,
        help="Extra RHS-normalized residual loss on comparable perturbation key species.",
    )
    parser.add_argument(
        "--shared-physics-rhs-floor-uM-per-h",
        type=float,
        default=0.001,
        help="RHS RMS floor used to normalize shared/key physics residuals.",
    )
    parser.add_argument(
        "--shared-integral-physics-weight",
        type=float,
        default=0.0,
        help=(
            "Integral trajectory consistency on teacher-comparable/shared species: "
            "C(t2)-C(t1) ≈ trapezoid integral of RHS(C)."
        ),
    )
    parser.add_argument(
        "--shared-key-integral-physics-weight",
        type=float,
        default=0.0,
        help="Integral trajectory consistency on comparable perturbation key species.",
    )
    parser.add_argument(
        "--critical-physics-weight",
        type=float,
        default=0.0,
        help=(
            "Extra mass-action residual weight on perturbation-derived key species "
            "(GTP/GDP/ATP/PPi/mRNA/ribosome/EFTu/deGFP nodes). This keeps the "
            "full 5772-state physics layer but emphasizes biologically interpreted nodes."
        ),
    )
    parser.add_argument("--physics-collocation-points", type=int, default=0)
    parser.add_argument("--physics-collocation-weight", type=float, default=None)
    parser.add_argument(
        "--critical-collocation-physics-weight",
        type=float,
        default=0.0,
        help="Extra collocation residual weight on perturbation-derived key species.",
    )
    parser.add_argument(
        "--integral-physics-weight",
        type=float,
        default=0.0,
        help="Weight for trapezoidal integral consistency: X(t2)-X(t1) ≈ ∫ RHS(X)dt.",
    )
    parser.add_argument(
        "--critical-integral-physics-weight",
        type=float,
        default=0.0,
        help="Extra integral consistency weight on perturbation-derived key species.",
    )
    parser.add_argument(
        "--integral-scale-floor-uM",
        type=float,
        default=0.001,
        help="Concentration scale floor for integral consistency loss, in uM.",
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.float64
    assets = args.assets
    traj_csv = args.trajectory_csv or (
        assets / "full_elementary_roadrunner_continuous_key_trajectory_gamma0p5_6h_13pts_rtol1e4.csv"
    )

    species = pd.read_csv(assets / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    lookup = {sid: i for i, sid in enumerate(species_ids)}

    if args.trajectory_csv is not None:
        traj = pd.read_csv(traj_csv)
        time_np = traj["time_s"].to_numpy(float)
        key_cols = [c for c in traj.columns if c.endswith("_uM")]
        key_names = [c[:-3] for c in key_cols]
        key_indices_np = np.array([lookup[name] for name in key_names if name in lookup], dtype=int)
        key_names = [name for name in key_names if name in lookup]
        key_target_np = traj[[f"{name}_uM" for name in key_names]].to_numpy(float) * 1e-6
    else:
        time_np = np.linspace(0.0, float(args.time_end_s), int(args.time_points))
        key_names = []
        key_indices_np = np.array([], dtype=int)
        key_target_np = np.zeros((len(time_np), 0), dtype=float)

    x0_np = np.maximum(np.load(assets / "initial_concentrations_x0_M.npy").astype(float), 0.0)
    # Scale all states from batch/full reference and continuous key maxima when available.
    batch_ref = np.load(assets / "full_elementary_long_probe_600s_61pts.npz")["concentrations_M"]
    scale_np = np.maximum(np.quantile(np.abs(batch_ref), 0.95, axis=0), 1e-9)
    for name, idx in zip(key_names, key_indices_np):
        scale_np[idx] = max(scale_np[idx], float(np.max(np.abs(key_target_np[:, key_names.index(name)]))), 1e-9)

    time = torch.tensor(time_np[:, None], dtype=dtype)
    if args.physics_collocation_points > 0:
        colloc_np = np.linspace(0.0, float(time_np[-1]), int(args.physics_collocation_points))
        supervised = set(np.round(time_np, 9).tolist())
        colloc_np = np.array([t for t in colloc_np if round(float(t), 9) not in supervised], dtype=float)
        if colloc_np.size == 0:
            colloc_np = np.linspace(0.0, float(time_np[-1]), int(args.physics_collocation_points))
        colloc_time = torch.tensor(colloc_np[:, None], dtype=dtype)
    else:
        colloc_np = np.zeros((0,), dtype=float)
        colloc_time = torch.empty((0, 1), dtype=dtype)
    key_target = torch.tensor(key_target_np, dtype=dtype)
    x0 = torch.tensor(x0_np, dtype=dtype)
    key_indices = torch.tensor(key_indices_np, dtype=torch.long)
    key_scale_np = np.maximum(np.max(np.abs(key_target_np), axis=0), 1e-9) if key_target_np.shape[1] else np.zeros((0,))
    key_scale = torch.tensor(key_scale_np, dtype=dtype)
    key_terminal_scale_np = (
        np.maximum(np.abs(key_target_np[-1]), float(args.key_terminal_scale_floor_uM) * 1e-6)
        if key_target_np.shape[1]
        else np.zeros((0,))
    )
    key_terminal_scale = torch.tensor(key_terminal_scale_np, dtype=dtype)

    teacher_available = args.teacher_snapshot is not None or args.teacher_trajectory is not None
    teacher_full_indices_np = np.array([], dtype=int)
    teacher_target_np = np.zeros((0,), dtype=float)
    teacher_target_traj_np = np.zeros((len(time_np), 0), dtype=float)
    teacher_full_traj_np = np.zeros((len(time_np), len(species_ids)), dtype=float)
    teacher_names: list[str] = []
    if teacher_available:
        teacher_data = np.load(args.teacher_trajectory or args.teacher_snapshot, allow_pickle=True)
        teacher_ids = [str(x) for x in teacher_data["species_id"].tolist()]
        teacher_lookup = {sid: i for i, sid in enumerate(teacher_ids)}
        teacher_x = np.maximum(teacher_data["concentrations_M"].astype(float), 0.0)
        teacher_time = teacher_data["time_s"].astype(float)
        map_path = args.teacher_map or (
            args.assets.parent / "teacher_to_full_mapping" / "full_species_teacher_mapping.csv"
        )
        mapping = pd.read_csv(map_path)
        shared_rows = []
        for row in mapping[mapping["teacher_supervision_available"]].itertuples(index=False):
            sid = str(row.species_id)
            if sid in teacher_lookup and sid in lookup:
                shared_rows.append((sid, int(row.species_index_full), teacher_lookup[sid]))
        teacher_names = [sid for sid, _fi, _ti in shared_rows]
        teacher_full_indices_np = np.array([fi for _sid, fi, _ti in shared_rows], dtype=int)
        if args.teacher_trajectory is not None:
            cols = []
            for _sid, _fi, ti in shared_rows:
                cols.append(np.interp(time_np, teacher_time, teacher_x[:, ti]))
            teacher_target_traj_np = np.stack(cols, axis=1) if cols else np.zeros((len(time_np), 0), dtype=float)
            teacher_target_np = teacher_target_traj_np[-1]
        else:
            teacher_target_np = np.array([teacher_x[-1, ti] for _sid, _fi, ti in shared_rows], dtype=float)
            teacher_target_traj_np = np.repeat(teacher_target_np[None, :], len(time_np), axis=0)
        if shared_rows:
            teacher_full_traj_np[:, teacher_full_indices_np] = teacher_target_traj_np
        for sid, fi, ti in shared_rows:
            scale_np[fi] = max(scale_np[fi], float(np.max(teacher_x[:, ti])), 1e-9)

    teacher_full_indices = torch.tensor(teacher_full_indices_np, dtype=torch.long)
    teacher_target = torch.tensor(teacher_target_np, dtype=dtype)
    teacher_target_traj = torch.tensor(teacher_target_traj_np, dtype=dtype)
    teacher_time_mask = torch.ones((len(time_np),), dtype=torch.bool)
    if args.teacher_train_max_s is not None:
        teacher_time_mask = torch.tensor(time_np <= float(args.teacher_train_max_s) + 1e-9, dtype=torch.bool)
    teacher_species_weight_np = np.ones((len(teacher_names),), dtype=float)
    teacher_boosted_species: list[str] = []
    if args.teacher_boost_regex is not None and len(teacher_names) > 0:
        boost_re = re.compile(args.teacher_boost_regex)
        for i, sid in enumerate(teacher_names):
            if boost_re.search(sid):
                teacher_species_weight_np[i] = float(args.teacher_boost_weight)
                teacher_boosted_species.append(sid)
        teacher_species_weight_np = teacher_species_weight_np / max(float(np.mean(teacher_species_weight_np)), 1e-30)
    teacher_species_weight = torch.tensor(teacher_species_weight_np, dtype=dtype)
    teacher_scale = torch.tensor(np.maximum(np.max(np.abs(teacher_target_traj_np), axis=0), 1e-9), dtype=dtype)
    teacher_full_mask_np = teacher_full_traj_np > 0.0
    # Important: teacher trajectories can contain late-rising products and
    # ribosome-mRNA complexes that are absent or tiny in the short batch
    # reference used to initialize scale_np. Create the model scale only after
    # teacher-based scale updates; otherwise these key biological nodes are
    # artificially capped by the network parameterization.
    scale = torch.tensor(scale_np, dtype=dtype)
    dx_scale = torch.clamp(scale / max(float(time_np[-1]), 1.0), min=1e-12)

    operator = FullMassActionOperator(assets, dtype=dtype)
    reaction_inventory = pd.read_csv(assets / "reaction_mass_action_inventory.csv")
    reaction_modules_np = build_reaction_modules(reaction_inventory)
    reaction_modules = {
        name: torch.tensor(indices, dtype=torch.long)
        for name, indices in reaction_modules_np.items()
        if len(indices) > 0
    }
    gtp_mechanism_modules = {
        module: reaction_modules[module]
        for module in GTP_MECHANISM_TO_FULL_MODULE.values()
        if module in reaction_modules
    }
    gtp_mechanism_available = args.mechanism_csv is not None and args.gtp_mechanism_weight > 0.0
    gtp_mechanism_names = list(GTP_MECHANISM_TO_FULL_MODULE.keys())
    gtp_mechanism_target = torch.empty((len(time_np), 0), dtype=dtype)
    gtp_mechanism_metric_weights = torch.ones((len(gtp_mechanism_names),), dtype=dtype)
    gtp_mechanism_metric_weights_list = [1.0 for _ in gtp_mechanism_names]
    if args.gtp_mechanism_metric_weights is not None:
        parts = [float(x.strip()) for x in args.gtp_mechanism_metric_weights.split(",") if x.strip()]
        if len(parts) != len(gtp_mechanism_names):
            raise SystemExit(
                "--gtp-mechanism-metric-weights must contain three values: "
                "tx_gtp_entry,tl_gtp_entry,tl_gtp_progression"
            )
        gtp_mechanism_metric_weights_list = parts
        gtp_mechanism_metric_weights = torch.tensor(parts, dtype=dtype)
    if gtp_mechanism_available:
        mech = pd.read_csv(args.mechanism_csv)
        if "time_s" in mech.columns:
            mech_time_s = mech["time_s"].to_numpy(float)
        elif "time_h" in mech.columns:
            mech_time_s = mech["time_h"].to_numpy(float) * 3600.0
        else:
            raise SystemExit("--mechanism-csv must contain time_s or time_h")
        missing_mech = [name for name in gtp_mechanism_names if name not in mech.columns]
        if missing_mech:
            raise SystemExit(f"--mechanism-csv missing columns: {missing_mech}")
        target_cols = [
            np.interp(time_np, mech_time_s, mech[name].to_numpy(float))
            for name in gtp_mechanism_names
        ]
        gtp_mechanism_target = torch.tensor(np.stack(target_cols, axis=1), dtype=dtype)
    supervised_reaction_modules: dict[str, torch.Tensor] = {}
    module_names = list(reaction_modules.keys())
    teacher_flux_available = teacher_available and args.teacher_flux_weight > 0.0
    module_flux_available = teacher_available and args.module_flux_weight > 0.0
    teacher_rhs_available = teacher_available and args.teacher_rhs_weight > 0.0 and teacher_full_indices.numel() > 0
    teacher_flux_indices = torch.empty((0,), dtype=torch.long)
    teacher_flux_target = torch.empty((len(time_np), 0), dtype=dtype)
    teacher_flux_scale = torch.empty((0,), dtype=dtype)
    module_flux_target = torch.empty((len(time_np), 0), dtype=dtype)
    module_flux_scale = torch.empty((0,), dtype=dtype)
    module_target_names: list[str] = []
    teacher_rhs_target = torch.empty((len(time_np), 0), dtype=dtype)
    teacher_rhs_scale = torch.empty((0,), dtype=dtype)
    teacher_rhs_mask = torch.empty((len(time_np), 0), dtype=torch.bool)
    if teacher_rhs_available:
        with torch.no_grad():
            teacher_full_traj_for_rhs = torch.tensor(np.maximum(teacher_full_traj_np, 0.0), dtype=dtype)
            if args.exchange_mode == "continuous":
                teacher_exchange_for_rhs, _teacher_exchange_meta = build_exchange(
                    time,
                    teacher_full_traj_for_rhs,
                    species_ids,
                    args.continuous_xml,
                    dtype,
                )
            else:
                teacher_exchange_for_rhs = torch.zeros_like(teacher_full_traj_for_rhs)
            teacher_rhs_full = operator(teacher_full_traj_for_rhs, teacher_exchange_for_rhs)
            teacher_rhs_target = teacher_rhs_full.index_select(1, teacher_full_indices).detach()
            teacher_rhs_scale = torch.clamp(
                torch.max(torch.abs(teacher_rhs_target), dim=0).values,
                min=float(args.teacher_rhs_scale_floor_uM_per_h) * 1e-6 / 3600.0,
            )
            teacher_rhs_mask = torch.tensor(teacher_full_mask_np[:, teacher_full_indices_np], dtype=torch.bool)
    if teacher_available and (args.teacher_flux_weight > 0.0 or args.module_flux_weight > 0.0):
        with torch.no_grad():
            teacher_full_traj = torch.tensor(np.maximum(teacher_full_traj_np, 0.0), dtype=dtype)
            teacher_all_flux = operator.flux(teacher_full_traj)
        if args.teacher_flux_weight > 0.0:
            rx_map = pd.read_csv(assets.parent / "teacher_to_full_mapping" / "full_reaction_teacher_mapping.csv")
            overlap_np = rx_map.loc[rx_map["teacher_reaction_overlap"], "reaction_index_full"].to_numpy(int)
            target_np = teacher_all_flux[:, overlap_np].detach().cpu().numpy()
            integrated_target = np.trapezoid(target_np, time_np, axis=0)
            active = (np.max(target_np, axis=0) > 1e-30) & (
                integrated_target >= float(args.teacher_flux_min_integrated_M)
            )
            overlap_np = overlap_np[active]
            target_np = target_np[:, active]
            teacher_flux_indices = torch.tensor(overlap_np, dtype=torch.long)
            teacher_flux_target = torch.tensor(target_np, dtype=dtype)
            teacher_flux_scale = torch.tensor(np.maximum(np.max(target_np, axis=0), 1e-30), dtype=dtype)
        if args.module_flux_weight > 0.0:
            for name, indices in reaction_modules_np.items():
                if len(indices) == 0:
                    continue
                target_for_reactions = teacher_all_flux[:, indices].detach().cpu().numpy()
                integrated_target = np.trapezoid(target_for_reactions, time_np, axis=0)
                active_indices = indices[
                    (np.max(target_for_reactions, axis=0) > 1e-30)
                    & (integrated_target >= float(args.module_flux_min_integrated_M))
                ]
                if len(active_indices) > 0:
                    supervised_reaction_modules[name] = torch.tensor(active_indices, dtype=torch.long)
            module_target, module_target_names = module_fluxes(teacher_all_flux, supervised_reaction_modules)
            module_flux_target = module_target.detach()
            module_flux_scale = torch.clamp(torch.max(module_flux_target, dim=0).values, min=1e-30)

    bottleneck_species_names = [sid for sid in PERTURBATION_KEY_SPECIES if sid in lookup]
    bottleneck_indices_np = np.array([lookup[sid] for sid in bottleneck_species_names], dtype=int)
    bottleneck_teacher_np = (
        teacher_full_traj_np[:, bottleneck_indices_np]
        if teacher_available and len(bottleneck_indices_np) > 0
        else np.zeros((len(time_np), 0), dtype=float)
    )
    bottleneck_indices = torch.tensor(bottleneck_indices_np, dtype=torch.long)
    bottleneck_teacher = torch.tensor(bottleneck_teacher_np, dtype=dtype)
    bottleneck_scale_floor_M = float(args.bottleneck_scale_floor_uM) * 1e-6
    bottleneck_scale = torch.tensor(
        np.maximum(np.max(np.abs(bottleneck_teacher_np), axis=0), bottleneck_scale_floor_M),
        dtype=dtype,
    )
    critical_weight_np = np.array(
        [CRITICAL_BOTTLENECK_WEIGHTS.get(sid, 1.0) for sid in bottleneck_species_names],
        dtype=float,
    )
    critical_weight_np = critical_weight_np / max(float(np.mean(critical_weight_np)), 1e-30)
    critical_weight = torch.tensor(critical_weight_np, dtype=dtype)
    shared_key_species_names = [sid for sid in PERTURBATION_KEY_SPECIES if sid in teacher_names]
    teacher_name_to_shared_col = {sid: i for i, sid in enumerate(teacher_names)}
    shared_key_shared_cols_np = np.array(
        [teacher_name_to_shared_col[sid] for sid in shared_key_species_names],
        dtype=int,
    )
    shared_key_full_indices_np = (
        teacher_full_indices_np[shared_key_shared_cols_np]
        if len(shared_key_shared_cols_np) > 0 and len(teacher_full_indices_np) > 0
        else np.array([], dtype=int)
    )
    shared_key_indices = torch.tensor(shared_key_full_indices_np, dtype=torch.long)
    shared_key_weight_np = np.array(
        [CRITICAL_BOTTLENECK_WEIGHTS.get(sid, 1.0) for sid in shared_key_species_names],
        dtype=float,
    )
    if shared_key_weight_np.size > 0:
        shared_key_weight_np = shared_key_weight_np / max(float(np.mean(shared_key_weight_np)), 1e-30)
    shared_key_weight = torch.tensor(shared_key_weight_np, dtype=dtype)
    bottleneck_derivative_teacher_np = (
        np.gradient(bottleneck_teacher_np, time_np, axis=0)
        if bottleneck_teacher_np.shape[1] > 0 and len(time_np) > 1
        else np.zeros_like(bottleneck_teacher_np)
    )
    bottleneck_derivative_teacher = torch.tensor(bottleneck_derivative_teacher_np, dtype=dtype)
    bottleneck_derivative_scale = torch.tensor(
        np.maximum(np.max(np.abs(bottleneck_derivative_teacher_np), axis=0), 1e-12),
        dtype=dtype,
    )
    plateau_mask = torch.tensor(time_np >= float(args.plateau_start_h) * 3600.0, dtype=torch.bool)
    model = HardICPositivePINN(x0, scale, float(time_np[-1]), args.width, args.depth, dtype)
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    report_steps = sorted(set([0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, args.steps]))
    exchange_meta: dict[str, object] = {}
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        pred, dpred_dt = time_derivative(model, time)
        if args.exchange_mode == "continuous":
            exchange, exchange_meta = build_exchange(time, pred, species_ids, args.continuous_xml, dtype)
        else:
            exchange = torch.zeros_like(pred)
            exchange_meta = {"gamma_h_inv": 0.0, "gamma_prod_h_inv": 0.0, "exchange_species_present": [], "exchange_species_missing": []}
        flux_pred = operator.flux(pred)
        rhs = torch.sparse.mm(operator.S, flux_pred.T).T + exchange
        if key_indices.numel() > 0 and args.key_weight > 0.0:
            key_pred = pred.index_select(1, key_indices)
            key_scaled = (key_pred - key_target) / key_scale
            key_loss = torch.mean(key_scaled * key_scaled)
        else:
            key_loss = torch.tensor(0.0, dtype=dtype)
        if key_indices.numel() > 0 and args.key_terminal_weight > 0.0:
            key_pred_end = pred[-1:, :].index_select(1, key_indices)
            key_terminal_scaled = (key_pred_end - key_target[-1:, :]) / key_terminal_scale
            key_terminal_loss = torch.mean(key_terminal_scaled * key_terminal_scaled)
        else:
            key_terminal_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_available and teacher_full_indices.numel() > 0 and args.teacher_weight > 0.0:
            teacher_pred = pred.index_select(1, teacher_full_indices)[teacher_time_mask]
            teacher_target_for_loss = teacher_target_traj[teacher_time_mask]
            if args.teacher_loss_mode == "log10":
                teacher_scaled = torch.log10(teacher_pred + args.teacher_log_eps_M) - torch.log10(
                    teacher_target_for_loss + args.teacher_log_eps_M
                )
                teacher_scaled = torch.clamp(teacher_scaled, min=-30.0, max=30.0)
            else:
                teacher_scaled = (teacher_pred - teacher_target_for_loss) / teacher_scale
            if teacher_species_weight.numel() == teacher_scaled.shape[1]:
                teacher_loss = torch.mean((teacher_scaled * teacher_scaled) * teacher_species_weight[None, :])
            else:
                teacher_loss = torch.mean(teacher_scaled * teacher_scaled)
        else:
            teacher_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_rhs_available:
            rhs_dpred = dpred_dt.index_select(1, teacher_full_indices)
            rhs_scaled = (rhs_dpred - teacher_rhs_target) / teacher_rhs_scale
            rhs_mask_now = teacher_rhs_mask & teacher_time_mask[:, None]
            if bool(torch.any(rhs_mask_now)):
                teacher_rhs_loss = torch.mean(torch.clamp(rhs_scaled[rhs_mask_now], min=-1e4, max=1e4) ** 2)
            else:
                teacher_rhs_loss = torch.tensor(0.0, dtype=dtype)
        else:
            teacher_rhs_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_flux_available and teacher_flux_indices.numel() > 0:
            selected_flux = flux_pred.index_select(1, teacher_flux_indices)
            flux_scaled = torch.log10(selected_flux + 1e-30) - torch.log10(teacher_flux_target + 1e-30)
            teacher_flux_loss = torch.mean(torch.clamp(flux_scaled, min=-30.0, max=30.0) ** 2)
        else:
            teacher_flux_loss = torch.tensor(0.0, dtype=dtype)
        if module_flux_available and module_flux_target.numel() > 0:
            module_pred, _module_names_now = module_fluxes(flux_pred, supervised_reaction_modules)
            module_scaled = torch.log10(module_pred + 1e-30) - torch.log10(module_flux_target + 1e-30)
            module_flux_loss = torch.mean(torch.clamp(module_scaled, min=-30.0, max=30.0) ** 2)
        else:
            module_flux_loss = torch.tensor(0.0, dtype=dtype)
        if gtp_mechanism_available and gtp_mechanism_target.numel() > 0:
            mech_proxy, mech_proxy_modules = module_fluxes(flux_pred, gtp_mechanism_modules)
            proxy_by_module = {name: mech_proxy[:, i] for i, name in enumerate(mech_proxy_modules)}
            mech_pred_cols = [
                proxy_by_module[GTP_MECHANISM_TO_FULL_MODULE[name]]
                for name in gtp_mechanism_names
            ]
            gtp_mechanism_pred = torch.stack(mech_pred_cols, dim=1)
            gtp_mechanism_loss = normalized_shape_loss(
                gtp_mechanism_pred,
                gtp_mechanism_target,
                gtp_mechanism_metric_weights,
            )
        else:
            gtp_mechanism_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_available and bottleneck_indices.numel() > 0 and args.bottleneck_state_weight > 0.0:
            bottleneck_pred = pred.index_select(1, bottleneck_indices)
            bottleneck_scaled = (bottleneck_pred - bottleneck_teacher) / bottleneck_scale
            bottleneck_state_loss = torch.mean(bottleneck_scaled * bottleneck_scaled)
        else:
            bottleneck_state_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_available and bottleneck_indices.numel() > 0 and args.critical_bottleneck_weight > 0.0:
            bottleneck_pred = pred.index_select(1, bottleneck_indices)
            critical_scaled = (bottleneck_pred - bottleneck_teacher) / bottleneck_scale
            critical_bottleneck_loss = torch.mean((critical_scaled * critical_scaled) * critical_weight[None, :])
        else:
            critical_bottleneck_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_available and bottleneck_indices.numel() > 0 and args.critical_terminal_weight > 0.0:
            bottleneck_pred_end = pred[-1:, :].index_select(1, bottleneck_indices)
            terminal_scaled = (bottleneck_pred_end - bottleneck_teacher[-1:, :]) / bottleneck_scale
            critical_terminal_loss = torch.mean((terminal_scaled * terminal_scaled) * critical_weight[None, :])
        else:
            critical_terminal_loss = torch.tensor(0.0, dtype=dtype)
        if (
            teacher_available
            and bottleneck_indices.numel() > 0
            and bool(torch.any(plateau_mask))
            and args.plateau_derivative_weight > 0.0
        ):
            bottleneck_dpred = dpred_dt.index_select(1, bottleneck_indices)
            plateau_scaled = (
                bottleneck_dpred[plateau_mask] - bottleneck_derivative_teacher[plateau_mask]
            ) / bottleneck_derivative_scale
            plateau_derivative_loss = torch.mean(torch.clamp(plateau_scaled, min=-1e4, max=1e4) ** 2)
        else:
            plateau_derivative_loss = torch.tensor(0.0, dtype=dtype)
        phys_scaled = (dpred_dt - rhs) / dx_scale
        phys_loss = torch.mean(torch.clamp(phys_scaled, min=-1e5, max=1e5) ** 2)
        residual = dpred_dt - rhs
        shared_phys_loss = torch.tensor(0.0, dtype=dtype)
        shared_key_phys_loss = torch.tensor(0.0, dtype=dtype)
        if teacher_full_indices.numel() > 0 and (args.shared_physics_weight > 0.0 or args.shared_key_physics_weight > 0.0):
            shared_residual = residual.index_select(1, teacher_full_indices)
            shared_rhs = rhs.index_select(1, teacher_full_indices)
            shared_rhs_scale = torch.clamp(
                torch.sqrt(torch.mean(shared_rhs.detach() ** 2, dim=0)),
                min=float(args.shared_physics_rhs_floor_uM_per_h) * 1e-6 / 3600.0,
            )
            shared_resid_norm = torch.clamp(shared_residual / shared_rhs_scale[None, :], min=-1e5, max=1e5)
            if args.shared_physics_weight > 0.0:
                shared_phys_loss = torch.mean(shared_resid_norm ** 2)
            if args.shared_key_physics_weight > 0.0 and shared_key_indices.numel() > 0:
                key_residual = residual.index_select(1, shared_key_indices)
                key_rhs = rhs.index_select(1, shared_key_indices)
                key_rhs_scale = torch.clamp(
                    torch.sqrt(torch.mean(key_rhs.detach() ** 2, dim=0)),
                    min=float(args.shared_physics_rhs_floor_uM_per_h) * 1e-6 / 3600.0,
                )
                key_resid_norm = torch.clamp(key_residual / key_rhs_scale[None, :], min=-1e5, max=1e5)
                if shared_key_weight.numel() == key_resid_norm.shape[1]:
                    shared_key_phys_loss = torch.mean((key_resid_norm ** 2) * shared_key_weight[None, :])
                else:
                    shared_key_phys_loss = torch.mean(key_resid_norm ** 2)
        if bottleneck_indices.numel() > 0 and args.critical_physics_weight > 0.0:
            critical_phys_scaled = phys_scaled.index_select(1, bottleneck_indices)
            critical_phys_loss = torch.mean(
                (torch.clamp(critical_phys_scaled, min=-1e5, max=1e5) ** 2) * critical_weight[None, :]
            )
        else:
            critical_phys_loss = torch.tensor(0.0, dtype=dtype)
        if (args.integral_physics_weight > 0.0 or args.critical_integral_physics_weight > 0.0) and pred.shape[0] > 1:
            dt = time[1:, :] - time[:-1, :]
            delta_pred = pred[1:, :] - pred[:-1, :]
            rhs_trap = 0.5 * dt * (rhs[1:, :] + rhs[:-1, :])
            integral_scale = torch.clamp(scale, min=float(args.integral_scale_floor_uM) * 1e-6)
            integral_scaled = (delta_pred - rhs_trap) / integral_scale
            if args.integral_physics_weight > 0.0:
                integral_phys_loss = torch.mean(torch.clamp(integral_scaled, min=-1e5, max=1e5) ** 2)
            else:
                integral_phys_loss = torch.tensor(0.0, dtype=dtype)
        else:
            integral_phys_loss = torch.tensor(0.0, dtype=dtype)
        if args.critical_integral_physics_weight > 0.0 and pred.shape[0] > 1 and bottleneck_indices.numel() > 0:
            critical_integral_scaled = integral_scaled.index_select(1, bottleneck_indices)
            critical_integral_phys_loss = torch.mean(
                (torch.clamp(critical_integral_scaled, min=-1e5, max=1e5) ** 2)
                * critical_weight[None, :]
            )
        else:
            critical_integral_phys_loss = torch.tensor(0.0, dtype=dtype)
        shared_integral_phys_loss = torch.tensor(0.0, dtype=dtype)
        shared_key_integral_phys_loss = torch.tensor(0.0, dtype=dtype)
        if (
            pred.shape[0] > 1
            and teacher_full_indices.numel() > 0
            and (args.shared_integral_physics_weight > 0.0 or args.shared_key_integral_physics_weight > 0.0)
        ):
            dt = time[1:, :] - time[:-1, :]
            delta_pred = pred[1:, :] - pred[:-1, :]
            rhs_trap = 0.5 * dt * (rhs[1:, :] + rhs[:-1, :])
            shared_delta_resid = (delta_pred - rhs_trap).index_select(1, teacher_full_indices)
            shared_rhs_trap = rhs_trap.index_select(1, teacher_full_indices)
            shared_integral_scale = torch.clamp(
                torch.sqrt(torch.mean(shared_rhs_trap.detach() ** 2, dim=0)),
                min=float(args.shared_physics_rhs_floor_uM_per_h) * 1e-6,
            )
            shared_integral_norm = torch.clamp(
                shared_delta_resid / shared_integral_scale[None, :],
                min=-1e5,
                max=1e5,
            )
            if args.shared_integral_physics_weight > 0.0:
                shared_integral_phys_loss = torch.mean(shared_integral_norm ** 2)
            if args.shared_key_integral_physics_weight > 0.0 and shared_key_indices.numel() > 0:
                key_delta_resid = (delta_pred - rhs_trap).index_select(1, shared_key_indices)
                key_rhs_trap = rhs_trap.index_select(1, shared_key_indices)
                key_integral_scale = torch.clamp(
                    torch.sqrt(torch.mean(key_rhs_trap.detach() ** 2, dim=0)),
                    min=float(args.shared_physics_rhs_floor_uM_per_h) * 1e-6,
                )
                key_integral_norm = torch.clamp(
                    key_delta_resid / key_integral_scale[None, :],
                    min=-1e5,
                    max=1e5,
                )
                if shared_key_weight.numel() == key_integral_norm.shape[1]:
                    shared_key_integral_phys_loss = torch.mean(
                        (key_integral_norm ** 2) * shared_key_weight[None, :]
                    )
                else:
                    shared_key_integral_phys_loss = torch.mean(key_integral_norm ** 2)
        if colloc_time.numel() > 0:
            colloc_pred, colloc_dpred_dt = time_derivative(model, colloc_time)
            if args.exchange_mode == "continuous":
                colloc_exchange, _colloc_meta = build_exchange(
                    colloc_time,
                    colloc_pred,
                    species_ids,
                    args.continuous_xml,
                    dtype,
                )
            else:
                colloc_exchange = torch.zeros_like(colloc_pred)
            colloc_rhs = operator(colloc_pred, colloc_exchange)
            colloc_scaled = (colloc_dpred_dt - colloc_rhs) / dx_scale
            colloc_phys_loss = torch.mean(torch.clamp(colloc_scaled, min=-1e5, max=1e5) ** 2)
            if bottleneck_indices.numel() > 0 and args.critical_collocation_physics_weight > 0.0:
                critical_colloc_scaled = colloc_scaled.index_select(1, bottleneck_indices)
                critical_colloc_phys_loss = torch.mean(
                    (torch.clamp(critical_colloc_scaled, min=-1e5, max=1e5) ** 2) * critical_weight[None, :]
                )
            else:
                critical_colloc_phys_loss = torch.tensor(0.0, dtype=dtype)
        else:
            colloc_phys_loss = torch.tensor(0.0, dtype=dtype)
            critical_colloc_phys_loss = torch.tensor(0.0, dtype=dtype)
        colloc_weight = args.physics_weight if args.physics_collocation_weight is None else float(args.physics_collocation_weight)
        loss = (
            args.key_weight * key_loss
            + args.key_terminal_weight * key_terminal_loss
            + args.teacher_weight * teacher_loss
            + args.teacher_rhs_weight * teacher_rhs_loss
            + args.teacher_flux_weight * teacher_flux_loss
            + args.module_flux_weight * module_flux_loss
            + args.gtp_mechanism_weight * gtp_mechanism_loss
            + args.bottleneck_state_weight * bottleneck_state_loss
            + args.critical_bottleneck_weight * critical_bottleneck_loss
            + args.critical_terminal_weight * critical_terminal_loss
            + args.plateau_derivative_weight * plateau_derivative_loss
            + args.physics_weight * phys_loss
            + args.shared_physics_weight * shared_phys_loss
            + args.shared_key_physics_weight * shared_key_phys_loss
            + args.critical_physics_weight * critical_phys_loss
            + colloc_weight * colloc_phys_loss
            + args.critical_collocation_physics_weight * critical_colloc_phys_loss
            + args.integral_physics_weight * integral_phys_loss
            + args.critical_integral_physics_weight * critical_integral_phys_loss
            + args.shared_integral_physics_weight * shared_integral_phys_loss
            + args.shared_key_integral_physics_weight * shared_key_integral_phys_loss
        )
        if step < args.steps:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step in report_steps:
            history.append(
                {
                    "step": int(step),
                    "total_loss": float(loss.detach()),
                    "key_state_scaled_rmse": float(torch.sqrt(key_loss).detach()),
                    "key_terminal_scaled_rmse": float(torch.sqrt(key_terminal_loss).detach()),
                    "teacher_shared_trajectory_scaled_rmse": float(torch.sqrt(teacher_loss).detach()),
                    "teacher_rhs_derivative_scaled_rmse": float(torch.sqrt(teacher_rhs_loss).detach()),
                    "teacher_shared_flux_log10_rmse": float(torch.sqrt(teacher_flux_loss).detach()),
                    "perturbation_module_flux_log10_rmse": float(torch.sqrt(module_flux_loss).detach()),
                    "gtp_mechanism_shape_rmse": float(torch.sqrt(gtp_mechanism_loss).detach()),
                    "perturbation_bottleneck_state_scaled_rmse": float(torch.sqrt(bottleneck_state_loss).detach()),
                    "critical_bottleneck_weighted_scaled_rmse": float(torch.sqrt(critical_bottleneck_loss).detach()),
                    "critical_terminal_weighted_scaled_rmse": float(torch.sqrt(critical_terminal_loss).detach()),
                    "plateau_bottleneck_derivative_scaled_rmse": float(torch.sqrt(plateau_derivative_loss).detach()),
                    "physics_scaled_rmse_clipped": float(torch.sqrt(phys_loss).detach()),
                    "shared_physics_rhs_normalized_rmse": float(torch.sqrt(shared_phys_loss).detach()),
                    "shared_key_physics_rhs_normalized_rmse": float(torch.sqrt(shared_key_phys_loss).detach()),
                    "critical_physics_scaled_rmse_clipped": float(torch.sqrt(critical_phys_loss).detach()),
                    "collocation_physics_scaled_rmse_clipped": float(torch.sqrt(colloc_phys_loss).detach()),
                    "critical_collocation_physics_scaled_rmse_clipped": float(
                        torch.sqrt(critical_colloc_phys_loss).detach()
                    ),
                    "integral_physics_scaled_rmse_clipped": float(torch.sqrt(integral_phys_loss).detach()),
                    "critical_integral_physics_scaled_rmse_clipped": float(
                        torch.sqrt(critical_integral_phys_loss).detach()
                    ),
                    "shared_integral_physics_rhs_normalized_rmse": float(
                        torch.sqrt(shared_integral_phys_loss).detach()
                    ),
                    "shared_key_integral_physics_rhs_normalized_rmse": float(
                        torch.sqrt(shared_key_integral_phys_loss).detach()
                    ),
                    "negative_state_fraction": float(torch.mean((pred < -1e-18).to(dtype)).detach()),
                }
            )

    with torch.no_grad():
        pred, dpred_dt = time_derivative(model, time)
        if args.exchange_mode == "continuous":
            exchange, exchange_meta = build_exchange(time, pred, species_ids, args.continuous_xml, dtype)
        else:
            exchange = torch.zeros_like(pred)
            exchange_meta = {"gamma_h_inv": 0.0, "gamma_prod_h_inv": 0.0, "exchange_species_present": [], "exchange_species_missing": []}
        flux_pred = operator.flux(pred)
        rhs = torch.sparse.mm(operator.S, flux_pred.T).T + exchange
        key_pred = pred.index_select(1, key_indices) if key_indices.numel() > 0 else torch.empty((pred.shape[0], 0), dtype=dtype)
        module_pred, module_pred_names = module_fluxes(flux_pred, supervised_reaction_modules)
        full_module_pred, full_module_pred_names = module_fluxes(flux_pred, reaction_modules)

    pred_np = pred.detach().cpu().numpy()
    key_pred_np = key_pred.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    dpred_np = dpred_dt.detach().cpu().numpy()

    key_rows = []
    for j, name in enumerate(key_names):
        err = key_pred_np[:, j] - key_target_np[:, j]
        key_rows.append(
            {
                "species_id": name,
                "target_end_uM": float(key_target_np[-1, j] * 1e6),
                "pred_end_uM": float(key_pred_np[-1, j] * 1e6),
                "end_abs_error_uM": float(abs(err[-1]) * 1e6),
                "trajectory_scaled_rmse": float(np.sqrt(np.mean((err / max(key_scale[j].item(), 1e-30)) ** 2))),
            }
        )

    teacher_rows = []
    if teacher_available and len(teacher_names) > 0:
        pred_end_np = pred_np[-1, teacher_full_indices_np]
        err = pred_end_np - teacher_target_np
        residual_np = dpred_np - rhs_np
        dx_scale_np = dx_scale.detach().cpu().numpy()
        for sid, target_i, pred_i, err_i, full_i in zip(
            teacher_names, teacher_target_np, pred_end_np, err, teacher_full_indices_np
        ):
            teacher_rows.append(
                {
                    "species_id": sid,
                    "teacher_end_uM": float(target_i * 1e6),
                    "pred_end_uM": float(pred_i * 1e6),
                    "end_abs_error_uM": float(abs(err_i) * 1e6),
                    "relative_error_eps1pM": float(abs(err_i) / max(abs(target_i), 1e-12)),
                    "mass_action_residual_rms_uM_per_h": float(
                        np.sqrt(np.mean(residual_np[:, full_i] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_rhs_rms_uM_per_h": float(
                        np.sqrt(np.mean(rhs_np[:, full_i] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_autograd_dxdt_rms_uM_per_h": float(
                        np.sqrt(np.mean(dpred_np[:, full_i] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_residual_scaled_rmse": float(
                        np.sqrt(np.mean((residual_np[:, full_i] / dx_scale_np[full_i]) ** 2))
                    ),
                }
            )

    module_rows = []
    if module_flux_available and module_flux_target.numel() > 0:
        module_pred_np = module_pred.detach().cpu().numpy()
        module_target_np = module_flux_target.detach().cpu().numpy()
        for j, name in enumerate(module_target_names):
            err_log = np.log10(module_pred_np[:, j] + 1e-30) - np.log10(module_target_np[:, j] + 1e-30)
            module_rows.append(
                {
                    "module": name,
                    "reaction_count": int(len(reaction_modules_np.get(name, []))),
                    "teacher_supervised_reaction_count": int(supervised_reaction_modules[name].numel()),
                    "teacher_end_flux_M_per_s": float(module_target_np[-1, j]),
                    "pred_end_flux_M_per_s": float(module_pred_np[-1, j]),
                    "trajectory_log10_rmse": float(np.sqrt(np.mean(np.clip(err_log, -30.0, 30.0) ** 2))),
                    "teacher_integrated_flux_M": float(np.trapezoid(module_target_np[:, j], time_np)),
                    "pred_integrated_flux_M": float(np.trapezoid(module_pred_np[:, j], time_np)),
                }
            )

    full_module_rows = []
    if full_module_pred.numel() > 0:
        full_module_pred_np = full_module_pred.detach().cpu().numpy()
        for j, name in enumerate(full_module_pred_names):
            full_module_rows.append(
                {
                    "module": name,
                    "full_elementary_reaction_count": int(len(reaction_modules_np.get(name, []))),
                    "pred_end_flux_M_per_s": float(full_module_pred_np[-1, j]),
                    "pred_integrated_flux_M": float(np.trapezoid(full_module_pred_np[:, j], time_np)),
                    "interpretation": "full-elementary PINN module indicator; not necessarily teacher-supervised if full-only latent chain reactions are involved",
                }
            )

    gtp_mechanism_rows = []
    if gtp_mechanism_available and gtp_mechanism_target.numel() > 0:
        mech_proxy, mech_proxy_modules = module_fluxes(flux_pred, gtp_mechanism_modules)
        mech_proxy_np = mech_proxy.detach().cpu().numpy()
        proxy_by_module_np = {name: mech_proxy_np[:, i] for i, name in enumerate(mech_proxy_modules)}
        mech_target_np = gtp_mechanism_target.detach().cpu().numpy()
        for j, metric in enumerate(gtp_mechanism_names):
            module = GTP_MECHANISM_TO_FULL_MODULE[metric]
            pred_series = proxy_by_module_np[module]
            target_series = mech_target_np[:, j]
            pred_scale = max(float(np.max(np.abs(pred_series))), 1e-30)
            target_scale = max(float(np.max(np.abs(target_series))), 1e-30)
            pred_norm = pred_series / pred_scale
            target_norm = target_series / target_scale
            gtp_mechanism_rows.append(
                {
                    "mechanism_metric": metric,
                    "full_elementary_proxy_module": module,
                    "teacher_target_end": float(target_series[-1]),
                    "proxy_end_M_per_s": float(pred_series[-1]),
                    "teacher_norm_end": float(target_norm[-1]),
                    "proxy_norm_end": float(pred_norm[-1]),
                    "normalized_shape_rmse": float(np.sqrt(np.mean((pred_norm - target_norm) ** 2))),
                    "interpretation": "shape/bottleneck-switching loss; target and proxy are not treated as same absolute unit",
                }
            )

    bottleneck_rows = []
    if teacher_available and len(bottleneck_species_names) > 0:
        bottleneck_pred_np = pred_np[:, bottleneck_indices_np]
        residual_np = dpred_np - rhs_np
        dx_scale_np = dx_scale.detach().cpu().numpy()
        for j, sid in enumerate(bottleneck_species_names):
            err = bottleneck_pred_np[:, j] - bottleneck_teacher_np[:, j]
            plateau_err = (
                bottleneck_derivative_teacher_np[plateau_mask.detach().cpu().numpy(), j]
                if np.any(plateau_mask.detach().cpu().numpy())
                else np.array([])
            )
            bottleneck_rows.append(
                {
                    "species_id": sid,
                    "role": "perturbation-derived key bottleneck state",
                    "teacher_end_uM": float(bottleneck_teacher_np[-1, j] * 1e6),
                    "pred_end_uM": float(bottleneck_pred_np[-1, j] * 1e6),
                    "trajectory_scaled_rmse": float(
                        np.sqrt(np.mean((err / max(float(bottleneck_scale[j]), 1e-30)) ** 2))
                    ),
                    "critical_weight": float(critical_weight_np[j]),
                    "terminal_scaled_abs_error": float(
                        abs(err[-1]) / max(float(bottleneck_scale[j]), 1e-30)
                    ),
                    "teacher_plateau_derivative_max_abs_uM_per_h": float(
                        np.max(np.abs(plateau_err)) * 1e6 * 3600.0 if plateau_err.size else 0.0
                    ),
                    "mass_action_residual_rms_uM_per_h": float(
                        np.sqrt(np.mean(residual_np[:, bottleneck_indices_np[j]] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_rhs_rms_uM_per_h": float(
                        np.sqrt(np.mean(rhs_np[:, bottleneck_indices_np[j]] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_autograd_dxdt_rms_uM_per_h": float(
                        np.sqrt(np.mean(dpred_np[:, bottleneck_indices_np[j]] ** 2)) * 1e6 * 3600.0
                    ),
                    "mass_action_residual_scaled_rmse": float(
                        np.sqrt(np.mean((residual_np[:, bottleneck_indices_np[j]] / dx_scale_np[bottleneck_indices_np[j]]) ** 2))
                    ),
                }
            )

    result = {
        "scope": "continuous teacher/key-supervised PINN on full 5772-state/7061-reaction elementary CRN",
        "important_limitation": "teacher supervision is applied only to mapped shared species; final physics remains the full elementary CRN residual",
        "parameterization": "same as batch PINN except q_exchange is added; all internal fluxes fixed by full mass-action A/k and rhs fixed by full S",
        "steps": int(args.steps),
        "physics_weight": float(args.physics_weight),
        "shared_physics_weight": float(args.shared_physics_weight),
        "shared_key_physics_weight": float(args.shared_key_physics_weight),
        "shared_physics_rhs_floor_uM_per_h": float(args.shared_physics_rhs_floor_uM_per_h),
        "shared_integral_physics_weight": float(args.shared_integral_physics_weight),
        "shared_key_integral_physics_weight": float(args.shared_key_integral_physics_weight),
        "critical_physics_weight": float(args.critical_physics_weight),
        "key_weight": float(args.key_weight),
        "key_terminal_weight": float(args.key_terminal_weight),
        "key_terminal_scale_floor_uM": float(args.key_terminal_scale_floor_uM),
        "teacher_weight": float(args.teacher_weight),
        "teacher_loss_mode": args.teacher_loss_mode,
        "teacher_log_eps_M": float(args.teacher_log_eps_M),
        "teacher_train_max_s": None if args.teacher_train_max_s is None else float(args.teacher_train_max_s),
        "teacher_boost_regex": args.teacher_boost_regex,
        "teacher_boost_weight": float(args.teacher_boost_weight),
        "teacher_boosted_species_count": int(len(teacher_boosted_species)),
        "teacher_boosted_species_preview": teacher_boosted_species[:50],
        "load_checkpoint": str(args.load_checkpoint) if args.load_checkpoint is not None else None,
        "save_checkpoint": bool(args.save_checkpoint),
        "teacher_rhs_weight": float(args.teacher_rhs_weight),
        "teacher_rhs_scale_floor_uM_per_h": float(args.teacher_rhs_scale_floor_uM_per_h),
        "teacher_flux_weight": float(args.teacher_flux_weight),
        "module_flux_weight": float(args.module_flux_weight),
        "mechanism_csv": str(args.mechanism_csv) if args.mechanism_csv is not None else None,
        "gtp_mechanism_weight": float(args.gtp_mechanism_weight),
        "gtp_mechanism_metric_weights": dict(zip(gtp_mechanism_names, gtp_mechanism_metric_weights_list)),
        "teacher_flux_min_integrated_M": float(args.teacher_flux_min_integrated_M),
        "module_flux_min_integrated_M": float(args.module_flux_min_integrated_M),
        "bottleneck_state_weight": float(args.bottleneck_state_weight),
        "critical_bottleneck_weight": float(args.critical_bottleneck_weight),
        "critical_terminal_weight": float(args.critical_terminal_weight),
        "plateau_derivative_weight": float(args.plateau_derivative_weight),
        "plateau_start_h": float(args.plateau_start_h),
        "bottleneck_scale_floor_uM": float(args.bottleneck_scale_floor_uM),
        "physics_collocation_points_requested": int(args.physics_collocation_points),
        "physics_collocation_points_used": int(len(colloc_np)),
        "physics_collocation_weight": float(
            args.physics_weight if args.physics_collocation_weight is None else args.physics_collocation_weight
        ),
        "critical_collocation_physics_weight": float(args.critical_collocation_physics_weight),
        "integral_physics_weight": float(args.integral_physics_weight),
        "critical_integral_physics_weight": float(args.critical_integral_physics_weight),
        "integral_scale_floor_uM": float(args.integral_scale_floor_uM),
        "exchange_mode": args.exchange_mode,
        "network": {"width": args.width, "depth": args.depth, "dtype": "float64"},
        **exchange_meta,
        "history": history,
        "final_key_state_scaled_rmse": history[-1]["key_state_scaled_rmse"],
        "final_key_terminal_scaled_rmse": history[-1]["key_terminal_scaled_rmse"],
        "final_teacher_shared_trajectory_scaled_rmse": history[-1]["teacher_shared_trajectory_scaled_rmse"],
        "final_teacher_rhs_derivative_scaled_rmse": history[-1]["teacher_rhs_derivative_scaled_rmse"],
        "final_teacher_shared_flux_log10_rmse": history[-1]["teacher_shared_flux_log10_rmse"],
        "final_perturbation_module_flux_log10_rmse": history[-1]["perturbation_module_flux_log10_rmse"],
        "final_gtp_mechanism_shape_rmse": history[-1]["gtp_mechanism_shape_rmse"],
        "final_perturbation_bottleneck_state_scaled_rmse": history[-1]["perturbation_bottleneck_state_scaled_rmse"],
        "final_critical_bottleneck_weighted_scaled_rmse": history[-1]["critical_bottleneck_weighted_scaled_rmse"],
        "final_critical_terminal_weighted_scaled_rmse": history[-1]["critical_terminal_weighted_scaled_rmse"],
        "final_plateau_bottleneck_derivative_scaled_rmse": history[-1]["plateau_bottleneck_derivative_scaled_rmse"],
        "final_physics_scaled_rmse_clipped": history[-1]["physics_scaled_rmse_clipped"],
        "final_shared_physics_rhs_normalized_rmse": history[-1]["shared_physics_rhs_normalized_rmse"],
        "final_shared_key_physics_rhs_normalized_rmse": history[-1]["shared_key_physics_rhs_normalized_rmse"],
        "final_critical_physics_scaled_rmse_clipped": history[-1]["critical_physics_scaled_rmse_clipped"],
        "final_collocation_physics_scaled_rmse_clipped": history[-1]["collocation_physics_scaled_rmse_clipped"],
        "final_critical_collocation_physics_scaled_rmse_clipped": history[-1][
            "critical_collocation_physics_scaled_rmse_clipped"
        ],
        "final_integral_physics_scaled_rmse_clipped": history[-1]["integral_physics_scaled_rmse_clipped"],
        "final_critical_integral_physics_scaled_rmse_clipped": history[-1][
            "critical_integral_physics_scaled_rmse_clipped"
        ],
        "final_shared_integral_physics_rhs_normalized_rmse": history[-1][
            "shared_integral_physics_rhs_normalized_rmse"
        ],
        "final_shared_key_integral_physics_rhs_normalized_rmse": history[-1][
            "shared_key_integral_physics_rhs_normalized_rmse"
        ],
        "finite_predicted_state_fraction": float(np.mean(np.isfinite(pred_np))),
        "negative_predicted_state_fraction_lt_minus_1e_18": float(np.mean(pred_np < -1e-18)),
        "finite_rhs_fraction": float(np.mean(np.isfinite(rhs_np))),
        "finite_autograd_derivative_fraction": float(np.mean(np.isfinite(dpred_np))),
        "key_species_count": int(len(key_names)),
        "teacher_shared_species_count": int(len(teacher_names)),
        "teacher_flux_reaction_count": int(teacher_flux_indices.numel()),
        "perturbation_module_fluxes": {name: int(len(indices)) for name, indices in reaction_modules_np.items()},
        "gtp_mechanism_mapping": GTP_MECHANISM_TO_FULL_MODULE,
        "teacher_supervised_perturbation_module_fluxes": {
            name: int(indices.numel()) for name, indices in supervised_reaction_modules.items()
        },
        "perturbation_bottleneck_species": bottleneck_species_names,
        "shared_key_physics_species": shared_key_species_names,
        "teacher_supervision_mode": "trajectory" if args.teacher_trajectory is not None else ("endpoint" if teacher_available else "none"),
        "time_end_h": float(time_np[-1] / 3600.0),
    }

    outdir = assets
    prefix = args.output_prefix
    (outdir / f"{prefix}_training_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(key_rows).to_csv(outdir / f"{prefix}_key_species_fit.csv", index=False)
    if teacher_rows:
        pd.DataFrame(teacher_rows).to_csv(outdir / f"{prefix}_teacher_shared_fit.csv", index=False)
    if module_rows:
        pd.DataFrame(module_rows).to_csv(outdir / f"{prefix}_perturbation_module_flux_fit.csv", index=False)
    if full_module_rows:
        pd.DataFrame(full_module_rows).to_csv(outdir / f"{prefix}_full_elementary_module_flux_indicators.csv", index=False)
    if gtp_mechanism_rows:
        pd.DataFrame(gtp_mechanism_rows).to_csv(outdir / f"{prefix}_gtp_mechanism_partition_fit.csv", index=False)
    if bottleneck_rows:
        pd.DataFrame(bottleneck_rows).to_csv(outdir / f"{prefix}_perturbation_bottleneck_state_fit.csv", index=False)
    np.savez_compressed(outdir / f"{prefix}_prediction.npz", time_s=time_np, prediction_M=pred_np)
    if args.save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "output_prefix": prefix,
                "time_s": time_np,
                "species_ids": species_ids,
                "args": vars(args),
            },
            outdir / f"{prefix}_checkpoint.pt",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
