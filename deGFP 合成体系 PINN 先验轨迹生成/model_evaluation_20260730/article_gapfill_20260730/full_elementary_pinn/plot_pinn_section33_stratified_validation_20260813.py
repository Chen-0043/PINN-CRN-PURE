#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for manuscript section 3.3/3.4.3 PINN validation and residual stratification."""

from __future__ import annotations

from pathlib import Path
import os
import re
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "article_33_20260813"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV = (
    ROOT
    / "continuous_6h_existing_teacher_plus_ribosome_mrna_gtp3mech_pinn_3500_noAA_phys1e7_w0p1_continuous_all_species_6h_validation.csv"
)
TOP_ERROR_CSV = ROOT / "continuous_6h_corrected_gtpresidual_pretrain_all3151_residual_diagnostic_teacher_shared_fit.csv"
PRIOR_NPZ = ROOT / "continuous_6h_trackpriority_tol15_finetune_200_prediction.npz"
CORRECTED_NPZ = ROOT / "continuous_6h_trackpriority_200_shooting_continue_add300_21300_1000_shooting_nodes.npz"
MODEL_DIR = ROOT.parent / "crn_constraint_map"
AXIS_COMPRESS_U_M = 1.0
DISPLAY_INTERVAL_QUANTILE = 0.95


def nice_symmetric_limit(axis_limit: float) -> float:
    if axis_limit < 1:
        return max(float(np.ceil(axis_limit * 10) / 10), axis_limit)
    magnitude = 10 ** np.floor(np.log10(axis_limit))
    return float(np.ceil(axis_limit / magnitude) * magnitude)


def compressed_change(v):
    return np.arcsinh(np.asarray(v, dtype=float) / AXIS_COMPRESS_U_M)


def compressed_ticks(max_abs):
    candidates = np.array([-1000, -100, -10, -1, 0, 1, 10, 100, 1000], dtype=float)
    ticks = candidates[np.abs(candidates) <= max_abs * 1.0001]
    if 0.0 not in ticks:
        ticks = np.append(ticks, 0.0)
    return np.sort(ticks)


def change_tick_label(v):
    if abs(v) < 1e-12:
        return "0"
    if abs(v) >= 100:
        return f"{int(v):d}"
    if abs(v) >= 1:
        return f"{v:g}"
    if abs(v) >= 0.01:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


FINAL_TEXT_METRICS = {
    "n_comparable": 3151,
    "mre_pct": 23.57,
    "r2": 0.9771,
    "prior_nrmsr_pct": 105.90,
    "corrected_nrmsr_pct": 47.76,
    "filtered_nrmsr_pct": 31.00,
    "initial_rms_uM": 0.689,
    "corrected_rms_uM": 0.203,
    "high_turnover_n": 13,
    "remaining_n": 41,
    "all_track_n": 54,
}

HIGH_TURNOVER_CATEGORIES = ["GTP核苷酸循环", "mRNA中间态", "能量再生", "氨酰化中间态"]


def parse_parameters(xml_path: Path) -> dict[str, float]:
    out = {}
    if not xml_path.exists():
        return out
    for m in re.finditer(r'<parameter id="([^"]+)" value="([^"]+)"', xml_path.read_text(errors="ignore")):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def setup_font() -> None:
    for path in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        p = Path(path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(p)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def log_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = df["teacher_6h_uM"].clip(lower=1e-9).to_numpy()
    y = df["pinn_6h_uM"].clip(lower=1e-9).to_numpy()

    ax.scatter(
        x,
        y,
        color="#4E8FB8",
        s=10,
        alpha=0.64,
        linewidths=0,
        rasterized=True,
    )
    lim = (1e-11, max(x.max(), y.max()) * 8.0)
    ax.plot(lim, lim, color="#111827", lw=1.2, ls="--", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    major_ticks = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5]
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"1e{int(np.log10(v))}" if v in major_ticks else ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"1e{int(np.log10(v))}" if v in major_ticks else ""))
    ax.set_xlabel("教师模型浓度 (μM)")
    ax.set_ylabel("CRN-PINN 预测浓度 (μM)")
    ax.set_title("A  6 h 验证集浓度重构", loc="left", fontsize=13, pad=8)
    ax.set_box_aspect(1.0)
    ax.text(
        0.04,
        0.96,
        f"n = {FINAL_TEXT_METRICS['n_comparable']}\nMRE = {FINAL_TEXT_METRICS['mre_pct']:.2f}%\nR2 = {FINAL_TEXT_METRICS['r2']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c7d2df", lw=0.8, alpha=0.95),
    )


def bar_panel(ax: plt.Axes) -> None:
    labels = ["先验轨迹\n91条", "物理校正后\n54条", "剔除高周转13条后\n41条"]
    values = [
        FINAL_TEXT_METRICS["prior_nrmsr_pct"],
        FINAL_TEXT_METRICS["corrected_nrmsr_pct"],
        FINAL_TEXT_METRICS["filtered_nrmsr_pct"],
    ]
    colors = ["#B65F4A", "#4E8FB8", "#2E9A72"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="none", width=0.58, edgecolor="white", linewidth=1.2, zorder=3)

    def draw_gradient_bar(bar, color: str) -> None:
        rgba = to_rgba(color)
        x0 = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        n_segments = 42
        for i in range(n_segments):
            y0 = height * i / n_segments
            segment_h = height / n_segments * 1.02
            frac = (i + 1) / n_segments
            alpha = 0.18 + 0.78 * (frac ** 1.18)
            lighten = 0.55 * (1.0 - frac)
            face = (
                rgba[0] + (1.0 - rgba[0]) * lighten,
                rgba[1] + (1.0 - rgba[1]) * lighten,
                rgba[2] + (1.0 - rgba[2]) * lighten,
                alpha,
            )
            ax.add_patch(Rectangle((x0, y0), width, segment_h, facecolor=face, edgecolor="none", zorder=2.8))
        ax.add_patch(Rectangle((x0, 0), width, height, facecolor="none", edgecolor="white", linewidth=1.2, zorder=3.2))

    for b, color in zip(bars, colors):
        draw_gradient_bar(b, color)

    ax.set_ylabel("NRMSR / 平均相对物理积分误差 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 120)
    ax.set_title("B  物理轨迹偏差的分层统计", loc="left", fontsize=13, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3.0, f"{v:.2f}%", ha="center", va="bottom", fontsize=11)


def trajectory_integral_detail(npz_path: Path, state_key: str) -> pd.DataFrame:
    """Return per-shared-species integral dynamics scales for a saved trajectory."""
    z = np.load(npz_path)
    x = np.maximum(z[state_key].astype(float), 0.0)
    time_s = z["time_s"].astype(float)
    species = pd.read_csv(ROOT / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    mapping = pd.read_csv(ROOT.parent / "teacher_to_full_mapping" / "full_species_teacher_mapping.csv")
    shared = set(mapping.loc[mapping["teacher_supervision_available"], "species_index_full"].astype(int))
    shared_idx = np.array(sorted(shared), dtype=int)

    stoich = sparse.load_npz(ROOT / "stoichiometric_matrix_S.npz").tocsr()
    reactants = sparse.load_npz(ROOT / "reactant_exponent_matrix_A.npz").tocsr()
    k = np.load(ROOT / "rate_constants_k.npy").astype(float)
    log_x = np.log(np.maximum(x, 1e-30))
    flux = np.exp(np.clip((reactants.T @ log_x.T).T + np.log(k)[None, :], -100, 100))

    params = parse_parameters(MODEL_DIR / "formal_model.xml")
    gamma = float(params.get("bg_gamma", 0.5 / 3600.0))
    gamma_prod = float(params.get("bg_gamma_prod", gamma))
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    exchange = np.zeros_like(x)
    for key, value in params.items():
        if key.startswith("bg_feed_") and key != "bg_feed_deGFP_m":
            sid = key.removeprefix("bg_feed_")
            if sid != "AA" and sid in lookup:
                exchange[:, lookup[sid]] += gamma * (value - x[:, lookup[sid]])
    if "deGFP_m" in lookup:
        exchange[:, lookup["deGFP_m"]] += gamma_prod * (0.0 - x[:, lookup["deGFP_m"]])

    rhs = (stoich @ flux.T).T + exchange
    dt = np.diff(time_s)[:, None]
    rhs_trap = 0.5 * dt * (rhs[1:] + rhs[:-1])
    delta = x[1:] - x[:-1]
    residual = delta - rhs_trap

    rhs_rms = np.sqrt(np.mean(rhs_trap[:, shared_idx] ** 2, axis=0)) * 1e6
    delta_rms = np.sqrt(np.mean(delta[:, shared_idx] ** 2, axis=0)) * 1e6
    residual_rms = np.sqrt(np.mean(residual[:, shared_idx] ** 2, axis=0)) * 1e6
    rel_resid = residual_rms / np.maximum(rhs_rms, 1e-12)
    return pd.DataFrame(
        {
            "species_id": [species_ids[i] for i in shared_idx],
            "rhs_rms_uM": rhs_rms,
            "delta_rms_uM": delta_rms,
            "residual_rms_uM": residual_rms,
            "relative_residual": rel_resid,
            "x_log": np.log10(np.maximum(rhs_rms, 1e-12)),
            "y_log": np.log10(np.maximum(delta_rms, 1e-12)),
        }
    )


def trajectory_point_detail(npz_path: Path, state_key: str) -> pd.DataFrame:
    """Return point-level signed physical and PINN adjacent changes."""
    z = np.load(npz_path)
    x = np.maximum(z[state_key].astype(float), 0.0)
    time_s = z["time_s"].astype(float)
    species = pd.read_csv(ROOT / "species_index.csv").sort_values("species_index")
    species_ids = species["species_id"].astype(str).tolist()
    mapping = pd.read_csv(ROOT.parent / "teacher_to_full_mapping" / "full_species_teacher_mapping.csv")
    shared = set(mapping.loc[mapping["teacher_supervision_available"], "species_index_full"].astype(int))
    shared_idx = np.array(sorted(shared), dtype=int)
    shared_species = [species_ids[i] for i in shared_idx]

    stoich = sparse.load_npz(ROOT / "stoichiometric_matrix_S.npz").tocsr()
    reactants = sparse.load_npz(ROOT / "reactant_exponent_matrix_A.npz").tocsr()
    k = np.load(ROOT / "rate_constants_k.npy").astype(float)
    log_x = np.log(np.maximum(x, 1e-30))
    flux = np.exp(np.clip((reactants.T @ log_x.T).T + np.log(k)[None, :], -100, 100))

    params = parse_parameters(MODEL_DIR / "formal_model.xml")
    gamma = float(params.get("bg_gamma", 0.5 / 3600.0))
    gamma_prod = float(params.get("bg_gamma_prod", gamma))
    lookup = {sid: i for i, sid in enumerate(species_ids)}
    exchange = np.zeros_like(x)
    for key, value in params.items():
        if key.startswith("bg_feed_") and key != "bg_feed_deGFP_m":
            sid = key.removeprefix("bg_feed_")
            if sid != "AA" and sid in lookup:
                exchange[:, lookup[sid]] += gamma * (value - x[:, lookup[sid]])
    if "deGFP_m" in lookup:
        exchange[:, lookup["deGFP_m"]] += gamma_prod * (0.0 - x[:, lookup["deGFP_m"]])

    rhs = (stoich @ flux.T).T + exchange
    dt = np.diff(time_s)[:, None]
    rhs_trap = 0.5 * dt * (rhs[1:] + rhs[:-1])
    delta = x[1:] - x[:-1]
    rhs_uM = rhs_trap[:, shared_idx] * 1e6
    delta_uM = delta[:, shared_idx] * 1e6
    return pd.DataFrame(
        {
            "species_id": np.tile(shared_species, rhs_uM.shape[0]),
            "x_change_uM": rhs_uM.reshape(-1),
            "y_change_uM": delta_uM.reshape(-1),
        }
    )


def t_marker3d(ax: plt.Axes, x0: float, y0: float, height: float, color: str, cap_width: float) -> None:
    ax.plot([x0, x0], [y0, y0], [0, height], color="white", lw=4.0, alpha=0.80, solid_capstyle="round", zorder=15)
    ax.plot([x0, x0], [y0, y0], [0, height], color=color, lw=2.2, alpha=0.98, solid_capstyle="round", zorder=16)
    ax.plot(
        [x0 - cap_width / 2, x0 + cap_width / 2],
        [y0, y0],
        [height, height],
        color="white",
        lw=5.2,
        alpha=0.82,
        solid_capstyle="round",
        zorder=17,
    )
    ax.plot(
        [x0 - cap_width / 2, x0 + cap_width / 2],
        [y0, y0],
        [height, height],
        color=color,
        lw=3.0,
        alpha=0.98,
        solid_capstyle="round",
        zorder=18,
    )


def dynamics_3d_panel(ax: plt.Axes) -> None:
    prior = trajectory_integral_detail(PRIOR_NPZ, "prediction_M")
    corrected = trajectory_integral_detail(CORRECTED_NPZ, "optimized_M")

    prior_91 = prior.nlargest(FINAL_TEXT_METRICS["all_track_n"] + 37, "rhs_rms_uM")
    corrected_54 = corrected.nlargest(FINAL_TEXT_METRICS["all_track_n"], "rhs_rms_uM").copy()
    high13_ids = set(corrected_54.nlargest(FINAL_TEXT_METRICS["high_turnover_n"], "relative_residual")["species_id"])
    corrected_41 = corrected_54.loc[~corrected_54["species_id"].isin(high13_ids)].copy()
    prior_points = trajectory_point_detail(PRIOR_NPZ, "prediction_M")
    corrected_points = trajectory_point_detail(CORRECTED_NPZ, "optimized_M")

    corrected_54_ids = set(corrected_54["species_id"])
    corrected_41_ids = set(corrected_41["species_id"])
    prior_91_ids = set(prior_91["species_id"])
    corrected_points["group"] = "background"
    corrected_points.loc[corrected_points["species_id"].isin(corrected_54_ids), "group"] = "corrected54"
    corrected_points.loc[corrected_points["species_id"].isin(corrected_41_ids), "group"] = "corrected41"

    bg = corrected_points.loc[corrected_points["group"] == "background"]
    c54_only = corrected_points.loc[corrected_points["group"] == "corrected54"]
    c41 = corrected_points.loc[corrected_points["group"] == "corrected41"]
    c54_all = corrected_points.loc[corrected_points["species_id"].isin(corrected_54_ids)]
    prior_91_points = prior_points.loc[prior_points["species_id"].isin(prior_91_ids)]
    # Use the same prior-trajectory 95% window as the pre-correction figure so
    # the shared 91 prior trajectories are shown on the same x/y scale.
    finite_changes = prior_points[["x_change_uM", "y_change_uM"]].to_numpy(dtype=float).reshape(-1)
    finite_changes = finite_changes[np.isfinite(finite_changes)]
    lower_q = (1.0 - DISPLAY_INTERVAL_QUANTILE) / 2.0
    upper_q = 1.0 - lower_q
    display_min = float(np.nanquantile(finite_changes, lower_q))
    display_max = float(np.nanquantile(finite_changes, upper_q))
    axis_limit = max(abs(display_min), abs(display_max), 1e-6) * 1.18
    corrected_points = corrected_points.loc[
        corrected_points["x_change_uM"].between(-axis_limit, axis_limit)
        & corrected_points["y_change_uM"].between(-axis_limit, axis_limit)
    ].copy()
    prior_points = prior_points.loc[
        prior_points["x_change_uM"].between(-axis_limit, axis_limit)
        & prior_points["y_change_uM"].between(-axis_limit, axis_limit)
    ].copy()
    corrected_points["x_plot"] = corrected_points["x_change_uM"]
    corrected_points["y_plot"] = corrected_points["y_change_uM"]
    prior_points["x_plot"] = prior_points["x_change_uM"]
    prior_points["y_plot"] = prior_points["y_change_uM"]
    tick_limit = nice_symmetric_limit(axis_limit)
    raw_ticks = np.linspace(-tick_limit, tick_limit, 5)
    xy_min = -tick_limit
    xy_max = tick_limit

    bg = corrected_points.loc[corrected_points["group"] == "background"]
    c54_only = corrected_points.loc[corrected_points["group"] == "corrected54"]
    c41 = corrected_points.loc[corrected_points["group"] == "corrected41"]
    c54_all = corrected_points.loc[corrected_points["species_id"].isin(corrected_54_ids)]
    prior_91_points = prior_points.loc[prior_points["species_id"].isin(prior_91_ids)]
    ax.scatter(c54_only["x_plot"], c54_only["y_plot"], np.zeros(len(c54_only)), s=5.0, c="#4E8FB8", alpha=0.58, linewidths=0, depthshade=False, rasterized=True)
    ax.scatter(c41["x_plot"], c41["y_plot"], np.zeros(len(c41)), s=6.0, c="#2E9A72", alpha=0.68, linewidths=0, depthshade=False, rasterized=True)
    ax.scatter(prior_91_points["x_plot"], prior_91_points["y_plot"], np.zeros(len(prior_91_points)), s=3.2, c="#B65F4A", alpha=0.24, linewidths=0, depthshade=False, rasterized=True)

    diagonal = np.linspace(xy_min, xy_max, 100)
    ax.plot(diagonal, diagonal, np.zeros_like(diagonal), "--", color="#111827", lw=1.2, alpha=0.85)

    bar_specs = [
        (prior_91_points, "#B65F4A", FINAL_TEXT_METRICS["prior_nrmsr_pct"], "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的91条先验轨迹"),
        (c54_all, "#4E8FB8", FINAL_TEXT_METRICS["corrected_nrmsr_pct"], "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的54条校正轨迹"),
        (c41, "#2E9A72", FINAL_TEXT_METRICS["filtered_nrmsr_pct"], "剔除高周转轴后的41条校正轨迹"),
    ]
    label_specs = []
    axis_span = xy_max - xy_min
    cap_width = 0.080 * axis_span
    for group, color, height, label in bar_specs:
        x0 = float(group["x_plot"].mean())
        y0 = float(group["y_plot"].mean())
        t_marker3d(ax, x0, y0, height, color, cap_width)
        label_specs.append((x0, y0, height, label))

    value_label_offsets = {
        "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的91条先验轨迹": (0.000 * axis_span, -0.014 * axis_span, 5.0),
        "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的54条校正轨迹": (0.050 * axis_span, 0.020 * axis_span, 7.0),
        "剔除高周转轴后的41条校正轨迹": (-0.044 * axis_span, -0.012 * axis_span, 6.0),
    }
    for x0, y0, height, label in label_specs:
        label_dx, label_dy, label_dz = value_label_offsets[label]
        value_text = ax.text(
            x0 + label_dx,
            y0 + label_dy,
            height + label_dz,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color="#334155",
            zorder=1000,
        )
        value_text.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white", alpha=0.92)])

    ax.set_xlabel(r"$\int_t^{t+\Delta t} R_i(\tau)d\tau$ ($\mu$M)", labelpad=8, fontsize=12)
    ax.set_ylabel(r"$\Delta C_i^{\mathrm{PINN}}$ ($\mu$M)", labelpad=10, fontsize=12)
    ax.set_zlabel("NRMSR (%)", labelpad=10, fontsize=12)
    ax.zaxis.set_rotate_label(True)
    ax.zaxis.label.set_clip_on(False)
    ax.set_title("(a)", loc="left", fontsize=17, pad=8)
    ax.set_xlim(xy_max, xy_min)
    ax.set_ylim(xy_min, xy_max)
    ax.set_xticks(raw_ticks)
    ax.set_yticks(raw_ticks)
    ax.set_xticklabels([change_tick_label(t) for t in raw_ticks])
    ax.set_yticklabels([change_tick_label(t) for t in raw_ticks])
    ax.view_init(elev=23, azim=-45)
    ax.set_zlim(0, 120)
    ax.set_zticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_box_aspect((1.0, 1.0, 0.72))
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)


def add_stratified_legend_box(fig: plt.Figure, ax: plt.Axes) -> None:
    """Draw a compact 3D legend sized by the text itself."""
    pos = ax.get_position()
    fig.text(
        pos.x0 + 0.61 * pos.width,
        pos.y0 + 0.890 * pos.height,
        r"红棕：$s_i^{\mathrm{rhs}}>1\,\mu$M 的91条先验轨迹" "\n"
        r"蓝色：$s_i^{\mathrm{rhs}}>1\,\mu$M 的54条校正轨迹" "\n"
        "绿色：剔除高周转轴后的41条校正轨迹",
        ha="left",
        va="top",
        fontsize=8.2,
        linespacing=1.12,
        color="#1f2937",
        zorder=1001,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cbd5e1", lw=0.8, alpha=0.86),
    )


def strata_panel(ax: plt.Axes, residual_df: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_title("C  偏差集中于高周转资源轴", loc="left", fontsize=13, pad=8)

    # Main track-count strip.
    left, bottom, width, height = 0.06, 0.76, 0.88, 0.12
    high_w = width * FINAL_TEXT_METRICS["high_turnover_n"] / FINAL_TEXT_METRICS["all_track_n"]
    remain_w = width - high_w
    ax.add_patch(Rectangle((left, bottom), high_w, height, color="#C95B55", transform=ax.transAxes))
    ax.add_patch(Rectangle((left + high_w, bottom), remain_w, height, color="#5AAE8A", transform=ax.transAxes))
    ax.text(left + high_w / 2, bottom + height / 2, "13条\n高周转轴", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="white")
    ax.text(left + high_w + remain_w / 2, bottom + height / 2, "41条\n其余轨迹", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="white")
    ax.text(left, bottom + height + 0.045, "sirhs > 1 μM 的54条物理显著轨迹", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="#334155")

    ax.add_patch(
        FancyArrowPatch(
            (left + 0.16, 0.69),
        (left + 0.16, 0.61),
            arrowstyle="-|>",
            mutation_scale=13,
            color="#64748b",
            lw=1.0,
            transform=ax.transAxes,
        )
    )
    ax.text(0.25, 0.63, "剔除集中偏差轨迹后", transform=ax.transAxes, ha="left", va="center", fontsize=11, color="#334155")
    ax.text(
        0.57,
        0.56,
        "其余41条轨迹  31.00%",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#166534",
        fontweight="bold",
    )

    y0 = 0.32
    ax.text(0.06, y0 + 0.13, "13条高偏差轨迹主要来源", transform=ax.transAxes, fontsize=11.5, color="#334155", fontweight="bold")
    for i, name in enumerate(HIGH_TURNOVER_CATEGORIES):
        y = y0 - i * 0.075
        ax.add_patch(Rectangle((0.065, y - 0.014), 0.018, 0.028, color="#C95B55", alpha=0.85, transform=ax.transAxes))
        ax.text(0.095, y, name, transform=ax.transAxes, ha="left", va="center", fontsize=10.5, color="#111827")

    if not residual_df.empty:
        top = residual_df.sort_values("mass_action_residual_scaled_rmse", ascending=False).head(6)
        names = [str(s)[:18] for s in top["species_id"]]
        vals = np.log10(top["mass_action_residual_scaled_rmse"].clip(lower=1).to_numpy())
        x0, yb, w, h = 0.66, 0.05, 0.29, 0.28
        inset = ax.inset_axes([x0, yb, w, h])
        inset.barh(np.arange(len(vals)), vals, color="#B65F4A", alpha=0.86)
        inset.set_yticks(np.arange(len(vals)))
        inset.set_yticklabels(names, fontsize=7.8)
        inset.invert_yaxis()
        inset.set_xlabel("log10 scaled residual", fontsize=8)
        inset.set_title("残差最高物种示例", fontsize=9, pad=3)
        inset.tick_params(axis="x", labelsize=8)
        inset.grid(axis="x", color="#e5e7eb", lw=0.6)
        inset.spines[["top", "right"]].set_visible(False)


def main() -> None:
    setup_font()
    validation = pd.read_csv(VALIDATION_CSV)

    fig = plt.figure(figsize=(15.2, 7.3), dpi=220)
    gs = GridSpec(1, 2, width_ratios=[1.20, 0.78], wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])

    dynamics_3d_panel(ax0)
    log_panel(ax1, validation)
    ax1.set_title("(b)", loc="left", fontsize=17, pad=8)

    fig.suptitle("CRN-PINN连续PURE系统验证与物理偏差分层", fontsize=18, y=0.985)
    note = (
        "注：a图散点为所有相邻时间区间带符号变化量分布的中央95%显示窗口，第一根 T 形标记为 $s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的91条先验轨迹的NRMSR，后两根 T 形标记分别为 $s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的54条校正轨迹及剔除高周转轴后的41条校正轨迹。"
        "b图散点使用最新可用的3151个可比物种6 h验证输出。"
    )
    fig.text(0.5, 0.015, "\n".join(textwrap.wrap(note, width=118)), ha="center", va="bottom", fontsize=9.5, color="#64748b")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.12, wspace=0.38)
    add_stratified_legend_box(fig, ax0)

    stem = FIG_DIR / "figure_section33_pinn_validation_residual_stratification_20260813"
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.18)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
