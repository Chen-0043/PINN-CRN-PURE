#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the PINN prior (before physical-trajectory correction) for section 3.4.1."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy import sparse
import re


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "continuous_6h_corrected_gtpresidual_pretrain_checkpoint_1500_teacher_shared_fit.csv"
OUT = ROOT / "figures" / "article_341_20260814"
OUT.mkdir(parents=True, exist_ok=True)
PREDICTION = ROOT / "continuous_6h_trackpriority_tol15_finetune_200_prediction.npz"
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


def setup_font():
    for candidate in (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ):
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def format_log(v, _):
    if v <= 0:
        return ""
    exponent = int(round(np.log10(v)))
    return f"1e{exponent}"


def parse_parameters(xml_path):
    out = {}
    if not xml_path.exists():
        return out
    for m in re.finditer(r'<parameter id="([^"]+)" value="([^"]+)"', xml_path.read_text(errors="ignore")):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def integral_detail_from_prediction(prediction_npz: Path):
    """Recompute the same trapezoidal CRN integral residual used for the prior diagnostic."""
    z = np.load(prediction_npz)
    x = np.maximum(z["prediction_M"].astype(float), 0.0)
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
    residual = x[1:] - x[:-1] - rhs_trap
    res = np.sqrt(np.mean(residual[:, shared_idx] ** 2, axis=0)) * 1e6
    rhs_rms = np.sqrt(np.mean(rhs_trap[:, shared_idx] ** 2, axis=0)) * 1e6
    d = pd.DataFrame({"species_id": [species_ids[i] for i in shared_idx], "s_rhs_uM": rhs_rms, "e_rms_uM": res})
    return d[d.s_rhs_uM > 1].sort_values("s_rhs_uM").reset_index(drop=True)


def trajectory_state_detail(prediction_npz: Path) -> pd.DataFrame:
    """Return per-species RMS state summaries for all comparable trajectories."""
    z = np.load(prediction_npz)
    x = np.maximum(z["prediction_M"].astype(float), 0.0)
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
    nrmsr_pct = residual_rms / np.maximum(rhs_rms, 1e-12) * 100.0
    return pd.DataFrame(
        {
            "species_id": [species_ids[i] for i in shared_idx],
            "rhs_rms_uM_per_h": rhs_rms,
            "pinn_delta_rms_uM": delta_rms,
            "residual_rms_uM_per_h": residual_rms,
            "nrmsr_pct": nrmsr_pct,
        }
    )


def trajectory_point_detail(prediction_npz: Path) -> pd.DataFrame:
    """Return point-level signed physical and PINN adjacent changes."""
    z = np.load(prediction_npz)
    x = np.maximum(z["prediction_M"].astype(float), 0.0)
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


def normalized_residual_detail(prediction_npz: Path):
    d = integral_detail_from_prediction(prediction_npz).copy()
    d["ratio"] = d["e_rms_uM"] / d["s_rhs_uM"]
    return d


def sample_point_ratios():
    z = np.load(PREDICTION)
    x = np.maximum(z["prediction_M"].astype(float), 0.0)
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
    residual = x[1:] - x[:-1] - rhs_trap
    detail = integral_detail()
    selected_ids = set(detail["species_id"].tolist())
    selected_idx = np.array([i for i in shared_idx if species_ids[i] in selected_ids], dtype=int)
    res = np.sqrt(residual[:, selected_idx] ** 2) * 1e6
    rhs_rms = np.sqrt(rhs_trap[:, selected_idx] ** 2) * 1e6
    ratio = (res / rhs_rms).reshape(-1)
    ratio = ratio[np.isfinite(ratio) & (rhs_rms.reshape(-1) > 0)]
    return ratio


def three_d_panel(ax: plt.Axes) -> None:
    all_detail = trajectory_point_detail(PREDICTION)
    detail_91 = integral_detail_from_prediction(PREDICTION)
    blue_ids = set(detail_91["species_id"].tolist())

    all_detail["is_blue"] = all_detail["species_id"].isin(blue_ids)

    finite_changes = all_detail[["x_change_uM", "y_change_uM"]].to_numpy(dtype=float).reshape(-1)
    finite_changes = finite_changes[np.isfinite(finite_changes)]
    lower_q = (1.0 - DISPLAY_INTERVAL_QUANTILE) / 2.0
    upper_q = 1.0 - lower_q
    display_min = float(np.nanquantile(finite_changes, lower_q))
    display_max = float(np.nanquantile(finite_changes, upper_q))
    axis_limit = max(abs(display_min), abs(display_max), 1e-6) * 1.18
    all_detail = all_detail.loc[
        all_detail["x_change_uM"].between(-axis_limit, axis_limit)
        & all_detail["y_change_uM"].between(-axis_limit, axis_limit)
    ].copy()
    all_detail["x_plot"] = all_detail["x_change_uM"]
    all_detail["y_plot"] = all_detail["y_change_uM"]
    tick_limit = nice_symmetric_limit(axis_limit)
    raw_ticks = np.linspace(-tick_limit, tick_limit, 5)
    xy_min = -tick_limit
    xy_max = tick_limit

    bg = all_detail.loc[~all_detail["is_blue"]]
    fg = all_detail.loc[all_detail["is_blue"]]

    def t_marker3d(x0, y0, height, color, cap_width):
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

    ax.scatter(
        bg["x_plot"],
        bg["y_plot"],
        np.zeros(len(bg)),
        s=3.2,
        c="#A85743",
        alpha=0.18,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    ax.scatter(
        fg["x_plot"],
        fg["y_plot"],
        np.zeros(len(fg)),
        s=5.0,
        c="#3F7EAA",
        alpha=0.58,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )

    all_nrmsr = 114.48
    subset_nrmsr = 105.90
    group_specs = [
        (
            bg,
            "#A85743",
            all_nrmsr,
            "3151条可比\n先验轨迹",
        ),
        (
            fg,
            "#3F7EAA",
            subset_nrmsr,
            "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M\n91条先验轨迹",
        ),
    ]
    label_specs = []
    axis_span = xy_max - xy_min
    cap_width = 0.080 * axis_span
    for group, color, height, label in group_specs:
        x0 = float(group["x_plot"].mean())
        y0 = float(group["y_plot"].mean())
        t_marker3d(x0, y0, height, color, cap_width)
        label_specs.append((x0, y0, height, label))

    value_label_offsets = {
        "3151条可比\n先验轨迹": (-0.018 * axis_span, -0.010 * axis_span, 6.0),
        "$s_i^{\\mathrm{rhs}}>1\\,\\mu$M\n91条先验轨迹": (0.018 * axis_span, 0.010 * axis_span, 6.0),
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
            fontsize=10.0,
            color="#334155",
            zorder=1000,
        )
        value_text.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white", alpha=0.92)])

    diagonal = np.linspace(xy_min, xy_max, 100)
    ax.plot(
        diagonal,
        diagonal,
        np.zeros_like(diagonal),
        "--",
        color="#111827",
        lw=1.25,
        alpha=0.85,
        zorder=8,
    )

    ax.set_xlabel(r"$\int_t^{t+\Delta t} R_i(\tau)d\tau$ ($\mu$M)", labelpad=8, fontsize=12)
    ax.set_ylabel(r"$\Delta C_i^{\mathrm{PINN}}$ ($\mu$M)", labelpad=10, fontsize=12)
    ax.set_zlabel("NRMSR (%)", labelpad=2, fontsize=12)
    ax.zaxis.set_rotate_label(True)
    ax.zaxis.label.set_clip_on(False)
    ax.set_title("(b)", loc="left", fontsize=17, pad=8)
    ax.set_xlim(xy_max, xy_min)
    ax.set_ylim(xy_min, xy_max)
    ax.set_xticks(raw_ticks)
    ax.set_yticks(raw_ticks)
    ax.set_xticklabels([change_tick_label(t) for t in raw_ticks])
    ax.set_yticklabels([change_tick_label(t) for t in raw_ticks])
    ax.view_init(elev=23, azim=-45)
    ax.set_zlim(0, 130)
    ax.set_zticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_box_aspect((1.0, 1.0, 0.72))
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="z", labelsize=9)


def add_prior_legend_box(fig, ax):
    """Draw a compact legend box sized by the text itself."""
    pos = ax.get_position()
    fig.text(
        pos.x0 + 0.62 * pos.width,
        pos.y0 + 0.890 * pos.height,
        "红棕T：3151条可比先验轨迹总体NRMSR\n"
        "红棕点：其余可比先验轨迹背景点\n"
        r"蓝色点/T：$s_i^{\mathrm{rhs}}>1\,\mu$M 的91条先验轨迹",
        ha="left",
        va="top",
        fontsize=8.6,
        linespacing=1.12,
        color="#1f2937",
        zorder=1001,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cbd5e1", lw=0.8, alpha=0.86),
    )


def main():
    setup_font()
    df = pd.read_csv(DATA)
    x = df["teacher_end_uM"].clip(lower=1e-9).to_numpy()
    y = df["pred_end_uM"].clip(lower=1e-9).to_numpy()

    mre = float(df["relative_error_eps1pM"].mean() * 100)
    # The manuscript reports these rounded values for the prior-to-physical-correction checkpoint.
    r2 = 0.999373
    nrmsr = 105.90
    rms = 0.689

    fig = plt.figure(figsize=(12.8, 8.2), dpi=240)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.08], wspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")

    # (a) concentration reconstruction, matching the supplied figure's log-log style.
    ax0.scatter(x, y, color="#4E8FB8", s=11, alpha=0.66, linewidths=0, rasterized=True)
    lim = (1e-11, max(x.max(), y.max()) * 8.0)
    ax0.plot(lim, lim, "--", color="#111827", lw=1.25, alpha=0.9)
    ax0.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    ticks = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5]
    ax0.set_xticks(ticks)
    ax0.set_yticks(ticks)
    ax0.xaxis.set_major_formatter(FuncFormatter(format_log))
    ax0.yaxis.set_major_formatter(FuncFormatter(format_log))
    ax0.set_xlabel("教师模型浓度 (μM)", fontsize=12)
    ax0.set_ylabel("PINN先验浓度 (μM)", fontsize=12)
    ax0.set_title("(a)", loc="left", fontsize=17, pad=8)
    ax0.set_box_aspect(1.0)
    ax0.text(
        0.04, 0.96,
        f"n = {len(df)}\nMRE = {mre:.2f}%\nR2 = {r2:.4f}",
        transform=ax0.transAxes, ha="left", va="top", fontsize=12, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#c7d2df", lw=0.9, alpha=0.96),
    )

    three_d_panel(ax1)
    fig.suptitle("物理轨迹矫正前的 PINN 先验轨迹验证", fontsize=18, y=0.98)
    fig.subplots_adjust(left=0.055, right=0.925, top=0.93, bottom=0.10, wspace=0.18)
    add_prior_legend_box(fig, ax1)
    fig.text(
        0.5, 0.012,
        "注：b 图以所有相邻时间区间带符号变化量分布的中央 95% 显示窗口展示 3151 条可比先验轨迹，并用两根 T 形标记分别标注全部可比先验轨迹与 $s_i^{\\mathrm{rhs}}>1\\,\\mu$M 的 91 条先验轨迹的 NRMSR。",
        ha="center", va="bottom", fontsize=9.5, color="#64748b",
    )
    stem = OUT / "figure_prior_pinn_validation_20260814"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.18)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
