from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from perturbation_curve_style_20260728 import BAR_ALPHA, BAR_LEGEND_Y, BAR_ORDER_COLORS, COLORS, LABELS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "technical_report_pure_20260713"
FIG_DIR = REPORT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / "atp_perturbation_full_stage_mechanism_curve_reorganized_20260728.png"
OUT_PDF = FIG_DIR / "atp_perturbation_full_stage_mechanism_curve_reorganized_20260728.pdf"
OUT_SVG = FIG_DIR / "atp_perturbation_full_stage_mechanism_curve_reorganized_20260728.svg"
OUT_PRE_CSV = FIG_DIR / "atp_0p5_vs_0p25_preoptimum_ratio_20260721.csv"
OUT_OVER_CSV = FIG_DIR / "atp_1x_vs_0p5_overoptimum_ratio_unified_time_20260723.csv"
OUT_OPT_CSV = FIG_DIR / "atp_0p5_optimum_timecourse_20260721.csv"
OUT_ADJ_CSV = FIG_DIR / "atp_0p25_0p5_1x_adjacent_state_6h_20260721.csv"


def setup_fonts() -> None:
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "STHeiti",
        "Songti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def endpoint_table() -> pd.DataFrame:
    compact = FIG_DIR / "atp_single_mechanism_evidence_6h_20260718.csv"
    if compact.exists():
        df = pd.read_csv(compact).copy()
        df["deGFP_peak_uM"] = df["deGFP_end_uM"]
        df["mRNA_t_end_uM"] = np.nan
        df["Ribosome_mRNA_end_uM"] = np.nan
        df["AA_end_uM"] = np.nan
        df["total_GTP_pool_end_uM"] = df["GTP_end_uM"]
        return df[
            [
                "scale",
                "deGFP_end_uM",
                "deGFP_peak_uM",
                "ATP_end_uM",
                "ATP_TX",
                "GTP_end_uM",
                "total_GTP_pool_end_uM",
                "PPi_end_uM",
            ]
        ].sort_values("scale")
    main = pd.read_csv(
        ROOT
        / "扰动模拟/results_final/atp_ctp_utp_main_fullsnap_serial_sparse7_20260711/synced_subset_metrics.csv"
    )
    main = main[main["component"].isin(["__baseline__", "ATP"])].copy()
    main.loc[main["component"].eq("__baseline__"), "component"] = "ATP"
    peak = pd.read_csv(ROOT / "扰动模拟/results_final/peak_points_fullsnap_20260712/synced_subset_metrics.csv")
    peak = peak[peak["component"].eq("ATP") & peak["scale"].eq(0.5)].copy()
    cols = [
        "component",
        "scale",
        "deGFP_end_uM",
        "deGFP_peak_uM",
        "mRNA_t_end_uM",
        "Ribosome_mRNA_end_uM",
        "ATP_end_uM",
        "GTP_end_uM",
        "PPi_end_uM",
        "AA_end_uM",
    ]
    out = (
        pd.concat([main[cols], peak[cols]], ignore_index=True)
        .sort_values("scale")
        .drop_duplicates("scale", keep="last")
        .reset_index(drop=True)
    )
    out["total_GTP_pool_end_uM"] = out["GTP_end_uM"]
    return out


def align_atp_endpoint_total_gtp(end: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    out = end.copy()
    if "total_GTP_pool_end_uM" not in out.columns:
        out["total_GTP_pool_end_uM"] = out["GTP_end_uM"]

    pre6 = pre[np.isclose(pre["time_h"], 6.0)]
    if not pre6.empty:
        row = pre6.iloc[0]
        out.loc[np.isclose(out["scale"], 0.25), "total_GTP_pool_end_uM"] = (
            float(row["GTP_low"]) + float(row["GTP_TX_low"])
        )
        out.loc[np.isclose(out["scale"], 0.5), "total_GTP_pool_end_uM"] = (
            float(row["GTP_high"]) + float(row["GTP_TX_high"])
        )

    records = pd.read_csv(REPORT / "report_audit_20260723/unified_perturbation_timeseries_records_20260723.csv")
    endpoints = records[(records["axis"].eq("ATP")) & np.isclose(records["time_h"], 6.0)]
    total_map = dict(zip(endpoints["scale"].astype(float), endpoints["total_GTP_pool"].astype(float)))
    for scale, value in total_map.items():
        out.loc[np.isclose(out["scale"], scale), "total_GTP_pool_end_uM"] = value
    return out


def read_species(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_reactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["reaction_id_str"] = df["reaction_id"].astype(str)
    df["reaction_class_str"] = df["reaction_class"].fillna("").astype(str)
    df["reactants_str"] = df["reactants"].fillna("").astype(str)
    return df


def species_sum_at(group: pd.DataFrame, selector) -> float:
    sp = group["species"].astype(str)
    return float(group.loc[selector(sp), "concentration_uM"].sum())


def summarize_species(sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in sp.groupby("time_h", sort=True):
        vals = dict(zip(g["species"].astype(str), g["concentration_uM"].astype(float)))
        ribo = float(vals.get("Ribosome_mRNA", np.nan))
        tlpool = species_sum_at(g, lambda s: s.str.startswith("TLchain_pos_"))
        charged = species_sum_at(
            g,
            lambda s: (s.str.match(r"^[A-Z][a-z]{2}tRNA") | s.str.startswith("fMettRNA"))
            & ~s.str.contains("degraded", regex=False),
        )
        rows.append(
            {
                "time_h": float(t),
                "deGFP": float(vals.get("deGFP_m", np.nan)),
                "mRNA_t": float(vals.get("mRNA_t", np.nan)),
                "Ribosome_mRNA": ribo,
                "TLchain_pos_pool": tlpool,
                "TLpool_per_Ribosome_mRNA": tlpool / ribo if ribo and np.isfinite(ribo) else np.nan,
                "ATP": float(vals.get("ATP", np.nan)),
                "ATP_TX": float(vals.get("ATP_TX", np.nan)),
                "GTP": float(vals.get("GTP", np.nan)),
                "GTP_TX": float(vals.get("GTP_TX", np.nan)),
                "PPi": float(vals.get("PPi", np.nan)),
                "charged_tRNA_pool": charged,
            }
        )
    return pd.DataFrame(rows)


def summarize_reactions(rx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in rx.groupby("time_h", sort=True):
        rid = g["reaction_id_str"]
        rcls = g["reaction_class_str"]
        react = g["reactants_str"]
        rate = g["rate_uM_per_h"].astype(float)
        gtp_react = react.str.contains("GTP", regex=False)
        tx = rcls.eq("transcription_or_mRNA_turnover") & gtp_react
        prog = rid.str.startswith("rxn_tl_chain_step")
        total = float(rate[gtp_react].sum())
        tx_val = float(rate[tx].sum())
        prog_val = float(rate[prog].sum())
        entry = total - tx_val - prog_val
        rows.append(
            {
                "time_h": float(t),
                "tx_gtp_entry": tx_val,
                "tl_gtp_entry": entry,
                "tl_gtp_progression": prog_val,
                "termination_release": float(rate[rid.eq("rxn_tl_chain_term")].sum()),
                "aa_tRNA_charging": float(rate[rcls.eq("aa_tRNA_charging_network")].abs().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_case(label: str, species_path: Path, reaction_path: Path) -> pd.DataFrame:
    sp = summarize_species(read_species(species_path))
    rx = summarize_reactions(read_reactions(reaction_path))
    out = sp.merge(rx, on="time_h", how="outer").sort_values("time_h")
    out.insert(0, "case", label)
    return out


def ratio_rows(abs_df: pd.DataFrame, high_label: str, low_label: str, comparison: str) -> pd.DataFrame:
    high = abs_df[abs_df["case"].eq(high_label)].copy()
    low = abs_df[abs_df["case"].eq(low_label)].copy()
    merged = high.merge(low, on="time_h", suffixes=("_high", "_low"))
    metrics = [
        "deGFP",
        "mRNA_t",
        "Ribosome_mRNA",
        "TLchain_pos_pool",
        "TLpool_per_Ribosome_mRNA",
        "ATP",
        "ATP_TX",
        "GTP",
        "GTP_TX",
        "PPi",
        "charged_tRNA_pool",
        "tx_gtp_entry",
        "tl_gtp_entry",
        "tl_gtp_progression",
        "termination_release",
        "aa_tRNA_charging",
    ]
    rows = []
    for _, r in merged.iterrows():
        row = {"comparison": comparison, "time_h": float(r["time_h"])}
        for m in metrics:
            lo = float(r[f"{m}_low"])
            hi = float(r[f"{m}_high"])
            row[f"{m}_low"] = lo
            row[f"{m}_high"] = hi
            row[f"{m}_ratio"] = np.nan if abs(lo) < 1e-30 else hi / lo
        row["unit_TLpool_progression_efficiency_ratio"] = (
            row["tl_gtp_progression_ratio"] / row["TLchain_pos_pool_ratio"]
            if np.isfinite(row["TLchain_pos_pool_ratio"]) and abs(row["TLchain_pos_pool_ratio"]) > 1e-30
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_preoptimum_ratio() -> pd.DataFrame:
    base025 = ROOT / "扰动模拟/results_final/atp_ctp_utp_main_fullsnap_serial_sparse7_20260711/sparse_snapshots"
    base05 = ROOT / "扰动模拟/results_final/peak_points_fullsnap_20260712/sparse_snapshots"
    c025 = summarize_case(
        "ATP_0p25",
        base025 / "species/species_snapshot_ATP_x0p25.csv.gz",
        base025 / "reactions/reaction_snapshot_ATP_x0p25.csv.gz",
    )
    c05 = summarize_case(
        "ATP_0p5",
        base05 / "species/species_snapshot_ATP_x0p5.csv.gz",
        base05 / "reactions/reaction_snapshot_ATP_x0p5.csv.gz",
    )
    out = ratio_rows(pd.concat([c025, c05], ignore_index=True), "ATP_0p5", "ATP_0p25", "ATP 0.5×/0.25×")
    out = out[out["time_h"].isin([0.2, 0.5, 1.0, 3.0, 4.0, 6.0])].copy()
    out.to_csv(OUT_PRE_CSV, index=False)
    return out


def load_overoptimum_ratio() -> pd.DataFrame:
    # Unified 2026-07-23 time grid; total_GTP_pool is GTP + GTP_TX.
    df = pd.read_csv(REPORT / "report_audit_20260723/unified_perturbation_timeseries_ratios_20260723.csv")
    out = df[(df["axis"].eq("ATP")) & (df["comparison"].eq("1x/0.5x"))].copy()
    out = out.sort_values("time_h")
    out.to_csv(OUT_OVER_CSV, index=False)
    return out


def load_optimum_timecourse() -> pd.DataFrame:
    df = pd.read_csv(REPORT / "report_audit_20260719/dense_atp_absolute_timesnap_20260719.csv")
    out = df[df["scale"].eq(0.5)].copy().sort_values("time_h")
    out.to_csv(OUT_OPT_CSV, index=False)
    return out


def build_adjacent_optimum_state(pre: pd.DataFrame, over: pd.DataFrame) -> pd.DataFrame:
    p = pre[np.isclose(pre["time_h"], 6.0)].iloc[0]
    o = over[np.isclose(over["time_h"], 6.0)].iloc[0]
    rows = [
        {
            "ATP_scale": "0.25×",
            "deGFP": p["deGFP_low"],
            "termination_release": p["termination_release_low"],
            "tl_gtp_progression": p["tl_gtp_progression_low"],
            "TLpool_per_Ribosome_mRNA": p["TLpool_per_Ribosome_mRNA_low"],
            "PPi": p["PPi_low"],
        },
        {
            "ATP_scale": "0.5×",
            "deGFP": p["deGFP_high"],
            "termination_release": p["termination_release_high"],
            "tl_gtp_progression": p["tl_gtp_progression_high"],
            "TLpool_per_Ribosome_mRNA": p["TLpool_per_Ribosome_mRNA_high"],
            "PPi": p["PPi_high"],
        },
        {
            "ATP_scale": "1×",
            "deGFP": o["deGFP_high"],
            "termination_release": o["termination_release_high"],
            "tl_gtp_progression": o["tl_gtp_progression_high"],
            "TLpool_per_Ribosome_mRNA": o["TLpool_per_Ribosome_mRNA_high"],
            "PPi": o["PPi_high"],
        },
    ]
    out = pd.DataFrame(rows)
    base = out[out["ATP_scale"].eq("0.5×")].iloc[0]
    for col in ["deGFP", "termination_release", "tl_gtp_progression", "TLpool_per_Ribosome_mRNA", "PPi"]:
        out[col + "_relative_to_0p5x"] = out[col] / base[col]
    out.to_csv(OUT_ADJ_CSV, index=False)
    return out


def main() -> None:
    setup_fonts()
    end = endpoint_table()
    pre = build_preoptimum_ratio()
    end = align_atp_endpoint_total_gtp(end, pre)
    opt = load_optimum_timecourse()
    over = load_overoptimum_ratio()
    adj = build_adjacent_optimum_state(pre, over)

    fig = plt.figure(figsize=(16.2, 10.4), dpi=220)
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.97, top=0.88, bottom=0.085, wspace=0.30, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    fig.suptitle("ATP 单独扰动对全代谢网络的阶段性影响", fontsize=18, weight="bold")

    scales = end["scale"].to_numpy(float)
    xpos = np.arange(len(scales))
    xticklabels = [f"{s:g}×" for s in scales]
    ax_a.plot(xpos, end["deGFP_end_uM"], "-o", color=COLORS["deGFP"], lw=2.5, ms=6, label=LABELS["deGFP_6h"])
    ax_a2 = ax_a.twinx()
    ax_a2.bar(
        xpos, end["PPi_end_uM"], width=0.55, color=COLORS["PPi"], alpha=BAR_ALPHA,
        edgecolor="none", linewidth=0, label=LABELS["PPi"], zorder=1,
    )
    ax_a.set_title("A  全剂量响应：0.25×不足，0.5×最优，之后平台/微降", loc="left", fontsize=11.4, weight="bold")
    ax_a.set_xlabel("ATP 初始投料倍率")
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels(xticklabels)
    ax_a.set_ylabel("deGFP (μM)")
    ax_a2.set_ylabel("PPi (μM)", color="#A65F00")
    ax_a2.tick_params(axis="y", labelcolor="#A65F00")
    ax_a.set_ylim(3.8, 6.25)
    ax_a2.set_ylim(0, end["PPi_end_uM"].max() * 1.25)
    ax_a.grid(False)
    lines, labels = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(
        lines + lines2, labels + labels2, frameon=False, fontsize=8, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    width = 0.18
    metrics = ["deGFP_end_uM", "PPi_end_uM", "ATP_end_uM", "ATP_TX", "total_GTP_pool_end_uM"]
    labels = [LABELS["deGFP"], LABELS["PPi"], LABELS["ATP"], LABELS["ATP_TX"], LABELS["total_GTP_pool"]]
    vals = pd.DataFrame(
        [end.loc[np.isclose(end["scale"], s), metrics].iloc[0] for s in [0.25, 0.5, 1.0, 4.0]],
        index=["0.25×", "0.5×", "1×", "4×"],
    )
    display_scales_b = pd.Series(
        {
            "deGFP_end_uM": 1.0,
            "PPi_end_uM": 0.2,
            "ATP_end_uM": 0.01,
            "ATP_TX": 0.01,
            "total_GTP_pool_end_uM": 0.1,
        }
    )
    plot_vals = vals.mul(display_scales_b, axis=1)
    xx = np.arange(len(labels))
    colors = BAR_ORDER_COLORS[:4]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for label, color, off in zip(vals.index, colors, offsets):
        ax_b.bar(
            xx + off, plot_vals.loc[label], width=width, color=color, alpha=BAR_ALPHA,
            edgecolor="none", linewidth=0, label=label,
        )
    ax_b.set_xticks(xx)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, max(plot_vals.max()) * 1.18)
    ax_b.set_ylabel("6 h 显示值（deGFP: μM；PPi ×0.2；ATP/ATP_TX ×0.01；总GTP浓度 ×0.1）")
    ax_b.set_title("B  资源状态：0.25× ATP/ATP_TX枯竭；高ATP产生冗余与PPi累积", loc="left", fontsize=11.4, weight="bold")
    ax_b.legend(
        frameon=False, fontsize=8, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )
    ax_b.grid(False)

    evidence = pd.read_csv(FIG_DIR / "atp_single_mechanism_evidence_6h_20260718.csv")
    evidence = evidence[evidence["scale"].isin([0.25, 0.5, 1.0])].copy().sort_values("scale")
    ribo_map = dict(zip(adj["ATP_scale"], adj["Ribosome_mRNA"] if "Ribosome_mRNA" in adj.columns else [np.nan] * len(adj)))
    if all(np.isnan(list(ribo_map.values()))):
        main_metrics = pd.read_csv(
            ROOT / "扰动模拟/results_final/atp_ctp_utp_main_fullsnap_serial_sparse7_20260711/synced_subset_metrics.csv"
        )
        peak_metrics = pd.read_csv(ROOT / "扰动模拟/results_final/peak_points_fullsnap_20260712/synced_subset_metrics.csv")
        ribo_map = {
            0.25: float(main_metrics[(main_metrics["component"].eq("ATP")) & (main_metrics["scale"].eq(0.25))]["Ribosome_mRNA_end_uM"].iloc[0]),
            0.5: float(peak_metrics[(peak_metrics["component"].eq("ATP")) & (peak_metrics["scale"].eq(0.5))]["Ribosome_mRNA_end_uM"].iloc[0]),
            1.0: float(main_metrics[(main_metrics["component"].eq("__baseline__")) & (main_metrics["scale"].eq(1.0))]["Ribosome_mRNA_end_uM"].iloc[0]),
        }
        evidence["Ribosome_mRNA_end_uM"] = evidence["scale"].map(ribo_map)
    charged_map = {
        0.25: float(pre[np.isclose(pre["time_h"], 6.0)]["charged_tRNA_pool_low"].iloc[0]),
        0.5: float(pre[np.isclose(pre["time_h"], 6.0)]["charged_tRNA_pool_high"].iloc[0]),
        1.0: float(over[np.isclose(over["time_h"], 6.0)]["charged_tRNA_pool_high"].iloc[0]),
    }
    evidence["charged_tRNA_pool"] = evidence["scale"].map(charged_map)
    dose_x = np.arange(len(evidence))
    ax_c.plot(dose_x, evidence["deGFP_end_uM"], "-o", color=COLORS["deGFP"], lw=2.2, ms=5.5, label=LABELS["deGFP"])
    ax_c.plot(
        dose_x,
        evidence["Ribosome_mRNA_end_uM"] * 100,
        "-s",
        color=COLORS["Ribosome_mRNA"],
        lw=2.2,
        ms=5.2,
        label=LABELS["Ribosome_mRNA"],
    )
    ax_c2 = ax_c.twinx()
    ax_c2.plot(dose_x, evidence["PPi_end_uM"], "--v", color=COLORS["PPi"], lw=2.0, ms=5.2, label=LABELS["PPi"])
    ax_c2.plot(dose_x, evidence["ATP_TX"], "--D", color=COLORS["ATP_TX"], lw=2.0, ms=5.0, label=LABELS["ATP_TX"])
    ax_c2.plot(
        dose_x,
        evidence["charged_tRNA_pool"],
        "--P",
        color=COLORS["charged_tRNA_pool"],
        lw=2.0,
        ms=5.0,
        label=LABELS["charged_tRNA_pool"],
    )
    ax_c.axvline(1, color="#d62728", lw=1.2, ls=":", alpha=0.9)
    ax_c.set_xticks(dose_x)
    ax_c.set_xticklabels([f"{s:g}×" for s in evidence["scale"]])
    ax_c.set_ylim(0, 10.8)
    ax_c2.set_yscale("log")
    ax_c2.set_ylim(0.08, max(evidence["PPi_end_uM"].max(), evidence["ATP_TX"].max(), evidence["charged_tRNA_pool"].max()) * 1.65)
    ax_c.set_ylabel("产出/承接绝对值（Ribosome_mRNA ×100）")
    ax_c2.set_ylabel("ATP_TX / PPi / charged-tRNA pool (μM, log)", color="#6b4f00")
    ax_c2.tick_params(axis="y", labelcolor="#6b4f00")
    ax_c.set_xlabel("ATP 初始投料倍率")
    ax_c.set_title("C  相邻三点绝对指标：0.5×高产出/高装载且PPi压力不过量", loc="left", fontsize=11.4, weight="bold")
    ax_c.grid(True, color="#D9D9D9", lw=0.7, alpha=0.85)
    lines, labels = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(
        lines + lines2, labels + labels2, frameon=False, fontsize=7.2,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    tover = over["time_h"].to_numpy(float)
    ax_d.plot(tover, over["TLchain_pos_pool_ratio"], "-s", lw=2.0, ms=4.8, color=COLORS["TLchain_pos_pool"], label=LABELS["TLchain_pos_pool"])
    ax_d.plot(tover, over["tl_gtp_progression_ratio"], "-^", lw=2.0, ms=4.8, color=COLORS["tl_gtp_progression"], label=LABELS["tl_gtp_progression"])
    ax_d.axhline(1, color="#555555", ls="--", lw=1)
    ax_d.axvspan(0.05, 0.25, color="#2ca02c", alpha=0.08, lw=0)
    ax_d.set_title("D  过最优统一时序：位置链短暂打开，随后推进优势消失", loc="left", fontsize=11.4, weight="bold")
    ax_d.set_xlabel("时间 (h)")
    ax_d.set_ylabel("倍数（ATP 1× / 0.5×）")
    ax_d.plot(tover, over["total_GTP_pool_ratio"], "--D", lw=1.9, ms=4.2, color=COLORS["total_GTP_pool"], label=LABELS["total_GTP_pool"])
    ax_d.axhline(1, color="#555555", ls="--", lw=1)
    ax_d.set_ylim(0.74, 1.30)
    ax_d.grid(True, color="#D9D9D9", lw=0.7, alpha=0.85)
    ax_d.legend(
        frameon=False, fontsize=7.4, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    ax_e.plot(tover, over["mRNA_t_ratio"], "-o", lw=2.1, ms=5, color=COLORS["mRNA_t"], label=LABELS["mRNA_t"])
    ax_e.plot(tover, over["Ribosome_mRNA_ratio"], "-s", lw=2.1, ms=4.8, color=COLORS["Ribosome_mRNA"], label=LABELS["Ribosome_mRNA"])
    ax_e.plot(tover, over["unit_TLpool_progression_efficiency_ratio"], "-^", lw=2.1, ms=4.8, color=COLORS["unit_TLpool_progression_efficiency"], label=LABELS["unit_TLpool_progression_efficiency"])
    ax_e.axhline(1, color="#555555", ls="--", lw=1)
    ax_e.axvspan(1.0, 6.0, color="#d62728", alpha=0.06, lw=0)
    ax_e.set_title("E  后期瓶颈：mRNA_t/Ribosome_mRNA与单位推进效率低于0.5×", loc="left", fontsize=11.4, weight="bold")
    ax_e.set_xlabel("时间 (h)")
    ax_e.set_ylabel("倍数（1× / 0.5×）")
    ax_e.set_ylim(0.88, 1.02)
    ax_e.grid(True, color="#D9D9D9", lw=0.7, alpha=0.85)
    ax_e.legend(
        frameon=False, fontsize=8, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    ax_f.plot(tover, over["tx_gtp_entry_ratio"], "-o", lw=2.1, ms=5, color=COLORS["tx_gtp_entry"], label=LABELS["tx_gtp_entry"])
    ax_f.plot(tover, over["tl_gtp_entry_ratio"], "-s", lw=2.1, ms=4.8, color=COLORS["tl_gtp_entry"], label=LABELS["tl_gtp_entry"])
    ax_f.plot(tover, over["tl_gtp_progression_ratio"], "-^", lw=2.1, ms=4.8, color=COLORS["tl_gtp_progression"], label=LABELS["tl_gtp_progression"])
    ax_f.axhline(1, color="#555555", ls="--", lw=1)
    ax_f.axvspan(0.05, 0.25, color="#2ca02c", alpha=0.08, lw=0)
    ax_f.axvspan(3.0, 6.0, color="#1f77b4", alpha=0.06, lw=0)
    ax_f.set_title("F  三部分GTP：推进早期占优，后期入口回补而推进低于0.5×", loc="left", fontsize=11.4, weight="bold")
    ax_f.set_xlabel("时间 (h)")
    ax_f.set_ylabel("GTP通量倍数（1× / 0.5×）")
    ax_f.set_ylim(0.78, 1.22)
    ax_f.grid(True, color="#D9D9D9", lw=0.7, alpha=0.85)
    ax_f.legend(
        frameon=False, fontsize=8, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, BAR_LEGEND_Y),
    )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_SVG)
    print(OUT_OPT_CSV)
    print(OUT_ADJ_CSV)
    print(OUT_OVER_CSV)


if __name__ == "__main__":
    main()
