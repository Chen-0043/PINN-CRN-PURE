#!/usr/bin/env python3
"""补跑文章扰动倍数缺失点，只保存6 h deGFP终点。

该脚本不生成全代谢网络时间点采样，也不覆盖历史结果。它扫描已有
CSV 中的 6 h deGFP 记录，按文章使用的统一倍数集合判断缺口；1x
直接使用基线，不重新模拟。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rerun_synced_ntp_aars_perturbations_20260711 as ntp_base  # noqa: E402
import run_final_total_aars_pool as total_aars  # noqa: E402


ARTICLE_SCALES = [0.0, 0.01, 0.03, 0.06, 0.1, 0.22, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
COMPONENTS = ["ATP", "GTP", "CTP", "UTP", "total_aars_pool"]
TOTAL_AARS_ALIASES = {"total_aars_pool", "total_aars", "Total_aaRS", "total_aaRS", "aaRS", "amino_acids"}
COMPONENT_ALIASES = {
    "ATP": {"ATP"},
    "GTP": {"GTP"},
    "CTP": {"CTP"},
    "UTP": {"UTP"},
    "total_aars_pool": TOTAL_AARS_ALIASES,
}


def close_scale(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)


def tag_float(x: float) -> str:
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def parse_scales(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_existing_degfp() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in ROOT.rglob("*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = set(df.columns)
        if not any(c in cols for c in ["scale", "扰动倍数", "倍数"]):
            continue
        scale_col = "scale" if "scale" in cols else ("扰动倍数" if "扰动倍数" in cols else "倍数")
        deg_col = None
        for candidate in ["deGFP_end_uM", "deGFP_m_end_uM", "deGFP"]:
            if candidate in cols:
                deg_col = candidate
                break
        if deg_col is None:
            continue
        if "time_h" in cols:
            time_num = pd.to_numeric(df["time_h"], errors="coerce")
            df = df[time_num.sub(6.0).abs() < 1e-6]
            if df.empty:
                continue
        comp_col = "component" if "component" in cols else ("axis" if "axis" in cols else None)
        for _, r in df.iterrows():
            try:
                scale = float(r[scale_col])
                degfp = float(r[deg_col])
            except Exception:
                continue
            if not any(close_scale(scale, x) for x in ARTICLE_SCALES):
                continue
            comp_raw = str(r[comp_col]) if comp_col else ""
            for comp, aliases in COMPONENT_ALIASES.items():
                if comp_raw in aliases:
                    rows.append(
                        {
                            "component": comp,
                            "scale": scale,
                            "deGFP_6h_uM": degfp,
                            "source": str(path.relative_to(ROOT)),
                            "source_kind": "existing",
                        }
                    )
    return pd.DataFrame(rows)


def first_existing(df: pd.DataFrame, component: str, scale: float) -> Dict[str, object] | None:
    if df.empty:
        return None
    sub = df[(df["component"] == component) & df["scale"].map(lambda x: close_scale(float(x), scale))]
    if sub.empty:
        return None
    # A forced rerun must be reflected in the returned conclusion table.
    sub = sub.copy()
    sub = sub[np.isfinite(pd.to_numeric(sub["deGFP_6h_uM"], errors="coerce"))]
    if sub.empty:
        return None
    sub["_source_priority"] = sub["source_kind"].map(
        {"new_endpoint_only": 0, "existing": 1}
    ).fillna(2)
    preferred = sub.sort_values(["_source_priority", "source"]).iloc[0].drop("_source_priority").to_dict()
    return preferred


def determine_missing(existing: pd.DataFrame, scales: Iterable[float], components: Iterable[str] = COMPONENTS) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for comp in components:
        for scale in scales:
            if close_scale(scale, 1.0):
                status = "baseline_1x_reused"
            elif first_existing(existing, comp, scale) is None:
                status = "missing_to_run"
            else:
                status = "existing_reused"
            rows.append({"component": comp, "scale": float(scale), "status": status})
    return pd.DataFrame(rows)


def simulate_endpoint(
    *,
    component: str,
    scale: float,
    params,
    hf_fit,
    keff_map,
    base_ic_uM,
    gamma_h: float,
    t_end_h: float,
    n_points: int,
) -> Dict[str, object]:
    if component == "total_aars_pool":
        row, _ts, _endpoint, _sparse, _reaction_table = total_aars.simulate_total_aars_condition(
            scale=float(scale),
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            base_ic_uM=base_ic_uM,
            gamma_h=float(gamma_h),
            t_end_h=float(t_end_h),
            n_points=int(n_points),
            snapshot_time_points_h=[],
        )
    else:
        row, _ts, _endpoint = ntp_base.simulate_condition(
            component=component,
            scale=float(scale),
            base_ic_uM=base_ic_uM,
            params=params,
            hf_fit=hf_fit,
            keff_map=keff_map,
            gamma_h=float(gamma_h),
            t_end_h=float(t_end_h),
            n_points=int(n_points),
        )
    return row


def build_complete_table(existing: pd.DataFrame, new_rows: pd.DataFrame, scales: Iterable[float], baseline_value: float, components: Iterable[str] = COMPONENTS) -> pd.DataFrame:
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    out_rows: List[Dict[str, object]] = []
    for comp in components:
        for scale in scales:
            if close_scale(scale, 1.0):
                out_rows.append(
                    {
                        "component": comp,
                        "scale": float(scale),
                        "deGFP_6h_uM": float(baseline_value),
                        "source_kind": "baseline_reused",
                        "source": "shared baseline 1x; not rerun",
                    }
                )
                continue
            got = first_existing(combined, comp, scale)
            if got is None:
                out_rows.append(
                    {
                        "component": comp,
                        "scale": float(scale),
                        "deGFP_6h_uM": np.nan,
                        "source_kind": "missing",
                        "source": "",
                    }
                )
            else:
                out_rows.append(got)
    result = pd.DataFrame(out_rows)
    result["scale"] = result["scale"].astype(float)
    return result.sort_values(["component", "scale"], kind="stable").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run missing article-scale perturbation endpoints for deGFP only.")
    ap.add_argument("--scales", default=",".join("%g" % x for x in ARTICLE_SCALES))
    ap.add_argument(
        "--components",
        default=",".join(COMPONENTS),
        help="Comma-separated subset of ATP,GTP,CTP,UTP,total_aars_pool.",
    )
    ap.add_argument("--outdir", type=Path, default=ROOT / "run_outputs" / "article_missing_scale_degfp_endpoints_20260902")
    ap.add_argument("--gamma-h", type=float, default=0.5)
    ap.add_argument("--t-end-h", type=float, default=6.0)
    ap.add_argument("--n-points", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Run all non-1x points even if an existing point is found.")
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
    components = [x.strip() for x in args.components.split(",") if x.strip()]
    invalid = sorted(set(components) - set(COMPONENTS))
    if invalid:
        raise ValueError(f"Unknown components: {invalid}")

    existing = read_existing_degfp()
    existing.to_csv(outdir / "existing_article_scale_degfp_sources.csv", index=False)

    plan = determine_missing(existing, scales, components)
    if args.force:
        plan.loc[~plan["scale"].map(lambda x: close_scale(float(x), 1.0)), "status"] = "missing_to_run"
    plan.to_csv(outdir / "missing_run_plan.csv", index=False)

    params, hf_fit, keff_map, base_ic_uM = ntp_base.load_sources()
    baseline_row, _baseline_ts, _baseline_endpoint = ntp_base.simulate_baseline(
        params=params,
        hf_fit=hf_fit,
        keff_map=keff_map,
        base_ic_uM=base_ic_uM,
        gamma_h=float(args.gamma_h),
        t_end_h=float(args.t_end_h),
        n_points=int(args.n_points),
    )
    baseline_value = float(baseline_row["deGFP_end_uM"])

    rows: List[Dict[str, object]] = []
    to_run = plan[plan["status"] == "missing_to_run"].copy()
    for idx, r in enumerate(to_run.itertuples(index=False), start=1):
        comp = str(r.component)
        scale = float(r.scale)
        label = f"{comp}_x{tag_float(scale)}"
        print(f"[RUN] {idx}/{len(to_run)} {label}", flush=True)
        try:
            row = simulate_endpoint(
                component=comp,
                scale=scale,
                params=params,
                hf_fit=hf_fit,
                keff_map=keff_map,
                base_ic_uM=base_ic_uM,
                gamma_h=float(args.gamma_h),
                t_end_h=float(args.t_end_h),
                n_points=int(args.n_points),
            )
            rec = {
                "component": comp,
                "scale": scale,
                "deGFP_6h_uM": float(row["deGFP_end_uM"]),
                "deGFP_peak_uM_with_two_points": float(row["deGFP_peak_uM"]),
                "runtime_s": float(row["runtime_s"]),
                "perturbation_kind": row.get("perturbation_kind", ""),
                "n_scaled_species": row.get("n_scaled_species", ""),
                "scaled_species": row.get("scaled_species", ""),
                "scaled_params": row.get("scaled_params", ""),
                "set_params": row.get("set_params", ""),
                "source_kind": "new_endpoint_only",
                "source": str((outdir / "new_missing_degfp_endpoints.csv").relative_to(ROOT)),
                "status": "ok",
            }
            print(f"[OK] {label} deGFP_6h={rec['deGFP_6h_uM']:.9g} uM runtime={rec['runtime_s']:.1f}s", flush=True)
        except Exception as exc:
            rec = {
                "component": comp,
                "scale": scale,
                "deGFP_6h_uM": np.nan,
                "source_kind": "new_endpoint_only",
                "source": str((outdir / "new_missing_degfp_endpoints.csv").relative_to(ROOT)),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"[FAILED] {label} {exc!r}", flush=True)
        rows.append(rec)
        pd.DataFrame(rows).to_csv(outdir / "new_missing_degfp_endpoints_partial.csv", index=False)

    new_df = pd.DataFrame(rows)
    new_df.to_csv(outdir / "new_missing_degfp_endpoints.csv", index=False)
    complete = build_complete_table(existing, new_df, scales, baseline_value, components)
    complete.to_csv(outdir / "article_5axis_scale_degfp_6h_complete.csv", index=False)

    coverage = (
        complete.assign(has_value=complete["deGFP_6h_uM"].notna())
        .groupby("component")["has_value"]
        .agg(["sum", "count"])
        .reset_index()
    )
    coverage["complete"] = coverage["sum"] == coverage["count"]
    coverage.to_csv(outdir / "coverage_summary.csv", index=False)

    summary = {
        "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "article_scales": scales,
        "components": components,
        "baseline_1x_deGFP_6h_uM": baseline_value,
        "n_existing_records_scanned": int(len(existing)),
        "n_missing_to_run": int(len(to_run)),
        "n_new_ok": int((new_df.get("status", pd.Series(dtype=str)) == "ok").sum()) if not new_df.empty else 0,
        "n_new_failed": int((new_df.get("status", pd.Series(dtype=str)) == "failed").sum()) if not new_df.empty else 0,
        "coverage_complete": bool(coverage["complete"].all()) if not coverage.empty else False,
        "outputs": {
            "existing_sources_csv": str(outdir / "existing_article_scale_degfp_sources.csv"),
            "missing_run_plan_csv": str(outdir / "missing_run_plan.csv"),
            "new_missing_degfp_endpoints_csv": str(outdir / "new_missing_degfp_endpoints.csv"),
            "complete_article_table_csv": str(outdir / "article_5axis_scale_degfp_6h_complete.csv"),
            "coverage_summary_csv": str(outdir / "coverage_summary.csv"),
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["coverage_complete"] and summary["n_new_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
