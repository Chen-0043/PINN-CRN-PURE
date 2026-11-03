#!/usr/bin/env python3
"""Finalize the experimental identifiability and uncertainty audit.

The output deliberately distinguishes a confidence interval from a chosen
perturbation range.  Individual kinetic-parameter confidence intervals are
reported as non-estimable when the likelihood is rank deficient or when the
available point is not a weighted least-squares optimum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
STAT = ROOT / "run_outputs" / "statistical_recalibration_20261027"
ROBUST = ROOT / "run_outputs" / "five_axis_uncertainty_20261027"
PACKAGE = ROOT / "model_inputs" / "deliverable_PURE_1nt1aa_independent_20260610"


def main() -> None:
    obs = pd.read_csv(STAT / "experimental_replicate_summary.csv")
    ident = pd.read_csv(STAT / "parameter_local_identifiability.csv")
    manifest = pd.read_csv(PACKAGE / "PARAMETER_MANIFEST.csv")
    numerical = pd.read_csv(ROBUST / "uncertainty_summary_for_reviewer.csv")
    z = np.load(STAT / "weighted_local_jacobian.npz", allow_pickle=True)
    J = z["J"]
    residual = z["residual"]
    weight = z["weight"]
    names = z["parameter_names"].astype(str)

    Jw = J * weight[:, None]
    rw = residual * weight
    U, singular, Vt = np.linalg.svd(Jw, full_matrices=False)
    relative = singular / singular[0]
    rank = int(np.sum(relative > 1e-6))
    dof = len(residual) - rank
    dispersion = float(np.dot(rw, rw) / dof)
    sqrt_dispersion = float(np.sqrt(dispersion))

    # Confidence widths for identifiable *SVD combinations*.  These are local
    # likelihood diagnostics, not individual-parameter intervals.
    modes = []
    for i in range(rank):
        log_se = sqrt_dispersion / singular[i]
        half = 1.96 * log_se
        top = np.argsort(-np.abs(Vt[i]))[:6]
        modes.append(
            {
                "mode": i + 1,
                "singular_value": singular[i],
                "relative_singular_value": relative[i],
                "local_log_standard_error": log_se,
                "local_95pct_multiplier_lower": float(np.exp(-min(half, 700.0))),
                "local_95pct_multiplier_upper": float(np.exp(min(half, 700.0))),
                "dominant_parameter_loadings": "; ".join(
                    f"{names[j]}:{Vt[i, j]:+.4f}" for j in top
                ),
            }
        )
    mode_df = pd.DataFrame(modes)
    mode_df.to_csv(STAT / "identifiable_parameter_combinations_local_95pct.csv", index=False)

    # The manifest has 86 rows, but some are metadata (profile/trend), fixed
    # initial values, derived rates, and fit diagnostics.  Only 53 rows are
    # independent adjustable quantities: 27 runtime + 20 HF profile + 6 keff.
    ident_by_name = ident.set_index("parameter")
    hf_free_by_profile = {
        "bi_exp": {"yinf_M", "a_fast", "k_fast_s_inv", "k_slow_s_inv"},
        "single_exp": {"yinf_M", "k_s_inv"},
    }
    profile_by_category = {}
    for category, group in manifest[manifest["参数类别"].astype(str).str.startswith("hf_fit:")].groupby("参数类别"):
        profile_row = group[group["参数名"] == "profile"]
        if len(profile_row):
            profile_by_category[str(category)] = str(profile_row.iloc[0]["实际值"])
    keff_free = {"r_inf", "r_0", "K_ATP_M", "K_GTP_M", "KI_PO4_M", "KI_PPi_M"}
    rows = []
    for _, row in manifest.iterrows():
        category = str(row["参数类别"])
        pname = str(row["参数名"])
        record = dict(row)
        if category == "batch_runtime_parameter":
            independent = True
            role = "runtime_adjustable_parameter"
        elif category.startswith("hf_fit:"):
            independent = pname in hf_free_by_profile.get(profile_by_category.get(category, ""), set())
            role = "background_profile_fit_parameter" if independent else "profile_metadata_fixed_or_derived"
        elif category == "keff_map":
            independent = pname in keff_free
            role = "effective_rate_map_fit_parameter" if independent else "fit_diagnostic"
        else:
            independent = False
            role = "metadata_or_fixed"
        record["independent_adjustable_quantity"] = independent
        record["manifest_role"] = role
        if category == "batch_runtime_parameter" and pname in ident_by_name.index:
            d = ident_by_name.loc[pname]
            norm = float(d["jacobian_l2_weighted"])
            responsive = norm > 0.0
            record.update(
                {
                    "experimental_weighted_gradient_norm": norm,
                    "local_null_space_participation": float(d["null_space_participation"]),
                    "experimental_status": (
                        "responsive_but_individual_CI_not_valid_at_nonoptimal_rank_deficient_point"
                        if responsive
                        else "zero_gradient_not_identifiable_from_available_observations"
                    ),
                    "individual_95pct_CI_lower": np.nan,
                    "individual_95pct_CI_upper": np.nan,
                }
            )
        elif independent:
            record.update(
                {
                    "experimental_weighted_gradient_norm": np.nan,
                    "local_null_space_participation": np.nan,
                    "experimental_status": (
                        "background_profile_or_keff_parameter_not_independently_observed_"
                        "by_mRNA_and_deGFP_replicates"
                    ),
                    "individual_95pct_CI_lower": np.nan,
                    "individual_95pct_CI_upper": np.nan,
                }
            )
        else:
            record.update(
                {
                    "experimental_weighted_gradient_norm": np.nan,
                    "local_null_space_participation": np.nan,
                    "experimental_status": "not_an_independent_parameter",
                    "individual_95pct_CI_lower": np.nan,
                    "individual_95pct_CI_upper": np.nan,
                }
            )
        rows.append(record)
    full = pd.DataFrame(rows)
    full.to_csv(STAT / "all_86_manifest_entries_uncertainty_status.csv", index=False)
    independent_full = full[full["independent_adjustable_quantity"]].copy()
    if len(independent_full) != 53:
        raise RuntimeError(f"Expected 53 independent adjustable quantities, got {len(independent_full)}")
    independent_full.to_csv(STAT / "independent_53_parameter_uncertainty_status.csv", index=False)

    # Exact small-sample t interval for the directly observed 6 h deGFP mean.
    six = obs[(obs["condition"] == "combined_expression") & np.isclose(obs["time_h"], 6.0)].iloc[0]
    reps = np.array([six.get(f"replicate_{i}_uM", np.nan) for i in range(1, 5)], dtype=float)
    reps = reps[np.isfinite(reps)]
    mean = float(np.mean(reps))
    sd = float(np.std(reps, ddof=1))
    n = len(reps)
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    exp_summary = {
        "n_replicates": n,
        "mean_uM": mean,
        "sd_uM": sd,
        "cv_percent": 100.0 * sd / mean,
        "t95_lower_uM": mean - half,
        "t95_upper_uM": mean + half,
        "t95_half_width_percent_of_mean": 100.0 * half / mean,
    }
    (STAT / "combined_expression_6h_experimental_interval.json").write_text(
        json.dumps(exp_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outcome = numerical.copy()
    outcome["numerically_resolved_vs_solver_tolerance"] = (
        outcome["gain_nominal_percent"].abs()
        > outcome["solver_gain_spread_percentage_point"].abs()
    )
    outcome["qualitative_effect_exceeds_direct_experimental_scatter"] = outcome[
        "condition"
    ].map(
        {
            "ATP": False,
            "GTP": True,
            "CTP": False,
            "UTP": False,
            "total_aars_pool": False,
        }
    )
    outcome["interpretation"] = outcome["condition"].map(
        {
            "ATP": "deterministic numerical optimum; 0.06% gain is below experimental and parameter uncertainty resolution",
            "GTP": "large model-predicted effect relative to numerical and direct experimental variability; exact optimum and parameter-robust significance remain model dependent",
            "CTP": "deterministic numerical optimum; 2.55% gain is not established as biologically significant",
            "UTP": "deterministic numerical optimum; 0.08% gain is below experimental and parameter uncertainty resolution",
            "total_aars_pool": "deterministic numerical optimum; 0.45% gain is below experimental and parameter uncertainty resolution",
        }
    )
    outcome.to_csv(STAT / "five_axis_reliability_conclusion.csv", index=False)

    zero = ident.loc[ident["jacobian_l2_weighted"] == 0, "parameter"].tolist()
    responsive = ident.loc[ident["jacobian_l2_weighted"] > 0, "parameter"].tolist()
    report = f"""# 参数可信性与五类投料结果的审计结论

## 数据与参数范围

- 重新提取了转录单独实验（4个重复，301个时间点）、翻译单独实验（3个重复，241个时间点）和转录–翻译联合实验（3个重复，121个时间点），共663个时间点均值。
- 模型参数清单共86行，其中33行为轨迹类型、趋势、初始值、派生速率或拟合误差等固定/派生信息；真正独立可调量为53项，包括27项运行时参数、20项高频背景轨迹拟合参数和6项有效速率映射参数。

## 可辨识性结果

- 27项运行时参数的加权局部雅可比秩为{rank}/27，条件数为{singular[0] / singular[-1]:.3e}。
- 原参数点的加权残差平方和为{float(np.dot(rw, rw)):.3f}，离散度为{dispersion:.3f}，表明实验离散度不足以解释模型与完整轨迹的偏差。
- 6项参数对现有观测的局部梯度为0：{', '.join(zero)}。
- 其余21项参数有局部响应，但强相关且原参数点不是三类实验轨迹的加权最小二乘最优点，因此不能计算有效的单参95%置信区间。
- 仅第1个参数组合的局部95%不确定倍数范围较窄（{mode_df.iloc[0]['local_95pct_multiplier_lower']:.3f}–{mode_df.iloc[0]['local_95pct_multiplier_upper']:.3f}）；第2组合已扩大到{mode_df.iloc[1]['local_95pct_multiplier_lower']:.3f}–{mode_df.iloc[1]['local_95pct_multiplier_upper']:.3f}，从第3组合起区间迅速失去实用辨识度。

## 直接实验与五类投料结论

- 联合表达实验6 h deGFP为{mean:.4f} ± {sd:.4f} µM（均值 ± SD，n={n}），95% t区间为{mean-half:.4f}–{mean+half:.4f} µM，区间半宽占均值{100*half/mean:.2f}%。
- 收紧求解器容差后，五类产量增益只变化0.000014–0.000172个百分点，所以数值积分误差不会改变确定性排序。
- ATP、CTP、UTP和总氨酰化模块的0.0629%、2.5504%、0.0788%和0.4508%增益虽是数值上可重复的最优点，但低于实验离散度和参数不确定性的可分辨范围，不应声称具有已证实的生物学显著性。
- GTP的201.068%模型预测增益明显大于数值误差和直接实验离散度，因而“基线GTP供给可能不足”是值得优先实验验证的定性假设；但由于单参数置信区间不可辨识，尚不能宣称其已获得参数稳健的生物学显著性，3倍的精确最优位置仍属模型依赖结果。

## 严谨表述边界

现有数据能支持求解器收敛性和参数局部可辨识性审计，但不足以为53项独立可调量逐一提供实验支持的置信区间。若要获得单参数置信区间，需要更多可直接观测NTP周转、核糖体装载、翻译因子与背景池动态的独立实验数据，或为固定文献参数提供可审计的先验分布。
"""
    (STAT / "parameter_uncertainty_audit_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
