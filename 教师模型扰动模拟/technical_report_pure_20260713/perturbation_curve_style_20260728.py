"""Shared naming, color, and grouping rules for perturbation figures."""

# Curves are grouped in this order wherever a panel contains multiple groups:
# output -> template/loading -> translation chain/flux -> resource/by-product.
GROUP_ORDER = (
    "产物输出",
    "模板与核糖体承接",
    "翻译位置链与GTP通量",
    "能量池与副产物",
)

LABELS = {
    "deGFP": "deGFP",
    "deGFP_6h": "deGFP",
    "deGFP_peak": "deGFP峰值",
    "mRNA_t": "mRNA_t",
    "Ribosome_mRNA": "Ribosome_mRNA",
    "TLchain_pos_pool": "TLchain_pos pool",
    "termination_release": "终止释放",
    "tx_gtp_entry": "转录GTP入口",
    "tl_gtp_entry": "翻译GTP入口",
    "tl_gtp_progression": "翻译GTP推进",
    "total_GTP_pool": "总GTP浓度",
    "charged_tRNA_pool": "charged-tRNA pool",
    "aa_tRNA_pool": "aa-tRNA pool",
    "EFTu_GTP_aa_tRNA": "EFTu-GTP-aa-tRNA",
    "Met_fMet_initiation_pool": "Met/fMet起始池",
    "unit_TLpool_progression_efficiency": "单位TLchain_pos pool推进效率",
    "ATP": "ATP",
    "ATP_TX": "ATP_TX",
    "GTP": "GTP",
    "PPi": "PPi",
}

# Okabe-Ito-inspired, color-vision-friendly palette. A variable keeps the same
# color across figures; perturbation conditions are distinguished by line style
# or dedicated condition colors when the plotted quantity is the same.
COLORS = {
    "deGFP": "#0072B2",
    "mRNA_t": "#009E73",
    "Ribosome_mRNA": "#7B61A8",
    "TLchain_pos_pool": "#E69F00",
    "termination_release": "#6C757D",
    "tx_gtp_entry": "#0072B2",
    "tl_gtp_entry": "#6C757D",
    "tl_gtp_progression": "#D55E00",
    "total_GTP_pool": "#1B8A8F",
    "charged_tRNA_pool": "#CC79A7",
    "aa_tRNA_pool": "#009E73",
    "EFTu_GTP_aa_tRNA": "#E69F00",
    "Met_fMet_initiation_pool": "#CC79A7",
    "unit_TLpool_progression_efficiency": "#7B61A8",
    "ATP": "#56B4E9",
    "ATP_TX": "#9ACAE6",
    "GTP": "#1B8A8F",
    "PPi": "#B8860B",
    "CTP_condition": "#0072B2",
    "UTP_condition": "#D55E00",
    "optimum_condition": "#0072B2",
    "overoptimum_condition": "#D55E00",
    "neutral": "#555555",
    "RNAP_ATP_wait": "#8C510A",
}

# Bar charts encode perturbation conditions by their displayed order.  The
# first/second/... condition always receives the first/second/... color,
# regardless of the actual fold values used by a perturbation.
BAR_ORDER_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
)

# Shared presentation for every bar-containing panel.
BAR_ALPHA = 0.90
BAR_LEGEND_Y = -0.17
