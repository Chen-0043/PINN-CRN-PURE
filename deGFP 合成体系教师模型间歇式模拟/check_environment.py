#!/usr/bin/env python3
"""Check packaged input files and Python runtime dependencies."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "fit_batch_then_continuous_full_crn_background.py",
    "run_batch_1nt1aa.py",
    "run_tx1nt_tlg1_micro3_longrun.py",
    "plot_results.py",
    "verify_results.py",
    "validate_platform_against_raw_data.py",
    "PURE_CRN_models/Models/CRN_PURE_MGaptTXwTL_with_deGFP_COPASI.xml",
    "PURE_CRN_models/Data_files/PURE_TXTL_initial_values_Final.csv",
    "external_PURE_CRN_models_sparse/Models/TXTL_PURE_MGapt-deGFP_model.xml",
    "output_full_crn_background/formal_mech_only_20260409_210539_rechecked.json",
    "output_full_crn_background/batch_then_continuous_summary.json",
    "output_full_crn_background/rs_legacy_module_migration_20260609/table_RS旧模块迁移清单_337反应_20260609.csv",
    "raw_literature_data/CRN_PURE_TX_mGaptv4_updated.xml",
    "raw_literature_data/MGapt_deGFP_Exp_PURE_Data/2022.08.19_OnePot_PURE_mGapt_uM.csv",
    "raw_literature_data/MGapt_deGFP_Exp_PURE_Data/2022.11.14_OnePotPURE_deGFPuM.csv",
    "raw_literature_data/Calibration_MGapt/2023.07.28_PUREwmRNA_updated.xlsx",
]
MODULES = ["numpy", "pandas", "scipy", "matplotlib", "tellurium", "libsbml"]


def main() -> None:
    files = {name: (ROOT / name).is_file() for name in REQUIRED_FILES}
    modules = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            modules[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            modules[name] = {"available": False, "error": str(exc)}
    report = {"package_root": str(ROOT), "files": files, "modules": modules}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(files.values()) or not all(item["available"] for item in modules.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
