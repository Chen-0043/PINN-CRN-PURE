#!/usr/bin/env python3
"""Reproduce the Section 3.4 PINN prior/correction workflow.

The bundled directory keeps the original relative layout so the legacy scripts
can still resolve their assets from Path(__file__). Use:

  python run_3_4_pipeline.py verify
  python run_3_4_pipeline.py plot
  python run_3_4_pipeline.py retrain-prior
  python run_3_4_pipeline.py correct-reference
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
MODEL = ROOT / "model_evaluation_20260730"
BASE = MODEL / "article_gapfill_20260730"
FULL = BASE / "full_elementary_pinn"
TEACHER = BASE / "existing_teacher_constraints"
MAPPING = BASE / "teacher_to_full_mapping"
CRN = BASE / "crn_constraint_map"
# Historical published runs interpolate this nine-point teacher trajectory.
# Its 5--6 h segment includes the held-out 6 h endpoint; see the audit report.
TEACHER_TRAJECTORY = TEACHER / "continuous_existing_teacher_train0to5_validate6h_all_species_trajectory.npz"


REQUIRED_FILES = [
    MODEL / "full_mass_action_pinn.py",
    MODEL / "train_continuous_full_crn_pinn_key.py",
    MODEL / "optimize_pinn_nodes_by_mass_action_shooting.py",
    FULL / "species_index.csv",
    FULL / "reaction_mass_action_inventory.csv",
    FULL / "stoichiometric_matrix_S.npz",
    FULL / "reactant_exponent_matrix_A.npz",
    FULL / "rate_constants_k.npy",
    FULL / "initial_concentrations_x0_M.npy",
    FULL / "continuous_6h_corrected_gtpresidual_pretrain_checkpoint_1500_checkpoint.pt",
    FULL / "continuous_6h_trackpriority_tol15_finetune_200_prediction.npz",
    FULL / "continuous_6h_trackpriority_200_shooting_nodes_w1_1000_continue_lr005_add300_train5h30_init.npz",
    FULL / "continuous_6h_trackpriority_200_shooting_continue_add300_train5h30_1000_shooting_nodes.npz",
    TEACHER_TRAJECTORY,
    TEACHER / "continuous_existing_teacher_train0to5_validate6h_all_species_trajectory.npz",
    TEACHER / "continuous_existing_teacher_train0to5_validate6h_all_species_6h_snapshot.csv",
    TEACHER / "continuous_existing_teacher_baseline_mechanism_metrics_gtp1x.csv",
    MAPPING / "full_species_teacher_mapping.csv",
    CRN / "formal_model.xml",
    FULL / "plot_prior_pinn_validation_20260814.py",
    FULL / "plot_pinn_section33_stratified_validation_20260813.py",
]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def verify_assets() -> None:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path.relative_to(ROOT)}")
        raise SystemExit(1)
    print("OK: Section 3.4 code and data assets are present.")


def verify_deps() -> None:
    missing = []
    for module in ["numpy", "pandas", "scipy", "matplotlib", "torch"]:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        print("Missing Python packages:", ", ".join(missing))
        print("Install with: python3 -m pip install -r requirements.txt")
        raise SystemExit(1)
    print("OK: Python dependencies are importable.")


def verify() -> None:
    verify_assets()
    verify_deps()


def plot() -> None:
    verify_assets()
    for module in ["numpy", "pandas", "scipy", "matplotlib"]:
        importlib.import_module(module)
    run([sys.executable, str(FULL / "plot_prior_pinn_validation_20260814.py")])
    run([sys.executable, str(FULL / "plot_pinn_section33_stratified_validation_20260813.py")])
    print("\nSVG outputs:")
    print(FULL / "figures" / "article_341_20260814" / "figure_prior_pinn_validation_20260814.svg")
    print(FULL / "figures" / "article_33_20260813" / "figure_section33_pinn_validation_residual_stratification_20260813.svg")


def retrain_prior(output_prefix: str, steps: int = 200) -> None:
    verify()
    run(
        [
            sys.executable,
            str(MODEL / "train_continuous_full_crn_pinn_key.py"),
            "--assets",
            str(FULL),
            "--continuous-xml",
            str(CRN / "formal_model.xml"),
            "--teacher-trajectory",
            str(TEACHER_TRAJECTORY),
            "--teacher-map",
            str(MAPPING / "full_species_teacher_mapping.csv"),
            "--teacher-weight",
            "1.0",
            "--teacher-loss-mode",
            "log10",
            "--teacher-train-max-s",
            "18000",
            "--load-checkpoint",
            str(FULL / "continuous_6h_corrected_gtpresidual_pretrain_checkpoint_1500_checkpoint.pt"),
            "--save-checkpoint",
            "--steps",
            str(steps),
            "--output-prefix",
            output_prefix,
            "--physics-weight",
            "1e-7",
            "--shared-physics-weight",
            "0.0001",
            "--shared-key-physics-weight",
            "0.0005",
            "--shared-physics-rhs-floor-uM-per-h",
            "0.01",
            "--shared-integral-physics-weight",
            "0.002",
            "--shared-key-integral-physics-weight",
            "0.008",
            "--physics-collocation-points",
            "40",
            "--physics-collocation-weight",
            "5e-8",
            "--mechanism-csv",
            str(TEACHER / "continuous_existing_teacher_baseline_mechanism_metrics_gtp1x.csv"),
            "--gtp-mechanism-weight",
            "0.3",
        ]
    )


def correct_reference(output_prefix: str, steps: int = 1000) -> None:
    verify()
    run(
        [
            sys.executable,
            str(MODEL / "optimize_pinn_nodes_by_mass_action_shooting.py"),
            "--assets",
            str(FULL),
            "--pinn",
            str(FULL / "continuous_6h_trackpriority_200_shooting_nodes_w1_1000_continue_lr005_add300_train5h30_init.npz"),
            "--output-prefix",
            output_prefix,
            "--steps",
            str(steps),
            "--lr",
            "0.005",
            "--teacher-weight",
            "1.0",
            "--teacher-trajectory",
            str(TEACHER_TRAJECTORY),
            "--teacher-train-max-s",
            "19800",
            "--integral-weight",
            "1.0",
            "--key-integral-weight",
            "2.0",
            "--pinn-proximity-weight",
            "0.005",
            "--rhs-floor-uM",
            "0.01",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("verify-assets")
    sub.add_parser("verify-deps")
    sub.add_parser("plot")
    p_prior = sub.add_parser("retrain-prior")
    p_prior.add_argument("--output-prefix", default="reproduced_continuous_6h_trackpriority_tol15_finetune_200")
    p_prior.add_argument("--steps", type=int, default=200)
    p_correct = sub.add_parser("correct-reference")
    p_correct.add_argument("--output-prefix", default="reproduced_continuous_6h_trackpriority_200_shooting_continue_add300_train5h30_1000")
    p_correct.add_argument("--steps", type=int, default=1000)
    sub.add_parser("smoke-simulate")
    args = parser.parse_args()

    if args.command == "verify":
        verify()
    elif args.command == "verify-assets":
        verify_assets()
    elif args.command == "verify-deps":
        verify_deps()
    elif args.command == "plot":
        plot()
    elif args.command == "retrain-prior":
        retrain_prior(args.output_prefix, args.steps)
    elif args.command == "correct-reference":
        correct_reference(args.output_prefix, args.steps)
    elif args.command == "smoke-simulate":
        retrain_prior("smoke_prior_one_step", 1)
        correct_reference("smoke_correct_one_step", 1)


if __name__ == "__main__":
    main()
