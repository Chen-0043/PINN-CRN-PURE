#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_continuous_coexpression import run  # noqa: E402

GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")
PANEL_ROOT = ROOT / "inputs/original_18p88_unified_shared_scan/tx3000_e0.24"
POOL = ROOT / "input_pool_override_sg_pure2_paper_mapped_20260424.json"
OUT = ROOT / "reproduced_results_verified"


def save(name: str, panel: Path) -> None:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    df, summary = run(72.0, 0.5, 900.0, 0.5, panel=panel, pool_json=POOL, energy_scale=0.24, formal_case=name)
    df.to_csv(out / "timeseries.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    for gene in GENES:
        save(f"single_{gene}", PANEL_ROOT / f"single_{gene}/panel.json")
    save("coexpression", PANEL_ROOT / "mux/panel.json")


if __name__ == "__main__":
    main()

