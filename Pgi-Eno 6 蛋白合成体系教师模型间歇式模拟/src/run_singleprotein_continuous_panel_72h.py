#!/usr/bin/env python3
"""Run the six validated single-protein teacher models to 72 h."""

import json
from pathlib import Path

from run_continuous_coexpression import ROOT, run


GENES = ("Pgi", "PfkA", "TpiA", "GAPDH", "Pgk", "Eno")
POOL = ROOT / "input_pool_override_sg_pure2_paper_mapped_20260424.json"
OUTPUT_ROOT = ROOT / "outputs" / "original_18p88_singleprotein_continuous_72h_dt0p5"


def main() -> None:
    for gene in GENES:
        panel = ROOT / "outputs" / "unified_shared_scan" / "tx3000_e0.24" / f"single_{gene}" / "panel.json"
        output_dir = OUTPUT_ROOT / gene
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"START {gene}", flush=True)
        data, summary = run(
            72.0,
            0.5,
            900.0,
            0.5,
            panel=panel,
            pool_json=POOL,
            energy_scale=0.24,
            formal_case=f"tx3000_e0.24/single_{gene}",
            detailed_state=False,
        )
        summary["single_gene_name"] = gene
        data.to_csv(output_dir / "singleprotein_continuous_timeseries.csv", index=False)
        (output_dir / "singleprotein_continuous_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"DONE {gene}: 72 h={data[f'{gene}_uM'].iloc[-1]:.6f} uM, "
            f"runtime={summary['runtime_s']:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
