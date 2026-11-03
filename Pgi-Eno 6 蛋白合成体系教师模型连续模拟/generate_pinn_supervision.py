#!/usr/bin/env python3
"""Re-integrate and map every represented six-protein teacher state."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 out=ROOT/'reproduced_supervision_source'
 subprocess.run([sys.executable,str(ROOT/'src/run_continuous_coexpression.py'),
  '--gamma-h','0.5','--t-end-h','6','--dt-s','0.125','--record-every-s','900',
  '--panel-json',str(ROOT/'inputs/original_18p88_unified_shared_scan/tx3000_e0.24/mux/panel.json'),
  '--pool-json',str(ROOT/'input_pool_override_sg_pure2_paper_mapped_20260424.json'),
  '--energy-scale','0.24','--formal-case','tx3000_e0.24','--detailed-state','--outdir',str(out)],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/build_sixprotein_teacher_supervision.py'),
  '--teacher-csv',str(out/'sixprotein_continuous_timeseries.csv'),
  '--assets',str(ROOT/'model_inputs/supervision_mapping'),
  '--output',str(out/'sixprotein_teacher_supervision_all_metabolites.npz')],cwd=ROOT,check=True)

if __name__=='__main__':main()
