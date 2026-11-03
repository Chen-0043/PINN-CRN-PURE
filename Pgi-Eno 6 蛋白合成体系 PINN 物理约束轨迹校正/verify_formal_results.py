#!/usr/bin/env python3
"""Recompute published numerical metrics from arrays, not from report labels."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
def close(actual,expected,name,tol=1e-8):
    if abs(actual-expected)>tol*max(1,abs(expected)):
        raise AssertionError(f'{name}: {actual} != {expected}')
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path);a=p.parse_args()
    expected=json.loads((ROOT/'formal_result_spec.json').read_text())
    checkpoint=ROOT/'inputs/figure4_current_prior/prior_pinn_checkpoint.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected['checkpoint_sha256']
    prediction=a.prediction or ROOT/'reference_results/final_correction_5p5/corrected_prediction_6h.npz'
    output=ROOT/'reproduced_results/verification/current_metrics.json'
    subprocess.run([sys.executable,str(ROOT/'evaluate_results.py'),'--prediction',str(prediction),'--output',str(output)],cwd=ROOT,check=True)
    actual=json.loads(output.read_text());target=expected['metrics']
    assert actual['concentration']['mapped_states']==actual['concentration']['states_entering_MRE']==9198
    assert actual['gt_1_uM']['count']==target['gt_1_uM']['count']==34
    for key in ['MRE_percent','R2']:close(actual['concentration'][key],target['concentration'][key],key)
    close(actual['gt_1_uM']['mean_NRMSR_percent'],target['gt_1_uM']['mean_NRMSR_percent'],'active NRMSR')
    for key in ['CTP','UTP']:close(actual['modules'][key],target['modules'][key],key)
    print('PASS: 9198-state MRE/R2, 34-trajectory NRMSR, CTP and UTP recomputed from numerical arrays.')
if __name__=='__main__':main()
