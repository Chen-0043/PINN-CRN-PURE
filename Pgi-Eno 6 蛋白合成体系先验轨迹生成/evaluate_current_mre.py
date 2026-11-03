#!/usr/bin/env python3
"""Evaluate all 9198 six-protein 6 h states using the deGFP MRE rule.

This same small script is intentionally included in both independent packages.
It does not change any training losses, weights, or concentration arrays.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced_results/current_mre_all9198.json')
    args=p.parse_args()
    prior=ROOT/'reference_results/final_prior/prior_prediction_6h.npz'
    path=args.prediction or (prior if prior.exists() else ROOT/'reference_results/final_correction_5p5/corrected_prediction_6h.npz')
    sup=np.load(ROOT/'model_assets/original_18p88_full_elementary/sixprotein_teacher_supervision_all_metabolites_dt0p125.npz',allow_pickle=True)
    z=np.load(path);ir=np.flatnonzero(np.isclose(z['time_s'],21600,rtol=0,atol=1e-8));it=np.flatnonzero(np.isclose(sup['time_s'],21600,rtol=0,atol=1e-8))
    assert len(ir)==len(it)==1
    state=z['prediction_M'][ir[0]].astype(float)
    members=sup['group_member_indices'].astype(int);offsets=sup['group_member_offsets'].astype(int)
    groups=sparse.csr_matrix((np.ones(len(members)),members,offsets),shape=(len(offsets)-1,len(state)))
    pred=np.r_[state[sup['direct_indices'].astype(int)],groups@state]
    target=np.r_[sup['direct_target_M'][it[0]],sup['group_target_M'][it[0]]].astype(float)
    assert len(pred)==len(target)==9198 and np.isfinite(pred).all() and np.isfinite(target).all()
    result={'states_entering_MRE':9198,'validation_time_h':6,'denominator_floor_M':1e-12,
        'MRE_percent':float(100*np.mean(np.abs(pred-target)/np.maximum(np.abs(target),1e-12))),
        'R2':float(1-np.sum((pred-target)**2)/np.sum((target-target.mean())**2)),
        'evaluation_only':True,'source_prediction':str(path)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
