#!/usr/bin/env python3
"""Rebuild six-protein Figure5 data from local computed concentration arrays."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pandas as pd
from scipy import sparse

DRAW=Path(__file__).resolve().parent
ROOT=DRAW.parent

def main():
    fresh=ROOT/'reproduced_results/final_correction_5p5/corrected_prediction_6h.npz'
    prediction=fresh if fresh.is_file() else ROOT/'reference_results/final_correction_5p5/corrected_prediction_6h.npz'
    metrics=ROOT/'reproduced_results/figure5/current_metrics.json'
    subprocess.run([sys.executable,str(ROOT/'evaluate_results.py'),'--prediction',str(prediction),'--output',str(metrics)],check=True,cwd=ROOT)
    sup=np.load(ROOT/'model_assets/original_18p88_full_elementary/sixprotein_teacher_supervision_all_metabolites_dt0p125.npz',allow_pickle=True)
    z=np.load(prediction)
    ir=np.flatnonzero(np.isclose(z['time_s'],21600,rtol=0,atol=1e-8))
    it=np.flatnonzero(np.isclose(sup['time_s'],21600,rtol=0,atol=1e-8))
    assert len(ir)==len(it)==1
    state=z['prediction_M'][ir[0]]
    members=sup['group_member_indices'].astype(int);offsets=sup['group_member_offsets'].astype(int)
    matrix=sparse.csr_matrix((np.ones(len(members)),members,offsets),shape=(len(offsets)-1,len(state)))
    actual=np.r_[state[sup['direct_indices'].astype(int)],matrix@state]
    target=np.r_[sup['direct_target_M'][it[0]],sup['group_target_M'][it[0]]]
    assert len(target)==len(actual)==9198
    pd.DataFrame({'teacher_6h_M':target,'new_corrected_6h_M':actual,
                  'teacher_6h_uM':target*1e6,'new_corrected_6h_uM':actual*1e6}).to_csv(DRAW/'六蛋白6h新接续浓度_9198对象.csv',index=False)
    result=json.loads(metrics.read_text())
    (DRAW/'本图使用的新结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
