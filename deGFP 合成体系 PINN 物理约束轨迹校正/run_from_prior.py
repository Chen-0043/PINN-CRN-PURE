#!/usr/bin/env python3
"""Rebuild the historical correction continuation from a supplied prior.

Each step consumes the preceding newly computed trajectory. The final teacher
mask includes interpolated 5.25/5.5 h values; see the audit report.
"""
from pathlib import Path
import argparse,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'model_evaluation_20260730/article_gapfill_20260730'
FULL=BASE/'full_elementary_pinn'

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,default=FULL/'continuous_6h_trackpriority_tol15_finetune_200_prediction.npz')
    args=p.parse_args();previous=args.prior.resolve()
    for number,(lr,prox,train_max) in enumerate([(.03,.01,18000),(.005,.005,18000),(.005,.005,19800)],1):
        if number==3:
            z=np.load(previous);t=np.unique(np.r_[z['time_s'],300.]);x=z['optimized_M']
            values=np.column_stack([np.interp(t,z['time_s'],x[:,i]) for i in range(x.shape[1])])
            previous=FULL/'from_prior_stage_3_interpolated_init.npz'
            np.savez_compressed(previous,time_s=t,optimized_M=values)
        prefix=f'from_prior_stage_{number}'
        command=[sys.executable,str(ROOT/'model_evaluation_20260730/optimize_pinn_nodes_by_mass_action_shooting.py'),
            '--assets',str(FULL),'--pinn',str(previous),'--output-prefix',prefix,'--steps','1000','--lr',str(lr),
            '--teacher-trajectory',str(BASE/'existing_teacher_constraints/continuous_existing_teacher_train0to5_validate6h_all_species_trajectory.npz'),
            '--teacher-train-max-s',str(train_max),'--teacher-weight','1','--integral-weight','1',
            '--key-integral-weight','2','--pinn-proximity-weight',str(prox),'--rhs-floor-uM','0.01']
        subprocess.run(command,cwd=ROOT,check=True)
        previous=FULL/(prefix+'_shooting_nodes.npz')
    print(previous)

if __name__=='__main__':main()
