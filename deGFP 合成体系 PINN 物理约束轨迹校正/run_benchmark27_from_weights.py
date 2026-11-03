#!/usr/bin/env python3
"""Time the documented 27-node/1000-step benchmark from fresh stage-2 nodes."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'model_evaluation_20260730/article_gapfill_20260730'
FULL=BASE/'full_elementary_pinn'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--use-current-run',action='store_true',help='Use stage 2 already regenerated during this validation run.');args=parser.parse_args()
 stage2=FULL/'from_prior_stage_2_shooting_nodes.npz'
 if not args.use_current_run or not stage2.exists():
  subprocess.run([sys.executable,str(ROOT/'run_correction_from_weights.py')],cwd=ROOT,check=True)
 z=np.load(stage2);t=np.unique(np.r_[z['time_s'],300.,21300.]);x=z['optimized_M']
 values=np.column_stack([np.interp(t,z['time_s'],x[:,j]) for j in range(x.shape[1])])
 out=ROOT/'性能对比/reproduced_results/benchmark27';out.mkdir(parents=True,exist_ok=True)
 initial=out/'initial_nodes.npz';np.savez_compressed(initial,time_s=t,optimized_M=values)
 cmd=[sys.executable,str(ROOT/'model_evaluation_20260730/optimize_pinn_nodes_by_mass_action_shooting.py'),
  '--assets',str(FULL),'--pinn',str(initial),'--output-prefix',str(out/'benchmark27'),
  '--steps','1000','--lr','0.005','--teacher-train-max-s','18000',
  '--teacher-trajectory',str(BASE/'existing_teacher_constraints/continuous_existing_teacher_train0to5_validate6h_all_species_trajectory.npz'),
  '--teacher-weight','1','--integral-weight','1','--key-integral-weight','2',
  '--pinn-proximity-weight','0.005','--rhs-floor-uM','0.01']
 with (out/'run.log').open('w') as log:
  started=time.perf_counter();subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True);elapsed=time.perf_counter()-started
 new=np.load(out/'benchmark27_shooting_nodes.npz');ref=np.load(FULL/'reviewer_benchmark_correction_1000_shooting_nodes.npz')
 result={'scope':'single 1000-step correction benchmark; includes Python startup/loading/optimization/evaluation/output; preceding stage reconstruction excluded',
  'nodes':len(t),'states':values.shape[1],'steps':1000,'wall_s':elapsed,'command':cmd,
  'input_source':'newly computed first/second correction stages, interpolated to the benchmark grid',
  'time_grid_equal':bool(np.array_equal(new['time_s'],ref['time_s'])),
  'maximum_absolute_difference_M':float(abs(new['optimized_M']-ref['optimized_M']).max())}
 (out/'measurement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
