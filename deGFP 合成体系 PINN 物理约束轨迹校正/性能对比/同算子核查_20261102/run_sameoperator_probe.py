from pathlib import Path
import argparse,json,subprocess,time,sys
p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--timeout',type=float,default=180);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent/'reproduced_results';base.mkdir(exist_ok=True)
command=[sys.executable,str(root/'model_evaluation_20260730/direct_full_elementary_mass_action_solve.py'),'--assets',str(root/'model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn'),'--output-prefix',str(base/('sameoperator_'+a.method.lower())),'--method',a.method,'--rtol','1e-6','--atol-uM','1e-6','--max-time-s','21600','--n-eval','25','--exchange-mode','continuous','--use-analytic-jac','--rate-unit','M','--state-unit','M']
t0=time.perf_counter()
with (base/('sameoperator_'+a.method.lower()+'.log')).open('w') as log:
 try:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout);status='completed' if result.returncode==0 else 'failed';returncode=result.returncode
 except subprocess.TimeoutExpired:
  status='timeout_without_complete_trajectory';returncode=None
report={'method':a.method,'command':command,'wall_time_s':time.perf_counter()-t0,'timeout_s':a.timeout,'status':status,'returncode':returncode,'rate_array_conversion':'none; k identical to PINN FullMassActionOperator','research_inputs':'unaltered bundled deGFP assets'}
(base/('sameoperator_'+a.method.lower()+'_run.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
