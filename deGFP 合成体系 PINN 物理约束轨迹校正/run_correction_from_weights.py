#!/usr/bin/env python3
"""Generate a prior from local trained weights, then rerun node correction."""
from pathlib import Path
import argparse,json,subprocess,sys

ROOT=Path(__file__).resolve().parent

def command():
 cfg=json.loads((ROOT/'prior_weight_replay_recipe.json').read_text())
 cfg.update(steps=0,output_prefix='prior_from_weights_for_correction',save_checkpoint=False)
 full=ROOT/'model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn'
 cfg['load_checkpoint']=str(full/'continuous_6h_trackpriority_tol15_finetune_200_checkpoint.pt')
 pathkeys={'assets','continuous_xml','trajectory_csv','teacher_snapshot','teacher_trajectory','teacher_map','mechanism_csv'}
 argv=[sys.executable,str(ROOT/'model_evaluation_20260730/train_continuous_full_crn_pinn_key.py')]
 for key,value in cfg.items():
  if value is None or value is False:continue
  flag=('--'+key.replace('_','-')).replace('eps-m','eps-M').replace('floor-m','floor-M').replace('-um','-uM')
  if value is True:argv.append(flag);continue
  if key in pathkeys:value=str(ROOT/str(value))
  argv.extend([flag,json.dumps(value) if isinstance(value,dict) else str(value)])
 prior=full/'prior_from_weights_for_correction_prediction.npz'
 return argv,[sys.executable,str(ROOT/'run_from_prior.py'),'--prior',str(prior)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();commands=command()
 if a.check:
  for c in commands:assert Path(c[1]).is_file()
  print(json.dumps({'input':'local trained neural-network checkpoint; no saved optimized nodes','commands':commands},ensure_ascii=False,indent=2));return
 for c in commands:subprocess.run(c,cwd=ROOT,check=True)

if __name__=='__main__':main()
