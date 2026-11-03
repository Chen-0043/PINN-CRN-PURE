#!/usr/bin/env python3
"""Reproduce current Figure5 from the trained current Figure4 weights."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'model_assets/original_18p88_full_elementary'
POOL=ROOT/'inputs/input_pool_override_sg_pure2_paper_mapped_20260424.json'
SUP=ASSETS/'sixprotein_teacher_supervision_all_metabolites_dt0p125.npz'
OUT=ROOT/'reproduced_results'

def plan(supervision):
 recipe=json.loads((ROOT/'historical_runtime/correction_recipe.json').read_text())
 trained=OUT/'full_history/trained_prior'
 common=['--assets',str(ASSETS),'--pool-json',str(POOL),'--teacher-supervision',str(supervision)]
 steps=[('trained_prior',[sys.executable,str(ROOT/'src/train_teacher_supervised_sixprotein_pinn.py'),*common,
  '--state-scale',str(ASSETS/'state_scale_from_degfp_probe_M.npy'),
  '--resume-checkpoint',str(ROOT/'inputs/figure4_current_prior/prior_pinn_checkpoint.pt'),
  '--output-dir',str(trained),'--steps','0','--teacher-train-max-s','19800','--energy-scale','0.24',
  '--teacher-derivative-weight','0.1','--terminal-teacher-weight','100','--terminal-derivative-weight','20',
  '--moiety-weight','1','--physics-weight','0','--include-all-teacher-metabolites'])]
 previous=trained/'prior_prediction_6h.npz'
 for i,item in enumerate(recipe['stages'],1):
  argv=list(item['argv']);dest=OUT/'full_history'/f'{i:02d}_{item["name"]}'
  for flag,value in [('--assets',ASSETS),('--pool-json',POOL),('--teacher-supervision',supervision),('--prior',previous),('--output-dir',dest)]:
   argv[argv.index(flag)+1]=str(value)
  steps.append((item['name'],[sys.executable,str(ROOT/'historical_runtime'/item['source']),*argv]))
  previous=dest/'corrected_prediction_6h.npz'
 internal=OUT/'internal_correction_5p5'
 final=OUT/'final_correction_5p5'
 steps.append(('internal_5p5',[sys.executable,str(ROOT/'src/correct_teacher_supervised_sixprotein_nodes.py'),*common,
  '--prior',str(previous),'--output-dir',str(internal),'--steps','2021','--lr','1e-4',
  '--teacher-train-max-s','19800','--energy-scale','0.24','--gamma-h','0.5','--teacher-weight','10',
  '--integral-weight','1','--key-integral-weight','2','--active-integral-weight','100',
  '--current-trajectory-nrmsr-weight','200','--prior-proximity-weight','0.005',
  '--optimization-seed-scope','unconstrained','--initial-state-mode','optimize_all',
  '--hidden-state-parameterization','softplus','--include-all-teacher-metabolites']))
 steps.append(('four_term_evaluation',[sys.executable,str(ROOT/'src/correct_teacher_supervised_sixprotein_nodes.py'),*common,
  '--prior',str(internal/'corrected_prediction_6h.npz'),'--output-dir',str(final),'--steps','0','--lr','1e-4',
  '--teacher-train-max-s','19800','--energy-scale','0.24','--gamma-h','0.5','--teacher-weight','1',
  '--integral-weight','1','--key-integral-weight','2','--active-integral-weight','0',
  '--current-trajectory-nrmsr-weight','0','--prior-proximity-weight','0.005','--optimization-seed-scope','all',
  '--initial-state-mode','optimize_all','--hidden-state-parameterization','softplus','--include-all-teacher-metabolites']))
 return steps

def main():
 os.environ['PYTHONPATH']=str(ROOT/'src')
 os.environ['PYTHONDONTWRITEBYTECODE']='1'
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--supervision',type=Path,default=SUP);a=p.parse_args()
 steps=plan(a.supervision.resolve())
 if a.check:
  for label,command in steps:
   assert Path(command[1]).is_file(),command[1]
  assert (ROOT/'inputs/figure4_current_prior/prior_pinn_checkpoint.pt').is_file()
  assert a.supervision.is_file()
  print(json.dumps({'computational_input':'trained neural-network weights, fixed model definitions and teacher data; no saved optimized node arrays',
   'prior_training_window_h':[0,5.5],'final_teacher_supervision_window_h':[0,5.5],
   'prior_is_current_figure4_checkpoint':True,'steps':steps},ensure_ascii=False,indent=2));return
 trace=[];logs=OUT/'full_history/logs';logs.mkdir(parents=True,exist_ok=True)
 for i,(label,command) in enumerate(steps):
  started=time.time();print(f'[{i+1}/{len(steps)}] {label}',flush=True)
  with (logs/f'{i:02d}.log').open('w') as log:
   subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  trace.append({'stage':label,'command':command,'runtime_s':time.time()-started})
  (OUT/'full_history/execution.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))
 # The publication's four-term numbers are evaluated on the final trajectory;
 # its proximity reference is the initial trajectory of the last segment.
 internal=json.loads((OUT/'internal_correction_5p5/physical_correction_report.json').read_text())
 h=internal['history'][-1]
 losses={'L_data':h['teacher_loss'],'L_int':h['integral_loss'],'L_key':h['key_integral_loss'],'L_prox':h['proximity_loss']}
 losses['L_total_four_term']=losses['L_data']+losses['L_int']+2*losses['L_key']+.005*losses['L_prox']
 losses['interpretation']='post-hoc four-term evaluation; actual optimization retains the documented auxiliary terms'
 (OUT/'four_reported_losses.json').write_text(json.dumps(losses,ensure_ascii=False,indent=2))
 subprocess.run([sys.executable,str(ROOT/'evaluate_results.py'),'--prediction',str(OUT/'final_correction_5p5/corrected_prediction_6h.npz')],cwd=ROOT,check=True)

if __name__=='__main__':main()
