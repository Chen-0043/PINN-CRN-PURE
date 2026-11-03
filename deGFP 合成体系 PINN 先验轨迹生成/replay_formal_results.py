#!/usr/bin/env python3
"""Replay formal first-stage weights without training, then compare arrays.

The same helper can live at the archive root or inside either standalone prior
package. All scientific inputs are resolved within that archive/package.
Second-stage outputs are independent optimized concentration nodes, not network
weights; use audit_pinn_conclusions.py for their physical re-evaluation.
"""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
NAMES={'degfp':'deGFP 合成体系 PINN 先验轨迹生成',
       'sixprotein':'Pgi-Eno 6 蛋白合成体系先验轨迹生成'}

def replay(system):
    local=(ROOT/'prior_initialization_recipe.json').exists()
    folder=ROOT if local else ROOT/NAMES[system]
    if local and ROOT.name!=NAMES[system]:
        raise ValueError('Selected system does not match this standalone folder')
    six=system=='sixprotein'
    cfg=dict(json.loads((folder/'prior_initialization_recipe.json').read_text())['stages'][-1])
    cfg['steps']=0
    full=folder/'model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn'
    if six:
        formal=folder/'reference_results/final_prior/prior_prediction_6h.npz'
        cfg['resume_checkpoint']=str(folder/'reference_results/final_prior/prior_pinn_checkpoint.pt')
        cfg['output_dir']=str(folder/'reproduced_results/frozen_formal_prior')
        output=Path(cfg['output_dir'])/'prior_prediction_6h.npz'
        script=folder/'src/train_teacher_supervised_sixprotein_pinn.py'
    else:
        stem='continuous_6h_trackpriority_tol15_finetune_200'
        formal=full/(stem+'_prediction.npz')
        cfg['load_checkpoint']=str(full/(stem+'_checkpoint.pt'))
        cfg['output_prefix']='replayed_frozen_formal_prior'
        cfg['save_checkpoint']=False
        output=full/'replayed_frozen_formal_prior_prediction.npz'
        script=folder/'model_evaluation_20260730/train_continuous_full_crn_pinn_key.py'
    pathkeys={'assets','continuous_xml','trajectory_csv','teacher_snapshot','teacher_trajectory','teacher_map','mechanism_csv','pool_json','teacher_supervision','state_scale'}
    command=[sys.executable,str(script)]
    for key,value in cfg.items():
        if value is None or value is False:continue
        flag=('--'+key.replace('_','-')).replace('eps-m','eps-M').replace('floor-m','floor-M').replace('-um','-uM')
        if value is True:command.append(flag);continue
        if key in pathkeys:value=str(folder/str(value))
        command.extend([flag,json.dumps(value) if isinstance(value,dict) else str(value)])
    print(json.dumps({'system':system,'steps':0,'command':command},ensure_ascii=False),flush=True)
    subprocess.run(command,cwd=folder,check=True)
    a=np.load(output);b=np.load(formal)
    result={'system':system,'same_time_grid':bool(np.array_equal(a['time_s'],b['time_s'])),
            'same_concentration_array':bool(np.array_equal(a['prediction_M'],b['prediction_M'])),
            'maximum_absolute_difference_M':float(np.max(np.abs(a['prediction_M']-b['prediction_M']))),
            'shape':list(a['prediction_M'].shape),'output':str(output.relative_to(folder)),
            'interpretation':'frozen trained weights; zero parameter-update steps'}
    out=folder/'reproduced_results/frozen_formal_prior_comparison.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if not (result['same_time_grid'] and result['same_concentration_array']):
        raise RuntimeError(f'Frozen replay differed from the saved formal result: {result}')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

def main():
    local=(ROOT/'prior_initialization_recipe.json').exists()
    default=next((k for k,v in NAMES.items() if local and ROOT.name==v),'all')
    p=argparse.ArgumentParser();p.add_argument('--system',choices=['all',*NAMES],default=default)
    args=p.parse_args()
    for system in NAMES if args.system=='all' else [args.system]:replay(system)

if __name__=='__main__':main()
