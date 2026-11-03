#!/usr/bin/env python3
"""Run recovered historical prior stages from a new random initialization.

Uses saved configuration JSON only, never archived neural-network weights.
Every continuation loads the immediately preceding newly trained stage.
"""
from pathlib import Path
import argparse,json,subprocess,sys

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('system',choices=['degfp','sixprotein'])
    p.add_argument('--steps-per-stage',type=int);p.add_argument('--print-only',action='store_true')
    args=p.parse_args();six=args.system=='sixprotein'
    folder=ROOT if (ROOT/'prior_initialization_recipe.json').exists() else ROOT/('Pgi-Eno 6 蛋白合成体系先验轨迹生成' if six else 'deGFP 合成体系 PINN 先验轨迹生成')
    recipe=json.loads((folder/'prior_initialization_recipe.json').read_text())
    script=folder/('src/train_teacher_supervised_sixprotein_pinn.py' if six else 'model_evaluation_20260730/train_continuous_full_crn_pinn_key.py')
    path_keys={'assets','continuous_xml','trajectory_csv','teacher_snapshot','teacher_trajectory','teacher_map','mechanism_csv','pool_json','teacher_supervision','state_scale'}
    previous=None
    for i,original in enumerate(recipe['stages'],1):
        config=dict(original)
        if args.steps_per_stage is not None:config['steps']=args.steps_per_stage
        checkpoint_key='resume_checkpoint' if six else 'load_checkpoint'
        config[checkpoint_key]=str(previous) if previous else None
        if six:
            output=folder/'reproduced_results/from_initialization'/f'stage_{i}'
            config['output_dir']=str(output)
            next_checkpoint=output/'prior_pinn_checkpoint.pt'
        else:
            prefix=f'from_initialization_stage_{i}'
            config['output_prefix']=prefix
            next_checkpoint=folder/'model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn'/(prefix+'_checkpoint.pt')
        command=[sys.executable,str(script)]
        for key,value in config.items():
            if value is None or value is False:continue
            flag='--'+key.replace('_','-')
            # argparse preserves the capital M in these legacy switches.
            for before,after in [('eps-m','eps-M'),('floor-m','floor-M'),('per-h','per-h')]:flag=flag.replace(before,after)
            flag=flag.replace('-um','-uM')
            if value is True:command.append(flag);continue
            if key in path_keys:value=str(folder/str(value))
            if isinstance(value,dict):value=json.dumps(value)
            command.extend([flag,str(value)])
        print(json.dumps({'stage':i,'command':command},ensure_ascii=False),flush=True)
        if not args.print_only:subprocess.run(command,cwd=folder,check=True)
        previous=next_checkpoint

if __name__=='__main__':main()
