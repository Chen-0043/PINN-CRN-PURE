"""Summarize this folder's endpoint integrations; never load another workflow."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reproduced_results/current_metrics.json');a=p.parse_args()
    fresh=ROOT/'run_outputs/article_missing_scale_degfp_endpoints_20260902/article_5axis_scale_degfp_6h_complete.csv'
    source=fresh if fresh.exists() else ROOT/'reproduced_results/five_axis_6h_endpoints_fresh_run.csv'
    frame=pd.read_csv(source)
    if 'deGFP_6h_uM' not in frame or 'component' not in frame or 'scale' not in frame:
        raise ValueError('Endpoint table lacks required columns: '+str(source))
    frame=frame[['component','scale','deGFP_6h_uM']].drop_duplicates(['component','scale'],keep='last')
    components=['ATP','GTP','CTP','UTP','total_aars_pool'];scales=[0,.01,.03,.06,.1,.22,.25,.5,1,2,3,4]
    expected={(c,float(s)) for c in components for s in scales}
    observed={(str(x.component),float(x.scale)) for x in frame.itertuples()}
    if observed!=expected or not np.isfinite(frame.deGFP_6h_uM).all():
        raise ValueError('Incomplete five-axis endpoint coverage')
    optima=frame.sort_values(['component','deGFP_6h_uM','scale'],ascending=[True,False,True]).groupby('component',as_index=False).first()
    result={'source':str(source.relative_to(ROOT)),'endpoint_rows':len(frame),'perturbation_objects':5,'scales':scales,'optima':optima.to_dict('records'),'calculation':'argmax of the local complete endpoint concentration table'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
