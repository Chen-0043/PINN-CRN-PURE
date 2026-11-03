from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
MODE='six_continuous'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reproduced_results/current_metrics.json');a=p.parse_args()
    if MODE=='degfp_batch':
        raw=pd.read_csv(ROOT/'raw_literature_data/MGapt_deGFP_Exp_PURE_Data/2022.11.14_OnePotPURE_deGFPuM.csv')
        path=ROOT/'reproduced_results/batch/batch_timeseries.csv'
        if not path.exists():path=ROOT/'results/batch/batch_timeseries.csv'
        model=pd.read_csv(path);ts=raw['Time'].to_numpy(float)*60;target=raw.iloc[:,1:].mean(axis=1).to_numpy(float);pred=np.interp(ts,model.time,model['[deGFP_m]']*1e6);mask=(ts>=0)&(ts<=21600)&(target>1e-12)
        result={'source':str(path.relative_to(ROOT)),'literature_points':int(mask.sum()),'MRE_percent':float(np.mean(abs(pred[mask]-target[mask])/target[mask])*100)}
    elif MODE=='degfp_continuous':
        path=ROOT/'reproduced_results/continuous/gamma_0p5_continuous_72h_timeseries.csv'
        if not path.exists():path=ROOT/'results/continuous/gamma_0p5_continuous_72h_timeseries.csv'
        frame=pd.read_csv(path);tail=frame[frame.time>=48*3600]['[deGFP_m]']*1e6
        result={'source':str(path.relative_to(ROOT)),'window_h':[48,72],'max_abs_deviation_from_window_mean_uM':float(abs(tail-tail.mean()).max()),'range_uM':float(tail.max()-tail.min())}
    elif MODE=='six_batch':
        lit=pd.read_csv(ROOT/'inputs/output_sixgene_single_gene_proxy_paperanchored_20260424/selected_sixgene_single_gene_vs_lit_paperanchored.csv')
        path=ROOT/'reproduced_results/seven_systems/shared_scan_results.csv'
        if not path.exists():path=ROOT/'reference_results/original_18p88_unified_shared_scan/shared_scan_results_ranked.csv'
        row=pd.read_csv(path).query("case == 'tx3000_e0.24'").iloc[0];err=[];detail=[]
        for _,x in lit.iterrows():
            for h in [3,6]:
                pred=float(row[f'{x.gene}_{h}h']);target=float(x[f'lit_{h}h_uM_single_proxy']);err.append(abs(pred-target)/target);detail.append({'system':x.gene,'time_h':h,'prediction_uM':pred,'reference_uM':target})
        for h in [3,6]:
            pred=float(row[f'all{h}_uM']);target=float(lit[f'all_GE_total_{h}h_uM_proxy'].iloc[0]);err.append(abs(pred-target)/target);detail.append({'system':'六蛋白共表达','time_h':h,'prediction_uM':pred,'reference_uM':target})
        result={'source':str(path.relative_to(ROOT)),'systems':7,'comparison_points':len(err),'MRE_percent':float(np.mean(err)*100),'detail':detail}
    else:
        result={'systems':{}}
        for condition in ['single_'+g for g in ['Pgi','PfkA','TpiA','GAPDH','Pgk','Eno']]+['coexpression']:
            path=ROOT/'reproduced_results_verified'/condition/'timeseries.csv'
            if not path.exists():
                reference=ROOT/'reference_results'
                path=(reference/'coexpression_72h/sixprotein_continuous_timeseries.csv' if condition=='coexpression' else
                      reference/'six_single_protein_72h'/condition.removeprefix('single_')/'singleprotein_continuous_timeseries.csv')
            frame=pd.read_csv(path);tail=frame[frame.time_h>=48].All_proteins_uM
            result['systems'][condition]={'source':str(path.relative_to(ROOT)),'max_abs_deviation_from_window_mean_uM':float(abs(tail-tail.mean()).max()),'range_uM':float(tail.max()-tail.min())}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
