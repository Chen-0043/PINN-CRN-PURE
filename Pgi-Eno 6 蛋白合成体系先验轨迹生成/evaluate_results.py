from pathlib import Path
import argparse,json,re,sys
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parent
STAGE='prior'
PREDICTION=None
def summarize(delta, integral, threshold):
    num = np.sqrt(np.mean((delta-integral)**2, axis=0))
    den = np.sqrt(np.mean(integral**2, axis=0))
    selected = den > threshold
    ratios = num[selected] / np.maximum(den[selected], 1e-12)
    return {'count': int(selected.sum()), 'mean_NRMSR_percent': float(ratios.mean()*100),
            'equation19_RMS_percent': float(np.sqrt(np.mean(ratios**2))*100)}

def concentration(pred, target, *, floor, exclude_small=False):
    selected = np.abs(target)>floor if exclude_small else np.ones(len(target),dtype=bool)
    relative = np.abs(pred-target)/np.maximum(np.abs(target),floor)
    return {'mapped_states':len(target), 'states_entering_MRE':int(selected.sum()),
            'MRE_percent':float(relative[selected].mean()*100),
            'R2':float(1-np.sum((pred-target)**2)/np.sum((target-target.mean())**2)),
            'MRE_floor_M':floor, 'MRE_excludes_small_targets':exclude_small}

def proximity_to(path, state_key, time, state):
    z=np.load(path);source=np.maximum(z[state_key],0)
    aligned=np.column_stack([np.interp(time,z['time_s'],source[:,i]) for i in range(source.shape[1])])
    return float(np.mean((np.log10(state+1e-12)-np.log10(aligned+1e-12))**2))

def sixprotein(prior_override=None):
    src=ROOT/'src'
    sys.path.insert(0,str(src))
    import torch
    from hybrid_resolution_pinn import HybridResolutionOperator
    from train_teacher_supervised_sixprotein_pinn import aggregate, load_supervision
    folder=src.parent;assets=folder/'model_assets/original_18p88_full_elementary'
    operator=HybridResolutionOperator(assets,folder/'inputs/input_pool_override_sg_pure2_paper_mapped_20260424.json',energy_scale=.24,gamma_h=.5,include_all_teacher_metabolites=True,dtype=torch.float64)
    sup=load_supervision(assets/'sixprotein_teacher_supervision_all_metabolites_dt0p125.npz',operator.n_species,torch.float64)
    idx=torch.tensor(sup['direct_indices_np'],dtype=torch.long)
    target=np.concatenate([sup['direct_target_np'][-1],sup['group_target_np'][-1]])
    out={}
    path=PREDICTION or ROOT/'reference_results/final_prior/prior_prediction_6h.npz'
    evaluations=[('prior',path,.24)]
    for label,path,energy_scale in evaluations:
        operator=HybridResolutionOperator(assets,folder/'inputs/input_pool_override_sg_pure2_paper_mapped_20260424.json',energy_scale=energy_scale,gamma_h=.5,include_all_teacher_metabolites=True,dtype=torch.float64)
        z=np.load(path);state=torch.tensor(z['prediction_M'],dtype=torch.float64);t=z['time_s']
        with torch.no_grad():
            rhs=operator(state)
            obs=torch.cat([state.index_select(1,idx),aggregate(sup['group_matrix'],state)],dim=1).numpy()
            observed_rhs=torch.cat([rhs.index_select(1,idx),aggregate(sup['group_matrix'],rhs)],dim=1).numpy()
        integral=.5*np.diff(t)[:,None]*(observed_rhs[1:]+observed_rhs[:-1])*1e6
        delta=np.diff(obs,axis=0)*1e6
        out[label]={'operator_energy_scale':energy_scale,
                    'concentration':concentration(obs[-1],target,floor=1e-12),
                    'historical_concentration_nonzero':concentration(obs[-1],target,floor=1e-15,exclude_small=True),
                    'positive':summarize(delta,integral,0),'gt_1_uM':summarize(delta,integral,1)}
    return out
def main():
    global PREDICTION
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path);p.add_argument('--output',type=Path,default=ROOT/'reproduced_results/current_metrics.json');a=p.parse_args();PREDICTION=a.prediction
    result=sixprotein()[STAGE];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
