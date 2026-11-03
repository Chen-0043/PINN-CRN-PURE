from pathlib import Path
import argparse,json,re,sys
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parent
STAGE='current_corrected'
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

def degfp():
    base=ROOT/'model_evaluation_20260730/article_gapfill_20260730'
    full=base/'full_elementary_pinn'
    ids=pd.read_csv(full/'species_index.csv').sort_values('species_index').species_id.to_numpy(str)
    mapping=pd.read_csv(base/'teacher_to_full_mapping/full_species_teacher_mapping.csv')
    mapping=mapping[mapping.teacher_supervision_available].copy()
    indices=mapping.species_index_full.to_numpy(int)
    S=sparse.load_npz(full/'stoichiometric_matrix_S.npz')
    A=sparse.load_npz(full/'reactant_exponent_matrix_A.npz')
    k=np.load(full/'rate_constants_k.npy')
    params={a:float(b) for a,b in re.findall(r'<parameter id="([^"]+)" value="([^"]+)"',(base/'crn_constraint_map/formal_model.xml').read_text())}
    lookup={x:i for i,x in enumerate(ids)}
    snap=pd.read_csv(base/'existing_teacher_constraints/continuous_existing_teacher_train0to5_validate6h_all_species_6h_snapshot.csv').set_index('species_id')
    target=snap.loc[mapping.species_id,'teacher_6h_M'].to_numpy(float)
    inventory=pd.read_csv(full/'reaction_mass_action_inventory.csv')
    involved=inventory.reactants.fillna('')+'; '+inventory.products.fillna('')
    rx_if=involved.str.contains(r'IF1|IF3',regex=True)
    def participants(texts):
        out=set()
        for text in texts:
            for token in text.split(';'):
                token=token.strip()
                if ' ' in token:out.add(token.split(' ',1)[1].strip())
        return out
    if_species=participants(involved[rx_if]) & set(mapping.species_id)
    fixed=pd.read_csv(full/'published_background_reaction_selection.csv')
    energy_species=participants((fixed.reactants.fillna('')+'; '+fixed.products.fillna(''))[fixed.in_background_set])
    module_indices={'IF1_IF3':np.array([lookup[x] for x in sorted(if_species)]),
                    'energy':np.array([lookup[x] for x in sorted(energy_species)])}
    paths={
        'prior':('continuous_6h_trackpriority_tol15_finetune_200_prediction.npz','prediction_M'),
        'current_corrected':('continuous_6h_trackpriority_200_shooting_continue_add300_train5h30_1000_shooting_nodes.npz','optimized_M'),
        'historical_module_source':('continuous_6h_trackpriority_200_shooting_continue_add300_21300_1000_shooting_nodes.npz','optimized_M')}
    paths={STAGE:paths[STAGE]}
    if PREDICTION is not None:paths[STAGE]=(str(PREDICTION.resolve()),paths[STAGE][1])
    out={}
    for label,(name,key) in paths.items():
        z=np.load(full/name);x=np.maximum(z[key],0);t=z['time_s']
        flux=np.exp(np.clip((A.T@np.log(np.maximum(x,1e-30)).T).T+np.log(k),-100,100))
        rhs=(S@flux.T).T
        for param,value in params.items():
            sid=param.removeprefix('bg_feed_')
            if param.startswith('bg_feed_') and sid!='AA' and sid in lookup and sid!='deGFP_m':
                rhs[:,lookup[sid]]+=params.get('bg_gamma',.5/3600)*(value-x[:,lookup[sid]])
        rhs[:,lookup['deGFP_m']]-=params.get('bg_gamma_prod',.5/3600)*x[:,lookup['deGFP_m']]
        integral=.5*np.diff(t)[:,None]*(rhs[1:]+rhs[:-1])*1e6
        delta=np.diff(x,axis=0)*1e6
        row={'source':name,'concentration':concentration(x[-1,indices],target,floor=1e-12),
             'positive':summarize(delta[:,indices],integral[:,indices],0),
             'gt_1_uM':summarize(delta[:,indices],integral[:,indices],1)}
        num=np.sqrt(np.mean((delta-integral)**2,axis=0));den=np.sqrt(np.mean(integral**2,axis=0))
        mask=den[indices]>1;ordered=indices[mask][np.argsort((num[indices]/np.maximum(den[indices],1e-12))[mask])]
        if label=='current_corrected':
            row['L_prox_references']={
                'to_final_segment_initial_state':proximity_to(full/'continuous_6h_trackpriority_200_shooting_nodes_w1_1000_continue_lr005_add300_train5h30_init.npz','optimized_M',t,x),
                'to_first_stage_neural_prior':proximity_to(full/'continuous_6h_trackpriority_tol15_finetune_200_prediction.npz','prediction_M',t,x)}
            keep=ordered[:-13]
            row['after_removing_13_largest']=summarize(delta[:,keep],integral[:,keep],0)
            active=indices[mask]
            for criterion,values in [('rhs',den),('absolute_residual',num)]:
                keep=active[np.argsort(values[active])][:-13]
                row['after_removing_13_largest_'+criterion]=summarize(delta[:,keep],integral[:,keep],0)
        row['modules']={}
        for module,idx in module_indices.items():
            ratios=num[idx]/np.maximum(den[idx],.01)
            row['modules'][module]={'count':len(idx),'direct_comparable_states':sum(int(i) in set(indices) for i in idx),
                'RMS_NRMSR_floor_0p01_percent':float(np.sqrt(np.mean(ratios**2))*100),
                'mean_NRMSR_floor_0p01_percent':float(np.mean(ratios)*100), 'species':ids[idx].tolist()}
        out[label]=row
    return out
def main():
    global PREDICTION
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path);p.add_argument('--output',type=Path,default=ROOT/'reproduced_results/current_metrics.json');a=p.parse_args();PREDICTION=a.prediction
    result=degfp()[STAGE];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
