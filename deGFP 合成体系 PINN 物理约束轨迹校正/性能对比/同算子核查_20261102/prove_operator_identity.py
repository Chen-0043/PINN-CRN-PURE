from pathlib import Path
import json,sys
import numpy as np,pandas as pd,torch
from scipy import sparse
R=Path(__file__).resolve().parents[2]
D=R/'model_evaluation_20260730';sys.path.insert(0,str(D))
from full_mass_action_pinn import FullMassActionOperator
from train_continuous_full_crn_pinn_key import build_exchange,DEFAULT_CONTINUOUS_XML
from generate_stiff_hidden_anchors import build_exchange_numpy,build_exchange_plan
F=D/'article_gapfill_20260730/full_elementary_pinn'
ids=pd.read_csv(F/'species_index.csv').sort_values('species_index').species_id.astype(str).tolist()
x=np.load(F/'initial_concentrations_x0_M.npy');k=np.load(F/'rate_constants_k.npy');A=sparse.load_npz(F/'reactant_exponent_matrix_A.npz');S=sparse.load_npz(F/'stoichiometric_matrix_S.npz');n=np.asarray(A.sum(0)).ravel()
cp=torch.load(F/'continuous_6h_trackpriority_tol15_finetune_200_checkpoint.pt',map_location='cpu',weights_only=False)
state=cp.get('model_state_dict',cp.get('state_dict',{}));initial=state.get('x0')
if initial is None: print('checkpoint keys',list(cp));raise RuntimeError('Need exact saved initial state')
plan=build_exchange_plan(ids,DEFAULT_CONTINUOUS_XML)
def np_rhs(x,rate):
 v=np.exp(np.clip(A.T@np.log(np.maximum(x,1e-30))+np.log(rate),-100,100))
 return S@v+build_exchange_numpy(x,*plan)
op=FullMassActionOperator(F,torch.float64)
t=torch.tensor([[0.0]],dtype=torch.float64);xx=torch.tensor(x[None,:],dtype=torch.float64)
with torch.no_grad():
 e,_=build_exchange(t,xx,ids,DEFAULT_CONTINUOUS_XML,torch.float64);y=op(xx,e).numpy()[0]
a=np_rhs(x,k);b=np_rhs(x,k*1e6**np.maximum(n-1,0))
(Path(__file__).resolve().parent/'reproduced_results').mkdir(exist_ok=True)
r={'initial_asset_equals_formal_checkpoint':bool(np.array_equal(x,initial.numpy())),
 'same_raw_k_numpy_vs_actual_pinn_allclose':bool(np.allclose(y,a,rtol=1e-12,atol=1e-30)),
 'same_raw_k_max_rhs_difference_M_s':float(np.max(np.abs(y-a))),
 'old_uM_flag_max_rhs_difference_M_s':float(np.max(np.abs(y-b))),
 'different_rate_constants_under_old_flag':int(np.count_nonzero(k!=k*1e6**np.maximum(n-1,0))),
 'old_declared_operator_identity_is_reproduced':bool(np.allclose(y,b,rtol=1e-12,atol=1e-30))}
(Path(__file__).resolve().parent/'reproduced_results'/'operator_identity_actual.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
assert r['initial_asset_equals_formal_checkpoint'] and r['same_raw_k_numpy_vs_actual_pinn_allclose']
