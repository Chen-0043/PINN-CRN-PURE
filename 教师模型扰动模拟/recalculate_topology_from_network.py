from pathlib import Path
import re, json, collections
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent / 'model_inputs/teacher_topology_network'
rxn=pd.read_csv(ROOT/'reaction_constraint_inventory.csv')
sp=pd.read_csv(ROOT/'species_index.csv')
OUT=Path(__file__).resolve().parent/'reproduced_results/topology_from_network.csv'
SUM=ROOT/'teacher_topology_rank_real_perturbable_with_pseudo_in_metrics_equal4_summary_20260824.json'

def parse_side(x):
    if pd.isna(x): return []
    out=[]
    for p in str(x).split(';'):
        p=p.strip()
        if not p: continue
        p=re.sub(r'^\s*[0-9.]+\s+','',p).strip()
        if p: out.append(p)
    return out
all_species=set(sp['species_id'].astype(str))
rxn_touched=[]
for _,r in rxn.iterrows():
    touched=set(parse_side(r['reactants'])+parse_side(r['products']))
    rxn_touched.append((r['reaction_id'], r['constraint_class'], touched))
module_patterns={
 'transcription': r'T7RNAP|DNA|mRNA|_TX\b',
 'translation_ribosome': r'Ribosome|RS30S|RS50S|RS70S|TLchain|Pept',
 'translation_factors': r'EFTu|EFG|IF1|IF2|IF3|RF1|RF2|RF3|RRF',
 'aa_tRNA': r'tRNA|[A-Z][a-z]{2}RS|\bAA\b',
 'energy_ntp': r'ATP|ADP|AMP|GTP|GDP|GMP|CTP|UTP|PPi|PO4|CP|CK|MK|NDK|PPiase',
 'output': r'deGFP',
}
# Ranked objects: exclude bookkeeping and model-state pseudo species from candidate/evaluated objects.
def is_pseudo_rank_object(s):
    return (
        s.startswith('Moiety_') or
        s.startswith('TLchain_pos_') or
        s.startswith('transcription_') or
        s.startswith('translation_') or
        s.startswith('rxn_') or
        re.match(r'^(elRS70S|termRS70S)', s) is not None
    )
# Keep broad real biochemical-ish objects and complexes; only pseudo state/bookkeeping removed from participants.
rank_species=[s for s in sorted(all_species) if not is_pseudo_rank_object(s)]

def features_for(seedset):
    seedset=set(seedset)&all_species
    direct=[]; neigh=set(seedset); cls=collections.Counter()
    for rid,cl,touched in rxn_touched:
        if touched & seedset:
            direct.append(rid); neigh.update(touched); cls[cl]+=1
    mods=set()
    for s in neigh:  # pseudo/state variables remain in metric calculation
        for m,pat in module_patterns.items():
            if re.search(pat,s): mods.add(m)
    return {
        'seed_species_count':len(seedset),
        'direct_reaction_count':len(direct),
        'one_hop_species_count':len(neigh),
        'module_coverage_count':len(mods),
        'module_coverage':';'.join(sorted(mods)),
        'effective_non_mass_action_reaction_count':int(cls['effective_non_mass_action']),
        'strict_mass_action_reaction_count':int(cls['strict_mass_action_monomial']),
        'zero_order_source_reaction_count':int(cls['zero_order_source']),
    }
rows=[]
for s in rank_species:
    rows.append({'object_type':'species','object_id':s,'display_name':s, **features_for([s])})
# Replace AA and aaRS individual perturbation interpretation with a total aminoacylation module object.
aars=[s for s in all_species if re.fullmatch(r'[A-Z][a-z]{2}RS',s)]
total_aars=['AA']+aars
rows.append({'object_type':'module','object_id':'total_aminoacylation_module','display_name':'总氨酰化模块', **features_for(total_aars)})
df=pd.DataFrame(rows)
# Remove AA and individual aaRS from displayed/evaluated perturbation objects because user wants module as article perturbation.
remove_as_individual=set(total_aars)
df=df[~((df['object_type']=='species') & (df['object_id'].isin(remove_as_individual)))].copy()

# Preserve the original two-pass table construction and its tie ordering.
old_metrics=['direct_reaction_count','one_hop_species_count','module_coverage_count','effective_non_mass_action_reaction_count']
for c in old_metrics:
 lo,hi=df[c].min(),df[c].max();df[c+'_norm']=0. if hi==lo else (df[c]-lo)/(hi-lo)
df['topology_priority_score_equal4']=sum(df[c+'_norm'] for c in old_metrics)/4.
df=df.sort_values('topology_priority_score_equal4',ascending=False).reset_index(drop=True)
intermediate=Path(__file__).resolve().parent/'reproduced_results/topology_first_pass.csv'
intermediate.parent.mkdir(exist_ok=True);df.to_csv(intermediate,index=False);df=pd.read_csv(intermediate)
df['qssa_background_reaction_count']=df.effective_non_mass_action_reaction_count+df.zero_order_source_reaction_count
metrics=['direct_reaction_count','one_hop_species_count','module_coverage_count','qssa_background_reaction_count']
for c in metrics:
 lo,hi=df[c].min(),df[c].max();df[c+'_norm']=0. if hi==lo else (df[c]-lo)/(hi-lo)
df['topology_priority_score_equal4_qssa_background']=df[[c+'_norm' for c in metrics]].mean(axis=1)
df=df.sort_values(['topology_priority_score_equal4_qssa_background','direct_reaction_count','one_hop_species_count'],ascending=[False,False,False]).reset_index(drop=True)
df['rank_equal4_qssa_background']=df.index+1
OUT.parent.mkdir(exist_ok=True);df.to_csv(OUT,index=False)
reference=pd.read_csv(Path(__file__).resolve().parent/'topology_priority/teacher_topology_rank_real_perturbable_qssa_background_equal4_20260824.csv').set_index('object_id')
comp=df.set_index('object_id').loc[reference.index]
report={'objects':len(df),'raw_feature_max_differences':{c:float(abs(comp[c]-reference[c]).max()) for c in metrics},'score_max_difference':float(abs(comp.topology_priority_score_equal4_qssa_background-reference.topology_priority_score_equal4_qssa_background).max()),'rank_max_difference':int(abs(comp.rank_equal4_qssa_background-reference.rank_equal4_qssa_background).max())}
import libsbml
document=libsbml.readSBMLFromFile(str(ROOT/'formal_model.xml'));model=document.getModel()
xmlids={x.getId() for x in model.getListOfSpecies()};xmlreactions={x.getId() for x in model.getListOfReactions()}
report['species_inventory_matches_XML']=xmlids==all_species
report['reaction_inventory_matches_XML']=xmlreactions==set(rxn.reaction_id)
report['reaction_participant_mismatches']=[]
for _,r in rxn.iterrows():
 reaction=model.getReaction(str(r.reaction_id));actual={x.getSpecies() for x in list(reaction.getListOfReactants())+list(reaction.getListOfProducts())}
 expected=set(parse_side(r.reactants)+parse_side(r.products))
 if actual!=expected:report['reaction_participant_mismatches'].append(str(r.reaction_id))
(OUT.parent/'topology_from_network_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
