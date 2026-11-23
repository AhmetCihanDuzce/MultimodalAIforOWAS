"""Replay the frozen test-set predictions from released features and LLM codes."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parent
OLD=BASE/'development'
OUT=BASE/'reproduced'
OUT.mkdir(exist_ok=True)
TEST=BASE/'test'
FROZEN=TEST/'frozen_models'
sys.path.insert(0,str(OLD))

from modular_followup.feature_interface import assemble as modular_assemble,predict_fitted
from ai_code_fusion_v1.code_models import COMPONENTS,VERSIONS,load_three_codes,majority,construct_codes
from ai_eight_condition_v1.eight_models import LEARNED,inputs,load_eight
from ai_ml_hybrid_v1.hybrid_models import assemble as hybrid_assemble,predict_base,fill_NA
from ml_improvement_v2.run_improvements import ImprovementStudy


def load_branch(branch):
    directory = FROZEN / branch
    config = json.loads((directory / "workflow.json").read_text())
    path = directory / "models.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["models_sha256"]:
        raise ValueError(f"Archived model checksum differs for {branch}")
    return config, joblib.load(path)

z=np.load(TEST/'features/new_features.npz',allow_pickle=False)
arrays={k:z[k] for k in z.files}
keys=pd.DataFrame({k:arrays[k] for k in ['dataset','frame_id']})
codes=TEST/'ai_codes.csv'
raw=pd.read_csv(codes,dtype=str,keep_default_na=False)
records=[]
def add(model,method,matrix,code=None,branch=''):
    matrix=np.asarray(matrix,dtype=int)
    assert matrix.shape==(len(keys),3)
    q=keys.copy();q['model']=model;q['method']=method;q['branch']=branch
    for j,c in enumerate(COMPONENTS):q['pred_'+c]=matrix[:,j]
    q['load']=1;q['predicted_code']=construct_codes(matrix) if code is None else code
    records.append(q)

with threadpool_limits(limits=1):
    config,models=load_branch('modular_followup')
    cache={}
    for k,entry in models.items():
        x,av=modular_assemble(arrays,entry['candidate'])
        cache[k]=predict_fitted(entry['model'],x,av,entry['candidate']['hierarchy'])[0]
    for name,mapping in config['systems'].items():
        add('IMAGE',name,np.column_stack([cache[mapping[c]] for c in COMPONENTS]),branch='modular_v1')


    config, models = load_branch("ml_improvement_v2")
    study = object.__new__(ImprovementStudy)
    study.arrays = arrays
    study.feature_cache = {}
    indices = np.arange(len(keys))
    for name, mapping in config["systems"].items():
        components=[]
        for component in COMPONENTS:
            entry=models[mapping[component]]
            components.append(study.infer(entry["model"], entry["candidate"], indices))
        add("IMAGE", name, np.column_stack(components), branch="image_v2")
    hybrid=load_branch('ai_ml_hybrid_v1')[1]
    x,av=hybrid_assemble(arrays);query=np.arange(len(keys))
    ml=np.column_stack([predict_base(hybrid['base_models'][c],x[c],av[c],query) for c in COMPONENTS])
    add('IMAGE','fixed_image_ML',ml,branch='fixed_hybrid_image_branch')
    three_models=load_branch('ai_code_fusion_v1')[1]
    eight_models=load_branch('ai_eight_condition_v1')[1]
    for version in VERSIONS:
        ai,archive=load_eight(codes,keys,version)
        three,_=load_three_codes(codes,keys,version)
        for c in COMPONENTS:assert np.array_equal(three[c],ai[c][:,:3])
        for condition in range(1,9):
            add(version,f'raw_condition_{condition}',np.column_stack([ai[c][:,condition-1] for c in COMPONENTS]),
                code=archive[str(condition)],branch='raw_AI')
        add(version,'component_majority',np.column_stack([majority(three[c]) for c in COMPONENTS]),branch='fixed_vote')
        add(version,'categorical_logistic',np.column_stack([three_models[version][c].predict(three[c]) for c in COMPONENTS]),branch='three_code_fusion')
        for method in ['ai_NA_filled_by_ML','component_source_selection','ai_codes_logistic','ai_ML_stacking']:
            predictions=[]
            for j,c in enumerate(COMPONENTS):
                if method=='ai_NA_filled_by_ML':pred=fill_NA(three[c][:,2],ml[:,j])
                elif method=='component_source_selection':pred=ml[:,j] if hybrid['source_selections'][version][c]=='ML' else three[c][:,2]
                else:
                    xx=np.column_stack([three[c],ml[:,j]]) if method=='ai_ML_stacking' else three[c]
                    pred=hybrid['meta_models'][version][c][method].predict(xx)
                predictions.append(pred)
            add(version,method,np.column_stack(predictions),branch='hybrid_v1')
        for method in LEARNED:
            matrix=np.column_stack([eight_models[version][c][method].predict(inputs(ai[c],ml[:,j],method))
                for j,c in enumerate(COMPONENTS)])
            add(version,method,matrix,branch='eight_condition_v1')
predictions=pd.concat(records,ignore_index=True)
assert not predictions.duplicated(['model','branch','method','dataset','frame_id']).any()
predictions.to_csv(OUT/'frozen_predictions.csv',index=False)

identity = ["dataset","frame_id","model","branch","method"]
def normalized(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).sort_values(identity).reset_index(drop=True)
pd.testing.assert_frame_equal(normalized(OUT/"frozen_predictions.csv"), normalized(TEST/"frozen_predictions.csv"))
print(f"Reproduced all {len(predictions)} held-out predictions from stored features, frozen classifiers and LLM codes.")
