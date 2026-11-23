"""Frozen predictions from fallible AI codes and previously cached image features."""
import argparse,hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
from ai_ml_hybrid_v1.predict_hybrid import infer as infer_image_branch
from ai_code_fusion_v1.code_models import COMPONENTS,VERSIONS,construct_codes
from ai_eight_condition_v1.eight_models import LEARNED,inputs,load_eight
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def infer(core,geometry,torso,codes,allow_development_smoke=False):
    seal=json.loads((HERE/'frozen_candidates/secondary_release.json').read_text())
    for rel,d in seal['sha256'].items():
        if sha(ROOT/rel)!=d:raise ValueError('Frozen source differs: '+rel)
    # The prior sealed runner supplies fixed image ML only; its other rows are discarded.
    old=infer_image_branch(core,geometry,torso,codes,allow_development_smoke)
    base=old[(old.model=='M56')&(old.method=='fixed_image_ML')].reset_index(drop=True)
    keys=base[['dataset','frame_id','timestamp_seconds']];ml=base[['pred_'+c for c in COMPONENTS]].to_numpy(int)
    models=joblib.load(HERE/'frozen_candidates/models.joblib');rows=[]
    for v in VERSIONS:
        ai,_=load_eight(codes,keys,v)
        for method in LEARNED:
            matrix=np.column_stack([models[v][c][method].predict(inputs(ai[c],ml[:,j],method)) for j,c in enumerate(COMPONENTS)])
            q=keys.copy();q['model']=v;q['method']=method
            for j,c in enumerate(COMPONENTS):q['pred_'+c]=matrix[:,j].astype(int)
            q['load']=1;q['predicted_code']=construct_codes(matrix)
            if 'worker_id' in base:q['worker_id']=base.worker_id
            q['evaluation_status']='technical_development_smoke' if allow_development_smoke else 'external_predictions_only_unscored';rows.append(q)
    return pd.concat(rows,ignore_index=True)
def main():
    p=argparse.ArgumentParser()
    for name in ['features','geometry','torso','ai-codes','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-development-smoke',action='store_true');a=p.parse_args()
    r=infer(a.features,a.geometry,a.torso,a.ai_codes,a.allow_development_smoke);a.output.parent.mkdir(parents=True,exist_ok=True);r.to_csv(a.output,index=False)
    print(json.dumps({'status':'inference_complete','rows':len(r),'model_fitting':False,'expert_labels_used':False,'new_AI_calls':False,'raw_photos_processed':False}))
if __name__=='__main__':main()
