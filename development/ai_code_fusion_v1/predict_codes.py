"""Frozen code fusion without fitting, expert labels, images or AI calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ai_code_fusion_v1.code_models import COMPONENTS,VERSIONS,load_three_codes,majority,construct_codes

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def infer(path,allow_development_smoke=False):
    release=json.loads((HERE/'frozen_candidates/secondary_release.json').read_text())
    for rel,digest in release['sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source differs: '+rel)
    raw=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    if not allow_development_smoke:
        if 'worker_id' not in raw.columns:raise ValueError('External inputs require worker_id')
        if raw.worker_id.str.strip().isin(['','W001']).any():raise ValueError('Blank/development worker is not external')
        if raw.worker_id.nunique()!=1:raise ValueError('One worker per inference run')
        if sha(path)=='9334f156d90241731e466721e898b36c227d21f17a448e3b6cb93c6bd328bba2':raise ValueError('Development input is not external')
    keys=raw[raw.combination.isin(['1','2','3'])][['dataset','frame_id']].drop_duplicates().reset_index(drop=True)
    if not len(keys) or keys.dataset.str.strip().eq('').any() or keys.frame_id.str.strip().eq('').any():raise ValueError('Missing identity')
    models=joblib.load(HERE/'frozen_candidates/models.joblib');rows=[]
    for version in VERSIONS:
        x,_=load_three_codes(path,keys,version)
        for method in ['component_majority','categorical_logistic']:
            pred=np.column_stack([majority(x[c]) if method=='component_majority' else models[version][c].predict(x[c]) for c in COMPONENTS])
            q=keys.copy();q['model']=version;q['method']=method
            for j,c in enumerate(COMPONENTS):q['pred_'+c]=pred[:,j].astype(int)
            q['load']=1;q['predicted_code']=construct_codes(pred)
            if 'worker_id' in raw.columns:q['worker_id']=raw.worker_id.iloc[0]
            q['evaluation_status']='technical_development_smoke' if allow_development_smoke else 'external_predictions_only_unscored'
            rows.append(q)
    return pd.concat(rows,ignore_index=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ai-codes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-development-smoke',action='store_true');args=p.parse_args()
    result=infer(args.ai_codes,args.allow_development_smoke)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(json.dumps({'status':'inference_complete','rows':len(result),'model_fitting':False,'expert_labels_used':False,'new_AI_calls':False,'raw_photos_processed':False}))

if __name__=='__main__':main()
