"""Sealed cached-feature and code inference; no labels or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ai_code_fusion_v1.code_models import COMPONENTS,VERSIONS,load_three_codes,construct_codes
from ai_ml_hybrid_v1.hybrid_models import SYSTEMS,assemble,predict_base,fill_NA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def infer(core_path,geometry_path,torso_path,codes_path,allow_development_smoke=False):
    seal=json.loads((HERE/'frozen_candidates/secondary_release.json').read_text())
    for rel,digest in seal['sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source differs: '+rel)
    core=np.load(core_path,allow_pickle=False);torso=np.load(torso_path,allow_pickle=False);geo=np.load(geometry_path,allow_pickle=False)
    identity=['dataset','frame_id','timestamp_seconds']
    for key in identity:
        if not np.array_equal(core[key],torso[key]):raise ValueError('Torso identity mismatch')
        if key not in geo.files:
            if not(allow_development_smoke and sha(geometry_path)=='cfe76b5fe49fe4e3e3148f872cf038017b259971f776649cad1e54845eab9f82'):raise ValueError('Geometry requires identity arrays')
        elif not np.array_equal(core[key],geo[key]):raise ValueError('Geometry identity mismatch')
    raw=pd.read_csv(codes_path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    if not allow_development_smoke:
        for z in [core,torso,geo]:
            if 'worker_id' not in z.files or np.isin(np.char.strip(z['worker_id'].astype(str)),['','W001']).any():raise ValueError('External feature files require non-development worker_id')
            if not np.array_equal(z['worker_id'],core['worker_id']):raise ValueError('Feature worker mismatch')
        if 'worker_id' not in raw or raw.worker_id.nunique()!=1 or raw.worker_id.iloc[0] in ['','W001']:raise ValueError('External AI codes require non-development worker_id')
        if not np.all(core['worker_id']==raw.worker_id.iloc[0]):raise ValueError('Code/feature worker mismatch')
    n=len(core['frame_id']);keys=pd.DataFrame({k:core[k] for k in identity})
    if not n or keys.duplicated(['dataset','frame_id']).any():raise ValueError('Empty/duplicate identities')
    if not np.isfinite(keys.timestamp_seconds).all() or keys.timestamp_seconds.lt(0).any():raise ValueError('Invalid elapsed seconds')
    arrays={k:core[k] for k in ['pose_front','pose_side','pose_detected_front','pose_detected_side']}
    arrays.update({k:torso[k] for k in ['torso_front','torso_side']});arrays.update({k:geo[k] for k in ['geometry_front','geometry_side','available_front','available_side']})
    x,av=assemble(arrays);bundle=joblib.load(HERE/'frozen_candidates/models.joblib');query=np.arange(n)
    ml=np.column_stack([predict_base(bundle['base_models'][c],x[c],av[c],query) for c in COMPONENTS]);rows=[]
    for v in VERSIONS:
        ai,archived=load_three_codes(codes_path,keys,v)
        for method in SYSTEMS:
            p=[]
            for j,c in enumerate(COMPONENTS):
                if method=='archived_both':pred=ai[c][:,2]
                elif method=='fixed_image_ML':pred=ml[:,j]
                elif method=='ai_NA_filled_by_ML':pred=fill_NA(ai[c][:,2],ml[:,j])
                elif method=='component_source_selection':pred=ml[:,j] if bundle['source_selections'][v][c]=='ML' else ai[c][:,2]
                else:
                    xx=np.column_stack([ai[c],ml[:,j]]) if method=='ai_ML_stacking' else ai[c]
                    pred=bundle['meta_models'][v][c][method].predict(xx)
                p.append(pred)
            matrix=np.column_stack(p);q=keys.copy();q['model']=v;q['method']=method
            for j,c in enumerate(COMPONENTS):q['pred_'+c]=matrix[:,j].astype(int)
            q['load']=1
            if method=='archived_both':
                q['predicted_code']=[s if len(s)==4 and s[0] in '1234' and s[1] in '123' and s[2] in '1234567' and s[3]=='1' else 'NA' for s in archived['3']]
            else:q['predicted_code']=construct_codes(matrix)
            if 'worker_id' in core.files:q['worker_id']=core['worker_id']
            q['evaluation_status']='technical_development_smoke' if allow_development_smoke else 'external_predictions_only_unscored'
            rows.append(q)
    return pd.concat(rows,ignore_index=True)

def main():
    p=argparse.ArgumentParser()
    for name in ['features','geometry','torso','ai-codes','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-development-smoke',action='store_true');a=p.parse_args()
    result=infer(a.features,a.geometry,a.torso,a.ai_codes,a.allow_development_smoke);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    print(json.dumps({'status':'inference_complete','rows':len(result),'model_fitting':False,'expert_labels_used':False,'new_AI_calls':False,'raw_photos_processed':False}))
if __name__=='__main__':main()
