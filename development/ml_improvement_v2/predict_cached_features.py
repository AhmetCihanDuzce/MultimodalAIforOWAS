"""Frozen v2 secondary inference on cached features, without labels or fitting.

This does not extract raw photographs or replace the original v1 external
runner. A development smoke run is technical parity only, never accuracy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from ml_improvement_v2.run_improvements import ImprovementStudy, COMPONENTS, sha

def load_inputs(core_path,geometry_path,torso_path,allow_smoke):
    core=np.load(core_path,allow_pickle=False)
    keys=['dataset','frame_id','timestamp_seconds','pose_front','pose_side',
          'pose_detected_front','pose_detected_side','embedding_front','embedding_side']
    arrays={k:core[k] for k in keys}
    if not allow_smoke:
        if 'worker_id' not in core.files:raise ValueError('External cached features require worker_id metadata')
        if np.isin(core['worker_id'].astype(str),['W001','']).any():raise ValueError('Development/blank worker_id is not external')
    if 'worker_id' in core.files:arrays['worker_id']=core['worker_id']
    for path,include in [(geometry_path,['geometry_front','geometry_side','available_front','available_side']),
                         (torso_path,['torso_front','torso_side'])]:
        z=np.load(path,allow_pickle=False)
        for k in ['dataset','frame_id','timestamp_seconds']:
            if k not in z.files:
                expected='cfe76b5fe49fe4e3e3148f872cf038017b259971f776649cad1e54845eab9f82'
                if not (allow_smoke and path==geometry_path and sha(path)==expected):
                    raise ValueError('External geometry/torso features require identity arrays: '+k)
            elif not np.array_equal(arrays[k],z[k]):raise ValueError('Feature identity mismatch: '+k)
        arrays.update({k:z[k] for k in include})
    n=len(arrays['frame_id'])
    if n==0:raise ValueError('No observations')
    if len(set(zip(arrays['dataset'].astype(str),arrays['frame_id'].astype(str))))!=n:raise ValueError('Duplicate observation identity')
    if not np.isfinite(arrays['timestamp_seconds']).all() or (arrays['timestamp_seconds']<0).any():raise ValueError('Invalid elapsed seconds')
    return arrays

def infer_cached(arrays):
    config=json.loads((HERE/'frozen_candidates/workflow.json').read_text())
    if sha(HERE/'protocol_freeze.json')!=config['protocol_freeze_sha256']:raise ValueError('Frozen protocol differs')
    if sha(HERE/'frozen_candidates/models.joblib')!=config['models_sha256']:raise ValueError('Frozen model hash differs')
    for rel,digest in config['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source differs: '+rel)
    models=joblib.load(HERE/'frozen_candidates/models.joblib')
    # Only the pure assembly/inference methods are used; no Benchmark init or fit.
    study=object.__new__(ImprovementStudy);study.arrays=arrays;study.feature_cache={}
    indices=np.arange(len(arrays['frame_id']))
    rows=[]
    for system,mapping in config['systems'].items():
        p={}
        for comp,key in mapping.items():
            entry=models[key];p[comp]=study.infer(entry['model'],entry['candidate'],indices)
        for i in indices:
            row={'dataset':str(arrays['dataset'][i]),'frame_id':str(arrays['frame_id'][i]),
                 'timestamp_seconds':float(arrays['timestamp_seconds'][i]),'method':system}
            if 'worker_id' in arrays:row['worker_id']=str(arrays['worker_id'][i])
            row.update({'pred_'+c:int(p[c][i]) for c in COMPONENTS})
            row['load']=1
            row['OWAS_code']=''.join(str(int(p[c][i])) for c in COMPONENTS)+'1' if all(p[c][i]>0 for c in COMPONENTS) else 'NA'
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True);parser.add_argument('--torso',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-development-smoke',action='store_true')
    args=parser.parse_args()
    arrays=load_inputs(args.features,args.geometry,args.torso,args.allow_development_smoke)
    result=infer_cached(arrays)
    result['evaluation_status']='technical_development_smoke' if args.allow_development_smoke else 'external_predictions_only_unscored'
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(json.dumps({'status':'inference_complete','rows':len(result),'observations':len(arrays['frame_id']),
                     'labels_read':False,'model_fitting':False,'raw_photos_processed':False,
                     'evaluation_status':str(result.evaluation_status.iloc[0])}))

if __name__=='__main__':main()
