"""Apply already frozen models to new-worker photographs without refitting.

Image metadata and expert references are separate. Labels are joined only after
all predictions have been generated. Training-image smoke tests are explicit.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from leg_followup.leg_features import extract_geometry
from leg_followup.run_followup import route,temperature_apply,CLASSES

OUT=ROOT/'leg_followup'
KEYS=['worker_id','recording_id','frame_id']


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--samples',type=Path,required=True)
 parser.add_argument('--references',type=Path)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--allow-training-smoke-test',action='store_true')
 args=parser.parse_args()
 if args.allow_training_smoke_test and args.references:raise ValueError('A training-image smoke test must not be reported as external evaluation')
 config=json.loads((OUT/'frozen_workflow/workflow.json').read_text())
 assert hashlib.sha256((OUT/'leg_features.py').read_bytes()).hexdigest()==config['feature_function_sha256']
 assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==config['protocol_sha256']
 models={}
 for name,info in config['classification_benchmarks'].items():
  path=OUT/'frozen_workflow'/info['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
  models[name]=joblib.load(path)
 for view,digest in config['model_hashes'].items():assert hashlib.sha256((OUT/'frozen_workflow'/(view+'_rf.joblib')).read_bytes()).hexdigest()==digest
 side_model=joblib.load(OUT/'frozen_workflow/side_rf.joblib');pair_model=joblib.load(OUT/'frozen_workflow/paired_rf.joblib')
 samples=pd.read_csv(args.samples,dtype=str,keep_default_na=False)
 required=KEYS+['timestamp_seconds','side_image_path','front_image_path']
 if not set(required)<=set(samples):raise ValueError('Missing required sample fields: '+str(set(required)-set(samples)))
 if samples.empty or samples[KEYS].duplicated().any():raise ValueError('Samples must contain unique nonempty observation keys')
 if any(not x for x in samples[KEYS].to_numpy().reshape(-1)):raise ValueError('Observation keys cannot be empty')
 if not args.allow_training_smoke_test and set(samples.worker_id)&set(config['training_worker_ids']):raise ValueError('A training worker cannot be called external validation')
 training_hashes=set(pd.read_csv(ROOT/'audit/image_manifest.csv').image_sha256)
 from PIL import Image
 import mediapipe as mp
 from extract_features import pose_features
 pose_asset=ROOT/'models/pose_landmarker_full.task'
 asset_record=next(x for x in json.loads((ROOT/'models/model_assets.json').read_text()) if x['filename']==pose_asset.name)
 assert hashlib.sha256(pose_asset.read_bytes()).hexdigest()==asset_record['sha256']
 options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_asset)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5)
 arrays={k:[] for k in ['geometry_front','geometry_side','generic_front','generic_side','quality_front','quality_side','available_front','available_side']}
 with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
  for row in samples.itertuples(index=False):
   for view in ['front','side']:
    path=Path(getattr(row,view+'_image_path'))
    if not path.is_absolute():path=args.samples.parent/path
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in training_hashes and not args.allow_training_smoke_test:raise ValueError('An image is identical to a training-source photograph')
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
    result=pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    landmarks=result.pose_landmarks[0] if result.pose_landmarks else []
    raw=[[p.x,p.y,p.z,p.visibility,p.presence] for p in landmarks]
    geometry,available,quality,_=extract_geometry(raw,rgb.shape[1],rgb.shape[0]);generic,_=pose_features(landmarks,rgb.shape[1],rgb.shape[0])
    for k,value in [('geometry',geometry),('generic',generic),('quality',quality),('available',available)]:arrays[k+'_'+view].append(value)
 arrays={k:np.asarray(v) for k,v in arrays.items()}
 pair_available=arrays['available_front']|arrays['available_side'];pair_quality=arrays['quality_front']|arrays['quality_side']
 all_predictions=[]
 for name,info in config['classification_benchmarks'].items():
  xs=[]
  for view in ['front','side']:
   x=arrays['geometry_'+view]
   if info['feature_set']=='generic':x=arrays['generic_'+view]
   elif info['feature_set']=='generic_plus_geometry':x=np.concatenate([arrays['generic_'+view],x],axis=1)
   xs.append(x)
  x=np.concatenate(xs,axis=1);pred=np.zeros(len(samples),int)
  if pair_available.any():pred[pair_available]=models[name].predict(x[pair_available])
  q=samples[KEYS].copy();q['benchmark']=name;q['pred_legs']=pred;all_predictions.append(q)
 def probabilities(model,x,available):
  p=np.zeros((len(samples),5))
  if available.any():
   raw=model.predict_proba(x[available])
   for j,c in enumerate(model.classes_):p[np.flatnonzero(available),np.searchsorted(CLASSES,c)]=raw[:,j]
  return p
 side=probabilities(side_model,arrays['geometry_side'],arrays['available_side'])
 pair=probabilities(pair_model,np.concatenate([arrays['geometry_front'],arrays['geometry_side']],axis=1),pair_available)
 side=temperature_apply(side,config['calibration']['side']['temperature']);pair=temperature_apply(pair,config['calibration']['paired']['temperature'])
 policies=[]
 for name,rule in config['policy_rules'].items():
  pred,stage,opened=route(side,pair,arrays['quality_side'],pair_quality,rule['side_threshold'],rule['paired_threshold'],rule['policy'],rule['force_referral'])
  q=samples[KEYS].copy();q['policy_rule']=name;q['pred_legs']=pred;q['stage']=stage;q['second_view_used']=opened
  q['primary_external_rule']=name==(config['primary_policy']+':'+config['primary_target']);policies.append(q)
 for view,p in [('side',side),('paired',pair)]:
  available=p.sum(axis=1)>0;pred=np.zeros(len(samples),int);pred[available]=CLASSES[np.argmax(p[available],axis=1)]
  q=samples[KEYS].copy();q['policy_rule']='unselective_'+view;q['pred_legs']=pred;q['stage']=np.where(available,view,'expert')
  q['second_view_used']=view=='paired';q['primary_external_rule']=False;policies.append(q)
 classifiers=pd.concat(all_predictions,ignore_index=True);policies=pd.concat(policies,ignore_index=True)
 args.output_dir.mkdir(parents=True,exist_ok=True)
 classifiers.to_csv(args.output_dir/'classifier_predictions.csv',index=False);policies.to_csv(args.output_dir/'policy_predictions.csv',index=False)
 status={'status':'training_image_smoke_test_only' if args.allow_training_smoke_test else 'frozen_new_data_predictions_generated',
         'sample_n':len(samples),'worker_ids':sorted(set(samples.worker_id)),'refitted':False,'expert_outcomes_simulated':False,
         'worker_identity_basis':'author metadata, not automatic identity recognition',
         'frozen_workflow_sha256':hashlib.sha256((OUT/'frozen_workflow/workflow.json').read_bytes()).hexdigest()}
 if args.references:
  refs=pd.read_csv(args.references,dtype=str,keep_default_na=False)
  if not set(KEYS+['consensus_both_code','reference_basis'])<=set(refs) or refs[KEYS].duplicated().any():raise ValueError('Reference file has missing fields or duplicate keys')
  refs['valid_reference']=refs.consensus_both_code.str.fullmatch(r'[1-4][1-3][1-7][1-3]')
  refs['ref_legs']=[int(c[2]) if ok else 0 for c,ok in zip(refs.consensus_both_code,refs.valid_reference)]
  refs['eligible_leg_scope']=refs.ref_legs.isin(CLASSES)&(refs.reference_basis=='paired_images_only')
  summary=[]
  for kind,prediction,group in [('classifier',classifiers,'benchmark'),('policy',policies,'policy_rule')]:
   merged=prediction.merge(refs[KEYS+['valid_reference','ref_legs','reference_basis','eligible_leg_scope']],on=KEYS,how='left',validate='many_to_one')
   merged.to_csv(args.output_dir/(kind+'_evaluation_rows.csv'),index=False)
   for name,q in merged.groupby(group):
    eligible=q.eligible_leg_scope.fillna(False).astype(bool);r=q[eligible];accepted=r.pred_legs>0;correct=r.pred_legs==r.ref_legs
    summary.append({'kind':kind,'name':name,'all_image_n':len(q),'eligible_scored_n':len(r),'unscored_or_outside_scope_n':int((~eligible).sum()),
                    'correct_scored_n':int(correct.sum()),'accuracy_all_eligible':float(correct.mean()) if len(r) else None,
                    'macro_f1_fixed5':f1_score(r.ref_legs,r.pred_legs,labels=CLASSES,average='macro',zero_division=0) if len(r) else None,
                    'classes_observed':','.join(map(str,sorted(set(r.ref_legs)))),'accepted_all_images':int((q.pred_legs>0).sum()),
                    'coverage_all_images':float((q.pred_legs>0).mean()),'accepted_eligible':int(accepted.sum()),
                    'accuracy_accepted_eligible':float(correct[accepted].mean()) if accepted.any() else None})
  pd.DataFrame(summary).to_csv(args.output_dir/'evaluation_summary.csv',index=False);status['external_reference_file_scored']=True
 (args.output_dir/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status,indent=2))


if __name__=='__main__':main()
