"""Exploratory leg-only improvements and train-selected selective camera policy."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score
from threadpoolctl import threadpool_limits
from run_experiments import Benchmark,GRIDS,SEED

OUT=ROOT/'leg_followup'
CLASSES=np.array([2,3,4,5,7])
GRID=[0,0.5,0.6,0.7,0.8,0.9,0.95,1.01]
ALIAS={'rf':'pose_RandomForest','svm':'pose_RBF_SVM'}


def temperature_apply(p,t):
 p=np.asarray(p);valid=p.sum(axis=1)>0
 out=np.zeros_like(p,dtype=float)
 out[valid]=softmax(np.log(np.clip(p[valid],1e-8,1))/float(t),axis=1)
 return out


def temperature_fit(p,y):
 valid=p.sum(axis=1)>0;p=p[valid];y=y[valid]
 counts=[int((y==c).sum()) for c in CLASSES]
 meta={'n':len(y),'class_counts':counts,'temperature':1.0,'status':'insufficient_support'}
 if len(y)<50 or min(counts)<5:return meta
 target=np.searchsorted(CLASSES,y)
 def loss(logt):
  pred=temperature_apply(p,np.exp(logt))
  return float(-np.log(np.clip(pred[np.arange(len(y)),target],1e-12,1)).mean())
 result=minimize_scalar(loss,bounds=(-2,2),method='bounded')
 logt=float(result.x) if result.fun<=loss(0) else 0.0
 meta.update(temperature=float(np.exp(logt)),status='fitted_on_training_oof',training_nll_before=loss(0),training_nll_after=loss(logt))
 return meta


def route(side,pair,side_quality,pair_quality,t1,t2,policy='cascade',force_referral=False):
 n=len(side);pred=np.zeros(n,dtype=int);stage=np.array(['expert']*n,dtype=object)
 front_used=np.zeros(n,bool)
 if force_referral:return pred,stage,front_used
 side_ok=(side.sum(axis=1)>0)&side_quality&(side.max(axis=1)>=t1)
 if policy!='paired_only':
  pred[side_ok]=CLASSES[np.argmax(side[side_ok],axis=1)];stage[side_ok]='side'
 if policy=='side_only':return pred,stage,front_used
 front_used[:]=True if policy=='paired_only' else False
 if policy!='paired_only':front_used[~side_ok]=True
 pair_ok=front_used&(pair.sum(axis=1)>0)&pair_quality&(pair.max(axis=1)>=t2)
 pred[pair_ok]=CLASSES[np.argmax(pair[pair_ok],axis=1)];stage[pair_ok]='paired'
 return pred,stage,front_used


def choose_rule(side,pair,sq,pq,y,ds,times,target,policy):
 choices=[]
 thresholds=[(a,b) for a in GRID for b in GRID] if policy=='cascade' else [(a,1.01) for a in GRID] if policy=='side_only' else [(1.01,b) for b in GRID]
 for t1,t2 in thresholds:
  pred,stage,opened=route(side,pair,sq,pq,t1,t2,policy)
  accepted=pred>0;n=int(accepted.sum())
  if n<20:continue
  accuracy=float((pred[accepted]==y[accepted]).mean())
  blocks=set(zip(ds[accepted],times[accepted]//60))
  if accuracy<target or len(blocks)<3 or len(set(y[accepted]))<2 or len(set(ds[accepted]))<2:continue
  choices.append({'side_threshold':t1,'paired_threshold':t2,'training_accepted':n,'training_correct':int((pred[accepted]==y[accepted]).sum()),
                  'training_accuracy':accuracy,'training_second_views':int(opened.sum()),'training_reference_classes':sorted(map(int,set(y[accepted]))),
                  'target':target,'policy':policy,'force_referral':False})
 if not choices:return {'side_threshold':1.01,'paired_threshold':1.01,'training_accepted':0,'target':target,'policy':policy,'force_referral':True,'reason':'no eligible training OOF threshold rule'}
 choices.sort(key=lambda x:(-x['training_accepted'],x['training_second_views'],-x['training_accuracy'],x['side_threshold'],x['paired_threshold']))
 return choices[0]


class LegStudy(Benchmark):
 def __init__(self):
  super().__init__()
  self.g=np.load(OUT/'geometry_features.npz')
  self.y=self.z['y_legs']
  self.indices=np.arange(len(self.df))
  (OUT/'results').mkdir(exist_ok=True)
  (OUT/'cache').mkdir(exist_ok=True)

 def leg_features(self,featureset,view):
  views=['front','side'] if view=='paired' else [view]
  xs=[]
  for v in views:
   x=self.g['geometry_'+v]
   if featureset=='generic_plus_geometry':x=np.concatenate([self.z['pose_'+v],x],axis=1)
   elif featureset=='generic':x=self.z['pose_'+v]
   xs.append(x)
  return np.concatenate(xs,axis=1),np.logical_or.reduce([self.g['available_'+v] for v in views]),np.logical_or.reduce([self.g['quality_'+v] for v in views])

 def fitted_predict(self,x,available,train,test,model,params):
  tr=train[available[train]];tv=test[available[test]]
  pred=np.zeros(len(test),int);prob=np.zeros((len(test),5))
  if not len(tr):return pred,prob,None
  estimator=self.estimator(ALIAS[model],params)
  if len(np.unique(self.y[tr]))==1:
   estimator=DummyClassifier(strategy='most_frequent')
   estimator.fit(np.zeros((len(tr),1)),self.y[tr])
   query=np.zeros((len(tv),1))
  else:
   estimator.fit(x[tr],self.y[tr]);query=x[tv]
  mask=available[test]
  if len(tv):
   pred[mask]=estimator.predict(query)
   if model=='rf':
    raw=estimator.predict_proba(query)
    for j,c in enumerate(estimator.classes_):prob[np.flatnonzero(mask),int(np.searchsorted(CLASSES,c))]=raw[:,j]
  return pred,prob,estimator

 def tune(self,featureset,view,model,train,test,block,fold):
  x,available,quality=self.leg_features(featureset,view)
  splits=self.splits(train,block,3,SEED+fold)
  candidates=GRIDS[ALIAS[model]];scores=[];oofs=[];probabilities=[]
  for params in candidates:
   oof=np.zeros(len(self.df),int);probs=np.zeros((len(self.df),5));ss=[]
   for tr,va,_ in splits:
    pred,prob,_=self.fitted_predict(x,available,tr,va,model,params)
    oof[va]=pred;probs[va]=prob
    ss.append(f1_score(self.y[va],pred,labels=CLASSES,average='macro',zero_division=0))
   scores.append(float(np.mean(ss)));oofs.append(oof);probabilities.append(probs)
  chosen=int(np.argmax(scores));params=candidates[chosen]
  pred,prob,estimator=self.fitted_predict(x,available,train,test,model,params)
  log={'feature_set':featureset,'view':view,'model':model,'fold':fold,'block_seconds':block,'parameters':params,
       'candidate_scores':[{'parameters':p,'macro_f1':s} for p,s in zip(candidates,scores)],
       'training_indices':train.tolist(),'test_indices':test.tolist(),
       'inner':[{'train':a.tolist(),'validation':b.tolist(),'purged':c.tolist()} for a,b,c in splits]}
  return {'pred':pred,'prob':prob,'estimator':estimator,'oof_pred':oofs[chosen],'oof_prob':probabilities[chosen],'available':available,'quality':quality,'log':log}

 def leg_report(self,p,directory):
  rows=[];classes=[]
  for (featureset,view,model),q in p.groupby(['feature_set','view','model'],sort=False):
   for subset in ['all','coverall','everyday']:
    r=q if subset=='all' else q[q.dataset==subset]
    if len(r)==0:continue
    y=r.target_legs.to_numpy();pred=r.pred_legs.to_numpy()
    rows.append({'feature_set':featureset,'view':view,'model':model,'subset':subset,'n':len(r),'correct':int((y==pred).sum()),
                 'accuracy':float((y==pred).mean()),'macro_f1':f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0),'unresolved':int((pred==0).sum())})
    for c in CLASSES:
     mask=y==c;classes.append({'feature_set':featureset,'view':view,'model':model,'subset':subset,'class':int(c),'support':int(mask.sum()),
                               'correct':int(((y==c)&(pred==c)).sum()),'recall':float((pred[mask]==c).mean()) if mask.any() else None})
  pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False);pd.DataFrame(classes).to_csv(directory/'class_metrics.csv',index=False)

 def improvement_cv(self,block,views):
  directory=OUT/'results'/('geometry_blocks_'+str(block)+'s');directory.mkdir(exist_ok=True)
  if (directory/'predictions.csv').exists():return
  rows=[];logs=[]
  for fold,(train,test,_) in enumerate(self.splits(self.indices,block,5,SEED),1):
   for featureset in ['leg_geometry','generic_plus_geometry']:
    for view in views:
     for model in ['svm','rf']:
      print('FOLLOWUP',block,fold,featureset,view,model,flush=True)
      result=self.tune(featureset,view,model,train,test,block,fold)
      q=self.df.iloc[test].copy();q['observation_index']=test;q['feature_set']=featureset;q['view']=view;q['model']=model;q['fold']=fold;q['pred_legs']=result['pred']
      rows.append(q);logs.append(result['log'])
      if block==60 and featureset=='leg_geometry' and model=='rf' and view in ['side','paired']:
       joblib.dump({k:v for k,v in result.items() if k!='estimator'},OUT/'cache'/f'fold_{fold}_{view}.joblib',compress=3)
      pd.concat(rows,ignore_index=True).to_csv(directory/'predictions.partial.csv',index=False)
  p=pd.concat(rows,ignore_index=True);p.to_csv(directory/'predictions.csv',index=False)
  (directory/'selection.json').write_text(json.dumps(logs,indent=2)+'\n');self.leg_report(p,directory)
  (directory/'predictions.partial.csv').unlink(missing_ok=True)
  print('GEOMETRY_CV_COMPLETE',block,flush=True)

 def transfer(self):
  directory=OUT/'results/geometry_transfer';directory.mkdir(exist_ok=True)
  if (directory/'predictions.csv').exists():return
  rows=[];logs=[]
  for fold,(source,target) in enumerate([('coverall','everyday'),('everyday','coverall')],101):
   train=np.flatnonzero(self.ds==source);test=np.flatnonzero(self.ds==target)
   for featureset in ['leg_geometry','generic_plus_geometry']:
    for model in ['svm','rf']:
     print('TRANSFER_FOLLOWUP',source,target,featureset,model,flush=True)
     result=self.tune(featureset,'paired',model,train,test,60,fold)
     q=self.df.iloc[test].copy();q['observation_index']=test;q['feature_set']=featureset;q['view']='paired';q['model']=model;q['source']=source;q['pred_legs']=result['pred']
     rows.append(q);logs.append(result['log'])
  p=pd.concat(rows,ignore_index=True);p.to_csv(directory/'predictions.csv',index=False)
  (directory/'selection.json').write_text(json.dumps(logs,indent=2)+'\n');self.leg_report(p,directory)

 def selective(self):
  directory=OUT/'results/selective';directory.mkdir(exist_ok=True)
  if (directory/'predictions.csv').exists() and 'unselective_side' in set(pd.read_csv(directory/'predictions.csv').policy):return
  records=[];logs=[];calibration_rows=[]
  for fold,(train,test,_) in enumerate(self.splits(self.indices,60,5,SEED),1):
   results={v:joblib.load(OUT/'cache'/f'fold_{fold}_{v}.joblib') for v in ['side','paired']}
   calibrated={};calibration={}
   for view,result in results.items():
    assert result['log']['training_indices']==train.tolist() and result['log']['test_indices']==test.tolist()
    meta=temperature_fit(result['oof_prob'][train],self.y[train]);calibration[view]=meta
    calibrated[view]={'train':temperature_apply(result['oof_prob'][train],meta['temperature']),
                      'test':temperature_apply(result['prob'],meta['temperature'])}
    valid=result['prob'].sum(axis=1)>0
    for k,i in enumerate(test):
     if valid[k]:
      raw=result['prob'][k];cal=calibrated[view]['test'][k];one=np.array([self.y[i]==c for c in CLASSES],float)
      calibration_rows.append({'fold':fold,'view':view,'observation_index':int(i),'raw_brier':float(((raw-one)**2).sum()),
                               'calibrated_brier':float(((cal-one)**2).sum()),'raw_nll':float(-np.log(max(raw[np.searchsorted(CLASSES,self.y[i])],1e-12))),
                               'calibrated_nll':float(-np.log(max(cal[np.searchsorted(CLASSES,self.y[i])],1e-12)))})
   st,pt=calibrated['side']['train'],calibrated['paired']['train']
   se,pe=calibrated['side']['test'],calibrated['paired']['test']
   sq,pq=results['side']['quality'],results['paired']['quality']
   for view in ['side','paired']:
    pred=results[view]['pred']
    for k,i in enumerate(test):
     records.append({'observation_index':int(i),'dataset':self.ds[i],'frame_id':self.df.iloc[i].frame_id,'fold':fold,'target':0.0,'policy':'unselective_'+view,
                     'ref_legs':int(self.y[i]),'pred_legs':int(pred[k]),'accepted':bool(pred[k]>0),'stage':view if pred[k]>0 else 'expert',
                     'second_view_used':view=='paired','side_confidence':float(se[k].max()),'paired_confidence':float(pe[k].max()),
                     'side_evidence_ok':bool(sq[i]),'paired_evidence_ok':bool(pq[i]),'side_threshold':0.0,'paired_threshold':0.0,'force_referral':False})
   for target in [0.9,0.8]:
    for policy in ['cascade','side_only','paired_only']:
     rule=choose_rule(st,pt,sq[train],pq[train],self.y[train],self.ds[train],self.t[train],target,policy)
     pred,stage,opened=route(se,pe,sq[test],pq[test],rule['side_threshold'],rule['paired_threshold'],policy,rule['force_referral'])
     for k,i in enumerate(test):
      records.append({'observation_index':int(i),'dataset':self.ds[i],'frame_id':self.df.iloc[i].frame_id,'fold':fold,'target':target,'policy':policy,
                      'ref_legs':int(self.y[i]),'pred_legs':int(pred[k]),'accepted':bool(pred[k]>0),'stage':stage[k],'second_view_used':bool(opened[k]),
                      'side_confidence':float(se[k].max()),'paired_confidence':float(pe[k].max()),'side_evidence_ok':bool(sq[i]),'paired_evidence_ok':bool(pq[i]),
                      'side_threshold':rule['side_threshold'],'paired_threshold':rule['paired_threshold'],'force_referral':rule['force_referral']})
     logs.append({'fold':fold,'training_indices':train.tolist(),'test_indices':test.tolist(),'calibration':calibration,'rule':rule})
  p=pd.DataFrame(records);p.to_csv(directory/'predictions.csv',index=False)
  (directory/'selection.json').write_text(json.dumps(logs,indent=2)+'\n')
  pd.DataFrame(calibration_rows).to_csv(directory/'calibration_predictions.csv',index=False)
  rows=[];class_rows=[]
  for (target,policy),q in p.groupby(['target','policy']):
   accepted=q.pred_legs>0;correct=(q.pred_legs==q.ref_legs)&accepted
   rows.append({'training_target':target,'policy':policy,'n':len(q),'accepted':int(accepted.sum()),'correct':int(correct.sum()),'wrong':int((accepted&~correct).sum()),
                'coverage':float(accepted.mean()),'accuracy_accepted':float(correct.sum()/accepted.sum()) if accepted.any() else None,
                'expert_referrals':int((~accepted).sum()),'second_views':int(q.second_view_used.sum()),'mean_views':float(1+q.second_view_used.mean()),
                'side_accepted':int((q.stage=='side').sum()),'paired_accepted':int((q.stage=='paired').sum())})
   for c in CLASSES:
    r=q[q.ref_legs==c];a=r.pred_legs>0;co=r.pred_legs==c
    class_rows.append({'target':target,'policy':policy,'class':int(c),'support':len(r),'accepted':int(a.sum()),'correct_accepted':int(co.sum()),'coverage':float(a.mean()),
                       'automated_recall':float(co.mean()),'accuracy_accepted':float(co.sum()/a.sum()) if a.any() else None})
  pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False);pd.DataFrame(class_rows).to_csv(directory/'class_metrics.csv',index=False)
  print('SELECTIVE_COMPLETE\n'+pd.DataFrame(rows).to_string(index=False),flush=True)

 def freeze_external(self):
  directory=OUT/'frozen_workflow';directory.mkdir(exist_ok=True)
  if (directory/'workflow.json').exists() and 'policy_rules' in json.loads((directory/'workflow.json').read_text()):return
  if (directory/'workflow.json').exists():
   (directory/'initial_cascade_recipe.json').write_bytes((directory/'workflow.json').read_bytes())
  train=self.indices;test=np.array([],int);results={};calibration={};probs={}
  for view in ['side','paired']:
   print('FREEZE',view,flush=True)
   result=self.tune('leg_geometry',view,'rf',train,test,60,999)
   joblib.dump(result['estimator'],directory/(view+'_rf.joblib'),compress=3)
   results[view]=result;meta=temperature_fit(result['oof_prob'][train],self.y);calibration[view]=meta
   probs[view]=temperature_apply(result['oof_prob'][train],meta['temperature'])
   (directory/(view+'_selection.json')).write_text(json.dumps(result['log'],indent=2)+'\n')
  rules={str(target):choose_rule(probs['side'],probs['paired'],results['side']['quality'],results['paired']['quality'],self.y,self.ds,self.t,target,'cascade') for target in [0.9,0.8]}
  policy_rules={f'{policy}:{target}':choose_rule(probs['side'],probs['paired'],results['side']['quality'],results['paired']['quality'],self.y,self.ds,self.t,target,policy) for policy in ['cascade','side_only','paired_only'] for target in [0.9,0.8]}
  benchmarks={}
  for featureset in ['generic','leg_geometry','generic_plus_geometry']:
   for model in ['svm','rf']:
    name=featureset+'_'+model+'_paired'
    if featureset=='leg_geometry' and model=='rf':
     filename='paired_rf.joblib';selection=results['paired']['log']
    else:
     print('FREEZE_BENCHMARK',name,flush=True)
     result=self.tune(featureset,'paired',model,train,test,60,999)
     filename=name+'.joblib';selection=result['log']
     joblib.dump(result['estimator'],directory/filename,compress=3)
     (directory/(name+'_selection.json')).write_text(json.dumps(selection,indent=2)+'\n')
    benchmarks[name]={'filename':filename,'feature_set':featureset,'model':model,'sha256':hashlib.sha256((directory/filename).read_bytes()).hexdigest(),'parameters':selection['parameters']}
  config={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'external_validation_status':'not_run_external_images_and_labels_not_supplied',
          'feature_set':'leg_geometry','model':'Random Forest','classes':CLASSES.tolist(),'calibration':calibration,'rules':rules,'primary_target':'0.9',
          'primary_policy':'cascade','policy_rules':policy_rules,'classification_benchmarks':benchmarks,
          'training_worker_ids':['W001'],
          'feature_function_sha256':hashlib.sha256((OUT/'leg_features.py').read_bytes()).hexdigest(),
          'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
          'training_reference_sha256':hashlib.sha256((ROOT/'inputs/expert_consensus_input.csv').read_bytes()).hexdigest(),
          'training_observations':len(train),'seed':SEED,'model_hashes':{v:hashlib.sha256((directory/(v+'_rf.joblib')).read_bytes()).hexdigest() for v in ['side','paired']}}
  (directory/'workflow.json').write_text(json.dumps(config,indent=2)+'\n')
  print('EXTERNAL_WORKFLOW_FROZEN',flush=True)


def main():
 freeze=OUT/'followup_freeze.json';digest=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
 if freeze.exists():assert json.loads(freeze.read_text())['protocol_sha256']==digest
 else:freeze.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest,'exploratory_after_pilot':True,'followup_accuracy_not_yet_evaluated':True},indent=2)+'\n')
 study=LegStudy()
 with threadpool_limits(limits=1):
  study.improvement_cv(60,['side','front','paired'])
  study.improvement_cv(30,['paired'])
  study.transfer();study.selective();study.freeze_external()
 print('LEG_FOLLOWUP_COMPLETE',flush=True)


if __name__=='__main__':main()
