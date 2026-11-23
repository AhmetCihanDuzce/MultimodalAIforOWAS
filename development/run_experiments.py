"""Nested, purged time-block OWAS benchmarks on frozen image features.

The chronological blocks group selected seconds, not physical task episodes.
All metrics retain unsuccessful predictions in the observation denominator.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SEED=20261029
CLASSES={'back':[1,2,3,4],'arms':[1,2,3],'legs':[2,3,4,5,7]}
METHODS=['pose_RBF_SVM','pose_RandomForest','frozen_MobileNetV3Small_linear_SVM','majority']
GRIDS={
 'pose_RBF_SVM':list(ParameterGrid({'C':[0.1,1,10],'gamma':['scale',0.1]})),
 'pose_RandomForest':list(ParameterGrid({'max_depth':[None,5],'min_samples_leaf':[1,3]})),
 'frozen_MobileNetV3Small_linear_SVM':list(ParameterGrid({'C':[0.01,0.1,1,10]}))}


class Benchmark:
 def __init__(self):
  self.z=np.load(ROOT/'features/features.npz',allow_pickle=False)
  self.df=pd.read_csv(ROOT/'audit/paired_manifest.csv',dtype={'frame_id':str})
  self.t=self.df.timestamp_seconds.to_numpy()
  self.ds=self.df.dataset.to_numpy()
  assert np.array_equal(self.z['frame_id'],self.df.frame_id.to_numpy())
  assert np.array_equal(self.z['dataset'],self.ds)
  assert np.array_equal(self.z['timestamp_seconds'],self.t)
  protocol_hash=hashlib.sha256((ROOT/'experiment_protocol.json').read_bytes()).hexdigest()
  assert protocol_hash==json.loads((ROOT/'audit/protocol_freeze.json').read_text())['protocol_sha256']
  assert protocol_hash==json.loads((ROOT/'features/metadata.json').read_text())['protocol_sha256']
  self.actions=pd.read_csv(ROOT/'inputs/owas_action_reference.csv',dtype={'OWAS_code':str}).set_index('OWAS_code').action.to_dict()
  self.root=ROOT/'results'; self.root.mkdir(exist_ok=True)

 def purge(self,train,test):
  keep=[i for i in train if not np.any((self.ds[test]==self.ds[i])&(np.abs(self.t[test]-self.t[i])<=10))]
  result=np.array(keep,dtype=int)
  assert len(set(result)&set(test))==0
  for i in result:
   assert not np.any((self.ds[test]==self.ds[i])&(np.abs(self.t[test]-self.t[i])<=10))
  return result

 def splits(self,indices,block_seconds,n_splits,seed):
  groups=np.array([str(self.ds[i])+'_t'+str(self.t[i]//block_seconds) for i in indices])
  n_splits=min(n_splits,len(set(groups)))
  if n_splits<2: raise ValueError('Not enough temporal groups for validation')
  output=[]
  for a,b in GroupKFold(n_splits,shuffle=True,random_state=seed).split(indices,groups=groups):
   raw,te=indices[a],indices[b]
   tr=self.purge(raw,te)
   assert len(tr)>0 and len(te)>0
   assert not (set(groups[a])&set(groups[b]))
   output.append((tr,te,np.setdiff1d(raw,tr)))
  return output

 def features(self,method,view):
  if method=='majority':return np.zeros((len(self.df),1)),np.ones(len(self.df),bool)
  key='pose' if method.startswith('pose_') else 'embedding'
  views=['front','side'] if view=='paired' else [view]
  x=np.concatenate([self.z[key+'_'+v] for v in views],axis=1)
  available=np.logical_or.reduce([self.z['pose_detected_'+v] for v in views]) if key=='pose' else np.ones(len(self.df),bool)
  return x,available

 def estimator(self,method,params):
  if method=='majority':return DummyClassifier(strategy='most_frequent')
  imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
  if method=='pose_RBF_SVM':
   return make_pipeline(imputer,StandardScaler(),SVC(kernel='rbf',class_weight='balanced',probability=False,**params))
  if method=='pose_RandomForest':
   return make_pipeline(imputer,RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',n_jobs=1,random_state=SEED,**params))
  return make_pipeline(imputer,StandardScaler(),LinearSVC(class_weight='balanced',max_iter=50000,random_state=SEED,dual='auto',**params))

 def fit_predict(self,x,available,y,train,test,method,params):
  pred=np.zeros(len(test),int)
  tr=train[available[train]]; valid=available[test]
  if len(tr)==0 or not valid.any():return pred,{'status':'no_usable_pose','train_usable':len(tr)}
  if len(np.unique(y[tr]))==1:
   pred[valid]=y[tr][0]
   return pred,{'status':'single_training_class','train_usable':len(tr)}
  estimator=self.estimator(method,params)
  estimator.fit(x[tr],y[tr])
  pred[valid]=estimator.predict(x[test[valid]])
  return pred,{'status':'fitted','train_usable':len(tr)}

 def fit_fold(self,method,view,train,test,block_seconds,fold):
  x,available=self.features(method,view)
  output={}; selections=[]
  inner=[] if method=='majority' else self.splits(train,block_seconds,3,SEED+fold)
  for component,classes in CLASSES.items():
   y=self.z['y_'+component]
   candidates=[{}] if method=='majority' else GRIDS[method]
   scores=[]
   for params in candidates:
    if method=='majority':scores.append(0.0);continue
    fold_scores=[]
    for it,iv,_ in inner:
     p,_=self.fit_predict(x,available,y,it,iv,method,params)
     fold_scores.append(f1_score(y[iv],p,labels=classes,average='macro',zero_division=0))
    scores.append(float(np.mean(fold_scores)))
   best=int(np.argmax(scores)); params=candidates[best]
   pred,status=self.fit_predict(x,available,y,train,test,method,params)
   output['pred_'+component]=pred
   selections.append({'fold':fold,'method':method,'view':view,'component':component,'parameters':params,
                      'inner_scores':[{'parameters':p,'mean_macro_f1':s} for p,s in zip(candidates,scores)],
                      'train_n':len(train),'test_n':len(test),'train_classes':sorted(map(int,np.unique(y[train[available[train]]]))),**status})
  return output,selections,inner

 def report(self,predictions,directory):
  directory.mkdir(exist_ok=True,parents=True)
  metric_rows=[]; class_rows=[]; action_rows=[]
  for (method,view),p in predictions.groupby(['method','view'],sort=False):
   for subset in ['all','coverall','everyday']:
    q=p if subset=='all' else p[p.dataset==subset]
    if q.empty:continue
    row={'method':method,'view':view,'subset':subset,'n':len(q)}
    for component,classes in CLASSES.items():
     y=q['target_'+component].astype(int).to_numpy(); pred=q['pred_'+component].astype(int).to_numpy()
     row[component+'_macro_f1']=f1_score(y,pred,labels=classes,average='macro',zero_division=0)
     row[component+'_accuracy']=accuracy_score(y,pred)
     row[component+'_unresolved']=int((pred==0).sum())
     recalls=recall_score(y,pred,labels=classes,average=None,zero_division=0)
     for c,rec in zip(classes,recalls):
      class_rows.append({'method':method,'view':view,'subset':subset,'component':component,'class':c,
                         'support':int((y==c).sum()),'correct':int(((y==c)&(pred==c)).sum()),'recall':rec})
    ref=q.ref_both_code.astype(str).to_numpy()
    if str(method).startswith('archived_'):
     pc=np.array([str(c) if str(c) in self.actions else 'NA' for c in q.archived_OWAS_code])
    else:
     pc=np.array([''.join(str(int(v)) for v in values)+'1' if all(int(v)>0 for v in values) else 'NA'
                  for values in q[['pred_back','pred_arms','pred_legs']].itertuples(index=False,name=None)])
    true_action=np.array([self.actions[c] for c in ref])
    pred_action=np.array([self.actions.get(c,0) for c in pc])
    row.update(exact_code_correct=int((pc==ref).sum()),exact_code_accuracy=float((pc==ref).mean()),
               action_correct=int((pred_action==true_action).sum()),action_accuracy=float((pred_action==true_action).mean()),
               complete_predictions=int((pc!='NA').sum()),high_risk_reference_n=int((true_action>=3).sum()),
               high_risk_predicted_low=int(((true_action>=3)&(pred_action>0)&(pred_action<3)).sum()),
               high_risk_unresolved=int(((true_action>=3)&(pred_action==0)).sum()))
    metric_rows.append(row)
    for index,true,pred in zip(q.index,true_action,pred_action):
     action_rows.append({'prediction_row':int(index),'method':method,'view':view,'subset':subset,'true_action':int(true),'pred_action':int(pred)})
  pd.DataFrame(metric_rows).to_csv(directory/'metrics.csv',index=False)
  pd.DataFrame(class_rows).to_csv(directory/'class_metrics.csv',index=False)
  pd.DataFrame(action_rows).to_csv(directory/'action_predictions.csv',index=False)
  return pd.DataFrame(metric_rows)

 def cv(self,views,block_seconds,directory):
  directory.mkdir(exist_ok=True,parents=True)
  outer=self.splits(np.arange(len(self.df)),block_seconds,5,SEED)
  assert sorted(np.concatenate([te for _,te,_ in outer]).tolist())==list(range(len(self.df)))
  records=[]; selections=[]; split_log=[]
  for fold,(tr,te,purged) in enumerate(outer,1):
   split_entry={'fold':fold,'train':tr.tolist(),'test':te.tolist(),'purged':purged.tolist()}
   for view in views:
    for method in METHODS:
     print('FIT',directory.name,'fold',fold,view,method,flush=True)
     preds,selected,inner=self.fit_fold(method,view,tr,te,block_seconds,fold)
     q=self.df.iloc[te].copy();q['observation_index']=te;q['fold']=fold;q['method']=method;q['view']=view
     for col,values in preds.items():q[col]=values
     records.append(q);selections.extend(selected)
     if inner:split_entry['inner']=[{'train':a.tolist(),'validation':b.tolist(),'purged':c.tolist()} for a,b,c in inner]
     partial=pd.concat(records,ignore_index=True)
     partial.to_csv(directory/'predictions.partial.csv',index=False)
   split_log.append(split_entry)
  result=pd.concat(records,ignore_index=True)
  result.to_csv(directory/'predictions.csv',index=False)
  (directory/'parameter_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
  (directory/'splits.json').write_text(json.dumps({'block_seconds':block_seconds,'purge_seconds':10,'folds':split_log},indent=2)+'\n')
  (directory/'predictions.partial.csv').unlink(missing_ok=True)
  metrics=self.report(result,directory)
  print(metrics[metrics.subset=='all'][['method','view','back_macro_f1','arms_macro_f1','legs_macro_f1','exact_code_correct','n']].to_string(index=False),flush=True)
  return result

 def experiment1(self):
  for block in [60,30]:
   path=self.root/'experiment1'/('blocks_'+str(block)+'s')
   if not (path/'predictions.csv').exists():self.cv(['paired'],block,path)
  print('EXPERIMENT_1_COMPLETE',flush=True)

 def experiment2(self):
  path=self.root/'experiment2'
  if not (path/'predictions.csv').exists():
   other=self.cv(['side','front'],60,path/'single_views')
   paired=pd.read_csv(self.root/'experiment1/blocks_60s/predictions.csv',dtype={'frame_id':str,'ref_both_code':str})
   combined=pd.concat([other,paired],ignore_index=True)
   combined.to_csv(path/'predictions.csv',index=False);self.report(combined,path)
  print('EXPERIMENT_2_COMPLETE',flush=True)

 def experiment3(self):
  path=self.root/'experiment3';path.mkdir(exist_ok=True)
  if (path/'predictions.csv').exists():return
  records=[];selections=[];split_log=[]
  for fold,(source,target) in enumerate([('coverall','everyday'),('everyday','coverall')],101):
   tr=np.flatnonzero(self.ds==source);te=np.flatnonzero(self.ds==target)
   for method in METHODS:
    print('FIT recording_transfer',source,'to',target,method,flush=True)
    preds,selected,inner=self.fit_fold(method,'paired',tr,te,60,fold)
    q=self.df.iloc[te].copy();q['observation_index']=te;q['fold']=fold;q['method']=method;q['view']='paired';q['source_recording']=source;q['test_recording']=target
    for col,values in preds.items():q[col]=values
    records.append(q);selections.extend(selected)
    if inner:split_log.append({'source':source,'target':target,'inner':[{'train':a.tolist(),'validation':b.tolist(),'purged':c.tolist()} for a,b,c in inner]})
  result=pd.concat(records,ignore_index=True);result.to_csv(path/'predictions.csv',index=False)
  (path/'parameter_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
  (path/'splits.json').write_text(json.dumps(split_log,indent=2)+'\n')
  self.report(result,path)
  print('EXPERIMENT_3_COMPLETE',flush=True)

 def experiment4(self):
  path=self.root/'experiment4';path.mkdir(exist_ok=True)
  ml=pd.read_csv(self.root/'experiment2/predictions.csv',dtype={'frame_id':str,'ref_both_code':str})
  ai=pd.read_csv(ROOT/'inputs/ai_coding_input.csv',dtype={'frame_id':str},keep_default_na=False)
  ai=ai[ai.combination.astype(str).isin(['1','2','3'])].copy()
  ai['view']=ai.combination.astype(str).map({'1':'side','2':'front','3':'paired'})
  ai['method']='archived_'+ai.model
  for component in CLASSES:
   ai['pred_'+component]=pd.to_numeric(ai[component],errors='coerce').fillna(0).astype(int)
   valid={'back':range(1,5),'arms':range(1,4),'legs':range(1,8)}[component]
   ai.loc[~ai['pred_'+component].isin(valid),'pred_'+component]=0
  ai['archived_OWAS_code']=ai.OWAS_code
  merged=ai[['dataset','frame_id','view','method','pred_back','pred_arms','pred_legs','archived_OWAS_code']].merge(self.df,on=['dataset','frame_id'],validate='many_to_one')
  assert len(merged)==181*2*3
  combined=pd.concat([ml,merged],ignore_index=True)
  combined.to_csv(path/'predictions.csv',index=False)
  self.report(combined,path)
  print('EXPERIMENT_4_COMPLETE',flush=True)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','1','2','3','4'],default='all');args=parser.parse_args()
 benchmark=Benchmark()
 with threadpool_limits(limits=1):
  for stage in range(1,5):
   if args.stage in ['all',str(stage)]:getattr(benchmark,'experiment'+str(stage))()


if __name__=='__main__':main()
