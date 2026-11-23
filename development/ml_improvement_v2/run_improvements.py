"""Three predeclared cached-feature experiments; never modify modular_v1."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_experiments import Benchmark, SEED
from modular_followup.models import CLASSES, base_estimator, make_candidates
from modular_followup.feature_interface import assemble as assemble_v1
from ml_improvement_v2.fusion_models import FrozenFeatureFusionLinearSVC, assemble_fusion_inputs

SYSTEMS=['modular_v1_reference','back_generic_matched','back_torso_matched',
         'embedding_pca_matched','fusion_pca_pose','pool_head_F1','pool_joint_fullcode']
COMPONENTS=list(CLASSES)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')

def checked_files():
    own=['protocol.json','run_improvements.py','fusion_models.py','torso_features.py','prepare_torso.py','torso_features.npz']
    files=[OUT/x for x in own]+[ROOT/x for x in [
        'features/features.npz','features/landmarks.jsonl','leg_followup/geometry_features.npz',
        'audit/paired_manifest.csv','inputs/owas_action_reference.csv','inputs/expert_consensus_input.csv',
        'run_experiments.py','modular_followup/models.py','modular_followup/feature_interface.py',
        'modular_followup/frozen_workflow/external_release.json','modular_followup/frozen_workflow/workflow.json',
        'modular_followup/frozen_workflow/models.joblib']]
    for name in ['blocks_60s','blocks_30s','recording_transfer','full_training']:
        files.extend(sorted((ROOT/'modular_followup/results'/name).glob('context_*.joblib')))
    return files

def freeze():
    hashes={str(p.relative_to(ROOT)):sha(p) for p in checked_files()}
    p=OUT/'protocol_freeze.json'
    if p.exists():
        assert json.loads(p.read_text())['sha256']==hashes,'Frozen development source changed'
    else:
        assert not (OUT/'results').exists(),'Must freeze before accuracy evaluation'
        write_json(p,{'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'new_v2_accuracy_evaluated':False,'prior_development_results_seen':True,
            'new_worker_data_seen':False,'raw_images_processed':False,'sha256':hashes})
    return sha(p)

def new_candidates(component):
    result=[]
    if component=='back':
        for c in make_candidates('back'):
            if c['family'] in ['pose_RF','pose_RBF_SVM']:
                d=copy.deepcopy(c);d['family']=c['family'].replace('pose_','torso_')
                d['id']=c['id'].replace('pose_','torso_',1);d['origin']='new'
                result.append(d)
    for family in ['pca_embedding_linear_SVM','pca_pose_linear_SVM']:
        for view in ['side','front','paired']:
            for k,c in enumerate([.1,1]):
                result.append({'id':f'{family}_{view}_{k}','family':family,'view':view,
                               'parameters':{'C':c},'hierarchy':False,'origin':'new'})
    return result

def choose(logs,predicate):
    pool=[c for c in logs if predicate(c)]
    assert pool
    return max(pool,key=lambda c:c['mean_macro_f1'])

def choose_combinations(options,oofs,inner,labels):
    records=[]
    for bits in itertools.product([0,1],repeat=3):
        selected={c:options[c][b] for c,b in zip(COMPONENTS,bits)}
        scores=[]
        for _,iv,_ in inner:
            correct=np.ones(len(iv),bool)
            for c in COMPONENTS:correct&=oofs[c][selected[c]['id']][iv]==labels[c][iv]
            scores.append(float(correct.mean()))
        records.append({'bits':list(bits),'candidate_ids':{c:d['id'] for c,d in selected.items()},
            'fold_fullcode_accuracy':scores,'mean_fullcode_accuracy':float(np.mean(scores)),
            'mean_component_macro_f1':float(np.mean([d['mean_macro_f1'] for d in selected.values()]))})
    best=max(records,key=lambda r:(r['mean_fullcode_accuracy'],r['mean_component_macro_f1']))
    return best,records

class ImprovementStudy(Benchmark):
    def __init__(self):
        super().__init__()
        self.arrays={k:self.z[k] for k in self.z.files}
        g=np.load(ROOT/'leg_followup/geometry_features.npz',allow_pickle=False)
        self.arrays.update({k:g[k] for k in g.files})
        t=np.load(OUT/'torso_features.npz',allow_pickle=False)
        for k in ['dataset','frame_id','timestamp_seconds']:
            assert np.array_equal(t[k],self.z[k]),k
        self.arrays.update({k:t[k] for k in ['torso_front','torso_side']})
        self.indices=np.arange(len(self.df))
        self.feature_cache={}
        self.freeze_sha=sha(OUT/'protocol_freeze.json')
        self.old_frozen_models=None

    def feature_inputs(self,candidate):
        key=(candidate['family'],candidate['view'])
        if key not in self.feature_cache:
            family,view=key
            if family.startswith('pca_'):
                x,av,dim=assemble_fusion_inputs(self.arrays,view,include_pose_flags=True)
            elif family.startswith('torso_'):
                views=['front','side'] if view=='paired' else [view]
                x=np.concatenate([np.concatenate([self.arrays['pose_'+v],self.arrays['torso_'+v]],axis=1) for v in views],axis=1)
                av=np.logical_or.reduce([self.arrays['pose_detected_'+v] for v in views]);dim=None
            else:
                x,av=assemble_v1(self.arrays,candidate);dim=None
            self.feature_cache[key]=(x,av,dim)
        return self.feature_cache[key]

    def fit(self,component,candidate,train):
        x,av,dim=self.feature_inputs(candidate);usable=train[av[train]]
        y=self.z['y_'+component]
        counts={str(c):int((y[usable]==c).sum()) for c in CLASSES[component]}
        support={'train_n':len(train),'usable_train_n':len(usable),'class_counts':counts}
        if not len(usable):return None,{**support,'status':'no_usable_input'}
        family=candidate['family']
        if len(np.unique(y[usable]))==1:
            model=DummyClassifier(strategy='most_frequent')
        elif family.startswith('pca_'):
            model=FrozenFeatureFusionLinearSVC(embedding_dim=dim,C=candidate['parameters']['C'],
                  n_components=16,use_pose=family=='pca_pose_linear_SVM')
        else:
            model=base_estimator(family.replace('torso_','pose_',1),candidate['parameters'])
        model.fit(x[usable],y[usable])
        support['status']='single_class_training_fallback' if len(np.unique(y[usable]))==1 else 'fitted'
        if hasattr(model,'fit_feature_summary_'):support['feature_fit']=model.fit_feature_summary_
        return model,support

    def infer(self,model,candidate,test):
        x,av,_=self.feature_inputs(candidate);pred=np.zeros(len(test),dtype=np.int16)
        ok=av[test]
        if model is not None and ok.any():pred[ok]=model.predict(x[test[ok]]).astype(np.int16)
        return pred

    def load_old(self,name,fold,train,test,block):
        saved=joblib.load(ROOT/'modular_followup/results'/name/f'context_{fold}.joblib')
        assert saved['train']==train.tolist() and saved['test']==test.tolist()
        assert saved['block_seconds']==block
        assert saved['protocol_freeze_sha256']==sha(ROOT/'modular_followup/protocol_freeze.json')
        inner=self.splits(train,block,3,SEED+fold)
        check=[{'train':a.tolist(),'validation':b.tolist(),'purged':c.tolist()} for a,b,c in inner]
        assert saved['inner']==check
        return saved,inner

    def context(self,name,fold,train,test,block,refit=False):
        directory=OUT/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        cache=directory/f'context_{fold}.joblib'
        if cache.exists():
            saved=joblib.load(cache)
            assert saved['train']==train.tolist() and saved['test']==test.tolist()
            assert saved['protocol_freeze_sha256']==self.freeze_sha
            if refit:assert (OUT/'frozen_candidates/models.joblib').exists(),'Incomplete freeze checkpoint'
            return saved
        old,inner=self.load_old(name,fold,train,test,block)
        all_logs={};oofs={};mapping={s:{} for s in SYSTEMS}
        for component,classes in CLASSES.items():
            # Old objects are copied in memory; source caches stay byte-identical.
            old_logs=copy.deepcopy([v for v in old['candidate_logs'][component] if not v['hierarchy']])
            for v in old_logs:v['origin']='v1'
            logs=old_logs.copy();preds={k:np.asarray(v).copy() for k,v in old['inner_oof_predictions'][component].items()}
            for candidate in new_candidates(component):
                pred=np.zeros(len(self.df),dtype=np.int16);scores=[];supports=[]
                for j,(it,iv,_) in enumerate(inner,1):
                    model,support=self.fit(component,candidate,it)
                    pred[iv]=self.infer(model,candidate,iv)
                    scores.append(float(f1_score(self.z['y_'+component][iv],pred[iv],labels=classes,average='macro',zero_division=0)))
                    supports.append({'inner_fold':j,**support})
                logs.append({**candidate,'fold_macro_f1':scores,'mean_macro_f1':float(np.mean(scores)),'training_support':supports})
                preds[candidate['id']]=pred
            index={c['id']:c for c in logs}
            primary_id=old['selections']['modular_flat'][component]['candidate_id']
            for sysname in ['modular_v1_reference','back_generic_matched','back_torso_matched']:
                mapping[sysname][component]=index[primary_id]
            if component=='back':
                mapping['back_generic_matched'][component]=choose(logs,lambda c:c['family'] in ['pose_RF','pose_RBF_SVM'])
                mapping['back_torso_matched'][component]=choose(logs,lambda c:c['family'] in ['torso_RF','torso_RBF_SVM'])
            mapping['embedding_pca_matched'][component]=choose(logs,lambda c:c['family']=='pca_embedding_linear_SVM')
            mapping['fusion_pca_pose'][component]=choose(logs,lambda c:c['family']=='pca_pose_linear_SVM')
            all_logs[component]=logs;oofs[component]=preds
            print('CANDIDATES_COMPLETE',name,fold,component,'new',len(new_candidates(component)),flush=True)
        options={}
        for component in COMPONENTS:
            afamily={'back':'torso_RF','arms':'pose_RF','legs':'combined_leg_RF'}[component]
            a=choose(all_logs[component],lambda c:c['family']==afamily and c['view']=='paired')
            b=choose(all_logs[component],lambda c:c['family']=='pca_pose_linear_SVM' and c['view']=='paired')
            options[component]=[a,b]
            mapping['pool_head_F1'][component]=max([a,b],key=lambda c:c['mean_macro_f1'])
        best,joint_scores=choose_combinations(options,oofs,inner,{c:self.z['y_'+c] for c in COMPONENTS})
        for c,bit in zip(COMPONENTS,best['bits']):mapping['pool_joint_fullcode'][c]=options[c][bit]
        predictions={s:{} for s in SYSTEMS};selections={s:{} for s in SYSTEMS};fitted={}
        model_cache={}
        for sysname in SYSTEMS:
            for component,candidate in mapping[sysname].items():
                key=component+':'+candidate['id']
                if key not in model_cache:
                    old_match=next((s for s,selection in old['selections'].items()
                         if selection[component]['candidate_id']==candidate['id']),None)
                    if old_match is not None and not refit:
                        pred=np.asarray(old['predictions'][old_match]['pred_'+component]).copy()
                        support=copy.deepcopy(old['selections'][old_match][component]['full_training_support'])
                        support['prediction_reused_from_v1_system']=old_match;model=None
                    else:
                        if refit and candidate.get('origin')=='v1':
                            if self.old_frozen_models is None:
                                self.old_frozen_models=joblib.load(ROOT/'modular_followup/frozen_workflow/models.joblib')
                            entry=self.old_frozen_models.get(key)
                        else:entry=None
                        if entry is not None:
                            model=entry['model'];support=copy.deepcopy(entry['training_support'])
                            support['model_reused_from_v1_frozen']=True
                        else:model,support=self.fit(component,candidate,train)
                        pred=self.infer(model,candidate,test)
                    model_cache[key]=(pred,support,model)
                pred,support,model=model_cache[key]
                predictions[sysname]['pred_'+component]=pred
                selections[sysname][component]={'candidate_id':candidate['id'],'candidate':candidate,
                    'full_training_support':support}
                if refit:fitted[key]={'model':model,'candidate':candidate,'training_support':support}
        saved={'name':name,'fold':fold,'block_seconds':block,'protocol_freeze_sha256':self.freeze_sha,
               'train':train.tolist(),'test':test.tolist(),'inner':old['inner'],
               'candidate_logs':all_logs,'inner_oof_predictions':oofs,'predictions':predictions,
               'selections':selections,'joint_options':{c:[x['id'] for x in v] for c,v in options.items()},
               'joint_scores':joint_scores,'selected_joint_bits':best['bits'],
               'v1_context_sha256':sha(ROOT/'modular_followup/results'/name/f'context_{fold}.joblib')}
        if refit:
            frozen=OUT/'frozen_candidates';frozen.mkdir(exist_ok=True)
            joblib.dump(fitted,frozen/'models.joblib',compress=3)
        temporary=cache.with_suffix('.joblib.building');joblib.dump(saved,temporary,compress=3);os.replace(temporary,cache)
        print('CONTEXT_COMPLETE',name,fold,'joint',best['bits'],flush=True)
        return saved

    def run(self,mode):
        if mode in ['60','30']:
            block=int(mode);name='blocks_'+mode+'s'
            contexts=[(i,a,b,c) for i,(a,b,c) in enumerate(self.splits(self.indices,block,5,SEED),1)]
        elif mode=='transfer':
            block=60;name='recording_transfer'
            contexts=[(i,np.flatnonzero(self.ds==a),np.flatnonzero(self.ds==b),np.array([],int))
                     for i,(a,b) in enumerate([('coverall','everyday'),('everyday','coverall')],101)]
        elif mode=='freeze':
            self.freeze_candidates();return
        else:raise ValueError(mode)
        records=[];splits=[]
        for fold,tr,te,removed in contexts:
            saved=self.context(name,fold,tr,te,block)
            splits.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'purged':removed.tolist(),'inner':saved['inner']})
            for system,pred in saved['predictions'].items():
                q=self.df.iloc[te].copy();q['observation_index']=te;q['fold']=fold;q['method']=system;q['view']='inner_selected'
                if mode=='transfer':q['source_recording']=str(self.ds[tr[0]])
                for c in COMPONENTS:q['selected_'+c]=saved['selections'][system][c]['candidate_id']
                for col,values in pred.items():q[col]=values
                records.append(q)
        directory=OUT/'results'/name
        result=pd.concat(records,ignore_index=True);result.to_csv(directory/'predictions.csv',index=False)
        write_json(directory/'splits.json',{'block_seconds':block,'purge_seconds':10,'folds':splits})
        metrics=self.report(result,directory)
        print('RUN_COMPLETE',name,metrics[metrics.subset=='all'][['method','exact_code_correct','back_macro_f1','arms_macro_f1','legs_macro_f1']].to_json(orient='records'),flush=True)

    def freeze_candidates(self):
        tr=self.indices;te=np.array([],int)
        saved=self.context('full_training',201,tr,te,60,refit=True)
        directory=OUT/'frozen_candidates'
        config={'version':'ml_improvement_v2_1','created_at_utc':datetime.now(timezone.utc).isoformat(),
                'status':'secondary exploratory candidates; original modular_v1 primary is unchanged',
                'external_validation':'not_run','new_worker_data_seen':False,'raw_pipeline_parity':'not_run',
                'training_worker_ids':['W001'],'classes':CLASSES,'load':1,
                'systems':{s:{c:c+':'+d['candidate_id'] for c,d in comps.items()} for s,comps in saved['selections'].items()},
                'models_file':'models.joblib','models_sha256':sha(directory/'models.joblib'),
                'protocol_freeze_sha256':self.freeze_sha,
                'software_versions':{n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','pandas','joblib']},
                'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'fusion_models.py',OUT/'torso_features.py',OUT/'run_improvements.py',ROOT/'modular_followup/models.py',ROOT/'modular_followup/feature_interface.py']},
                'in_sample_accuracy_reported':False,'input_note':'cached feature predictor only; no fresh-photo or external validation claimed'}
        write_json(directory/'workflow.json',config)
        print('FROZEN_SECONDARY_CANDIDATES_COMPLETE',len(joblib.load(directory/'models.joblib')),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['60','30','transfer','freeze','all'],default='all')
    args=parser.parse_args();freeze();study=ImprovementStudy()
    with threadpool_limits(limits=1):
        for mode in (['60','30','transfer','freeze'] if args.mode=='all' else [args.mode]):study.run(mode)

if __name__=='__main__':main()
