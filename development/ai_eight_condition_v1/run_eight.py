"""Matched three/eight-code experiments on pinned nested ML crossfit inputs."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,sys,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ai_code_fusion_v1.run_fusion import Study as CodeStudy
from ai_code_fusion_v1.code_models import COMPONENTS,CLASSES,VERSIONS,SEED,construct_codes
from ai_eight_condition_v1.eight_models import SYSTEMS,LEARNED,FAMILIES,SETS,spec,candidates,inputs,fit_model,load_eight,dependency_audit
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')

def freeze():
    for rel in ['ai_ml_hybrid_v1/protocol_freeze.json','ai_ml_hybrid_v1/frozen_candidates/secondary_release.json']:
        for p,d in json.loads((ROOT/rel).read_text())['sha256'].items():assert sha(ROOT/p)==d,p
    paths=[HERE/n for n in ['protocol.json','eight_models.py','run_eight.py','predict_eight.py','seal_secondary.py']]
    paths.extend(ROOT/n for n in ['audit/paired_manifest.csv','inputs/ai_coding_input.csv','inputs/expert_consensus_input.csv',
        'inputs/owas_action_reference.csv','ai_code_fusion_v1/code_models.py','ai_code_fusion_v1/run_fusion.py',
        'ai_ml_hybrid_v1/protocol.json','ai_ml_hybrid_v1/protocol_freeze.json','ai_ml_hybrid_v1/hybrid_models.py',
        'ai_ml_hybrid_v1/frozen_candidates/secondary_release.json'])
    for name in ['blocks_60s','blocks_30s','recording_transfer','full_training']:
        paths.extend(sorted((ROOT/'ai_ml_hybrid_v1/results'/name).glob('context_*.joblib')))
        if name!='full_training':paths.append(ROOT/'ai_ml_hybrid_v1/results'/name/'predictions.csv')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};target=HERE/'protocol_freeze.json'
    if target.exists():assert json.loads(target.read_text())['sha256']==hashes,'Frozen eight-condition source changed'
    else:
        assert not (HERE/'results').exists(),'Freeze before new accuracy'
        write_json(target,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'eight_condition_accuracy_evaluated':False,
            'prior_development_results_seen':True,'new_worker_data_seen':False,'new_AI_calls':False,'sha256':hashes})
    return sha(target)

class EightStudy(CodeStudy):
    def __init__(self):
        super().__init__();self.freeze_sha=sha(HERE/'protocol_freeze.json')
        for v in VERSIONS:self.x[v],self.archived[v]=load_eight(ROOT/'inputs/ai_coding_input.csv',self.df,v)
        write_json(HERE/'AI_dependency_audit.json',{'status':'PASS','groups':dependency_audit(ROOT/'inputs/ai_coding_input.csv'),
            'expert_labels_used_as_predictors':False,'interpretation':'sequential AI conditions, not independent raters'})

    def old_context(self,name,fold,train,test,block):
        path=ROOT/'ai_ml_hybrid_v1/results'/name/f'context_{fold}.joblib'
        digest=json.loads((HERE/'protocol_freeze.json').read_text())['sha256'][str(path.relative_to(ROOT))]
        assert sha(path)==digest
        e=joblib.load(path)
        assert e['train']==train.tolist() and e['test']==test.tolist() and e['block_seconds']==block
        assert e['freeze_sha256']==sha(ROOT/'ai_ml_hybrid_v1/protocol_freeze.json')
        inner=self.splits(train,block,3,SEED+fold)
        for k,(record,(it,iv,purge)) in enumerate(zip(e['inner'],inner),1):
            assert record['train']==it.tolist() and record['validation']==iv.tolist() and record['purged']==purge.tolist()
            assert record['further_seed']==SEED+fold+10000+k
            further=self.splits(it,block,3,record['further_seed'])
            assert len(record['further'])==len(further)==3
            for d,(ft,fv,fp) in zip(record['further'],further):
                assert d['train']==ft.tolist() and d['validation']==fv.tolist() and d['purged']==fp.tolist()
            assert record['ML_predictions'].shape==(len(iv),3) and record['meta_training_ML'].shape==(len(it),3)
        assert len(e['inner'])==len(inner)==3 and e['meta_training_ML'].shape==(len(train),3)
        return e,path,digest

    def fit_context(self,name,fold,train,test,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True);target=directory/f'context_{fold}.joblib'
        if target.exists():
            e=joblib.load(target);assert e['freeze_sha256']==self.freeze_sha and e['train']==train.tolist() and e['test']==test.tolist();return e
        old,path,digest=self.old_context(name,fold,train,test,block)
        e={'context':name,'fold':fold,'block_seconds':block,'train':train.tolist(),'test':test.tolist(),
            'freeze_sha256':self.freeze_sha,'ML_cache_source':str(path.relative_to(ROOT)),'ML_cache_sha256':digest,
            'test_ML':old['test_ML'],'heads':{}}
        for v in VERSIONS:
            e['heads'][v]={}
            for j,c in enumerate(COMPONENTS):
                ai=self.x[v][c];y=self.y[c];head={}
                for method in LEARNED:
                    source,family=spec(method);logs=[]
                    for params in candidates(family):
                        folds=[]
                        for record in old['inner']:
                            it=np.array(record['train'],int);iv=np.array(record['validation'],int)
                            xt=inputs(ai[it],record['meta_training_ML'][:,j],method)
                            xv=inputs(ai[iv],record['ML_predictions'][:,j],method)
                            model=fit_model(xt,y[it],c,family,params);p=model.predict(xv).astype(int)
                            folds.append({'predictions':p,'macro_f1':float(f1_score(y[iv],p,labels=CLASSES[c],average='macro',zero_division=0)),
                                'train_class_counts':{str(k):int((y[it]==k).sum()) for k in CLASSES[c]}})
                        logs.append({'parameters':params,'folds':folds,'mean_macro_f1':float(np.mean([d['macro_f1'] for d in folds]))})
                    best=int(np.argmax([d['mean_macro_f1'] for d in logs]));params=logs[best]['parameters']
                    xt=inputs(ai[train],old['meta_training_ML'][:,j],method);xq=inputs(ai[test],old['test_ML'][:,j],method)
                    model=fit_model(xt,y[train],c,family,params)
                    head[method]={'selected_parameters':params,'candidates':logs,
                        'train_class_counts':{str(k):int((y[train]==k).sum()) for k in CLASSES[c]},
                        'input_width':xt.shape[1],'encoded_width':model.steps[-1][1].n_features_in_ if hasattr(model.steps[-1][1],'n_features_in_') else len(model.steps[0][1].get_feature_names_out()),
                        'classes':model.classes_.astype(int).tolist(),
                        'test_predictions':model.predict(xq).astype(int) if len(test) else np.array([],int),
                        'model':model if name=='full_training' else None}
                e['heads'][v][c]=head
        joblib.dump(e,target,compress=3);print('FITTED',name,fold,'train',len(train),'test',len(test),flush=True);return e

    def prediction_rows(self,name,e):
        test=np.array(e['test'],int);ml=e['test_ML'];parts=[]
        for v in VERSIONS:
            for method in SYSTEMS:
                q=self.df.iloc[test].copy();q['observation_index']=test;q['context']=name;q['fold']=e['fold'];q['model']=v;q['method']=method
                for j,c in enumerate(COMPONENTS):
                    a=self.x[v][c][test]
                    for k in range(8):q[f'input_{c}_condition{k+1}']=a[:,k]
                    for k,view in enumerate(['side','front','both']):q[f'input_{c}_{view}']=a[:,k]
                    q['input_'+c+'_ML']=ml[:,j]
                    # Preserve inherited table's literal three-AI-input semantics;
                    # actual_input_diagnostics.csv separately handles all3/8/9 inputs.
                    q['all_inputs_NA_'+c]=np.all(a[:,:3]==0,axis=1).astype(int)
                    if method=='archived_both':p=a[:,2]
                    elif method=='fixed_image_ML':p=ml[:,j]
                    elif method=='ai_NA_filled_by_ML':p=np.where(a[:,2]>0,a[:,2],ml[:,j])
                    else:p=e['heads'][v][c][method]['test_predictions']
                    q['pred_'+c]=p
                q['predicted_code']=[s if s in self.actions else 'NA' for s in self.archived[v]['3'][test]] if method=='archived_both' else construct_codes(q[['pred_'+c for c in COMPONENTS]].to_numpy())
                q['source_recording']=str(self.ds[e['train'][0]]) if name=='recording_transfer' else 'pooled';q['test_recording']=q.dataset;parts.append(q)
        return pd.concat(parts,ignore_index=True)

    def evaluate(self,name,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        if name=='recording_transfer':entries=[(101+i,np.flatnonzero(self.ds==a),np.flatnonzero(self.ds==b),np.array([],int)) for i,(a,b) in enumerate([('coverall','everyday'),('everyday','coverall')])]
        else:entries=[(i,tr,te,p) for i,(tr,te,p) in enumerate(self.splits(np.arange(181),block,5,SEED),1)]
        parts=[];splitlog=[]
        for fold,tr,te,purge in entries:
            e=self.fit_context(name,fold,tr,te,block);parts.append(self.prediction_rows(name,e))
            old,_,_=self.old_context(name,fold,tr,te,block)
            splitlog.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'purged':purge.tolist(),
                'ML_cache_source':e['ML_cache_source'],'ML_cache_sha256':e['ML_cache_sha256'],
                'inner':[{'train':d['train'],'validation':d['validation'],'purged':d['purged'],'further_seed':d['further_seed'],
                    'further':[{'train':f['train'],'validation':f['validation'],'purged':f['purged']} for f in d['further']]} for d in old['inner']]})
        p=pd.concat(parts,ignore_index=True);assert p.groupby(['model','method']).size().eq(181).all() and not p.duplicated(['model','method','observation_index']).any()
        p.to_csv(directory/'predictions.csv',index=False);write_json(directory/'splits.json',{'block_seconds':block,'purge_seconds':10,'folds':splitlog})
        self.report(p,directory)
        contrasts=[];diagnostics=[]
        pairs=[]
        for family in FAMILIES:pairs.extend([(f'AI3_{family}',f'AI8_{family}','3_to_8'),(f'AI8_{family}',f'AI8ML_{family}','8_to_8ML')])
        for source in SETS:
            pairs.extend([(source+'_Logistic',source+'_RF','family_Logistic_to_RF'),(source+'_Logistic',source+'_RBF_SVM','family_Logistic_to_RBF_SVM')])
        for v in VERSIONS:
            for baseline,method,contrast in pairs:
                for subset in ['all','coverall','everyday']:
                    b=p[(p.model==v)&(p.method==baseline)];m=p[(p.model==v)&(p.method==method)]
                    if subset!='all':b=b[b.dataset==subset];m=m[m.dataset==subset]
                    b=b.set_index('observation_index').sort_index();m=m.set_index('observation_index').sort_index()
                    a=b.predicted_code.eq(b.ref_both_code).to_numpy();z=m.predicted_code.eq(m.ref_both_code).to_numpy()
                    contrasts.append({'context':name,'model':v,'contrast':contrast,'method':method,'baseline':baseline,'subset':subset,'n':len(b),
                        'both_correct':int((a&z).sum()),'baseline_only_correct':int((a&~z).sum()),'method_only_correct':int((~a&z).sum()),
                        'neither_correct':int((~a&~z).sum()),'baseline_correct':int(a.sum()),'method_correct':int(z.sum())})
            for method in LEARNED:
                q=p[(p.model==v)&(p.method==method)];source,_=spec(method);width=3 if source=='AI3' else 8
                for c in COMPONENTS:
                    xx=q[[f'input_{c}_condition{k}' for k in range(1,width+1)]].to_numpy(int)
                    if source=='AI8ML':xx=np.column_stack([xx,q['input_'+c+'_ML']])
                    out=q['pred_'+c].to_numpy(int);missing=np.all(xx==0,axis=1)
                    diagnostics.append({'context':name,'model':v,'method':method,'component':c,'actual_input_width':xx.shape[1],
                        'all_actual_inputs_NA_n':int(missing.sum()),'all_NA_resolved_n':int(((out>0)&missing).sum()),
                        'all_NA_correct_n':int(((out==q['target_'+c])&missing).sum()),
                        'output_absent_from_actual_inputs_n':int(((out>0)&~(xx==out[:,None]).any(axis=1)).sum())})
        pd.DataFrame(contrasts).to_csv(directory/'matched_contrasts.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(directory/'actual_input_diagnostics.csv',index=False)

    def frozen_models(self):
        e=self.fit_context('full_training',201,np.arange(181),np.array([],int),60);directory=HERE/'frozen_candidates';directory.mkdir(exist_ok=True)
        bundle={v:{c:{m:e['heads'][v][c][m]['model'] for m in LEARNED} for c in COMPONENTS} for v in VERSIONS}
        joblib.dump(bundle,directory/'models.joblib',compress=3)
        write_json(directory/'workflow.json',{'version':'ai_eight_condition_v1','role':'secondary exploratory candidates',
            'new_worker_data_seen':False,'external_validation':'not_run','training_observations':181,
            'models_sha256':sha(directory/'models.joblib'),'protocol_freeze_sha256':self.freeze_sha,'systems':LEARNED,
            'selected_parameters':{v:{c:{m:e['heads'][v][c][m]['selected_parameters'] for m in LEARNED} for c in COMPONENTS} for v in VERSIONS},
            'old_image_branch_release_sha256':sha(ROOT/'ai_ml_hybrid_v1/frozen_candidates/secondary_release.json'),
            'limits':'Cached codes/features only; earlier primary and secondary unchanged; no raw-photo parity, new-worker score or calibrated confidence.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');a=parser.parse_args();freeze()
    if a.freeze_only:print('PROTOCOL_FROZEN');return
    warnings.filterwarnings('error',category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        s=EightStudy()
        for name,block in [('blocks_60s',60),('blocks_30s',30),('recording_transfer',60)]:s.evaluate(name,block)
        s.frozen_models()
    write_json(HERE/'status.json',{'status':'experiments_complete','new_worker_data_seen':False,'new_AI_calls':False,
        'raw_photos_processed':False,'manuscripts_modified':False,'prior_releases_modified':False,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib']}})
    print('EIGHT_CONDITION_COMPLETE',flush=True)
if __name__=='__main__':main()
