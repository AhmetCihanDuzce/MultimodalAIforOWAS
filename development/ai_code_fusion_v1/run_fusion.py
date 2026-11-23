"""Predeclared three-code AI fusion; fixed archives, nested training only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ai_code_fusion_v1.code_models import SEED,COMPONENTS,CLASSES,DOMAINS,VERSIONS,SYSTEMS,C_GRID,load_three_codes,majority,fit_meta,construct_codes

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')

def freeze():
    paths=[HERE/n for n in ['protocol.json','code_models.py','run_fusion.py','predict_codes.py','seal_secondary.py']]
    paths.extend(ROOT/n for n in ['inputs/ai_coding_input.csv','inputs/expert_consensus_input.csv',
        'inputs/owas_action_reference.csv','audit/paired_manifest.csv','run_experiments.py',
        'modular_followup/frozen_workflow/external_release.json',
        'ml_improvement_v2/frozen_candidates/secondary_release.json'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    target=HERE/'protocol_freeze.json'
    if target.exists():assert json.loads(target.read_text())['sha256']==hashes,'Frozen source changed'
    else:
        assert not (HERE/'results').exists(),'Freeze before fusion evaluation'
        write_json(target,{'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'fusion_accuracy_evaluated':False,'prior_AI_ML_results_seen':True,
            'new_worker_data_seen':False,'new_AI_calls':False,'sha256':hashes})
    return sha(target)

class Study:
    def __init__(self):
        self.df=pd.read_csv(ROOT/'audit/paired_manifest.csv',dtype={'frame_id':str,'ref_both_code':str})
        assert len(self.df)==181 and not self.df.duplicated(['dataset','frame_id']).any()
        self.t=self.df.timestamp_seconds.to_numpy();self.ds=self.df.dataset.to_numpy()
        self.y={c:self.df['target_'+c].to_numpy(dtype=int) for c in COMPONENTS}
        self.actions=pd.read_csv(ROOT/'inputs/owas_action_reference.csv',dtype={'OWAS_code':str}).set_index('OWAS_code').action.to_dict()
        self.x={};self.archived={}
        for v in VERSIONS:self.x[v],self.archived[v]=load_three_codes(ROOT/'inputs/ai_coding_input.csv',self.df,v)
        self.freeze_sha=sha(HERE/'protocol_freeze.json')

    def purge(self,train,test):
        return np.array([i for i in train if not np.any((self.ds[test]==self.ds[i])&(np.abs(self.t[test]-self.t[i])<=10))],dtype=int)

    def splits(self,indices,block,n_splits,seed):
        groups=np.array([str(self.ds[i])+'_t'+str(self.t[i]//block) for i in indices])
        count=min(n_splits,len(set(groups)))
        if count<2:raise ValueError('Insufficient groups')
        result=[]
        for a,b in GroupKFold(count,shuffle=True,random_state=seed).split(indices,groups=groups):
            tr,te=indices[a],indices[b]
            purged=self.purge(tr,te)
            if not len(purged):raise ValueError('No purged training rows')
            result.append((purged,te,np.setdiff1d(tr,purged)))
        return result

    def fit_context(self,name,fold,train,test,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        target=directory/f'context_{fold}.joblib'
        if target.exists():
            entry=joblib.load(target)
            assert entry['freeze_sha256']==self.freeze_sha
            assert entry['train']==train.tolist() and entry['test']==test.tolist()
            return entry
        inner=self.splits(train,block,3,SEED+fold)
        entry={'fold':fold,'context':name,'block_seconds':block,'train':train.tolist(),'test':test.tolist(),
            'freeze_sha256':self.freeze_sha,'inner':[{'train':a.tolist(),'validation':b.tolist(),'purged':p.tolist()} for a,b,p in inner],
            'heads':{}}
        for version in VERSIONS:
            entry['heads'][version]={}
            for component in COMPONENTS:
                x=self.x[version][component];y=self.y[component];logs=[]
                for C in C_GRID:
                    folds=[]
                    for it,iv,_ in inner:
                        model=fit_meta(x[it],y[it],component,C)
                        p=model.predict(x[iv]).astype(int)
                        folds.append({'predictions':p.tolist(),'macro_f1':float(f1_score(y[iv],p,labels=CLASSES[component],average='macro',zero_division=0)),
                            'train_class_counts':{str(c):int((y[it]==c).sum()) for c in CLASSES[component]}})
                    logs.append({'C':C,'folds':folds,'mean_macro_f1':float(np.mean([d['macro_f1'] for d in folds]))})
                best=int(np.argmax([d['mean_macro_f1'] for d in logs]));C=logs[best]['C']
                model=fit_meta(x[train],y[train],component,C)
                entry['heads'][version][component]={'selected_C':C,'candidates':logs,'model':model,
                    'train_class_counts':{str(c):int((y[train]==c).sum()) for c in CLASSES[component]},
                    'test_predictions':model.predict(x[test]).astype(int).tolist() if len(test) else []}
        joblib.dump(entry,target,compress=3)
        print('FITTED',name,fold,'train',len(train),'test',len(test),flush=True)
        return entry

    def prediction_rows(self,name,entry):
        test=np.array(entry['test'],dtype=int);parts=[]
        for version in VERSIONS:
            for method in SYSTEMS:
                q=self.df.iloc[test].copy()
                q['observation_index']=test;q['context']=name;q['fold']=entry['fold'];q['model']=version;q['method']=method
                for j,c in enumerate(COMPONENTS):
                    x=self.x[version][c][test]
                    for k,v in enumerate(['side','front','both']):q['input_'+c+'_'+v]=x[:,k]
                    q['all_inputs_NA_'+c]=np.all(x==0,axis=1).astype(int)
                    if method.startswith('archived_'):p=x[:,SYSTEMS.index(method)]
                    elif method=='component_majority':p=majority(x)
                    else:p=np.array(entry['heads'][version][c]['test_predictions'],int)
                    q['pred_'+c]=p
                if method.startswith('archived_'):
                    codes=self.archived[version][str(SYSTEMS.index(method)+1)][test]
                    q['predicted_code']=[c if c in self.actions else 'NA' for c in codes]
                else:q['predicted_code']=construct_codes(q[['pred_'+c for c in COMPONENTS]].to_numpy())
                q['source_recording']=str(self.ds[entry['train'][0]]) if name=='recording_transfer' else 'pooled'
                q['test_recording']=q.dataset
                parts.append(q)
        return pd.concat(parts,ignore_index=True)

    def report(self,predictions,directory):
        metrics=[];class_rows=[];actions=[];contrasts=[];diagnostics=[]
        for (version,method),p in predictions.groupby(['model','method'],sort=False):
            for subset in ['all','coverall','everyday']:
                q=p if subset=='all' else p[p.dataset==subset]
                row={'context':str(q.context.iloc[0]),'model':version,'method':method,'subset':subset,'n':len(q)}
                for c in COMPONENTS:
                    y=q['target_'+c].to_numpy(int);pred=q['pred_'+c].to_numpy(int)
                    precision,recall,fscore,support=precision_recall_fscore_support(y,pred,labels=CLASSES[c],zero_division=0)
                    row[c+'_macro_f1']=float(np.mean(fscore));row[c+'_correct']=int((y==pred).sum())
                    row[c+'_accuracy']=float((y==pred).mean());row[c+'_unresolved']=int((pred==0).sum())
                    for k,pr,re,f,su in zip(CLASSES[c],precision,recall,fscore,support):
                        class_rows.append({**{k:row[k] for k in ['context','model','method','subset']},'component':c,'class':k,
                            'support':int(su),'correct':int(((y==k)&(pred==k)).sum()),'predicted_n':int((pred==k).sum()),'precision':float(pr),'recall':float(re),'f1':float(f)})
                    missing=q['all_inputs_NA_'+c].to_numpy(bool)
                    diagnostics.append({**{k:row[k] for k in ['context','model','method','subset']},'component':c,
                        'all_three_inputs_NA_n':int(missing.sum()),'all_NA_output_resolved_n':int(((pred>0)&missing).sum()),
                        'all_NA_output_correct_n':int(((y==pred)&missing).sum()),
                        'output_absent_from_three_inputs_n':int(((pred>0)&~(q[['input_'+c+'_'+v for v in ['side','front','both']]].to_numpy()==pred[:,None]).any(axis=1)).sum())})
                true=q.ref_both_code.map(self.actions).to_numpy(int);predact=q.predicted_code.map(self.actions).fillna(0).to_numpy(int)
                exact=q.predicted_code.to_numpy()==q.ref_both_code.to_numpy()
                row.update(exact_code_correct=int(exact.sum()),exact_code_accuracy=float(exact.mean()),
                    complete_predictions=int(q.predicted_code.ne('NA').sum()),action_correct=int((true==predact).sum()),
                    action_accuracy=float((true==predact).mean()),high_risk_reference_n=int((true>=3).sum()),
                    high_risk_predicted_low=int(((true>=3)&(predact>0)&(predact<3)).sum()),high_risk_unresolved=int(((true>=3)&(predact==0)).sum()))
                metrics.append(row)
                for obs,t,pa in zip(q.observation_index,true,predact):
                    actions.append({**{k:row[k] for k in ['context','model','method','subset']},'observation_index':int(obs),'true_action':int(t),'pred_action':int(pa)})
                if method!='archived_both':
                    base=predictions[(predictions.model==version)&(predictions.method=='archived_both')].set_index('observation_index').loc[q.observation_index]
                    a=base.predicted_code.to_numpy()==base.ref_both_code.to_numpy();b=exact
                    contrasts.append({**{k:row[k] for k in ['context','model','method','subset']},'baseline':'archived_both','n':len(q),
                        'both_correct':int((a&b).sum()),'baseline_only_correct':int((a&~b).sum()),'method_only_correct':int((~a&b).sum()),
                        'neither_correct':int((~a&~b).sum()),'baseline_correct':int(a.sum()),'method_correct':int(b.sum())})
        for name,records in [('metrics',metrics),('class_metrics',class_rows),('action_predictions',actions),('paired_contrasts',contrasts),('input_NA_diagnostics',diagnostics)]:
            pd.DataFrame(records).to_csv(directory/(name+'.csv'),index=False)

    def evaluate(self,name,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        indices=np.arange(len(self.df))
        if name=='recording_transfer':
            outer=[(np.flatnonzero(self.ds==a),np.flatnonzero(self.ds==b),np.array([],int)) for a,b in [('coverall','everyday'),('everyday','coverall')]]
            entries=[(101+i,tr,te,p) for i,(tr,te,p) in enumerate(outer)]
        else:entries=[(i,tr,te,p) for i,(tr,te,p) in enumerate(self.splits(indices,block,5,SEED),1)]
        parts=[];splits=[]
        for fold,tr,te,purge in entries:
            entry=self.fit_context(name,fold,tr,te,block);parts.append(self.prediction_rows(name,entry))
            splits.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'purged':purge.tolist(),'inner':entry['inner']})
        result=pd.concat(parts,ignore_index=True)
        assert result.groupby(['model','method']).size().eq(181).all()
        assert not result.duplicated(['model','method','observation_index']).any()
        result.to_csv(directory/'predictions.csv',index=False)
        write_json(directory/'splits.json',{'block_seconds':block,'purge_seconds':10,'folds':splits})
        self.report(result,directory)

    def frozen_models(self):
        entry=self.fit_context('full_training',201,np.arange(len(self.df)),np.array([],int),60)
        directory=HERE/'frozen_candidates';directory.mkdir(exist_ok=True)
        models={v:{c:entry['heads'][v][c]['model'] for c in COMPONENTS} for v in VERSIONS}
        joblib.dump(models,directory/'models.joblib',compress=3)
        write_json(directory/'workflow.json',{'version':'ai_code_fusion_v1','role':'secondary exploratory code-fusion candidates',
            'new_worker_data_seen':False,'external_validation':'not_run','new_AI_calls':False,
            'models_sha256':sha(directory/'models.joblib'),'protocol_freeze_sha256':self.freeze_sha,
            'training_observations':181,'models':VERSIONS,'systems':['component_majority','categorical_logistic'],
            'selected_C':{v:{c:entry['heads'][v][c]['selected_C'] for c in COMPONENTS} for v in VERSIONS},
            'source_sha256':{'ai_code_fusion_v1/code_models.py':sha(HERE/'code_models.py')},
            'input_schema':'dataset,frame_id,model,combination,back,arms,legs,load,OWAS_code plus worker_id for external; conditions1/2/3 only',
            'limits':'Code fusion only. No image processing, new-worker result or calibrated-confidence claim. modular_v1 primary unchanged.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    freeze()
    if args.freeze_only:print('PROTOCOL_FROZEN',flush=True);return
    warnings.filterwarnings('error',category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        study=Study()
        for name,block in [('blocks_60s',60),('blocks_30s',30),('recording_transfer',60)]:study.evaluate(name,block)
        study.frozen_models()
    write_json(HERE/'status.json',{'status':'experiments_complete','new_AI_calls':False,'raw_photos_processed':False,
        'new_worker_data_seen':False,'manuscripts_modified':False,'original_ML_releases_modified':False,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib']}})
    print('AI_CODE_FUSION_COMPLETE',flush=True)

if __name__=='__main__':main()
