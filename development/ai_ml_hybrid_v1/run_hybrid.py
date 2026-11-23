"""Three hybrid comparisons with fully nested, purged base cross-fitting."""
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
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ai_code_fusion_v1.run_fusion import Study as CodeStudy
from ai_code_fusion_v1.code_models import COMPONENTS,CLASSES,VERSIONS,SEED,construct_codes
from ai_ml_hybrid_v1.hybrid_models import SYSTEMS,C_GRID,assemble,fit_base,predict_base,fit_meta,fill_NA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')

def freeze():
    paths=[HERE/n for n in ['protocol.json','hybrid_models.py','run_hybrid.py','predict_hybrid.py','seal_secondary.py']]
    paths.extend(ROOT/n for n in ['features/features.npz','ml_improvement_v2/torso_features.npz',
        'leg_followup/geometry_features.npz','audit/paired_manifest.csv','inputs/ai_coding_input.csv',
        'inputs/expert_consensus_input.csv','inputs/owas_action_reference.csv','modular_followup/models.py',
        'ai_code_fusion_v1/code_models.py','ai_code_fusion_v1/run_fusion.py',
        'modular_followup/frozen_workflow/external_release.json',
        'ml_improvement_v2/frozen_candidates/secondary_release.json',
        'ai_code_fusion_v1/frozen_candidates/secondary_release.json'])
    paths.extend(ROOT/'ai_code_fusion_v1/results'/n/'predictions.csv' for n in ['blocks_60s','blocks_30s','recording_transfer'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    path=HERE/'protocol_freeze.json'
    if path.exists():assert json.loads(path.read_text())['sha256']==hashes,'Frozen hybrid source changed'
    else:
        assert not (HERE/'results').exists(),'Freeze before hybrid accuracy'
        write_json(path,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'hybrid_accuracy_evaluated':False,
            'prior_results_and_error_overlap_seen':True,'new_worker_data_seen':False,'new_AI_calls':False,'sha256':hashes})
    return sha(path)

class HybridStudy(CodeStudy):
    def __init__(self):
        super().__init__()
        self.freeze_sha=sha(HERE/'protocol_freeze.json')
        core=np.load(ROOT/'features/features.npz',allow_pickle=False)
        torso=np.load(ROOT/'ml_improvement_v2/torso_features.npz',allow_pickle=False)
        geo=np.load(ROOT/'leg_followup/geometry_features.npz',allow_pickle=False)
        for k in ['dataset','frame_id','timestamp_seconds']:
            assert np.array_equal(core[k],self.df[k].to_numpy())
            assert np.array_equal(torso[k],core[k])
        arrays={k:core[k] for k in ['pose_front','pose_side','pose_detected_front','pose_detected_side']}
        arrays.update({k:torso[k] for k in ['torso_front','torso_side']})
        arrays.update({k:geo[k] for k in ['geometry_front','geometry_side','available_front','available_side']})
        self.bx,self.bav=assemble(arrays);self.base_cache={}

    def base(self,train,query):
        key=tuple(map(int,train))
        if key not in self.base_cache:
            models={};support={}
            for c in COMPONENTS:models[c],support[c]=fit_base(self.bx[c],self.bav[c],self.y[c],train,c)
            self.base_cache[key]=(models,support)
        models,support=self.base_cache[key]
        pred=np.column_stack([predict_base(models[c],self.bx[c],self.bav[c],query) for c in COMPONENTS])
        return pred,models,support

    def crossfit(self,train,block,seed):
        folds=self.splits(train,block,3,seed);pred=np.zeros((len(train),3),int);covered=np.zeros(len(train),int)
        positions={int(i):j for j,i in enumerate(train)};records=[]
        for it,iv,purge in folds:
            p,_,support=self.base(it,iv);pos=np.array([positions[int(i)] for i in iv])
            pred[pos]=p;covered[pos]+=1
            records.append({'train':it.tolist(),'validation':iv.tolist(),'purged':purge.tolist(),
                'ML_predictions':p,'base_support':support})
        assert np.all(covered==1),'Every meta training row must have exactly one crossfit prediction'
        return pred,records

    def fit_context(self,name,fold,train,test,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'context_{fold}.joblib'
        if path.exists():
            entry=joblib.load(path)
            assert entry['freeze_sha256']==self.freeze_sha and entry['train']==train.tolist() and entry['test']==test.tolist()
            return entry
        outer_p,models,support=self.base(train,test)
        train_ml,inner=self.crossfit(train,block,SEED+fold)
        for k,record in enumerate(inner,1):
            it=np.array(record['train'],int)
            inner_ml,further=self.crossfit(it,block,SEED+fold+10000+k)
            record['meta_training_ML']=inner_ml;record['further_seed']=SEED+fold+10000+k;record['further']=further
        entry={'context':name,'fold':fold,'block_seconds':block,'train':train.tolist(),'test':test.tolist(),
            'freeze_sha256':self.freeze_sha,'outer_base_models':models,'outer_base_support':support,
            'test_ML':outer_p,'meta_training_ML':train_ml,'inner':inner,'versions':{}}
        for version in VERSIONS:
            entry['versions'][version]={}
            for j,c in enumerate(COMPONENTS):
                ai=self.x[version][c];y=self.y[c]
                ai_scores=[];ml_scores=[]
                for record in inner:
                    iv=np.array(record['validation'],int)
                    ai_scores.append(float(f1_score(y[iv],ai[iv,2],labels=CLASSES[c],average='macro',zero_division=0)))
                    ml_scores.append(float(f1_score(y[iv],record['ML_predictions'][:,j],labels=CLASSES[c],average='macro',zero_division=0)))
                choice='ML' if np.mean(ml_scores)>np.mean(ai_scores) else 'AI'
                head={'source_selection':{'AI_fold_F1':ai_scores,'ML_fold_F1':ml_scores,
                    'AI_mean_F1':float(np.mean(ai_scores)),'ML_mean_F1':float(np.mean(ml_scores)),'chosen':choice}}
                for kind in ['ai_codes_logistic','ai_ML_stacking']:
                    candidates=[]
                    for C in C_GRID:
                        folds=[]
                        for record in inner:
                            it=np.array(record['train'],int);iv=np.array(record['validation'],int)
                            xt=ai[it];xv=ai[iv]
                            if kind=='ai_ML_stacking':
                                xt=np.column_stack([xt,record['meta_training_ML'][:,j]])
                                xv=np.column_stack([xv,record['ML_predictions'][:,j]])
                            model=fit_meta(xt,y[it],c,C);p=model.predict(xv).astype(int)
                            folds.append({'predictions':p,'macro_f1':float(f1_score(y[iv],p,labels=CLASSES[c],average='macro',zero_division=0)),
                                'train_class_counts':{str(k):int((y[it]==k).sum()) for k in CLASSES[c]}})
                        candidates.append({'C':C,'folds':folds,'mean_macro_f1':float(np.mean([d['macro_f1'] for d in folds]))})
                    best=int(np.argmax([d['mean_macro_f1'] for d in candidates]));C=candidates[best]['C']
                    xt=ai[train];xq=ai[test]
                    if kind=='ai_ML_stacking':xt=np.column_stack([xt,train_ml[:,j]]);xq=np.column_stack([xq,outer_p[:,j]])
                    model=fit_meta(xt,y[train],c,C)
                    head[kind]={'selected_C':C,'candidates':candidates,'model':model,
                        'train_class_counts':{str(k):int((y[train]==k).sum()) for k in CLASSES[c]},
                        'test_predictions':model.predict(xq).astype(int) if len(test) else np.array([],int)}
                entry['versions'][version][c]=head
        joblib.dump(entry,path,compress=3)
        print('FITTED',name,fold,'train',len(train),'test',len(test),'fixed_base_fits_cached',len(self.base_cache),flush=True)
        return entry

    def prediction_rows(self,name,entry):
        test=np.array(entry['test'],int);ml=entry['test_ML'];parts=[]
        for version in VERSIONS:
            for method in SYSTEMS:
                q=self.df.iloc[test].copy();q['observation_index']=test;q['context']=name;q['fold']=entry['fold'];q['model']=version;q['method']=method
                for j,c in enumerate(COMPONENTS):
                    ai=self.x[version][c][test];head=entry['versions'][version][c]
                    for k,v in enumerate(['side','front','both']):q['input_'+c+'_'+v]=ai[:,k]
                    q['input_'+c+'_ML']=ml[:,j];q['all_inputs_NA_'+c]=np.all(ai==0,axis=1).astype(int)
                    q['selected_source_'+c]=head['source_selection']['chosen']
                    if method=='archived_both':pred=ai[:,2]
                    elif method=='fixed_image_ML':pred=ml[:,j]
                    elif method=='ai_NA_filled_by_ML':pred=fill_NA(ai[:,2],ml[:,j])
                    elif method=='component_source_selection':pred=ml[:,j] if head['source_selection']['chosen']=='ML' else ai[:,2]
                    else:pred=head[method]['test_predictions']
                    q['pred_'+c]=pred
                if method=='archived_both':q['predicted_code']=[c if c in self.actions else 'NA' for c in self.archived[version]['3'][test]]
                else:q['predicted_code']=construct_codes(q[['pred_'+c for c in COMPONENTS]].to_numpy())
                q['source_recording']=str(self.ds[entry['train'][0]]) if name=='recording_transfer' else 'pooled';q['test_recording']=q.dataset
                parts.append(q)
        return pd.concat(parts,ignore_index=True)

    def evaluate(self,name,block):
        directory=HERE/'results'/name;directory.mkdir(parents=True,exist_ok=True)
        if name=='recording_transfer':
            entries=[(101+i,np.flatnonzero(self.ds==a),np.flatnonzero(self.ds==b),np.array([],int)) for i,(a,b) in enumerate([('coverall','everyday'),('everyday','coverall')])]
        else:entries=[(i,tr,te,p) for i,(tr,te,p) in enumerate(self.splits(np.arange(181),block,5,SEED),1)]
        parts=[];splits=[];selections=[]
        for fold,tr,te,purge in entries:
            entry=self.fit_context(name,fold,tr,te,block);parts.append(self.prediction_rows(name,entry))
            splits.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'purged':purge.tolist(),
                'inner':[{'train':d['train'],'validation':d['validation'],'purged':d['purged'],
                    'further_seed':d['further_seed'],'further':[{'train':f['train'],'validation':f['validation'],'purged':f['purged']} for f in d['further']]} for d in entry['inner']]})
            for v in VERSIONS:
                for c in COMPONENTS:selections.append({'context':name,'fold':fold,'model':v,'component':c,'test_n':len(te),**entry['versions'][v][c]['source_selection']})
        p=pd.concat(parts,ignore_index=True)
        assert p.groupby(['model','method']).size().eq(181).all() and not p.duplicated(['model','method','observation_index']).any()
        p.to_csv(directory/'predictions.csv',index=False)
        write_json(directory/'splits.json',{'block_seconds':block,'purge_seconds':10,'folds':splits})
        write_json(directory/'source_selections.json',selections)
        self.report(p,directory)
        rows=[]
        for v in VERSIONS:
            for c in COMPONENTS:
                q=p[(p.model==v)&(p.method=='ai_NA_filled_by_ML')];mask=q['input_'+c+'_both'].eq(0)
                rows.append({'context':name,'model':v,'component':c,'AI_NA_n':int(mask.sum()),
                    'filled_n':int((mask&q['pred_'+c].gt(0)).sum()),'filled_correct_n':int((mask&q['pred_'+c].eq(q['target_'+c])).sum())})
        pd.DataFrame(rows).to_csv(directory/'NA_fill_diagnostics.csv',index=False)
        contrasts=[]
        for v in VERSIONS:
            for baseline in ['fixed_image_ML','ai_codes_logistic']:
                for method in ['ai_NA_filled_by_ML','component_source_selection','ai_ML_stacking']:
                    for subset in ['all','coverall','everyday']:
                        b=p[(p.model==v)&(p.method==baseline)];m=p[(p.model==v)&(p.method==method)]
                        if subset!='all':b=b[b.dataset==subset];m=m[m.dataset==subset]
                        b=b.set_index('observation_index').sort_index();m=m.set_index('observation_index').sort_index()
                        a=b.predicted_code.eq(b.ref_both_code).to_numpy();z=m.predicted_code.eq(m.ref_both_code).to_numpy()
                        contrasts.append({'context':name,'model':v,'method':method,'baseline':baseline,'subset':subset,'n':len(b),
                            'both_correct':int((a&z).sum()),'baseline_only_correct':int((a&~z).sum()),'method_only_correct':int((~a&z).sum()),
                            'neither_correct':int((~a&~z).sum()),'baseline_correct':int(a.sum()),'method_correct':int(z.sum())})
        pd.DataFrame(contrasts).to_csv(directory/'additional_paired_contrasts.csv',index=False)

    def frozen_models(self):
        e=self.fit_context('full_training',201,np.arange(181),np.array([],int),60)
        directory=HERE/'frozen_candidates';directory.mkdir(exist_ok=True)
        bundle={'base_models':e['outer_base_models'],
            'meta_models':{v:{c:{k:e['versions'][v][c][k]['model'] for k in ['ai_codes_logistic','ai_ML_stacking']} for c in COMPONENTS} for v in VERSIONS},
            'source_selections':{v:{c:e['versions'][v][c]['source_selection']['chosen'] for c in COMPONENTS} for v in VERSIONS}}
        joblib.dump(bundle,directory/'models.joblib',compress=3)
        write_json(directory/'workflow.json',{'version':'ai_ml_hybrid_v1','role':'secondary exploratory fixed-image-branch hybrid candidates',
            'new_worker_data_seen':False,'external_validation':'not_run','models_sha256':sha(directory/'models.joblib'),
            'protocol_freeze_sha256':self.freeze_sha,'training_observations':181,'systems':SYSTEMS,
            'source_selections':bundle['source_selections'],
            'selected_C':{v:{c:{k:e['versions'][v][c][k]['selected_C'] for k in ['ai_codes_logistic','ai_ML_stacking']} for c in COMPONENTS} for v in VERSIONS},
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'hybrid_models.py',ROOT/'modular_followup/models.py',ROOT/'ai_code_fusion_v1/code_models.py']},
            'limits':'Cached features and archived-format codes only; no raw-photo parity/new-worker score. Prior ML primary/secondary and AI-code releases unchanged.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args();freeze()
    if args.freeze_only:print('PROTOCOL_FROZEN');return
    warnings.filterwarnings('error',category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        study=HybridStudy()
        for name,block in [('blocks_60s',60),('blocks_30s',30),('recording_transfer',60)]:study.evaluate(name,block)
        study.frozen_models()
    write_json(HERE/'status.json',{'status':'experiments_complete','new_AI_calls':False,'raw_photos_processed':False,
        'new_worker_data_seen':False,'manuscripts_modified':False,'prior_releases_modified':False,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','joblib']}})
    print('AI_ML_HYBRID_COMPLETE',flush=True)
if __name__=='__main__':main()
