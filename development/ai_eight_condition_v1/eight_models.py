"""Eight fallible categorical AI outputs; expert targets never enter features."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from ai_code_fusion_v1.code_models import COMPONENTS,CLASSES,DOMAINS,VERSIONS,SEED,parse_component

SETS=['AI3','AI8','AI8ML'];FAMILIES=['Logistic','RF','RBF_SVM']
LEARNED=[s+'_'+f for f in FAMILIES for s in SETS]
SYSTEMS=['archived_both','fixed_image_ML','ai_NA_filled_by_ML']+LEARNED
def spec(method):
    for s in SETS:
        if method.startswith(s+'_'):return s,method[len(s)+1:]
    raise ValueError(method)
def candidates(family):
    return [{'max_depth':d} for d in [3,5]] if family=='RF' else [{'C':c} for c in [.01,.1,1.,10.]]
def inputs(ai,ml,method):
    s,_=spec(method)
    return ai[:,:3] if s=='AI3' else ai if s=='AI8' else np.column_stack([ai,ml])
def fit_model(x,y,component,family,params):
    if not len(y):raise ValueError('Empty training rows')
    encoder=OneHotEncoder(categories=[np.array(DOMAINS[component],int) for _ in range(x.shape[1])],
        drop=None,handle_unknown='error',sparse_output=False,dtype=np.float64)
    if len(np.unique(y))==1:clf=DummyClassifier(strategy='most_frequent')
    elif family=='Logistic':clf=LogisticRegression(**params,l1_ratio=0.,solver='lbfgs',class_weight='balanced',max_iter=5000,tol=.0001,random_state=SEED)
    elif family=='RF':clf=RandomForestClassifier(**params,n_estimators=200,min_samples_leaf=3,class_weight='balanced_subsample',random_state=SEED,n_jobs=1)
    elif family=='RBF_SVM':clf=SVC(**params,kernel='rbf',gamma='scale',class_weight='balanced',probability=False)
    else:raise ValueError(family)
    model=make_pipeline(encoder,clf);model.fit(x,y);return model
def load_eight(path,identities,version):
    raw=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    raw=raw[(raw.model==version)&raw.combination.isin([str(i) for i in range(1,9)])]
    if raw.duplicated(['dataset','frame_id','combination']).any():raise ValueError('Duplicate AI observation')
    expected=set(map(tuple,identities[['dataset','frame_id']].to_numpy()));arrays={c:[] for c in COMPONENTS};archive={}
    for condition in range(1,9):
        q=raw[raw.combination==str(condition)]
        if set(map(tuple,q[['dataset','frame_id']].to_numpy()))!=expected:raise ValueError('Missing/extra AI identities')
        q=identities[['dataset','frame_id']].merge(q,on=['dataset','frame_id'],how='left',validate='one_to_one',sort=False)
        if not q.load.eq('1').all():raise ValueError('Only load1 task evaluated')
        for c in COMPONENTS:arrays[c].append(parse_component(q[c],c))
        archive[str(condition)]=q.OWAS_code.to_numpy(dtype=str)
    return {c:np.column_stack(v) for c,v in arrays.items()},archive
def dependency_audit(path):
    raw=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    records=[]
    for version in VERSIONS:
        d=raw[raw.model==version]
        for condition,source in [(4,1),(5,2),(6,3),(7,2),(8,1)]:
            a=d[d.combination==str(condition)].set_index(['dataset','frame_id']).sort_index()
            b=d[d.combination==str(source)].set_index(['dataset','frame_id']).sort_index()
            if not a.index.equals(b.index):raise ValueError('Prior identity mismatch')
            for c in COMPONENTS+['load','OWAS_code']:
                if not a['prior_'+c].eq(b[c]).all():raise ValueError('Prior differs from archived AI '+str(condition)+' '+c)
            source_name={1:'side',2:'front',3:'both'}[source]
            if not a.prior_source.eq(source_name).all():raise ValueError('Prior source mismatch')
            records.append({'model':version,'condition':condition,'prior_condition':source,'rows':len(a),'all_prior_components_and_code_match':True})
    return records
