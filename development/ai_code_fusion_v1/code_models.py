"""Pure three-code features, deterministic voting, and train-only meta models."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

SEED=20261029
COMPONENTS=['back','arms','legs']
CLASSES={'back':[1,2,3,4],'arms':[1,2,3],'legs':[2,3,4,5,7]}
DOMAINS={'back':list(range(5)),'arms':list(range(4)),'legs':list(range(8))}
VERSIONS=['M56','M61']
SYSTEMS=['archived_side','archived_front','archived_both','component_majority','categorical_logistic']
C_GRID=[.01,.1,1.,10.]

def parse_component(values,component):
    out=[]
    for value in values:
        s=str(value).strip()
        if s in ['NA','0']:out.append(0);continue
        if s not in [str(c) for c in DOMAINS[component] if c]:
            raise ValueError('Invalid '+component+' code: '+repr(s))
        out.append(int(s))
    return np.asarray(out,dtype=np.int16)

def load_three_codes(path,identities,version):
    df=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    df=df[(df.model==version)&df.combination.isin(['1','2','3'])].copy()
    if df.duplicated(['dataset','frame_id','combination']).any():raise ValueError('Duplicate AI input')
    expected=set(map(tuple,identities[['dataset','frame_id']].to_numpy()))
    arrays={};archived={}
    for condition in ['1','2','3']:
        q=df[df.combination==condition]
        if set(map(tuple,q[['dataset','frame_id']].to_numpy()))!=expected:raise ValueError('AI identity mismatch')
        ordered=identities[['dataset','frame_id']].merge(q,on=['dataset','frame_id'],validate='one_to_one',how='left',sort=False)
        for c in COMPONENTS:arrays.setdefault(c,[]).append(parse_component(ordered[c],c))
        if not ordered.load.eq('1').all():raise ValueError('This protocol requires archived load1')
        archived[condition]=ordered.OWAS_code.to_numpy(dtype=str)
    return {c:np.stack(v,axis=1) for c,v in arrays.items()},archived

def majority(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError('Exactly three codes are required')
    out=np.zeros(len(x),dtype=np.int16)
    for i,row in enumerate(x):
        values,counts=np.unique(row[row>0],return_counts=True)
        if len(counts) and counts.max()>=2:out[i]=values[np.argmax(counts)]
    return out

def fit_meta(x,y,component,C):
    if not len(y):raise ValueError('No training rows')
    categories=[np.array(DOMAINS[component],dtype=int)]*3
    encoder=OneHotEncoder(categories=categories,drop=None,handle_unknown='error',sparse_output=False,dtype=np.float64)
    classifier=DummyClassifier(strategy='most_frequent') if len(np.unique(y))==1 else LogisticRegression(
        C=C,l1_ratio=0.,solver='lbfgs',class_weight='balanced',max_iter=5000,tol=.0001,random_state=SEED)
    model=make_pipeline(encoder,classifier)
    model.fit(x,y)
    return model

def construct_codes(predictions):
    return np.array([''.join(str(int(v)) for v in row)+'1' if all(v>0 for v in row) else 'NA'
                     for row in np.asarray(predictions)],dtype=str)
