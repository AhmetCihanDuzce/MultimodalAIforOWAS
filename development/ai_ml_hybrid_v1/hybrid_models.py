"""Fixed image branch and categorical meta models; no inference-time fitting."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from modular_followup.models import base_estimator
from ai_code_fusion_v1.code_models import CLASSES,COMPONENTS,DOMAINS,SEED

SYSTEMS=['archived_both','fixed_image_ML','ai_NA_filled_by_ML','component_source_selection','ai_codes_logistic','ai_ML_stacking']
C_GRID=[.01,.1,1.,10.]

def assemble(arrays):
    x={};av={}
    x['back']=np.concatenate([np.concatenate([arrays['pose_'+v],arrays['torso_'+v]],axis=1) for v in ['front','side']],axis=1)
    av['back']=np.logical_or(arrays['pose_detected_front'],arrays['pose_detected_side'])
    x['arms']=arrays['pose_side'];av['arms']=arrays['pose_detected_side'].astype(bool)
    x['legs']=np.concatenate([np.concatenate([arrays['pose_'+v],arrays['geometry_'+v]],axis=1) for v in ['front','side']],axis=1)
    av['legs']=np.logical_or(arrays['available_front'],arrays['available_side'])
    return x,av

def fit_base(x,av,y,train,component):
    usable=train[av[train]]
    support={'train':train.tolist(),'usable_train':usable.tolist(),
        'train_n':len(train),'usable_train_n':len(usable),
        'class_counts':{str(k):int((y[usable]==k).sum()) for k in CLASSES[component]}}
    if not len(usable):return None,{**support,'status':'no_usable_training_rows'}
    if len(np.unique(y[usable]))==1:model=DummyClassifier(strategy='most_frequent');status='single_training_class'
    elif component=='arms':model=base_estimator('pose_RBF_SVM',{'C':10});status='fitted'
    else:model=base_estimator('pose_RF',{'max_depth':5});status='fitted'
    model.fit(x[usable],y[usable]);return model,{**support,'status':status}

def predict_base(model,x,av,query):
    pred=np.zeros(len(query),int);ok=av[query]
    if model is not None and ok.any():pred[ok]=model.predict(x[query[ok]]).astype(int)
    return pred

def fit_meta(x,y,component,C):
    if len(y)==0:raise ValueError('Empty meta training')
    encoder=OneHotEncoder(categories=[np.array(DOMAINS[component],int) for _ in range(x.shape[1])],
        drop=None,handle_unknown='error',sparse_output=False,dtype=np.float64)
    classifier=DummyClassifier(strategy='most_frequent') if len(np.unique(y))==1 else LogisticRegression(
        C=C,l1_ratio=0.,solver='lbfgs',class_weight='balanced',max_iter=5000,tol=.0001,random_state=SEED)
    model=make_pipeline(encoder,classifier);model.fit(x,y);return model

def fill_NA(ai,ml):return np.where(np.asarray(ai)>0,ai,ml).astype(int)
