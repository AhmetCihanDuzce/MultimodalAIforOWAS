"""Two recording OWAS study. Original observations are never overwritten.

All analyses are exploratory. Frame-independent tests and ordered-block
resampling are conditional sensitivity calculations, not worker-population
inference. Lux is categorical and recording periods are not randomized.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import csv,io,re,json,hashlib,math,itertools,platform
import numpy as np
import pandas as pd
from scipy.stats import binomtest,chi2
from owas_core import LOOKUP,wilson,kappa,holm,ffh

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'owas-data-results'
OUT=ROOT/'generated'/'results'
OUT.parent.mkdir(parents=True,exist_ok=True)
MODELS=['M56','M61'];SETS=['coverall','everyday'];VIEWS=['both','side','front']
COMPONENTS={'back':4,'arms':3,'legs':7}
LUX_GROUPS=[(1,400),(6,250),(11,100)]
PRIOR={4:1,5:2,6:3,7:2,8:1}
CONTRASTS=[(1,3),(2,3),(3,4),(3,5),(3,6),(1,7),(2,8)]
ENDPOINTS=['posture','complete','action','back','arms','legs']

def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig',na_rep='NA',float_format='%.12g')

def read_inputs():
    """Read deidentified public CSV inputs; preserve issued and prior values.

    No images, private workbook metadata, or original file paths are required.
    CSV records were exported from the previously audited archived matrices.
    """
    reference=[];rows=[];audit=[];prior_audit=[];refs={}
    input_dir=DATA/'inputs'
    def read_csv(name):
        with (input_dir/name).open(encoding='utf-8-sig',newline='') as handle:
            return list(csv.DictReader(handle))
    raw_references=read_csv('expert_consensus_input.csv')
    expected_ref_fields={'dataset','frame_id','illuminance_lux','ref_front_code','ref_side_code','ref_both_code'}
    assert raw_references and set(raw_references[0])==expected_ref_fields
    for item in raw_references:
        dataset=item['dataset'];fid=item['frame_id'];lux=int(item['illuminance_lux'])
        assert dataset in SETS and re.fullmatch(r'\d{5}',fid) and lux in [100,250,400]
        r=dict(dataset=dataset,frame_id=fid,frame_key=f'{dataset}:{fid}',illuminance_lux=lux)
        for view in ['front','side','both']:
            code=item['ref_'+view+'_code']
            assert re.fullmatch(r'[1-4][1-3][1-7]1',code),(dataset,fid,code)
            r['ref_'+view+'_code']=code;r['ref_'+view+'_action']=LOOKUP[code]
            for comp,digit in zip(['back','arms','legs','load'],code):r['ref_'+view+'_'+comp]=digit
        assert r['frame_key'] not in refs
        refs[r['frame_key']]=r;reference.append(r)
    assert len(reference)==181
    assert Counter(r['dataset'] for r in reference)=={'coverall':69,'everyday':112}
    assert Counter((r['dataset'],r['illuminance_lux']) for r in reference)=={('coverall',100):15,('coverall',250):22,('coverall',400):32,('everyday',100):37,('everyday',250):36,('everyday',400):39}
    raw_rows=read_csv('ai_coding_input.csv')
    expected_original={'frame_id','back','arms','legs','load','OWAS_code','status','short_note','back_evidence','arms_evidence','legs_evidence','back_prior_influence','arms_prior_influence','legs_prior_influence','prior_source','prior_back','prior_arms','prior_legs','prior_load','prior_OWAS_code','prior_status','dataset','model','combination'}
    assert raw_rows and set(raw_rows[0])==expected_original
    by={};seen=set()
    for original in raw_rows:
        dataset=original['dataset'];model=original['model'];condition=int(original['combination']);fid=original['frame_id']
        assert dataset in SETS and model in MODELS and 1<=condition<=8
        key=(dataset,model,condition,fid)
        assert key not in seen;seen.add(key)
        reference_key=f'{dataset}:{fid}';assert reference_key in refs
        r={k:v for k,v in original.items() if k not in ['dataset','model','combination']}
        r.update(refs[reference_key]);r.update(model=model,combination=condition)
        for comp,mx in COMPONENTS.items():assert r[comp]=='NA' or r[comp] in list(map(str,range(1,mx+1)))
        assert r['load']=='1'
        complete=all(r[c]!='NA' for c in COMPONENTS)
        assert r['OWAS_code']==(''.join(r[c] for c in ['back','arms','legs','load']) if complete else 'NA')
        assert r['status']==('ok' if complete else 'partial_na')
        r['action']=LOOKUP[r['OWAS_code']] if complete else None
        r['posture_match']=int(r['OWAS_code']==r['ref_both_code'])
        r['complete_match']=int(complete)
        r['action_match']=int(r['action']==r['ref_both_action'])
        for comp in COMPONENTS:r[comp+'_match']=int(r[comp]==r['ref_both_'+comp])
        rows.append(r);by.setdefault((dataset,model,condition),{})[fid]=r
    assert len(rows)==2896 and len(by)==32
    for dataset in SETS:
        ids={r['frame_id'] for r in reference if r['dataset']==dataset}
        for model in MODELS:
            for condition in range(1,9):
                selected=by[dataset,model,condition]
                assert set(selected)==ids
                audit.append(dict(dataset=dataset,model=model,combination=condition,input_file='ai_coding_input.csv',n=len(selected)))
            for condition,baseline in PRIOR.items():
                comparisons=0
                for fid,r in by[dataset,model,condition].items():
                    for field in ['back','arms','legs','load','OWAS_code']:
                        comparisons+=1
                        assert r['prior_'+field]==by[dataset,model,baseline][fid][field],(dataset,model,condition,fid,field)
                prior_audit.append(dict(dataset=dataset,model=model,combination=condition,prior_condition=baseline,frames=len(ids),compared_values=comparisons,mismatches=0))
    assert sum(r['compared_values'] for r in prior_audit)==9050
    rows=sorted(rows,key=lambda r:(r['dataset'],r['model'],r['combination'],r['frame_id']))
    save('expert_consensus',reference);save('coding_matrix',rows);save('input_audit',audit);save('prior_provenance_audit',prior_audit)
    save('action_lookup_load1',[dict(OWAS_code=k,action=v) for k,v in LOOKUP.items()])
    manifest={str(f.relative_to(DATA)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(input_dir.glob('*.csv'))}
    (OUT/'input_sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return rows,reference

def groups(rows):
    for dataset in SETS+['pooled']:
        selected=[r for r in rows if dataset=='pooled' or r['dataset']==dataset]
        for model in MODELS:
            for condition in range(1,9):
                yield dataset,model,condition,[r for r in selected if r['model']==model and r['combination']==condition]

def metrics(rr,view):
    n=len(rr);complete=[r for r in rr if r['OWAS_code']!='NA']
    exact=sum(r['OWAS_code']==r['ref_'+view+'_code'] for r in rr)
    amatch=sum(r['action']==r['ref_'+view+'_action'] for r in complete)
    high=[r for r in rr if r['ref_'+view+'_action']>=3]
    lo,hi=wilson(exact,n)
    return dict(n=n,complete_n=len(complete),exact_n=exact,exact_pct=100*exact/n,complete_pct=100*len(complete)/n,
        conditional_exact_pct=100*exact/len(complete) if complete else None,exact_wilson_low=lo,exact_wilson_high=hi,
        posture_kappa=kappa([r['ref_'+view+'_code'] for r in complete],[r['OWAS_code'] for r in complete]),
        action_exact_n=amatch,action_exact_pct=100*amatch/n,
        action_kappa=kappa([r['ref_'+view+'_action'] for r in complete],[r['action'] for r in complete],True),
        action_below_n=sum(r['action']<r['ref_'+view+'_action'] for r in complete),action_above_n=sum(r['action']>r['ref_'+view+'_action'] for r in complete),
        high_reference_n=len(high),high_retained_n=sum(r['action'] is not None and r['action']>=3 for r in high),
        high_lower_n=sum(r['action'] is not None and r['action']<3 for r in high),high_unresolved_n=sum(r['action'] is None for r in high))

def descriptive(rows,reference):
    summary=[];lights=[];components=[];classes=[];confusions=[]
    for dataset,model,condition,rr in groups(rows):
        for view in VIEWS:
            key=dict(dataset=dataset,model=model,combination=condition,reference=view)
            summary.append(dict(**key,**metrics(rr,view)))
            for lux in [100,250,400]:lights.append(dict(**key,illuminance_lux=lux,**metrics([r for r in rr if r['illuminance_lux']==lux],view)))
            for comp,mx in COMPONENTS.items():
                valid=[r for r in rr if r[comp]!='NA'];equal=sum(r[comp]==r['ref_'+view+'_'+comp] for r in rr)
                recall=[];f1=[]
                for code in map(str,range(1,mx+1)):
                    ref_n=sum(r['ref_'+view+'_'+comp]==code for r in rr);pred_n=sum(r[comp]==code for r in rr)
                    tp=sum(r[comp]==code and r['ref_'+view+'_'+comp]==code for r in rr)
                    na=sum(r[comp]=='NA' and r['ref_'+view+'_'+comp]==code for r in rr)
                    if ref_n:recall.append(tp/ref_n);f1.append(2*tp/(ref_n+pred_n))
                    classes.append(dict(**key,component=comp,category=code,reference_n=ref_n,predicted_n=pred_n,matched_n=tp,unresolved_reference_n=na,
                        recall_pct=100*tp/ref_n if ref_n else None,precision_pct=100*tp/pred_n if pred_n else None,f1_pct=200*tp/(ref_n+pred_n) if ref_n+pred_n else None))
                    for predicted in list(map(str,range(1,mx+1)))+['NA']:
                        confusions.append(dict(**key,component=comp,reference_category=code,predicted_category=predicted,n=sum(r['ref_'+view+'_'+comp]==code and r[comp]==predicted for r in rr)))
                components.append(dict(**key,component=comp,n=len(rr),coded_n=len(valid),exact_n=equal,exact_pct=100*equal/len(rr),
                    conditional_exact_pct=100*equal/len(valid) if valid else None,macro_recall_pct=100*np.mean(recall),macro_f1_pct=100*np.mean(f1),
                    kappa=kappa([r['ref_'+view+'_'+comp] for r in valid],[r[comp] for r in valid])))
    save('performance',summary);save('illuminance_performance',lights);save('component_performance',components);save('category_performance',classes);save('confusion_matrices',confusions)
    composition=[];agreements=[]
    for dataset in SETS+['pooled']:
        rr=[r for r in reference if dataset=='pooled' or r['dataset']==dataset]
        for lux in [0,100,250,400]:
            ss=rr if not lux else [r for r in rr if r['illuminance_lux']==lux]
            for comp,mx in COMPONENTS.items():
                for code in map(str,range(1,mx+1)):
                    composition.append(dict(dataset=dataset,illuminance_lux=lux or 'all',component=comp,category=code,n=sum(r['ref_both_'+comp]==code for r in ss),total=len(ss)))
        for a,b in [('front','both'),('side','both'),('front','side')]:
            n=sum(r['ref_'+a+'_code']==r['ref_'+b+'_code'] for r in rr)
            agreements.append(dict(dataset=dataset,reference_a=a,reference_b=b,n=len(rr),matches=n,agreement_pct=100*n/len(rr),kappa=kappa([r['ref_'+a+'_code'] for r in rr],[r['ref_'+b+'_code'] for r in rr])))
    save('reference_composition',composition);save('reference_agreement',agreements)
    weighting=[]
    for model in MODELS:
        for c in range(1,9):
            scores={r['dataset']:r for r in summary if r['model']==model and r['combination']==c and r['reference']=='both'}
            weighting.append(dict(model=model,combination=c,frame_weighted_pct=scores['pooled']['exact_pct'],
                equal_recording_weight_pct=(scores['coverall']['exact_pct']+scores['everyday']['exact_pct'])/2,
                coverall_pct=scores['coverall']['exact_pct'],everyday_pct=scores['everyday']['exact_pct']))
    save('recording_weight_sensitivity',weighting)
    return summary,components

def endpoint(r,name):return r[name+'_match']
def paired(a,b):
    both=sum(x and y for x,y in zip(a,b));gain=sum(not x and y for x,y in zip(a,b));loss=sum(x and not y for x,y in zip(a,b));n=len(a)
    return dict(n=n,a_match_n=sum(a),b_match_n=sum(b),both_match_n=both,gains=gain,losses=loss,delta_pp=100*(gain-loss)/n,
        p_raw=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.)

def comparisons(rows):
    versions=[];directed=[];component_tests=[];omnibus=[];lighting=[];shared=[];priors=[];transitions=[]
    by={(ds,m,c):rr for ds,m,c,rr in groups(rows)}
    for dataset in SETS+['pooled']:
        for c in range(1,9):
            aa=sorted(by[dataset,'M56',c],key=lambda r:r['frame_key']);bb=sorted(by[dataset,'M61',c],key=lambda r:r['frame_key'])
            assert [r['frame_key'] for r in aa]==[r['frame_key'] for r in bb]
            for ep in ENDPOINTS:versions.append(dict(dataset=dataset,combination=c,endpoint=ep,**paired([endpoint(r,ep) for r in aa],[endpoint(r,ep) for r in bb])))
            selected=[(a,b) for a,b in zip(aa,bb) if a['complete_match'] and b['complete_match']]
            if selected:shared.append(dict(dataset=dataset,combination=c,**paired([a['posture_match'] for a,b in selected],[b['posture_match'] for a,b in selected])))
        for m in MODELS:
            for a,b in CONTRASTS:
                aa=sorted(by[dataset,m,a],key=lambda r:r['frame_key']);bb=sorted(by[dataset,m,b],key=lambda r:r['frame_key'])
                for ep in ENDPOINTS:directed.append(dict(dataset=dataset,model=m,condition_a=a,condition_b=b,endpoint=ep,**paired([endpoint(r,ep) for r in aa],[endpoint(r,ep) for r in bb])))
                for x,y in zip(aa,bb):
                    status=lambda r:'exact' if r['posture_match'] else 'mismatch' if r['complete_match'] else 'unresolved'
                    transitions.append(dict(dataset=dataset,model=m,condition_a=a,condition_b=b,frame_key=x['frame_key'],state_a=status(x),state_b=status(y)))
            for ep in ['posture','complete','action']:
                arr=np.array([[endpoint(r,ep) for r in sorted(by[dataset,m,c],key=lambda r:r['frame_key'])] for c in range(1,9)]).T
                cs=arr.sum(0);rs=arr.sum(1);den=8*cs.sum()-(rs*rs).sum();q=7*(8*(cs*cs).sum()-cs.sum()**2)/den if den else 0.
                omnibus.append(dict(dataset=dataset,model=m,endpoint=ep,n=len(arr),q=q,df=7,p_raw=chi2.sf(q,7)))
            for c in range(1,9):
                rr=by[dataset,m,c]
                for a,b in itertools.combinations(COMPONENTS,2):component_tests.append(dict(dataset=dataset,model=m,combination=c,component_a=a,component_b=b,**paired([r[a+'_match'] for r in rr],[r[b+'_match'] for r in rr])))
                totals=[sum(r['illuminance_lux']==v for r in rr) for v in [100,250,400]]
                for ep in ENDPOINTS:
                    successes=[sum(endpoint(r,ep) for r in rr if r['illuminance_lux']==v) for v in [100,250,400]]
                    lighting.append(dict(dataset=dataset,model=m,combination=c,endpoint=ep,n100=totals[0],n250=totals[1],n400=totals[2],match100=successes[0],match250=successes[1],match400=successes[2],p_raw=ffh(successes,totals)))
                if c in PRIOR:
                    complete=[r for r in rr if r['prior_OWAS_code']!='NA' and r['OWAS_code']!='NA']
                    wrong=[r for r in rr if r['prior_OWAS_code']!='NA' and r['prior_OWAS_code']!=r['ref_both_code']]
                    priors.append(dict(dataset=dataset,model=m,combination=c,prior_condition=PRIOR[c],shared_complete_n=len(complete),same_code_n=sum(r['OWAS_code']==r['prior_OWAS_code'] for r in complete),
                        prior_wrong_n=len(wrong),corrected_n=sum(r['posture_match'] for r in wrong),same_wrong_n=sum(r['OWAS_code']==r['prior_OWAS_code'] for r in wrong),
                        different_wrong_n=sum(r['complete_match'] and not r['posture_match'] and r['OWAS_code']!=r['prior_OWAS_code'] for r in wrong),unresolved_n=sum(not r['complete_match'] for r in wrong)))
    # Explicit families include both recordings and pooled summaries. Component
    # version comparisons and within-version component comparisons share H2.
    families={}
    for r in versions:
        family='H2_component_agreement' if r['endpoint'] in COMPONENTS else 'H1_version_posture' if r['endpoint']=='posture' else 'secondary_'+r['endpoint']
        families.setdefault(family,[]).append(r)
    for r in directed:
        family='secondary_component_information' if r['endpoint'] in COMPONENTS else 'H3_information_posture' if r['endpoint']=='posture' else 'secondary_'+r['endpoint']
        families.setdefault(family,[]).append(r)
    families['H2_component_agreement']+=component_tests
    families['H4_illuminance']=lighting
    families['omnibus_sensitivity']=omnibus
    families['shared_complete_sensitivity']=shared
    for family,rr in families.items():
        for r,p in zip(rr,holm([r['p_raw'] for r in rr])):
            r.update(p_holm=p,family=family,family_n=len(rr),assumption='independent frame pairs; exploratory sensitivity only')
    for name,data in [('version_tests',versions),('information_tests',directed),('component_tests',component_tests),('omnibus_tests',omnibus),('illuminance_tests',lighting),('shared_complete_tests',shared)]:save(name,data)
    save('multiplicity_families',[dict(family=k,tests=len(v),adjustment='Holm',scope='Exploratory; independent-frame sensitivity') for k,v in families.items()])
    save('previous_code_transitions',priors);save('information_frame_transitions',transitions)

def balanced_illuminance(rows):
    result=[];audit=[]
    for dataset,model,c,rr in groups(rows):
        if dataset=='pooled':continue
        for comp in COMPONENTS:
            common=set.intersection(*[{r['ref_both_'+comp] for r in rr if r['illuminance_lux']==lux} for lux in [100,250,400]])
            counts=Counter(r['ref_both_'+comp] for r in rr if r['ref_both_'+comp] in common)
            den=sum(counts.values());weights={k:v/den for k,v in counts.items()}
            for lux in [100,250,400]:
                standardized=0.;eligible=0
                for cat,w in weights.items():
                    ss=[r for r in rr if r['illuminance_lux']==lux and r['ref_both_'+comp]==cat]
                    n=len(ss);k=sum(r[comp+'_match'] for r in ss);eligible+=n;standardized+=w*k/n
                    audit.append(dict(dataset=dataset,model=model,combination=c,component=comp,illuminance_lux=lux,category=cat,n=n,matched=k,weight=w))
                result.append(dict(dataset=dataset,model=model,combination=c,component=comp,illuminance_lux=lux,standardized_match_pct=100*standardized,
                    common_categories='|'.join(sorted(common)),eligible_n=eligible,total_n=sum(r['illuminance_lux']==lux for r in rr),weight_reference_n=den))
    save('illuminance_standardized_components',result);save('illuminance_standardization_audit',audit)

def candidate_codes(r):
    note=r['short_note'];found={c:[] for c in COMPONENTS}
    numeric=r'([1-7](?:\s*(?:/|,|\bor\b|\band\b)\s*[1-7])*)'
    # Candidate clauses in both observed word orders; descriptive-only text is not translated.
    for part in re.split(r'[.;]',note):
        if not re.search(r'candidates?',part,re.I):continue
        kind='prior-informed' if re.search(r'prior[- ]informed',part,re.I) else 'visual' if re.search(r'visual',part,re.I) else 'unspecified'
        for prefix in re.finditer(r'\b(back|arms?|legs?)\s+(?:visual|prior[- ]informed)\s+candidates?\s*[:=]?\s*'+numeric,part,re.I):
            comp={'arm':'arms','leg':'legs'}.get(prefix[1].lower(),prefix[1].lower());vals=list(map(int,re.findall(r'\d',prefix[2])))
            found[comp].append((kind,vals,part.strip()))
        for marker in re.finditer(r'(?:visual|prior[- ]informed)\s+candidates?\s*:?\s*(.*)',part,re.I):
            for m in re.finditer(r'\b(back|arms?|legs?)\s*[:=]?\s*'+numeric,marker[1],re.I):
                comp={'arm':'arms','leg':'legs'}.get(m[1].lower(),m[1].lower());vals=list(map(int,re.findall(r'\d',m[2])))
                found[comp].append((kind,vals,part.strip()))
    result={}
    for comp,entries in found.items():
        if r[comp]!='NA':continue
        vals=sorted({v for kind,vs,text in entries for v in vs})
        assert all(1<=v<=COMPONENTS[comp] for v in vals),(comp,entries)
        sources=sorted({kind for kind,vs,text in entries})
        result[comp]=dict(values=vals,source='|'.join(sources),excerpt=' || '.join(dict.fromkeys(text for kind,vs,text in entries)))
    return result

def candidates(rows):
    audit=[];derived=[];compatibility=[]
    for original in rows:
        extracted=candidate_codes(original)
        if extracted:
            available=all(e['values'] for e in extracted.values())
            compatible=all(int(original['ref_both_'+c]) in extracted[c]['values'] if c in extracted else original[c]==original['ref_both_'+c] for c in COMPONENTS) if available else None
            compatibility.append(dict(dataset=original['dataset'],frame_key=original['frame_key'],model=original['model'],combination=original['combination'],
                missing_components=len(extracted),all_missing_have_candidates=int(available),reference_posture_compatible=int(compatible) if available else None,
                complete_code_candidates=math.prod(len(e['values']) for e in extracted.values()) if available else 0,
                has_alternatives=int(any(len(e['values'])>1 for e in extracted.values()))))
        for comp,e in extracted.items():
            vals=e['values'];ref=int(original['ref_both_'+comp])
            audit.append(dict(dataset=original['dataset'],frame_id=original['frame_id'],frame_key=original['frame_key'],model=original['model'],combination=original['combination'],illuminance_lux=original['illuminance_lux'],component=comp,
                candidate_status='none' if not vals else 'single' if len(vals)==1 else 'alternatives',candidate_values='|'.join(map(str,vals)),candidate_n=len(vals),candidate_source=e['source'],
                candidate_excerpt=e['excerpt'],short_note=original['short_note'],reference_code=ref,reference_in_candidates=int(ref in vals) if vals else None,
                equals_prior=int(str(vals[0])==original['prior_'+comp]) if len(vals)==1 and e['source']=='prior-informed' else None))
        for policy in ['visual_single','all_single']:
            r=dict(original);replaced=[]
            for comp,e in extracted.items():
                if len(e['values'])==1 and (policy=='all_single' or e['source']=='visual'):
                    r[comp]=str(e['values'][0]);replaced.append(comp)
            complete=all(r[c]!='NA' for c in COMPONENTS)
            r['OWAS_code']=''.join(r[c] for c in ['back','arms','legs','load']) if complete else 'NA'
            r['action']=LOOKUP[r['OWAS_code']] if complete else None
            r['status']='ok' if complete else 'partial_na'
            r['posture_match']=int(r['OWAS_code']==r['ref_both_code'])
            r['complete_match']=int(complete)
            r['action_match']=int(r['action']==r['ref_both_action'])
            for comp in COMPONENTS:r[comp+'_match']=int(r[comp]==r['ref_both_'+comp])
            r.update(policy=policy,replaced_components='|'.join(replaced),original_OWAS_code=original['OWAS_code'],
                     original_back=original['back'],original_arms=original['arms'],original_legs=original['legs'])
            derived.append(r)
    save('candidate_audit',audit);save('candidate_derived_matrix',derived);save('candidate_frame_compatibility',compatibility)
    summaries=[];outcomes=[]
    for ds in SETS+['pooled']:
        for model in MODELS+['pooled']:
            for comp in list(COMPONENTS)+['all']:
                aa=[a for a in audit if (ds=='pooled' or a['dataset']==ds) and (model=='pooled' or a['model']==model) and (comp=='all' or a['component']==comp)]
                ss=[a for a in aa if a['candidate_status']=='single'];alt=[a for a in aa if a['candidate_status']=='alternatives']
                summaries.append(dict(dataset=ds,model=model,component=comp,unresolved_n=len(aa),single_n=len(ss),single_matched_n=sum(a['reference_in_candidates'] for a in ss),
                    alternatives_n=len(alt),alternatives_covered_n=sum(a['reference_in_candidates'] for a in alt),no_candidate_n=sum(a['candidate_status']=='none' for a in aa),
                    prior_single_n=sum(a['candidate_source']=='prior-informed' for a in ss)))
        for model in MODELS:
            for c in range(1,9):
                orig=[r for r in rows if (ds=='pooled' or r['dataset']==ds) and r['model']==model and r['combination']==c]
                for policy in ['visual_single','all_single']:
                    rr=[r for r in derived if (ds=='pooled' or r['dataset']==ds) and r['model']==model and r['combination']==c and r['policy']==policy]
                    a=metrics(orig,'both');b=metrics(rr,'both')
                    outcomes.append(dict(dataset=ds,model=model,combination=c,policy=policy,n=len(rr),original_complete_n=a['complete_n'],complete_n=b['complete_n'],original_exact_n=a['exact_n'],exact_n=b['exact_n'],
                        new_complete_n=b['complete_n']-a['complete_n'],new_exact_n=b['exact_n']-a['exact_n'],new_mismatch_n=(b['complete_n']-a['complete_n'])-(b['exact_n']-a['exact_n'])))
    save('candidate_summary',summaries);save('candidate_policy_performance',outcomes)
    widths=[];sources=[]
    for ds in SETS+['pooled']:
        aa=[a for a in audit if ds=='pooled' or a['dataset']==ds]
        for width in sorted({a['candidate_n'] for a in aa if a['candidate_n']>0}):
            ss=[a for a in aa if a['candidate_n']==width]
            widths.append(dict(dataset=ds,candidate_n=width,n=len(ss),covered_n=sum(a['reference_in_candidates'] for a in ss)))
        for source in sorted({a['candidate_source'] for a in aa if a['candidate_n']==1}):
            ss=[a for a in aa if a['candidate_n']==1 and a['candidate_source']==source]
            sources.append(dict(dataset=ds,source=source,n=len(ss),matched_n=sum(a['reference_in_candidates'] for a in ss),equal_prior_n=sum(a['equals_prior']==1 for a in ss)))
    save('candidate_set_sizes',widths);save('candidate_sources',sources)

def ordered_bootstrap(rows,replicates=10000):
    """Resample whole frame vectors in each set x lux stratum, never individual outputs.

    Circular moving blocks keep each ordered position equally likely. Frame IDs
    give order only; no claim about elapsed time or verified stationarity is made.
    Lengths 1, 3 and 5 are explicit sensitivity assumptions; all intervals are
    pointwise, not simultaneous confidence bands or hypothesis tests.
    """
    refs=sorted({(r['dataset'],r['frame_id'],r['illuminance_lux']) for r in rows})
    framekeys=[ds+':'+fid for ds,fid,lux in refs];pos={k:i for i,k in enumerate(framekeys)}
    cols=[(m,c,ep) for m in MODELS for c in range(1,9) for ep in ENDPOINTS]
    cp={k:i for i,k in enumerate(cols)};X=np.zeros((len(refs),len(cols)))
    for r in rows:
        for ep in ENDPOINTS:X[pos[r['frame_key']],cp[r['model'],r['combination'],ep]]=endpoint(r,ep)
    intervals=[];delta_intervals=[];interactions=[]
    for length in [1,3,5]:
        rng=np.random.default_rng(20261027+length);weights=np.zeros((replicates,len(refs)))
        for dataset in SETS:
            for lux in [100,250,400]:
                inds=np.array([i for i,(ds,fid,l) in enumerate(refs) if ds==dataset and l==lux]);n=len(inds)
                starts=rng.integers(0,n,size=(replicates,math.ceil(n/length)))
                sample=((starts[:,:,None]+np.arange(length))%n).reshape(replicates,-1)[:,:n]
                for j in range(n):weights[np.arange(replicates),inds[sample[:,j]]]+=1
        boot={};points={}
        for dataset in SETS+['pooled']:
            inds=np.array([i for i,(ds,fid,lux) in enumerate(refs) if dataset=='pooled' or ds==dataset]);n=len(inds)
            boot[dataset]=100*(weights[:,inds]@X[inds])/n;points[dataset]=100*X[inds].mean(0)
            qs=np.quantile(boot[dataset],[.025,.975],axis=0)
            for j,(model,c,ep) in enumerate(cols):intervals.append(dict(dataset=dataset,model=model,combination=c,endpoint=ep,block_length=length,estimate_pct=points[dataset][j],low=qs[0,j],high=qs[1,j],replicates=replicates))
            for ep in ENDPOINTS:
                for c in range(1,9):
                    j=cp['M61',c,ep];i=cp['M56',c,ep];values=boot[dataset][:,j]-boot[dataset][:,i]
                    lo,hi=np.quantile(values,[.025,.975]);delta_intervals.append(dict(dataset=dataset,contrast_type='version',model='M61 minus M56',condition_a=c,condition_b=c,endpoint=ep,block_length=length,estimate_pp=points[dataset][j]-points[dataset][i],low=lo,high=hi))
                for model in MODELS:
                    for a,b in CONTRASTS:
                        i=cp[model,a,ep];j=cp[model,b,ep];values=boot[dataset][:,j]-boot[dataset][:,i]
                        lo,hi=np.quantile(values,[.025,.975]);delta_intervals.append(dict(dataset=dataset,contrast_type='information',model=model,condition_a=a,condition_b=b,endpoint=ep,block_length=length,estimate_pp=points[dataset][j]-points[dataset][i],low=lo,high=hi))
        for ep in ENDPOINTS:
            for model in MODELS:
                for a,b in CONTRASTS:
                    i=cp[model,a,ep];j=cp[model,b,ep]
                    values=(boot['everyday'][:,j]-boot['everyday'][:,i])-(boot['coverall'][:,j]-boot['coverall'][:,i]);lo,hi=np.quantile(values,[.025,.975])
                    interactions.append(dict(model=model,condition_a=a,condition_b=b,endpoint=ep,block_length=length,estimate_pp=(points['everyday'][j]-points['everyday'][i])-(points['coverall'][j]-points['coverall'][i]),low=lo,high=hi,interpretation='difference between observed set-specific contrasts, not clothing effect'))
    save('ordered_bootstrap_intervals',intervals);save('ordered_bootstrap_contrasts',delta_intervals);save('recording_contrast_differences',interactions)
    (OUT/'resampling_settings.json').write_text(json.dumps(dict(replicates=replicates,block_lengths=[1,3,5],seeds={str(k):20261027+k for k in [1,3,5]},strata='dataset x illuminance_lux',sampling='circular moving blocks of frame-ID order; whole frame vectors',scope='Conditional pointwise sensitivity intervals. No verified temporal scale, stationarity or worker-population coverage.'),indent=2)+'\n')

def main():
    OUT.mkdir(exist_ok=True)
    rows,reference=read_inputs();print('Input audit:',len(rows),'outputs;',len(reference),'frame pairs',flush=True)
    descriptive(rows,reference);print('Descriptive tables complete',flush=True)
    comparisons(rows);balanced_illuminance(rows);print('Paired, component and lux sensitivity calculations complete',flush=True)
    candidates(rows);print('Candidate extraction and scoring complete',flush=True)
    ordered_bootstrap(rows);print('Ordered-frame dependence sensitivity complete',flush=True)
    qa=dict(status='PASS',original_outputs=len(rows),unique_frame_pairs=len(reference),reference_codes=3*len(reference),constant_load=1,
        expected_primary_rows=2896,prior_field_checks=9050,missing_frame_ids=0,duplicate_ids=0,provider_identifiers_verified=False,
        scope='One worker, two recording sets. Not a randomized clothing or illuminance experiment.',python=platform.python_version())
    assert len(rows)==2896 and len(reference)==181
    (OUT/'analysis_QA.json').write_text(json.dumps(qa,indent=2)+'\n')

if __name__=='__main__':main()
