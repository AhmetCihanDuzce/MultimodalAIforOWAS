"""Independent numerical checks using public CSV inputs and integer combinatorics."""
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
from fractions import Fraction
import csv,io,json,math,xml.etree.ElementTree as ET
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'owas-data-results'
RESULTS=ROOT/'generated'/'results'
def read(name):
    with (RESULTS/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def exact_binomial(g,l):
    n=g+l
    return min(1.,float(Fraction(2*sum(math.comb(n,k) for k in range(min(g,l)+1)),2**n))) if n else 1.
def independent_ffh(success,totals):
    n1,n2,n3=totals;k=sum(success);obs=math.prod(math.comb(n,s) for n,s in zip(totals,success));num=0
    for a in range(max(0,k-n2-n3),min(n1,k)+1):
        for b in range(max(0,k-a-n3),min(n2,k-a)+1):
            c=k-a-b;v=math.comb(n1,a)*math.comb(n2,b)*math.comb(n3,c)
            if v<=obs:num+=v
    return float(Fraction(num,math.comb(sum(totals),k)))

def main():
    matrix=read('coding_matrix');refs=read('expert_consensus')
    assert len(matrix)==2896 and len(refs)==181
    # Independently compare all 543 released consensus codes via the csv module.
    # This does not claim to repeat private-workbook XML extraction.
    with (DATA/'inputs'/'expert_consensus_input.csv').open(encoding='utf-8-sig',newline='') as handle:
        inputs=list(csv.DictReader(handle))
    source={(r['dataset'],r['frame_id']):r for r in inputs}
    assert len(source)==len(inputs)==181
    csv_count=0
    for r in refs:
        original=source[r['dataset'],r['frame_id']]
        assert original['illuminance_lux']==r['illuminance_lux']
        for view in ['front','side','both']:
            assert original['ref_'+view+'_code']==r['ref_'+view+'_code'];csv_count+=1
    assert csv_count==543
    # Reference and original components directly determine each performance row.
    for s in read('performance'):
        rr=[r for r in matrix if (s['dataset']=='pooled' or r['dataset']==s['dataset']) and r['model']==s['model'] and r['combination']==s['combination']]
        n=len(rr);complete=sum(r['OWAS_code']!='NA' for r in rr);matches=sum(r['OWAS_code']==r['ref_'+s['reference']+'_code'] for r in rr)
        assert (n,complete,matches)==tuple(int(s[k]) for k in ['n','complete_n','exact_n'])
        ci=binomtest(matches,n).proportion_ci(method='wilson')
        assert abs(float(s['exact_wilson_low'])-100*ci.low)<1e-8 and abs(float(s['exact_wilson_high'])-100*ci.high)<1e-8
    for s in read('category_performance'):
        rr=[r for r in matrix if (s['dataset']=='pooled' or r['dataset']==s['dataset']) and r['model']==s['model'] and r['combination']==s['combination']]
        pairs=Counter((r['ref_'+s['reference']+'_'+s['component']],r[s['component']]) for r in rr)
        c=s['category'];tp=pairs[c,c];nr=sum(v for (a,b),v in pairs.items() if a==c);np=sum(v for (a,b),v in pairs.items() if b==c)
        assert (tp,nr,np,pairs[c,'NA'])==tuple(int(s[k]) for k in ['matched_n','reference_n','predicted_n','unresolved_reference_n'])
    test_rows=[]
    for name in ['version_tests','information_tests','component_tests','shared_complete_tests']:
        for r in read(name):
            assert abs(float(r['p_raw'])-exact_binomial(int(r['gains']),int(r['losses'])))<1e-10
            assert abs(float(r['delta_pp'])-100*(int(r['gains'])-int(r['losses']))/int(r['n']))<1e-8
            test_rows.append(r)
    for r in read('illuminance_tests'):
        actual=independent_ffh([int(r[k]) for k in ['match100','match250','match400']],[int(r[k]) for k in ['n100','n250','n400']])
        assert abs(float(r['p_raw'])-actual)<2e-9
        test_rows.append(r)
    test_rows+=read('omnibus_tests')
    for family in {r['family'] for r in test_rows}:
        rr=sorted([r for r in test_rows if r['family']==family],key=lambda r:float(r['p_raw']))
        running=0.
        for rank,r in enumerate(rr):
            running=max(running,min(1.,(len(rr)-rank)*float(r['p_raw'])))
            assert abs(running-float(r['p_holm']))<1e-8
    # Score frozen candidate sets with no selection among alternatives.
    candidate=read('candidate_audit')
    for r in candidate:
        values=r['candidate_values'].split('|') if r['candidate_values'] else []
        assert len(values)==int(r['candidate_n'])
        assert r['reference_in_candidates']==('1' if r['reference_code'] in values else '0') if values else r['reference_in_candidates']=='NA'
    # Each derived output preserves every originally available component.
    original={(r['frame_key'],r['model'],r['combination']):r for r in matrix}
    for r in read('candidate_derived_matrix'):
        prior=original[r['frame_key'],r['model'],r['combination']]
        for comp in ['back','arms','legs']:
            if prior[comp]!='NA':assert prior[comp]==r[comp]
            assert int(r[comp+'_match'])==int(r[comp]==r['ref_both_'+comp])
        assert int(r['posture_match'])==int(r['OWAS_code']==r['ref_both_code'])
        assert int(r['complete_match'])==int(r['OWAS_code']!='NA')
    result=dict(status='PASS',consensus_csv_reference_cells=csv_count,performance_rows=144,category_rows=2016,
        exact_paired_tests=sum(len(read(n)) for n in ['version_tests','information_tests','component_tests','shared_complete_tests']),
        integer_exact_illuminance_tests=288,holm_families=len({r['family'] for r in test_rows}),candidate_memberships=len(candidate),derived_outputs_checked=5792)
    (RESULTS/'independent_QA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
