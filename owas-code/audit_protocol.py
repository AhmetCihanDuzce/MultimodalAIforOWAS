"""Audit archived metadata against supplied prompts without rewriting observations.

Only deterministic metadata normalizations are proposed. Free-text evidence
and contradictory resolved-code evidence are retained for source review.
No photographs are recoded and no posture component is changed.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'owas-data-results'
OUT=ROOT/'generated'/'audit'
PRIOR={4:1,5:2,6:3,7:2,8:1}
EVIDENCE={'side_visual','front_visual','both_visual','insufficient'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (DATA/'inputs/ai_coding_input.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    by={(r['dataset'],r['model'],int(r['combination']),r['frame_id']):r for r in rows}
    assert len(by)==len(rows)==2896
    flags=[];normalizations=[]
    def record(r,field,issue,normal='',basis=''):
        item={k:r[k] for k in ['dataset','model','combination','frame_id']}
        item.update(field=field,raw_value=r[field],issue=issue,normalized_value=normal,
                    normalization_status='deterministic_metadata_only' if normal else 'source_review_required',basis=basis)
        flags.append(item)
        if normal:normalizations.append(item.copy())
    for r in rows:
        c=int(r['combination'])
        if c in PRIOR:
            base=by[r['dataset'],r['model'],PRIOR[c],r['frame_id']]
            assert all(r['prior_'+k]==base[k] for k in ['back','arms','legs','load','OWAS_code'])
            if r['prior_status']!='matched_valid':
                record(r,'prior_status','prior_availability_used_as_validity','matched_valid','Unique matched baseline; all five supplied code fields equal the frozen source, including valid NA values.')
        for component in ['back','arms','legs']:
            key=component+'_evidence';ev=r[key]
            if ev not in EVIDENCE:
                record(r,key,'free_text_outside_evidence_enum',basis='Free text is retained; the actual supporting view is not inferred from the condition label.')
            if r[component]=='NA' and ev!='insufficient':
                record(r,key,'primary_NA_with_resolved_evidence','insufficient','The prompt defines evidence=insufficient for a primary NA; this does not recode the image.')
            elif r[component]!='NA' and ev=='insufficient':
                record(r,key,'resolved_primary_with_insufficient_evidence',basis='The archived code and evidence label conflict; the true visual support cannot be determined without the source record or images.')
            key=component+'_prior_influence'
            if c in PRIOR and r['prior_'+component]=='NA' and r[key]!='not_applicable':
                record(r,key,'influence_flag_without_substantive_prior','not_applicable','The matched prior component is NA and supplies no substantive category for this component.')
    fields=['dataset','model','combination','frame_id','field','raw_value','issue','normalized_value','normalization_status','basis']
    for name,records in [('protocol_metadata_flags.csv',flags),('protocol_metadata_normalization.csv',normalizations)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records)
    result={'status':'PASS_WITH_ARCHIVED_SCHEMA_DEVIATIONS','rows_checked':len(rows),
            'flagged_fields':len(flags),'flagged_rows':len({(r['dataset'],r['model'],r['combination'],r['frame_id']) for r in flags}),
            'issue_counts':dict(sorted(Counter(r['issue'] for r in flags).items())),
            'deterministic_metadata_normalizations':len(normalizations),'fields_requiring_source_review':len(flags)-len(normalizations),
            'raw_input_rewritten':False,'posture_codes_changed':0,'scientific_metric_effect':'None; metadata audit and proposed normalizations are separate from the original-output analysis.',
            'ai_input_sha256':hashlib.sha256((DATA/'inputs/ai_coding_input.csv').read_bytes()).hexdigest()}
    (OUT/'protocol_metadata_QA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
