"""Rebuild the reported comparisons from stored predictions and consensus codes."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproduced"
OUT.mkdir(exist_ok=True)
ID = ["dataset", "frame_id"]

def read(relative):
    return pd.read_csv(ROOT / relative, dtype=str, keep_default_na=False)

def comparison_counts(rows):
    return {
        "correct": int(rows.correct.sum()),
        "action_correct": int(rows.action_correct.sum()),
        "complete": int(rows.pred_action.ne(0).sum()),
        "high_low": int((rows.ref_action.ge(3) & rows.pred_action.isin([1, 2])).sum()),
        "high_NA": int((rows.ref_action.ge(3) & rows.pred_action.eq(0)).sum()),
    }

lookup = read("development/inputs/owas_action_reference.csv").set_index("OWAS_code").action.astype(int)
dev_ref = read("development/audit/paired_manifest.csv").rename(columns={"ref_both_code":"reference_both"})
test_ref = read("test/references.csv")
assert len(dev_ref) == 181 and len(test_ref) == 113
assert not dev_ref.duplicated(ID).any() and not test_ref.duplicated(ID).any()

def scored(predictions, reference):
    rows = predictions.merge(reference[ID + ["reference_both"]], on=ID, validate="many_to_one", how="left")
    assert rows.reference_both.notna().all()
    digits = rows[["pred_back", "pred_arms", "pred_legs"]]
    code = digits.pred_back + digits.pred_arms + digits.pred_legs + "1"
    rows["predicted_code"] = code.where(code.isin(lookup.index), "NA")
    rows["ref_action"] = rows.reference_both.map(lookup)
    assert rows.ref_action.notna().all()
    rows["pred_action"] = rows.predicted_code.map(lookup).fillna(0).astype(int)
    rows["correct"] = rows.predicted_code.eq(rows.reference_both)
    rows["action_correct"] = rows.pred_action.eq(rows.ref_action)
    return rows

dev = scored(read("development/held_out_predictions.csv.gz"), dev_ref)
test = scored(read("test/frozen_predictions.csv"), test_ref)
n_test = len(test_ref)
assert len(test) == 66 * n_test and not test.duplicated(["branch", "model", "method"]+ID).any()
def subset(model, method):
    rows = test[(test.model == model) & (test.method == method)].set_index(ID).sort_index()
    assert rows.index.is_unique and len(rows) == n_test
    return rows

def save(name, records):
    pd.DataFrame(records).to_csv(OUT / (name + ".csv"), index=False)

classes=[]
for name, ref in (("development", dev_ref), ("third_job", test_ref)):
    for pos, (part, max_code) in enumerate([("Back",4), ("Arms",3), ("Legs",7)]):
        for code in range(1, max_code+1):
            classes.append(dict(dataset_group=name,component=part,code=code,n=int(ref.reference_both.str[pos].eq(str(code)).sum())))
save("class_support", classes)

raw=[]
for name, path, ref in (("development","development/inputs/ai_coding_input.csv",dev_ref),
                        ("third_job","test/ai_codes.csv",test_ref)):
    responses=read(path).merge(ref[ID+["reference_both"]],on=ID,validate="many_to_one")
    for (model, condition), rows in responses.groupby(["model","combination"]):
        assert len(rows)==len(ref) and not rows.duplicated(ID).any()
        raw.append(dict(dataset_group=name,model=model,condition=int(condition),n=len(rows),
                        correct=int(rows.OWAS_code.eq(rows.reference_both).sum()),complete=int(rows.OWAS_code.ne("NA").sum())))
save("raw_AI_summary",raw)

declared=read("published_tables/integrated_model_summary.csv")
integrated=[]
for entry in declared.itertuples():
    method, model = entry.method, entry.model
    branch = {"component_majority":"ai_code_fusion_v1","ai_codes_logistic":"ai_ml_hybrid_v1","ai_ML_stacking":"ai_ml_hybrid_v1"}.get(method,"ai_eight_condition_v1")
    row=dict(model=model,method=method,label=entry.label)
    for phase,label in [("blocks_60s","main"),("blocks_30s","sensitivity"),("recording_transfer","transfer")]:
        group=dev[(dev.phase==phase)&(dev.branch==branch)&(dev.method==method)&(dev.model==model)]
        assert len(group)==181 and not group.duplicated(ID).any()
        row.update({label+"_"+k:v for k,v in comparison_counts(group).items()})
    test_method={"archived_both":"raw_condition_3","ai_codes_logistic":"AI3_Logistic"}.get(method,method)
    test_model="IMAGE" if method=="fixed_image_ML" else model
    row.update({"third_"+k:v for k,v in comparison_counts(subset(test_model,test_method)).items()})
    integrated.append(row)
save("integrated_model_summary",integrated)
primary=[]
for phase,rows in dev[(dev.branch=="modular_followup")&(dev.method=="modular_flat")].groupby("phase"):
    assert len(rows)==181
    primary.append(dict(phase=phase,n=len(rows),exact=int(rows.correct.sum())))
primary.append(dict(phase="test",n=n_test,exact=int(subset("IMAGE","modular_flat").correct.sum())))
save("primary_image_model",primary)
assert {r["phase"]:r["exact"] for r in primary}=={"blocks_60s":45,"blocks_30s":41,"recording_transfer":27,"test":32}

changes=[]
for entry in read("published_tables/paired_test_transitions.csv").itertuples():
    before=subset(entry.archive,"raw_condition_3")
    after=subset(entry.archive,entry.method)
    assert before.index.equals(after.index)
    gain=~before.correct & after.correct
    repaired=int((gain & before.pred_action.ne(0)).sum())
    recovered=int((gain & before.pred_action.eq(0)).sum())
    lost=int((before.correct & ~after.correct).sum())
    changes.append(dict(archive=entry.archive,method=entry.method,label=entry.label,baseline=int(before.correct.sum()),
                        final=int(after.correct.sum()),corrected=repaired,completed=recovered,lost=lost,
                        change_pp=100*(repaired+recovered-lost)/n_test))
save("paired_test_transitions",changes)
all_test=[]
for (branch,model,method),rows in test.groupby(["branch","model","method"]):
    counts=comparison_counts(rows)
    high=int(rows.ref_action.ge(3).sum())
    assert high==40
    all_test.append(dict(branch=branch,model=model,method=method,high_n=high,retained=high-counts["high_low"]-counts["high_NA"],
                         low_to_high=int((rows.ref_action.le(2)&rows.pred_action.ge(3)).sum()),**counts))
save("test_outcomes",all_test)

def compare(name,keys):
    actual = pd.read_csv(OUT/(name+".csv")).sort_values(keys).reset_index(drop=True)
    published = pd.read_csv(ROOT/"published_tables"/(name+".csv")).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[published.columns],published,check_dtype=False)

compare("paired_test_transitions", ["archive","method"])
compare("raw_AI_summary", ["dataset_group","model","condition"])
compare("class_support", ["dataset_group","component","code"])
compare("integrated_model_summary", ["method","model"])
metrics=read("test/metrics.csv").set_index(["branch","model","method"])
for row in all_test:
    stored=metrics.loc[(row["branch"],row["model"],row["method"])]
    for local,archived in [("correct","full_correct"),("complete","complete_n"),("action_correct","action_correct"),
                           ("high_low","high_to_low_n"),("high_NA","high_unresolved_n"),("retained","high_retained_n")]:
        assert row[local]==int(stored[archived])
lux_rows=read("development/inputs/ai_coding_input.csv")
lux_rows=lux_rows[lux_rows.combination=="3"].merge(dev_ref[ID+["illuminance_lux","reference_both"]],on=ID,validate="many_to_one")
lux_counts=[]
for (dataset,model,lux),group in lux_rows.groupby(["dataset","model","illuminance_lux"]):
    lux_counts.append(dict(dataset=dataset,model=model,illuminance_lux=int(lux),n=len(group),correct=int(group.OWAS_code.eq(group.reference_both).sum())))
save("development_paired_AI_lux_counts",lux_counts)
compare("development_paired_AI_lux_counts", ["dataset","model","illuminance_lux"])
for method, expected, up in [("AI8_RBF_SVM",20,2),("AI8ML_RBF_SVM",4,2),("raw_condition_3",13,0),("ai_NA_filled_by_ML",21,0)]:
    row=next(r for r in all_test if r["method"]==method and r["model"]=="M61")
    assert (row["high_low"],row["low_to_high"])==(expected,up)
for method,code in [("raw_condition_3","4151"),("ai_ML_stacking","4251")]:
    example=subset("M56",method).loc[("job3","00129")]
    assert (example.predicted_code,example.reference_both,example.pred_action,example.ref_action)==(code,"4251",4,4)
print("Verified 181 development and 113 test observations, all 66 test configurations, Tables 3–5 and supplementary class counts.")
