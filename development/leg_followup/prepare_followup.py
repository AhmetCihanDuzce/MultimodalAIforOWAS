"""Create the leg features and deterministic error-review atlas."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from leg_followup.leg_features import extract_geometry,FEATURE_NAMES,CORE_POINTS

OUT=ROOT/'leg_followup'


def draw_case(row,lookup,path):
 fig,axes=plt.subplots(1,2,figsize=(12,4.7))
 for view,ax in zip(['front','side'],axes):
  im=Image.open(ROOT/getattr(row,view+'_image_path'));ax.imshow(im);ax.set_axis_off()
  raw=lookup[(row.dataset,row.frame_id,view)];a=np.asarray(raw['image_landmarks'])
  if a.shape==(33,5):
   xy=a[:,:2]*im.size;conf=np.minimum(a[:,3],a[:,4])
   for i,j in [(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]:
    color='#00b8ff' if i%2 else '#ff7f00'
    if i in [11,12]:color='#ffffff'
    ax.plot(xy[[i,j],0],xy[[i,j],1],color=color,linewidth=2.0,linestyle='-' if min(conf[i],conf[j])>=0.5 else '--',alpha=0.95)
   for i in range(23,33):
    ax.scatter(*xy[i],s=16,c='#00ff66' if conf[i]>=0.5 else '#ffff00',edgecolors='black',linewidth=0.4)
    if i in [25,26,27,28]:ax.annotate(str(i),xy[i],xytext=(4,-6),textcoords='offset points',color='white',fontsize=8,bbox={'facecolor':'black','alpha':0.65,'pad':0.5})
   _,_,quality,stats=extract_geometry(a,*im.size)
   ax.set_title(f'{view}: core visible {stats["visible_core"]}/6 | evidence gate {quality}',fontsize=10)
  else:
   ax.set_title(view+': NO POSE DETECTED',fontsize=10,color='#a00000')
  ax.set_xlim(0,im.size[0]);ax.set_ylim(im.size[1],0)
 fig.suptitle(f'{row.case_id}  {row.dataset}  second {int(row.timestamp_seconds)}  | consensus leg {row.target_legs}, baseline RF {row.pred_legs}\n{row.selection_reason}',fontsize=12)
 fig.text(.5,.018,'Solid line: endpoints confidence >=0.5; dashed/yellow: low confidence. Model confidence is not verified landmark accuracy.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.04,1,.89]);fig.savefig(path,dpi=130,pil_kwargs={'quality':88});plt.close(fig)


def main():
 (OUT/'audit/cases').mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(ROOT/'audit/paired_manifest.csv',dtype={'frame_id':str})
 lookup={}
 for line in (ROOT/'features/landmarks.jsonl').read_text().splitlines():
  r=json.loads(line);lookup[(r['dataset'],r['frame_id'],r['view'])]=r
 arrays={};quality_rows=[]
 for view in ['front','side']:
  x=[];available=[];quality=[]
  for r in df.itertuples():
   raw=lookup[(r.dataset,r.frame_id,view)]
   f,a,q,s=extract_geometry(raw['image_landmarks']);x.append(f);available.append(a);quality.append(q)
   quality_rows.append({'dataset':r.dataset,'frame_id':r.frame_id,'view':view,'ref_legs':r.target_legs,'pose_detected':a,'evidence_ok':q,**s})
  arrays['geometry_'+view]=np.asarray(x);arrays['available_'+view]=np.asarray(available);arrays['quality_'+view]=np.asarray(quality)
 np.savez_compressed(OUT/'geometry_features.npz',**arrays)
 (OUT/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2)+'\n')
 pd.DataFrame(quality_rows).to_csv(OUT/'audit/pose_quality.csv',index=False)
 pred=pd.read_csv(ROOT/'results/experiment2/predictions.csv',dtype={'frame_id':str})
 pred=pred[(pred.method=='pose_RandomForest')&(pred.view=='paired')]
 df=df.merge(pred[['dataset','frame_id','pred_legs']],on=['dataset','frame_id'],validate='one_to_one')
 df['observation_index']=np.arange(len(df))
 selected=[];used=set()
 def choose(candidates,reason,limit):
  candidates=candidates.sort_values(['dataset','timestamp_seconds'])
  first=[]
  for _,g in candidates.groupby('dataset',sort=True):
   if len(g):first.append(g.index[0])
  order=first+[i for i in candidates.index if i not in first]
  count=0
  for i in order:
   if i in used:continue
   row=df.loc[i].to_dict();row['selection_reason']=reason;selected.append(row);used.add(i);count+=1
   if count>=limit:break
 for a,b in [(2,3),(3,2),(5,4),(7,3),(7,2)]:choose(df[(df.target_legs==a)&(df.pred_legs==b)],f'error {a} -> {b}',2)
 choose(df[df.pred_legs==0],'no pose in either view',2)
 for c in [2,3,4,5,7]:choose(df[(df.target_legs==c)&(df.pred_legs==c)],f'correct comparison class {c}',1)
 cases=pd.DataFrame(selected);cases['case_id']=[f'C{i+1:02d}' for i in range(len(cases))]
 cases.to_csv(OUT/'audit/visual_cases.csv',index=False)
 for row in cases.itertuples():draw_case(row,lookup,OUT/'audit/cases'/f'{row.case_id}.jpg')
 for start in range(0,len(cases),4):
  ids=cases.case_id.tolist()[start:start+4]
  images=[Image.open(OUT/'audit/cases'/f'{c}.jpg') for c in ids]
  canvas=Image.new('RGB',(images[0].width,sum(im.height for im in images)),'white');y=0
  for im in images:canvas.paste(im,(0,y));y+=im.height
  canvas.save(OUT/'audit'/f'contact_{start//4+1:02d}.jpg',quality=90)
 meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'geometry_dimensions_per_view':len(FEATURE_NAMES),
       'case_count':len(cases),'review_selection':'purposive errors plus correct controls; not a sample for estimating landmark accuracy',
       'feature_function_sha256':hashlib.sha256((OUT/'leg_features.py').read_bytes()).hexdigest(),
       'quality_by_view':{v:{'pose_available':int(arrays['available_'+v].sum()),'evidence_ok':int(arrays['quality_'+v].sum())} for v in ['front','side']}}
 (OUT/'feature_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
