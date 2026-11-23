"""Deterministic image-plane leg geometry from unchanged pose landmarks."""
from __future__ import annotations
import numpy as np

LEG_POINTS=list(range(23,33))
CORE_POINTS=[23,24,25,26,27,28]
ANGLE_TRIPLES=[(11,23,25),(12,24,26),(23,25,27),(24,26,28),(25,27,31),(26,28,32)]
SEGMENTS=[(23,25),(24,26),(25,27),(26,28),(29,31),(30,32)]
SCALAR_NAMES=['knee_angle_asymmetry','hip_angle_asymmetry','ankle_angle_asymmetry',
              'ankle_dx','ankle_dy','ankle_distance','knee_dx','knee_dy','knee_distance',
              'mean_ankle_below_hip','hip_ankle_height_asymmetry','mean_thigh_shank_ratio']
QUALITY_NAMES=['visible_core_fraction','mean_core_confidence','min_core_confidence',
               'complete_left_leg','complete_right_leg','leg_scale_available']
FEATURE_NAMES=([f'point_{i}_{c}' for i in LEG_POINTS for c in ['x','y']]
 +[f'point_{i}_confidence' for i in LEG_POINTS]
 +[f'angle_{a}_{b}_{c}' for a,b,c in ANGLE_TRIPLES]
 +[f'unit_{a}_{b}_{c}' for a,b in SEGMENTS for c in ['x','y']]
 +SCALAR_NAMES+QUALITY_NAMES)


def extract_geometry(landmarks,width=1920,height=1080):
 a=np.asarray(landmarks,dtype=float)
 if a.shape!=(33,5):
  empty=np.full(len(FEATURE_NAMES),np.nan,dtype=np.float32)
  empty[20:30]=0;empty[-6:]=0
  return empty,False,False,{'visible_core':0,'mean_core_confidence':0.0}
 confidence=np.minimum(a[:,3],a[:,4]);valid=confidence>=0.5
 xy=a[:,:2]*[width,height]
 hip=xy[[23,24]].mean(axis=0);shoulder=xy[[11,12]].mean(axis=0)
 torso=np.linalg.norm(shoulder-hip)
 if not np.isfinite(torso) or torso<1:
  return extract_geometry([],width,height)
 lengths=[]
 for i,j,k in [(23,25,27),(24,26,28)]:
  if valid[[i,j,k]].all():lengths.append(np.linalg.norm(xy[i]-xy[j])+np.linalg.norm(xy[j]-xy[k]))
 scale=np.mean(lengths) if lengths and np.mean(lengths)>1 else torso
 normalized=(xy-hip)/scale;normalized[~valid]=np.nan
 def angle(i,j,k):
  u,v=normalized[i]-normalized[j],normalized[k]-normalized[j]
  den=np.linalg.norm(u)*np.linalg.norm(v)
  return np.arccos(np.clip(np.dot(u,v)/den,-1,1))/np.pi if np.isfinite(den) and den>1e-9 else np.nan
 angles=[angle(*t) for t in ANGLE_TRIPLES]
 directions=[]
 for i,j in SEGMENTS:
  u=normalized[j]-normalized[i];d=np.linalg.norm(u)
  directions.extend((u/d).tolist() if np.isfinite(d) and d>1e-9 else [np.nan,np.nan])
 ad=normalized[28]-normalized[27];kd=normalized[26]-normalized[25]
 ratios=[]
 for i,j,k in [(23,25,27),(24,26,28)]:
  thigh=np.linalg.norm(normalized[j]-normalized[i]);shank=np.linalg.norm(normalized[k]-normalized[j])
  if np.isfinite(thigh+shank) and shank>1e-9:ratios.append(thigh/shank)
 ankle_y=np.mean(normalized[[27,28],1])
 left_height=normalized[27,1]-normalized[23,1];right_height=normalized[28,1]-normalized[24,1]
 scalars=[abs(angles[2]-angles[3]),abs(angles[0]-angles[1]),abs(angles[4]-angles[5]),
          ad[0],ad[1],np.linalg.norm(ad),kd[0],kd[1],np.linalg.norm(kd),
          ankle_y,abs(left_height-right_height),np.mean(ratios) if ratios else np.nan]
 core=confidence[CORE_POINTS]
 complete_left=bool(valid[[23,25,27]].all());complete_right=bool(valid[[24,26,28]].all())
 quality=[valid[CORE_POINTS].mean(),core.mean(),core.min(),complete_left,complete_right,bool(lengths)]
 features=np.concatenate([normalized[LEG_POINTS].reshape(-1),confidence[LEG_POINTS],angles,directions,scalars,quality]).astype(np.float32)
 assert len(features)==len(FEATURE_NAMES) and not np.isinf(features).any()
 evidence_ok=bool(valid[CORE_POINTS].sum()>=4 and (complete_left or complete_right))
 return features,True,evidence_ok,{'visible_core':int(valid[CORE_POINTS].sum()),'mean_core_confidence':float(core.mean()),'complete_left':complete_left,'complete_right':complete_right}
