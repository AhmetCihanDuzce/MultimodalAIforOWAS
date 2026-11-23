"""Frozen frame-independent MediaPipe features; no OWAS model fitting."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
JOINTS = [(11,13,15),(12,14,16),(13,11,23),(14,12,24),
          (11,23,25),(12,24,26),(23,25,27),(24,26,28)]
FEATURE_NAMES = ([f'landmark_{i}_{c}' for i in range(11,33) for c in ['x','y']]
                 + [f'landmark_{i}_confidence' for i in range(11,33)]
                 + [f'angle_{a}_{b}_{c}' for a,b,c in JOINTS])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_features(landmarks, width: int, height: int):
    if not landmarks:
        return np.array([np.nan]*44 + [0.0]*22 + [np.nan]*8, np.float32), False
    xy = np.array([[p.x*width,p.y*height] for p in landmarks], dtype=float)
    confidence = np.array([min(p.visibility,p.presence) for p in landmarks], dtype=float)
    hip = xy[[23,24]].mean(axis=0)
    shoulder = xy[[11,12]].mean(axis=0)
    scale = np.linalg.norm(shoulder-hip)
    if not np.isfinite(scale) or scale < 1:
        return np.array([np.nan]*44 + [0.0]*22 + [np.nan]*8, np.float32), False
    normalized = (xy-hip)/scale
    normalized[confidence < 0.5] = np.nan
    angles = []
    for a,b,c in JOINTS:
        u,v = normalized[a]-normalized[b],normalized[c]-normalized[b]
        denom = np.linalg.norm(u)*np.linalg.norm(v)
        if np.isfinite(denom) and denom > 1e-10:
            angles.append(np.arccos(np.clip(np.dot(u,v)/denom,-1,1))/np.pi)
        else:
            angles.append(np.nan)
    return np.concatenate([normalized[11:33].reshape(-1),confidence[11:33],angles]).astype(np.float32),True


def main():
    out = ROOT / 'features'
    out.mkdir(exist_ok=True)
    protocol = ROOT/'experiment_protocol.json'
    freeze = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'protocol_sha256':sha(protocol),
              'author_metadata_sha256':sha(ROOT/'author_metadata.json'),
              'expert_input_sha256':sha(ROOT/'inputs/expert_consensus_input.csv'),
              'accuracy_evaluated':False}
    freeze_path=ROOT/'audit/protocol_freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())['protocol_sha256']==freeze['protocol_sha256']
    else:
        freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
    pairs = list(csv.DictReader((ROOT/'audit/paired_manifest.csv').open()))
    metadata = {'feature_names':FEATURE_NAMES,
                'model_assets':json.loads((ROOT/'models/model_assets.json').read_text()),
                'protocol_sha256':freeze['protocol_sha256'],
                'versions':{name:importlib.metadata.version(name) for name in ['mediapipe','numpy','Pillow','scikit-learn']},
                'parameters':{'running_mode':'IMAGE','num_poses':1,'min_pose_detection_confidence':0.5,'min_pose_presence_confidence':0.5,'landmark_confidence_threshold':0.5,'embedding_l2_normalize':True,'embedding_quantize':False}}
    arrays = {key:[] for key in ['pose_front','pose_side','embedding_front','embedding_side','pose_detected_front','pose_detected_side']}
    landmark_path = out/'landmarks.jsonl'
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/pose_landmarker_full.task')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1,
        min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5)
    embedding_options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/mobilenet_v3_small.tflite')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,l2_normalize=True,quantize=False)
    n=0
    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose, \
         mp.tasks.vision.ImageEmbedder.create_from_options(embedding_options) as embedder, \
         landmark_path.open('w') as raw:
        for r in pairs:
            for view in ['front','side']:
                path=ROOT/r[view+'_image_path']
                with Image.open(path) as image:
                    rgb=np.asarray(image.convert('RGB'))
                mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                result=pose.detect(mp_image)
                landmarks=result.pose_landmarks[0] if result.pose_landmarks else []
                features,detected=pose_features(landmarks,rgb.shape[1],rgb.shape[0])
                embedding=np.asarray(embedder.embed(mp_image).embeddings[0].embedding,dtype=np.float32)
                assert np.isfinite(embedding).all() and embedding.ndim==1
                arrays['pose_'+view].append(features)
                arrays['pose_detected_'+view].append(detected)
                arrays['embedding_'+view].append(embedding)
                record={'dataset':r['dataset'],'frame_id':r['frame_id'],'timestamp_seconds':int(r['timestamp_seconds']),
                        'view':view,'pose_detected':bool(detected),'image_sha256':sha(path),
                        'image_landmarks':[[p.x,p.y,p.z,p.visibility,p.presence] for p in landmarks]}
                raw.write(json.dumps(record)+'\n')
                raw.flush()
                n+=1
                if n%50==0 or n==362:
                    print('EXTRACTED',n,'/362',flush=True)
    for key in arrays:
        arrays[key]=np.array(arrays[key])
    arrays.update(dataset=np.array([r['dataset'] for r in pairs]),
                  frame_id=np.array([r['frame_id'] for r in pairs]),
                  timestamp_seconds=np.array([int(r['timestamp_seconds']) for r in pairs]),
                  **{'y_'+c:np.array([int(r['target_'+c]) for r in pairs]) for c in ['back','arms','legs']})
    np.savez_compressed(out/'features.npz',**arrays)
    metadata.update({'observations':len(pairs),'images':n,
                     'pose_feature_dimensions_per_view':int(arrays['pose_front'].shape[1]),
                     'embedding_dimensions_per_view':int(arrays['embedding_front'].shape[1]),
                     'detected_front':int(arrays['pose_detected_front'].sum()),
                     'detected_side':int(arrays['pose_detected_side'].sum()),
                     'detected_any_view':int((arrays['pose_detected_front']|arrays['pose_detected_side']).sum()),
                     'detected_both_views':int((arrays['pose_detected_front']&arrays['pose_detected_side']).sum()),
                     'features_sha256':sha(out/'features.npz')})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['feature_names','model_assets']}),flush=True)


if __name__=='__main__':
    main()
