"""Run frozen modular OWAS systems; never fit models or use reference labels for inference.

Image predictions are persisted before an optional expert-reference file is read.
The smoke-test option is explicitly incompatible with reference evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from modular_followup.feature_interface import assemble, predict_fitted
from modular_followup.models import CLASSES
from leg_followup.leg_features import extract_geometry

OUT = ROOT / 'modular_followup'
KEYS = ['worker_id', 'recording_id', 'frame_id']
SAMPLE_FIELDS = KEYS + ['timestamp_seconds', 'side_image_path', 'front_image_path']
REFERENCE_BASIS = ['paired_images_only', 'paired_images_plus_video', 'uncertain']
OWAS_PATTERN = re.compile(r'[1-4][1-3][1-7][1-3]')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_samples(path):
    samples = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(SAMPLE_FIELDS) - set(samples)
    if missing:
        raise ValueError('Missing required sample fields: ' + ', '.join(sorted(missing)))
    if samples.empty:
        raise ValueError('Samples must contain at least one observation; the template is header-only')
    for key in KEYS:
        samples[key] = samples[key].str.strip()
    if samples[KEYS].eq('').any().any() or samples[KEYS].duplicated().any():
        raise ValueError('Samples must have unique, nonempty worker/recording/frame keys')
    for view in ['side', 'front']:
        if samples[view + '_image_path'].str.strip().eq('').any():
            raise ValueError('Both synchronized image paths are required for every observation')
    times = pd.to_numeric(samples.timestamp_seconds, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('timestamp_seconds must contain finite nonnegative video elapsed seconds')
    return samples


def load_frozen():
    release_path = OUT / 'frozen_workflow/external_release.json'
    if not release_path.is_file():
        raise ValueError('The complete external release has not yet been sealed')
    release = json.loads(release_path.read_text())
    if release.get('version') != 'modular_v1' or release.get('new_worker_data_seen') is not False or release.get('external_validation') != 'not_run':
        raise ValueError('External release must record a modular_v1 freeze before new-worker data were seen')
    required_release_paths = {
        'modular_followup/predict_external.py', 'modular_followup/protocol.json',
        'modular_followup/frozen_workflow/workflow.json',
        'modular_followup/EXTERNAL_PROTOCOL_EN.txt',
        'modular_followup/external_templates/samples_TEMPLATE.csv',
        'modular_followup/external_templates/expert_reference_TEMPLATE.csv'}
    if not isinstance(release.get('sha256'), dict) or not required_release_paths <= set(release['sha256']):
        raise ValueError('External release seal omits required runner, workflow or protocol files')
    for relpath, digest in release['sha256'].items():
        relative = Path(relpath)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('External release contains a non-project-relative path')
        if sha(ROOT / relative) != digest:
            raise ValueError('External release hash mismatch: ' + relpath)
    config_path = OUT / 'frozen_workflow/workflow.json'
    config = json.loads(config_path.read_text())
    model_path = OUT / 'frozen_workflow' / config['models_file']
    if sha(model_path) != config['models_sha256']:
        raise ValueError('Frozen model-file hash mismatch')
    if sha(OUT / 'protocol_freeze.json') != config['development_freeze_sha256']:
        raise ValueError('Development-freeze hash mismatch')
    for relpath, digest in config['source_code_sha256'].items():
        if sha(ROOT / relpath) != digest:
            raise ValueError('Frozen source-code hash mismatch: ' + relpath)
    if config['primary_system'] != 'modular_flat' or config['classes'] != CLASSES or config['load'] != 1:
        raise ValueError('Frozen primary system or component scope differs from modular_v1')
    for info in config['model_assets']:
        if sha(ROOT / 'models' / info['filename']) != info['sha256']:
            raise ValueError('Frozen feature-extractor asset hash mismatch: ' + info['filename'])
    models = joblib.load(model_path)
    for system, mapping in config['systems'].items():
        if set(mapping) != set(CLASSES) or any(key not in models for key in mapping.values()):
            raise ValueError('Frozen system mapping is incomplete: ' + system)
    return config, models


def extract_new_features(samples, samples_path, allow_training_smoke_test):
    """Use exact frozen image assets, IMAGE mode, and original feature functions."""
    from PIL import Image
    import mediapipe as mp
    from extract_features import pose_features
    original = pd.read_csv(ROOT / 'audit/image_manifest.csv', dtype=str, keep_default_na=False)
    training_hashes = set(original.image_sha256)
    training_pixel_hashes = set(original.pixel_sha256)
    arrays = {prefix + '_' + view: [] for view in ['front', 'side'] for prefix in
              ['pose', 'embedding', 'pose_detected', 'geometry', 'available']}
    image_audit = []
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / 'models/pose_landmarker_full.task')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5)
    embed_options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / 'models/mobilenet_v3_small.tflite')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, l2_normalize=True, quantize=False)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose, \
         mp.tasks.vision.ImageEmbedder.create_from_options(embed_options) as embedder:
        for row in samples.itertuples(index=False):
            for view in ['front', 'side']:
                path = Path(getattr(row, view + '_image_path'))
                if not path.is_absolute():
                    path = samples_path.parent / path
                digest = sha(path)
                if digest in training_hashes and not allow_training_smoke_test:
                    raise ValueError('An external photograph is identical to a training-source file')
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
                    pixel_digest = hashlib.sha256(str(image.size).encode() + rgb.tobytes()).hexdigest()
                if pixel_digest in training_pixel_hashes and not allow_training_smoke_test:
                    raise ValueError('External image pixels match a training image, including possible re-encoding')
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose.detect(mp_image)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
                generic, detected = pose_features(landmarks, rgb.shape[1], rgb.shape[0])
                raw = [[p.x, p.y, p.z, p.visibility, p.presence] for p in landmarks]
                geometry, available, _, _ = extract_geometry(raw, rgb.shape[1], rgb.shape[0])
                embedding = np.asarray(embedder.embed(mp_image).embeddings[0].embedding, dtype=np.float32)
                if embedding.shape != (1024,) or not np.isfinite(embedding).all():
                    raise ValueError('Unexpected frozen embedding output')
                for prefix, value in [('pose', generic), ('embedding', embedding), ('pose_detected', detected),
                                      ('geometry', geometry), ('available', available)]:
                    arrays[prefix + '_' + view].append(value)
                image_audit.append({key: getattr(row, key) for key in KEYS} | {
                    'view': view, 'image_sha256': digest, 'pixel_sha256': pixel_digest,
                    'width': rgb.shape[1], 'height': rgb.shape[0], 'pose_detected': bool(detected)})
    return {key: np.asarray(value) for key, value in arrays.items()}, pd.DataFrame(image_audit)


def infer_systems(samples, arrays, config, models):
    component_cache = {}
    for key, info in models.items():
        x, available = assemble(arrays, info['candidate'])
        component_cache[key] = predict_fitted(info['model'], x, available, info['candidate']['hierarchy'])
    records = []
    for name, mapping in config['systems'].items():
        q = samples[KEYS + ['timestamp_seconds']].copy()
        q['system'] = name
        q['primary_external_system'] = name == config['primary_system']
        for component, key in mapping.items():
            q['pred_' + component] = component_cache[key][0]
            q['selected_' + component] = models[key]['candidate']['id']
            if component == 'legs':
                q['pred_leg_family'] = component_cache[key][1]
        q['pred_OWAS_code'] = [''.join(map(str, map(int, values))) + '1' if all(int(v) > 0 for v in values) else 'NA'
                               for values in q[['pred_back', 'pred_arms', 'pred_legs']].itertuples(index=False, name=None)]
        records.append(q)
    return pd.concat(records, ignore_index=True)


def read_references(path, samples):
    refs = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = KEYS + ['consensus_both_code', 'reference_basis']
    if not set(required) <= set(refs):
        raise ValueError('Reference file lacks required fields: ' + ', '.join(sorted(set(required) - set(refs))))
    for key in KEYS:
        refs[key] = refs[key].str.strip()
    if refs[KEYS].eq('').any().any() or refs[KEYS].duplicated().any():
        raise ValueError('Reference keys must be unique and nonempty')
    if not refs.reference_basis.isin(REFERENCE_BASIS).all():
        raise ValueError('reference_basis must be paired_images_only, paired_images_plus_video, or uncertain')
    for field in ['expert1_both_code', 'expert2_both_code', 'consensus_both_code']:
        if field not in refs:
            continue
        refs[field] = refs[field].str.strip()
        invalid = refs[field].ne('') & ~refs[field].map(lambda value: OWAS_PATTERN.fullmatch(value) is not None)
        if invalid.any():
            raise ValueError('Nonempty ' + field + ' must be a valid four-digit OWAS code')
    sample_keys = set(map(tuple, samples[KEYS].to_numpy()))
    if any(tuple(row) not in sample_keys for row in refs[KEYS].itertuples(index=False, name=None)):
        raise ValueError('Reference file contains observation keys absent from the sample file')
    # Do not let additional spreadsheet columns overwrite system predictions
    # or introduce merge suffixes for fields that belong to the inference file.
    allowed = required + [field for field in ['expert1_both_code', 'expert2_both_code', 'uncertainty_note'] if field in refs]
    refs = refs[allowed].copy()
    refs['reference_row_present'] = True
    refs['valid_consensus'] = refs.consensus_both_code.map(lambda value: OWAS_PATTERN.fullmatch(value) is not None)
    for position, component in enumerate(['back', 'arms', 'legs', 'load']):
        refs['ref_' + component] = [int(code[position]) if valid else 0 for code, valid in zip(refs.consensus_both_code, refs.valid_consensus)]
    return refs


def evaluate(predictions, refs, output_dir):
    q = predictions.merge(refs, on=KEYS, how='left', validate='many_to_one')
    q['reference_row_present'] = q.reference_row_present.eq(True)
    q['valid_consensus'] = q.valid_consensus.eq(True)
    for component in ['back', 'arms', 'legs', 'load']:
        q['ref_' + component] = q['ref_' + component].fillna(0).astype(int)
    q['primary_reference_basis'] = q.reference_basis.eq('paired_images_only') & q.valid_consensus
    for component, classes in CLASSES.items():
        q['eligible_' + component] = q.primary_reference_basis & q['ref_' + component].isin(classes)
    q['eligible_full_code'] = q[['eligible_' + component for component in CLASSES]].all(axis=1) & q.ref_load.eq(1)
    actions = pd.read_csv(ROOT / 'inputs/owas_action_reference.csv', dtype={'OWAS_code': str}).set_index('OWAS_code').action.to_dict()
    q['pred_action'] = q.pred_OWAS_code.map(actions).fillna(0).astype(int)
    q['ref_action'] = q.consensus_both_code.map(actions).fillna(0).astype(int)
    q.to_csv(output_dir / 'evaluation_rows.csv', index=False)
    full_rows, component_rows, class_rows = [], [], []
    for system, system_rows in q.groupby('system', sort=False):
        groups = [('all', 'all', system_rows)]
        groups += [('worker', worker, worker_rows) for worker, worker_rows in system_rows.groupby('worker_id', sort=False)]
        groups += [('worker_recording', str(worker) + '/' + str(recording), recording_rows)
                   for (worker, recording), recording_rows in system_rows.groupby(['worker_id', 'recording_id'], sort=False)]
        for subset_type, subset_id, all_rows in groups:
            meta = {'system': system, 'primary_external_system': system == 'modular_flat', 'subset_type': subset_type, 'subset_id': subset_id}
            r = all_rows[all_rows.eligible_full_code]
            complete = all_rows.pred_OWAS_code.ne('NA')
            high = r.ref_action >= 3
            low = (r.ref_action > 0) & (r.ref_action < 3)
            full_rows.append(meta | {
                'all_image_n': len(all_rows), 'reference_rows_n': int(all_rows.reference_row_present.sum()),
                'missing_reference_rows_n': int((~all_rows.reference_row_present).sum()),
                'blank_or_missing_consensus_n': int((~all_rows.valid_consensus).sum()),
                'nonprimary_reference_basis_n': int((all_rows.valid_consensus & ~all_rows.primary_reference_basis).sum()),
                'primary_reference_outside_full_scope_n': int((all_rows.primary_reference_basis & ~all_rows.eligible_full_code).sum()),
                'eligible_full_code_n': len(r), 'full_code_correct': int(r.pred_OWAS_code.eq(r.consensus_both_code).sum()),
                'full_code_accuracy': float(r.pred_OWAS_code.eq(r.consensus_both_code).mean()) if len(r) else None,
                'full_code_unresolved_eligible': int(r.pred_OWAS_code.eq('NA').sum()),
                'complete_predictions_all_images': int(complete.sum()), 'complete_prediction_rate_all_images': float(complete.mean()),
                'action_correct': int(r.pred_action.eq(r.ref_action).sum()),
                'action_accuracy': float(r.pred_action.eq(r.ref_action).mean()) if len(r) else None,
                'high_risk_reference_n': int(high.sum()),
                'high_risk_predicted_low': int((high & r.pred_action.between(1, 2)).sum()),
                'high_risk_unresolved': int((high & r.pred_action.eq(0)).sum()),
                'lower_risk_reference_n': int(low.sum()), 'lower_risk_predicted_high': int((low & (r.pred_action >= 3)).sum())})
            for component, classes in CLASSES.items():
                r = all_rows[all_rows['eligible_' + component]]
                target = r['ref_' + component]
                pred = r['pred_' + component]
                component_rows.append(meta | {'component': component, 'all_image_n': len(all_rows),
                    'eligible_component_n': len(r), 'unscored_component_n': len(all_rows) - len(r),
                    'correct': int(pred.eq(target).sum()), 'accuracy': float(pred.eq(target).mean()) if len(r) else None,
                    'macro_f1_fixed_classes': float(f1_score(target, pred, labels=classes, average='macro', zero_division=0)) if len(r) else None,
                    'observed_reference_classes': ','.join(map(str, sorted(set(target)))),
                    'unresolved_eligible': int(pred.eq(0).sum()),
                    'outputs_all_images': int(all_rows['pred_' + component].gt(0).sum()),
                    'output_rate_all_images': float(all_rows['pred_' + component].gt(0).mean())})
                for code in classes:
                    mask = target.eq(code)
                    n = int(mask.sum())
                    correct = int((mask & pred.eq(code)).sum())
                    class_rows.append(meta | {'component': component, 'class': code, 'support': n, 'correct': correct,
                        'recall': correct / n if n else None, 'unresolved': int((mask & pred.eq(0)).sum())})
    pd.DataFrame(full_rows).to_csv(output_dir / 'full_code_metrics.csv', index=False)
    pd.DataFrame(component_rows).to_csv(output_dir / 'component_metrics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / 'class_metrics.csv', index=False)
    return {'reference_rows_n': len(refs), 'primary_endpoint_scored': True,
            'scored_observations_primary_system': int(q[q.system.eq('modular_flat')].eligible_full_code.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-training-smoke-test', action='store_true')
    args = parser.parse_args()
    if args.allow_training_smoke_test and args.references:
        raise ValueError('Training-image smoke tests cannot include reference evaluation')
    samples = read_samples(args.samples)
    config, models = load_frozen()
    if not args.allow_training_smoke_test and set(samples.worker_id) & set(config['training_worker_ids']):
        raise ValueError('An existing training worker cannot be called external validation')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new or empty output directory to preserve prediction records')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        arrays, image_audit = extract_new_features(samples, args.samples, args.allow_training_smoke_test)
        predictions = infer_systems(samples, arrays, config, models)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    image_audit.to_csv(args.output_dir / 'image_audit.csv', index=False)
    np.savez_compressed(args.output_dir / 'extracted_features.npz', **arrays)
    status = {'status': 'training_image_smoke_test_only' if args.allow_training_smoke_test else 'frozen_new_data_predictions_generated',
        'sample_n': len(samples), 'worker_ids': sorted(set(samples.worker_id)), 'refitted': False,
        'expert_outcomes_simulated': False, 'external_validation_completed': False,
        'worker_identity_basis': 'author metadata; no automatic identity recognition',
        'feature_execution': 'both views eagerly computed for all comparators; not an adaptive acquisition experiment',
        'reference_file_read_only_after_predictions_persisted': True,
        'frozen_workflow_sha256': sha(OUT / 'frozen_workflow/workflow.json'),
        'external_release_sha256': sha(OUT / 'frozen_workflow/external_release.json'),
        'predictions_sha256': sha(args.output_dir / 'predictions.csv'),
        'runner_sha256': sha(__file__),
        'versions': {name: importlib.metadata.version(name) for name in ['mediapipe', 'numpy', 'pandas', 'scikit-learn', 'Pillow']}}
    # Save prediction-only state even if a later optional reference file is invalid.
    (args.output_dir / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
    if args.references:
        refs = read_references(args.references, samples)
        status.update(evaluate(predictions, refs, args.output_dir))
        status['reference_file_sha256'] = sha(args.references)
        status['external_reference_file_scored'] = True
        status['external_validation_completed'] = status['scored_observations_primary_system'] > 0
        (args.output_dir / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
