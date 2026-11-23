"""Prepare torso features using only immutable cached landmarks and identities."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from torso_features import extract_torso, FEATURE_NAMES, run_geometry_checks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = [ROOT/'features/landmarks.jsonl', ROOT/'features/features.npz',
              ROOT/'audit/paired_manifest.csv', ROOT/'audit/image_manifest.csv']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    with (ROOT/'audit/paired_manifest.csv').open(newline='') as stream:
        pairs = list(csv.DictReader(stream))
    with (ROOT/'audit/image_manifest.csv').open(newline='') as stream:
        images = list(csv.DictReader(stream))
    dimensions = {}
    for image in images:
        key = (image['dataset'], image['frame_id'], image['view'])
        if key in dimensions:
            raise ValueError(f'Duplicate image identity: {key}')
        dimensions[key] = (int(image['width']), int(image['height']))
    lookup = {}
    for line in (ROOT/'features/landmarks.jsonl').read_text().splitlines():
        record = json.loads(line)
        key = (record['dataset'], record['frame_id'], record['view'])
        if key in lookup:
            raise ValueError(f'Duplicate cached landmark identity: {key}')
        lookup[key] = record
    expected_keys = {(r['dataset'], r['frame_id'], v) for r in pairs for v in ['front', 'side']}
    assert set(lookup) == set(dimensions) == expected_keys
    with np.load(ROOT/'features/features.npz') as old:
        for column in ['dataset', 'frame_id']:
            np.testing.assert_array_equal(old[column], [r[column] for r in pairs])
        np.testing.assert_array_equal(old['timestamp_seconds'],
                                      [int(r['timestamp_seconds']) for r in pairs])
        arrays = {column: old[column].copy()
                  for column in ['dataset', 'frame_id', 'timestamp_seconds']}
        for view in ['front', 'side']:
            arrays['torso_'+view] = np.stack([
                extract_torso(lookup[(r['dataset'], r['frame_id'], view)]['image_landmarks'],
                              *dimensions[(r['dataset'], r['frame_id'], view)])
                for r in pairs])
            arrays['available_'+view] = old['pose_detected_'+view].copy()
            recorded = [lookup[(r['dataset'], r['frame_id'], view)]['pose_detected'] for r in pairs]
            np.testing.assert_array_equal(arrays['available_'+view], recorded)
            assert arrays['torso_'+view].shape == (len(pairs), len(FEATURE_NAMES))
            assert not np.isinf(arrays['torso_'+view]).any()
        arrays['feature_names'] = np.asarray(FEATURE_NAMES)
    feature_path = HERE/'torso_features.npz'
    np.savez_compressed(feature_path, **arrays)
    (HERE/'torso_feature_names.json').write_text(json.dumps(FEATURE_NAMES, indent=2)+'\n')
    checks = run_geometry_checks()
    checks.update({'identities_aligned_n': len(pairs), 'cached_landmark_records_n': len(lookup),
                   'availability_exactly_matches_generic_pose': True,
                   'source_hashes_unchanged': all(sha(ROOT/p) == h for p, h in source_hashes.items())})
    (HERE/'torso_feature_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'observations': len(pairs), 'features_per_view': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'source_sha256': source_hashes,
        'feature_code_sha256': sha(HERE/'torso_features.py'),
        'preparation_code_sha256': sha(HERE/'prepare_torso.py'),
        'output_sha256': sha(feature_path),
        'availability_definition': 'Unchanged generic-pose flag copied exactly; trusted torso scale is only a feature.',
        'geometry_scope': 'Image-plane features only; no z, triangulation, measured twisting, force, new images, label use or estimator fitting.',
        'view_summary': {v: {
            'generic_pose_available': int(arrays['available_'+v].sum()),
            'trusted_torso_scale_available': int(arrays['torso_'+v][:, -1].sum()),
            'finite_geometry_values': int(np.isfinite(arrays['torso_'+v][:, :29]).sum()),
            'geometry_values': int(arrays['torso_'+v][:, :29].size),
        } for v in ['front', 'side']},
        'checks_status': checks['status'],
    }
    (HERE/'torso_feature_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in ['observations', 'features_per_view', 'view_summary', 'checks_status']}, indent=2))


if __name__ == '__main__':
    main()
