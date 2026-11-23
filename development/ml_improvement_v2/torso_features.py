"""Deterministic projected torso features from frozen 2D pose landmarks.

This representation adds geometric inductive features, not new image evidence,
measured axial rotation, force estimates, or a reconstructed 3D body.
"""
from __future__ import annotations

import numpy as np

CORE_POINTS = (11, 12, 23, 24)
GEOMETRY_NAMES = [
    'torso_direction_x', 'torso_direction_y',
    'torso_signed_lean_pi', 'torso_absolute_lean_pi',
    'left_trunk_direction_x', 'left_trunk_direction_y',
    'right_trunk_direction_x', 'right_trunk_direction_y',
    'left_trunk_length', 'right_trunk_length',
    'shoulder_axis_direction_x', 'shoulder_axis_direction_y',
    'hip_axis_direction_x', 'hip_axis_direction_y',
    'shoulder_width', 'hip_width', 'shoulder_to_hip_width_ratio',
    'shoulder_hip_axis_dot', 'shoulder_hip_axis_cross',
    'shoulder_right_minus_left_y', 'hip_right_minus_left_y',
    'left_minus_right_trunk_length', 'absolute_trunk_length_asymmetry',
    'left_shoulder_to_right_hip_length', 'right_shoulder_to_left_hip_length',
    'left_minus_right_diagonal_length', 'signed_projected_torso_area',
    'torso_shoulder_axis_cross', 'torso_hip_axis_cross',
]
QUALITY_NAMES = [
    *[f'point_{i}_confidence' for i in CORE_POINTS],
    'visible_torso_fraction', 'mean_torso_confidence', 'min_torso_confidence',
    'all_four_torso_points_visible', 'trusted_torso_scale_available',
]
FEATURE_NAMES = GEOMETRY_NAMES + QUALITY_NAMES
assert len(FEATURE_NAMES) == 38


def extract_torso(landmarks, width=1920, height=1080):
    """Return one float32 vector of 38 projected torso and quality features.

    Input columns are normalized x, normalized y, z, visibility, presence.
    The z column is never used. A point is trusted only when x, y and its
    confidence are finite and min(visibility, presence) >= 0.5. Scaled lengths
    and area require all four trusted torso points and a midpoint torso length
    of at least one pixel. Local segment directions can remain available when
    their own two endpoints are trusted; they do not use an invalid midpoint.

    The function does not decide model availability. The caller must preserve
    the existing generic-pose availability flag independently.
    """
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('Image dimensions must be finite and positive.')
    a = np.asarray([] if landmarks is None else landmarks, dtype=float)
    if a.size == 0:
        return np.concatenate([
            np.full(len(GEOMETRY_NAMES), np.nan),
            np.zeros(len(QUALITY_NAMES)),
        ]).astype(np.float32)
    if a.shape != (33, 5):
        raise ValueError(f'Expected (33, 5) landmarks, received {a.shape}.')
    confidence = np.minimum(a[:, 3], a[:, 4])
    confidence[~np.isfinite(confidence)] = np.nan
    xy = a[:, :2] * np.array([width, height], dtype=float)
    trusted = (np.isfinite(xy).all(axis=1) & np.isfinite(confidence)
               & (confidence >= 0.5))
    xy[~trusted] = np.nan
    all_trusted = bool(trusted[list(CORE_POINTS)].all())
    shoulder = (xy[11] + xy[12]) / 2
    hip = (xy[23] + xy[24]) / 2
    torso = shoulder - hip
    raw_scale = np.linalg.norm(torso)
    scale_ok = bool(all_trusted and np.isfinite(raw_scale) and raw_scale >= 1)
    scale = raw_scale if scale_ok else np.nan

    def vector(i, j):
        return xy[j] - xy[i]

    def unit(v):
        length = np.linalg.norm(v)
        return v / length if np.isfinite(length) and length >= 1 else np.full(2, np.nan)

    def length(i, j):
        return np.linalg.norm(vector(i, j)) / scale

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    torso_unit = unit(torso) if scale_ok else np.full(2, np.nan)
    left_unit = unit(vector(23, 11))
    right_unit = unit(vector(24, 12))
    shoulder_unit = unit(vector(11, 12))
    hip_unit = unit(vector(23, 24))
    lean = np.arctan2(torso[0], -torso[1]) / np.pi if scale_ok else np.nan
    left_length, right_length = length(23, 11), length(24, 12)
    shoulder_width, hip_width = length(11, 12), length(23, 24)
    ratio = (shoulder_width / hip_width
             if scale_ok and np.linalg.norm(vector(23, 24)) >= 1 else np.nan)
    left_diagonal, right_diagonal = length(11, 24), length(12, 23)
    signed_area = np.nan
    if scale_ok:
        polygon = (xy[[11, 12, 24, 23]] - hip) / scale
        signed_area = 0.5 * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    geometry = [
        *torso_unit, lean, abs(lean), *left_unit, *right_unit,
        left_length, right_length, *shoulder_unit, *hip_unit,
        shoulder_width, hip_width, ratio,
        np.dot(shoulder_unit, hip_unit), cross(shoulder_unit, hip_unit),
        vector(11, 12)[1] / scale, vector(23, 24)[1] / scale,
        left_length - right_length, abs(left_length - right_length),
        left_diagonal, right_diagonal, left_diagonal - right_diagonal,
        signed_area, cross(torso_unit, shoulder_unit), cross(torso_unit, hip_unit),
    ]
    core_confidence = confidence[list(CORE_POINTS)]
    quality = [
        *core_confidence, float(trusted[list(CORE_POINTS)].mean()),
        float(core_confidence.mean()), float(core_confidence.min()),
        float(all_trusted), float(scale_ok),
    ]
    features = np.asarray(geometry + quality, dtype=np.float32)
    if features.shape != (len(FEATURE_NAMES),) or np.isinf(features).any():
        raise FloatingPointError('Invalid torso-feature shape or infinite feature.')
    return features


def run_geometry_checks():
    """Check geometric properties and masking without any reference labels."""
    # Four visibly asymmetric points: signed orientation should be retained.
    a = np.zeros((33, 5), dtype=float)
    a[:, 3:] = 0.1
    a[11, :2] = [.34, .23]
    a[12, :2] = [.53, .27]
    a[23, :2] = [.41, .57]
    a[24, :2] = [.54, .58]
    a[list(CORE_POINTS), 3:] = .95
    original = extract_torso(a)
    translated = a.copy()
    translated[:, :2] += [.17, -.11]
    np.testing.assert_allclose(original, extract_torso(translated), atol=1e-6, rtol=1e-6, equal_nan=True)
    scaled = a.copy()
    scaled[:, :2] *= 1.7
    np.testing.assert_allclose(original, extract_torso(scaled), atol=1e-6, rtol=1e-6, equal_nan=True)
    # Changing only z cannot affect this explicitly two-dimensional feature set.
    changed_z = a.copy()
    changed_z[:, 2] = np.arange(33) * 100
    np.testing.assert_array_equal(original, extract_torso(changed_z))
    low = a.copy()
    low[11, 3] = .2
    low_shifted = low.copy()
    low_shifted[11, :2] = [1000, -5000]
    np.testing.assert_allclose(extract_torso(low), extract_torso(low_shifted),
                               atol=0, rtol=0, equal_nan=True)
    empty = extract_torso([])
    assert np.isnan(empty[:len(GEOMETRY_NAMES)]).all()
    assert np.array_equal(empty[len(GEOMETRY_NAMES):], np.zeros(len(QUALITY_NAMES)))
    below = a.copy()
    below[list(CORE_POINTS), 3:] = .49
    below_features = extract_torso(below)
    assert np.isnan(below_features[:len(GEOMETRY_NAMES)]).all()
    reflected = a.copy()
    reflected[:, 0] = 1 - reflected[:, 0]
    mirror = extract_torso(reflected)
    np.testing.assert_allclose(mirror[0], -original[0], atol=1e-6)
    np.testing.assert_allclose(mirror[2], -original[2], atol=1e-6)
    np.testing.assert_allclose(mirror[3], original[3], atol=1e-6)
    np.testing.assert_allclose(mirror[26], -original[26], atol=1e-6)
    assert not np.allclose(mirror, original, equal_nan=True)
    return {
        'status': 'PASS', 'dimensions': len(FEATURE_NAMES),
        'translation_invariance': True, 'positive_uniform_scale_invariance': True,
        'z_ignored': True, 'low_confidence_coordinates_cannot_leak': True,
        'missing_pose_preserves_nan_geometry': True,
        'horizontal_reflection_retains_signed_orientation_information': True,
        'scope': 'synthetic geometric invariance/masking checks, not pose accuracy or model validation',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(run_geometry_checks(), indent=2))
