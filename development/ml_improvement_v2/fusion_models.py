"""Train-only fusion of frozen image embeddings and cached 2D pose features.

No image decoder, pose SDK, network call, or pretrained-weight update is used.
The input matrix is [all selected-view embeddings | all selected-view poses].
The embedding block is standardized, reduced with deterministic full-SVD PCA,
then standardized again.  The pose block is median-imputed with missingness
indicators and standardized.  Each block is divided by sqrt(its final width),
so its average squared training norm is at most approximately one.  Constant
columns have zero variance and contribute zero to that norm.

Every fitted transformation is learned exclusively in fit(X_train, y_train).
The frozen embedding extractor itself is NOT refitted here.  Pose coordinates
remain 2D image-plane features, not measured 3D posture or physical loading.

Set use_pose=False for the matched PCA-embedding-only ablation.  It accepts
the same raw matrix and intentionally ignores every column after embedding_dim.
LinearSVC decision_function values are margins, not calibrated probabilities.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


SEED = 20261029


def assemble_fusion_inputs(arrays, view, include_pose_flags=True):
    """Return (X, available, embedding_dim) for side, front, or paired views.

    Paired embedding columns are front then side; paired pose columns are
    front then side. Each pose block includes its binary pipeline-detection
    flag when include_pose_flags=True, giving 75 rather than 74 pose columns
    per view in the actual OWAS cache. A missing pose does not reject a row because its image
    embedding is still available.  Non-finite embedding rows are flagged as
    unavailable rather than silently imputed.  The current cache has none.
    """
    if view not in {'front', 'side', 'paired'}:
        raise ValueError(f'Unknown view: {view!r}')
    views = ['front', 'side'] if view == 'paired' else [view]
    embeddings = [np.asarray(arrays['embedding_' + v], dtype=float) for v in views]
    poses = []
    for v in views:
        pose = np.asarray(arrays['pose_' + v], dtype=float)
        if include_pose_flags:
            flag = np.asarray(arrays['pose_detected_' + v], dtype=float)
            if flag.ndim != 1 or len(flag) != len(pose) or not np.isin(flag, [0, 1]).all():
                raise ValueError('Pose-detected flags must be one binary value per row')
            pose = np.concatenate([pose, flag[:, None]], axis=1)
        poses.append(pose)
    if any(a.ndim != 2 for a in embeddings + poses):
        raise ValueError('Every embedding and pose block must be a 2D matrix')
    if len({len(a) for a in embeddings + poses}) != 1:
        raise ValueError('Embedding and pose blocks must have matching row counts')
    embedding = np.concatenate(embeddings, axis=1)
    pose = np.concatenate(poses, axis=1)
    available = np.isfinite(embedding).all(axis=1)
    return np.concatenate([embedding, pose], axis=1), available, embedding.shape[1]


class FrozenFeatureFusionLinearSVC(ClassifierMixin, BaseEstimator):
    """Small classifier with all preprocessing confined to training rows.

    Parameters
    ----------
    embedding_dim : int
        Number of leading raw embedding columns: 1024 per selected view.
    C : float
        Balanced LinearSVC regularization setting.
    n_components : int
        Requested PCA dimension. Actual dimension is min(request, n_train-1,
        embedding_dim), with a minimum of one when fit receives >=2 rows.
    use_pose : bool
        Include the separately scaled pose block, or retain embedding only.
    random_state : int
        LinearSVC seed. PCA uses deterministic full SVD.
    max_iter : int
        LinearSVC iteration limit.

    A class-absent or no-input training policy belongs to the caller.  This
    estimator explicitly requires two or more observed classes and two rows;
    the experiment runner can use its existing training-only dummy fallback.
    """

    def __init__(self, embedding_dim=1024, C=0.1, n_components=16,
                 use_pose=True, random_state=SEED, max_iter=50000):
        self.embedding_dim = embedding_dim
        self.C = C
        self.n_components = n_components
        self.use_pose = use_pose
        self.random_state = random_state
        self.max_iter = max_iter

    def _array(self, x, fitting=False):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError('X must be a 2D matrix')
        if not isinstance(self.embedding_dim, Integral) or self.embedding_dim < 1:
            raise ValueError('embedding_dim must be a positive integer')
        if x.shape[1] < self.embedding_dim:
            raise ValueError('X contains fewer columns than embedding_dim')
        if self.use_pose and x.shape[1] == self.embedding_dim:
            raise ValueError('use_pose=True requires at least one pose column')
        if not fitting:
            check_is_fitted(self, ['classifier_', 'n_features_in_'])
            if x.shape[1] != self.n_features_in_:
                raise ValueError('Raw feature width differs from fitted training width')
        if not np.isfinite(x[:, :self.embedding_dim]).all():
            raise ValueError('Frozen image embeddings must be finite')
        if np.isinf(x[:, self.embedding_dim:]).any():
            raise ValueError('Pose block permits NaN but not infinity')
        return x

    def fit(self, x, y):
        x = self._array(x, fitting=True)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError('y must be a one-dimensional label for every X row')
        if len(x) < 2 or len(np.unique(y)) < 2:
            raise ValueError('Use the runner dummy fallback for <2 rows or classes')
        if not isinstance(self.n_components, Integral) or self.n_components < 1:
            raise ValueError('n_components must be a positive integer')
        self.n_features_in_ = x.shape[1]
        self.embedding_scaler_ = StandardScaler()
        embedding = self.embedding_scaler_.fit_transform(x[:, :self.embedding_dim])
        self.n_components_ = min(self.n_components, len(x) - 1, self.embedding_dim)
        self.pca_ = PCA(n_components=self.n_components_, svd_solver='full')
        reduced = self.pca_.fit_transform(embedding)
        self.pca_scaler_ = StandardScaler()
        reduced = self.pca_scaler_.fit_transform(reduced)
        self.embedding_block_width_ = reduced.shape[1]
        self.embedding_block_divisor_ = float(np.sqrt(self.embedding_block_width_))
        blocks = [reduced / self.embedding_block_divisor_]
        self.pose_block_width_ = 0
        self.pose_block_divisor_ = None
        if self.use_pose:
            self.pose_imputer_ = SimpleImputer(strategy='median', add_indicator=True,
                                              keep_empty_features=True)
            pose = self.pose_imputer_.fit_transform(x[:, self.embedding_dim:])
            self.pose_scaler_ = StandardScaler()
            pose = self.pose_scaler_.fit_transform(pose)
            self.pose_block_width_ = pose.shape[1]
            self.pose_block_divisor_ = float(np.sqrt(self.pose_block_width_))
            blocks.append(pose / self.pose_block_divisor_)
        else:
            # Clear stale fitted state if a previously fitted instance changes
            # use_pose via set_params and is then fitted again.
            for attr in ('pose_imputer_', 'pose_scaler_'):
                if hasattr(self, attr):
                    delattr(self, attr)
        transformed = np.concatenate(blocks, axis=1)
        if not np.isfinite(transformed).all():
            raise ValueError('Training preprocessing produced non-finite values')
        self.classifier_ = LinearSVC(C=self.C, class_weight='balanced',
                                     max_iter=self.max_iter,
                                     random_state=self.random_state, dual='auto')
        self.classifier_.fit(transformed, y)
        self.classes_ = self.classifier_.classes_.copy()
        self.fit_feature_summary_ = {
            'training_n': len(x), 'raw_features': self.n_features_in_,
            'embedding_raw_features': self.embedding_dim,
            'pose_raw_features': x.shape[1] - self.embedding_dim,
            'pca_requested': self.n_components, 'pca_actual': self.n_components_,
            'pose_used': bool(self.use_pose),
            'pose_transformed_features': self.pose_block_width_,
            'final_features': transformed.shape[1],
            'embedding_block_divisor': self.embedding_block_divisor_,
            'pose_block_divisor': self.pose_block_divisor_,
            'preprocessing_fit_on_training_only': True,
            'pca_solver': 'full', 'margins_are_probabilities': False,
        }
        return self

    def transform_features(self, x):
        """Transform with frozen training statistics; do not update any state."""
        x = self._array(x)
        embedding = self.embedding_scaler_.transform(x[:, :self.embedding_dim])
        embedding = self.pca_.transform(embedding)
        embedding = self.pca_scaler_.transform(embedding)
        blocks = [embedding / self.embedding_block_divisor_]
        if self.use_pose:
            pose = self.pose_imputer_.transform(x[:, self.embedding_dim:])
            pose = self.pose_scaler_.transform(pose)
            blocks.append(pose / self.pose_block_divisor_)
        result = np.concatenate(blocks, axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Preprocessing produced non-finite values')
        return result

    def predict(self, x):
        return self.classifier_.predict(self.transform_features(x))

    def decision_function(self, x):
        """Return uncalibrated LinearSVC margins, not class probabilities."""
        return self.classifier_.decision_function(self.transform_features(x))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True  # Only pose columns may contain NaN.
        return tags


def smoke_checks():
    """Small synthetic checks of shape, state isolation, and serialization.

    These are technical checks only. They use no OWAS observations, references,
    raw images, or performance selection and report no predictive accuracy.
    """
    import io
    import joblib
    from sklearn.base import clone, is_classifier

    rng = np.random.default_rng(20261030)
    embedding = rng.normal(size=(24, 10))
    pose = rng.normal(size=(24, 8))
    pose[:, 0] = np.nan  # Entire training feature is missing.
    pose[::3, 1] = np.nan
    x = np.concatenate([embedding, pose], axis=1)
    y = np.arange(24) % 3
    model = FrozenFeatureFusionLinearSVC(embedding_dim=10).fit(x, y)
    assert is_classifier(model)
    assert model.n_components_ == 10
    assert model.transform_features(x).shape[1] == 20  # PCA10 + pose8+2 indicators.
    held_out = np.concatenate([rng.normal(size=(7, 10)), rng.normal(size=(7, 8))], axis=1)
    held_out[:, 10] = np.nan
    held_out[0, 14] = np.nan  # Newly missing test feature absent from indicator fit.
    before = {
        'embedding_mean': model.embedding_scaler_.mean_.copy(),
        'components': model.pca_.components_.copy(),
        'pca_mean': model.pca_scaler_.mean_.copy(),
        'imputation': model.pose_imputer_.statistics_.copy(),
        'pose_mean': model.pose_scaler_.mean_.copy(),
        'training_transformed': model.transform_features(x).copy(),
    }
    assert model.transform_features(held_out).shape == (7, 20)
    prediction = model.predict(held_out)
    assert prediction.shape == (7,)
    assert model.decision_function(held_out).shape == (7, 3)
    extreme = held_out.copy()
    extreme[:, :10] += 1000
    model.predict(extreme)
    for key, value in [
        ('embedding_mean', model.embedding_scaler_.mean_),
        ('components', model.pca_.components_),
        ('pca_mean', model.pca_scaler_.mean_),
        ('imputation', model.pose_imputer_.statistics_),
        ('pose_mean', model.pose_scaler_.mean_),
        ('training_transformed', model.transform_features(x)),
    ]:
        assert np.array_equal(value, before[key], equal_nan=True), key
    restored_buffer = io.BytesIO()
    joblib.dump(model, restored_buffer)
    restored_buffer.seek(0)
    restored = joblib.load(restored_buffer)
    assert np.array_equal(restored.predict(held_out), prediction)
    clone_model = clone(model)
    assert clone_model.get_params() == model.get_params()
    assert not hasattr(clone_model, 'classifier_')
    small = clone(model).fit(x[:4], y[:4])
    assert small.n_components_ == 3  # Training-row cap, not test-row cap.
    all_missing = x.copy()
    all_missing[:, 10:] = np.nan
    missing_model = clone(model).fit(all_missing, y)
    assert np.isfinite(missing_model.transform_features(all_missing)).all()
    assert missing_model.predict(held_out).shape == (7,)
    embedding_only = FrozenFeatureFusionLinearSVC(embedding_dim=10, use_pose=False).fit(x, y)
    a = held_out.copy()
    b = held_out.copy()
    b[:, 10:] = 100000  # Explicitly ignored in the matched ablation.
    assert np.array_equal(embedding_only.transform_features(a), embedding_only.transform_features(b))
    assert embedding_only.transform_features(x).shape == (24, 10)
    arrays = {f'{k}_{v}': x[:, :10] if k == 'embedding' else x[:, 10:]
              for k in ['embedding', 'pose'] for v in ['front', 'side']}
    arrays.update({f'pose_detected_{v}': np.ones(24, dtype=bool) for v in ['front', 'side']})
    paired, available, embedding_dim = assemble_fusion_inputs(arrays, 'paired')
    assert paired.shape == (24, 38) and embedding_dim == 20 and available.all()
    assert np.array_equal(paired[:, :10], arrays['embedding_front'])
    assert np.array_equal(paired[:, 20:28], arrays['pose_front'], equal_nan=True)
    return {
        'status': 'PASS', 'raw_images_processed': 0, 'owas_rows_used': 0,
        'checks': ['sklearn_classifier_and_clone', 'training_only_pca_dimension',
                   'all_missing_training_pose_column', 'new_missing_test_column',
                   'all_missing_pose_block', 'test_transform_no_training_state_update',
                   'joblib_round_trip', 'embedding_only_ignores_pose',
                   'paired_block_column_order'],
    }


if __name__ == '__main__':
    import json
    print(json.dumps(smoke_checks(), indent=2))
