"""Small, serializable classifiers, including an actual conditional hierarchy."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SEED = 20261029
CLASSES = {'back': [1, 2, 3, 4], 'arms': [1, 2, 3], 'legs': [2, 3, 4, 5, 7]}


def base_estimator(family, params):
    impute = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    if family in ['pose_RF', 'combined_leg_RF']:
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
              max_depth=params['max_depth'], class_weight='balanced_subsample',
              random_state=SEED, n_jobs=1)
        return make_pipeline(impute, clf)
    if family == 'pose_RBF_SVM':
        return make_pipeline(impute, StandardScaler(), SVC(C=params['C'], gamma='scale',
                             kernel='rbf', class_weight='balanced', probability=False))
    if family == 'embedding_linear_SVM':
        return make_pipeline(impute, StandardScaler(), LinearSVC(C=params['C'], class_weight='balanced',
                             max_iter=50000, random_state=SEED, dual='auto'))
    raise ValueError(family)


def fit_node(x, y, family, params):
    if len(y) == 0:
        return None
    clf = DummyClassifier(strategy='most_frequent') if len(np.unique(y)) == 1 else base_estimator(family, params)
    clf.fit(x, y)
    return clf


class HierarchicalLegRF(ClassifierMixin, BaseEstimator):
    """Classify appearance family, then its subcode; never route by true test label."""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, x, y):
        params = {'max_depth': self.max_depth}
        family = np.array([0 if c in [2, 3] else 1 if c in [4, 5] else 2 for c in y])
        self.coarse_ = fit_node(x, family, 'combined_leg_RF', params)
        self.branches_ = {}
        self.training_support_ = {'coarse': {str(c): int((family == c).sum()) for c in [0, 1, 2]}}
        for branch, codes in [(0, [2, 3]), (1, [4, 5])]:
            mask = np.isin(y, codes)
            self.branches_[branch] = fit_node(x[mask], y[mask], 'combined_leg_RF', params)
            self.training_support_[str(branch)] = {str(c): int((y[mask] == c).sum()) for c in codes}
        self.classes_ = np.unique(y)
        return self

    def predict_details(self, x):
        coarse = self.coarse_.predict(x).astype(int)
        prediction = np.zeros(len(x), dtype=int)
        prediction[coarse == 2] = 7
        for branch in [0, 1]:
            mask = coarse == branch
            if mask.any() and self.branches_[branch] is not None:
                prediction[mask] = self.branches_[branch].predict(x[mask]).astype(int)
        return prediction, coarse

    def predict(self, x):
        return self.predict_details(x)[0]


def make_candidates(component, hierarchy=False):
    families = ['pose_RF', 'pose_RBF_SVM', 'embedding_linear_SVM']
    if component == 'legs':
        families += ['combined_leg_RF']
    if hierarchy:
        families = ['combined_leg_RF']
    result = []
    for view in ['side', 'front', 'paired']:
        for family in families:
            params = [{'max_depth': d} for d in [5, None]] if family.endswith('_RF') else [{'C': c} for c in ([1, 10] if family == 'pose_RBF_SVM' else [.1, 1])]
            for k, param in enumerate(params):
                ident = family + '_' + view + '_' + str(k) + ('_hierarchy' if hierarchy else '')
                result.append({'id': ident, 'family': family, 'view': view, 'parameters': param, 'hierarchy': hierarchy})
    return result
