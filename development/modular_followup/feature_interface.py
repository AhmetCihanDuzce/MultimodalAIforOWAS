"""Feature assembly shared by development and frozen image inference."""
import numpy as np


def assemble(arrays, candidate):
    views = ['front', 'side'] if candidate['view'] == 'paired' else [candidate['view']]
    xs, availability = [], []
    for view in views:
        if candidate['family'] == 'embedding_linear_SVM':
            x = arrays['embedding_' + view]
            available = np.ones(len(x), dtype=bool)
        else:
            x = arrays['pose_' + view]
            available = arrays['pose_detected_' + view].astype(bool)
            if candidate['family'] == 'combined_leg_RF':
                x = np.concatenate([x, arrays['geometry_' + view]], axis=1)
                available = arrays['available_' + view].astype(bool)
        xs.append(x)
        availability.append(available)
    return np.concatenate(xs, axis=1), np.logical_or.reduce(availability)


def predict_fitted(model, x, available, hierarchy=False):
    pred = np.zeros(len(x), dtype=int)
    coarse = np.full(len(x), -1, dtype=int)
    if model is not None and available.any():
        if hierarchy:
            pred[available], coarse[available] = model.predict_details(x[available])
        else:
            pred[available] = model.predict(x[available]).astype(int)
    return pred, coarse
