"""Persisted prediction chain used by training, evaluation and deployment."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class Predictor(ClassifierMixin, BaseEstimator):
    """An already fitted preprocessor, model and optional score postprocessor.

    Groups remain external to the feature matrix. In particular, perturbing a
    feature for EC does not implicitly change the person's protected group.
    """

    accepts_sensitive_features = True

    def __init__(self, preprocessor, model, postprocessor=None, threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
        if hasattr(preprocessor, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(preprocessor.feature_names_in_)
            self.n_features_in_ = len(self.feature_names_in_)

    def fit(self, X, y=None):
        raise RuntimeError("Predictor wraps fitted components; fit TrustworthyClassifier instead.")

    def __sklearn_is_fitted__(self):
        return True

    def _transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame with original feature names.")
        if self.postprocessor is not None and not X.index.is_unique:
            raise ValueError("Postprocessing requires unique stable observation indices.")
        if hasattr(self, "feature_names_in_"):
            names = list(self.feature_names_in_)
            if set(X.columns) != set(names) or not X.columns.is_unique:
                raise ValueError("Prediction features must match the fitted feature schema.")
            X = X.loc[:, names]
        return self.preprocessor.transform(X)

    def predict_proba(self, X, *, sensitive_features=None):
        if sensitive_features is not None:
            if isinstance(sensitive_features, pd.Series) and not sensitive_features.index.equals(X.index):
                raise ValueError("sensitive_features index must match X, including its order.")
            groups = np.asarray(sensitive_features)
            if groups.ndim != 1 or len(groups) != len(X) or pd.isna(groups).any():
                raise ValueError("sensitive_features must be complete and aligned with X.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{type(self.model).__name__} does not expose probabilities.")
        probabilities = np.asarray(self.model.predict_proba(self._transform(X)), dtype=float)
        labels = list(self.model.classes_)
        if probabilities.ndim != 2 or probabilities.shape != (len(X), len(labels)):
            raise ValueError("Model probabilities have an invalid shape.")
        scores = probabilities[:, labels.index(1)] if 1 in labels else np.zeros(len(X))
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Model probabilities must be finite and in [0, 1].")
        if self.postprocessor is not None:
            if sensitive_features is None:
                raise ValueError("This predictor requires sensitive_features for postprocessing.")
            scores = np.asarray(self.postprocessor.transform(
                scores, sensitive_features=sensitive_features, index=X.index,
            ), dtype=float).reshape(-1)
            if len(scores) != len(X) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Postprocessed probabilities must be finite and in [0, 1].")
        return np.column_stack((1.0 - scores, scores))

    def predict(self, X, *, sensitive_features=None):
        if self.postprocessor is None:
            predictions = np.asarray(self.model.predict(self._transform(X))).reshape(-1)
            if not np.isin(predictions, [0, 1]).all():
                raise ValueError("The current binary classification contract requires labels 0 and 1.")
            return predictions.astype(int)
        return (self.predict_proba(X, sensitive_features=sensitive_features)[:, 1] >= self.threshold).astype(int)
