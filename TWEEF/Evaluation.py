"""Model-independent evaluation on original tabular variables.

All ranking and calibration preparation belongs before evaluation. In particular,
these functions never refit a predictor or select a feature ranking on test.
"""
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from TWEEF.Constants import METRICS, EPSILON


DEFAULT_EC = {
    "loss": "bernoulli_kl", "loss_threshold": 0.05,
    "feature_unit": "original", "ranking_method": "permutation", "repeats": 3,
}


def make_tabular_preprocessor(X):
    """An inductive baseline; callers may inject a model-specific transformer."""
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
        raise ValueError("X must be a DataFrame with unique original variable names.")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [name for name in X if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", MinMaxScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=True)


def normalize_ec_config(config=None):
    result = deepcopy(DEFAULT_EC)
    if config is not None:
        if not hasattr(config, "items"):
            raise TypeError("effective_complexity must be a configuration mapping.")
        unknown = set(config).difference(result)
        if unknown:
            raise ValueError(f"Unknown effective complexity options: {sorted(unknown)!r}")
        result.update(config)
    if result["loss"] not in {"bernoulli_kl", "class_disagreement"}:
        raise ValueError("EC loss must be 'bernoulli_kl' or 'class_disagreement'.")
    threshold = result["loss_threshold"]
    if not isinstance(threshold, Real) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("EC loss_threshold must be finite and nonnegative.")
    if result["loss"] == "class_disagreement" and threshold > 1:
        raise ValueError("Class disagreement threshold must be in [0, 1].")
    if not isinstance(result["repeats"], Integral) or result["repeats"] < 1:
        raise ValueError("EC repeats must be a positive integer.")
    if result["feature_unit"] != "original" or result["ranking_method"] != "permutation":
        raise ValueError("This protocol requires original variables and a frozen permutation ranking.")
    return result


def _aligned(values, X, name):
    if isinstance(values, pd.Series) and not values.index.equals(X.index):
        raise ValueError(f"{name} index must match X, including its order.")
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != len(X) or pd.isna(array).any():
        raise ValueError(f"{name} must be complete, one-dimensional and aligned with X.")
    return array


def _call(predictor, method, X, groups):
    # FittedPredictor explicitly advertises support. Raw sklearn estimators do
    # not receive metadata they do not implement; exceptions are not swallowed.
    function = getattr(predictor, method)
    if getattr(predictor, "accepts_sensitive_features", False):
        return function(X, sensitive_features=groups)
    return function(X)


def positive_probabilities(predictor, X, groups=None):
    if not hasattr(predictor, "predict_proba"):
        raise ValueError("bernoulli_kl requires predict_proba; choose class_disagreement explicitly.")
    try:
        probabilities = np.asarray(_call(predictor, "predict_proba", X, groups), dtype=float)
    except AttributeError as exc:
        raise ValueError("bernoulli_kl requires predict_proba; no automatic loss substitution.") from exc
    classes = list(getattr(predictor, "classes_", [0, 1]))
    if probabilities.ndim != 2 or probabilities.shape != (len(X), len(classes)):
        raise ValueError("predict_proba returned an invalid shape.")
    if (not np.isfinite(probabilities).all() or
            ((probabilities < 0) | (probabilities > 1)).any() or
            not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)):
        raise ValueError("predict_proba must return finite probability distributions.")
    if 1 not in classes or any(value not in (0, 1) for value in classes):
        if classes == [0]:
            return np.zeros(len(X))
        raise ValueError("The Bernoulli protocol requires binary labels 0 and 1.")
    return probabilities[:, classes.index(1)]


def prediction_loss(reference, restricted, loss):
    """KL in nats or a class disagreement fraction, with explicit units."""
    p = np.asarray(reference)
    q = np.asarray(restricted)
    if p.shape != q.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("Prediction vectors must have the same nonempty shape.")
    if loss == "class_disagreement":
        return float(np.mean(p != q))
    if loss != "bernoulli_kl":
        raise ValueError("Unknown prediction loss.")
    p, q = p.astype(float), q.astype(float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (
        (p < 0).any() or (p > 1).any() or (q < 0).any() or (q > 1).any()
    ):
        raise ValueError("KL requires finite probabilities in [0, 1].")
    p, q = np.clip(p, EPSILON, 1 - EPSILON), np.clip(q, EPSILON, 1 - EPSILON)
    value = float(np.mean(rel_entr(p, q) + rel_entr(1 - p, 1 - q)))
    if value < -1e-12 or not np.isfinite(value):
        raise ArithmeticError("Invalid KL divergence.")
    return max(0.0, value)


def frozen_ranking(predictor, X, y, random_state=0, repeats=5, groups=None):
    """Permutation importance in accuracy, computed on pre-test data only.

    Permute an entire original variable. Sensitive group metadata stays attached
    to the observation even when a predictor input encodes a related variable.
    Negative importances remain negative; ties keep original column order.
    """
    y = _aligned(y, X, "y")
    if groups is not None:
        _aligned(groups, X, "sensitive_features")
    if not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("Ranking repeats must be a positive integer.")
    baseline = np.mean(_call(predictor, "predict", X, groups) == y)
    rng = np.random.default_rng(random_state)
    importances = []
    for name in X:
        scores = []
        for _ in range(repeats):
            perturbed = X.copy()
            perturbed[name] = X[name].to_numpy()[rng.permutation(len(X))]
            scores.append(np.mean(_call(predictor, "predict", perturbed, groups) == y))
        importances.append(float(baseline - np.mean(scores)))
    order = np.argsort(-np.asarray(importances), kind="stable")
    return [X.columns[i] for i in order]


def effective_complexity(predictor, X, ranking, ec_config=None, random_state=0, groups=None):
    """Return the EC score and its reproducible loss curve on raw variables."""
    config = normalize_ec_config(ec_config)
    if not isinstance(X, pd.DataFrame) or X.empty or X.columns.has_duplicates:
        raise ValueError("EC requires nonempty original data with unique variable names.")
    if len(ranking) != len(X.columns) or set(ranking) != set(X.columns):
        raise ValueError("The frozen ranking must contain each original variable exactly once.")
    if groups is not None:
        _aligned(groups, X, "sensitive_features")
    probabilistic = config["loss"] == "bernoulli_kl"
    def predictions(data):
        return (positive_probabilities(predictor, data, groups) if probabilistic
                else np.asarray(_call(predictor, "predict", data, groups)))
    reference = predictions(X)
    n_features = len(ranking)
    rng = np.random.default_rng(random_state)
    curves = []
    k_selected = n_features
    for k in range(1, n_features + 1):
        losses = []
        for _ in range(config["repeats"]):
            restricted = X.copy()
            for name in ranking[k:]:
                restricted[name] = X[name].to_numpy()[rng.permutation(len(X))]
            losses.append(prediction_loss(reference, predictions(restricted), config["loss"]))
        mean_loss = float(np.mean(losses))
        curves.append({"k": k, "mean_loss": mean_loss, "losses": losses})
        threshold = config["loss_threshold"]
        tolerance = 4 * np.finfo(float).eps * max(abs(mean_loss), abs(threshold))
        if mean_loss <= threshold + tolerance:
            k_selected = k
            break
    return 1.0 - k_selected / n_features, {
        "k": k_selected, "n_original_features": n_features,
        "loss": config["loss"], "loss_units": "nats" if probabilistic else "fraction",
        "loss_threshold": config["loss_threshold"], "curve": curves,
        "ranking": list(ranking), "random_state": random_state,
    }


def evaluate_metrics(predictor, X, y, groups, ranking, ec_config=None, random_state=0):
    y = _aligned(y, X, "y")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("This evaluation protocol requires binary labels 0 and 1.")
    predictions = np.asarray(_call(predictor, "predict", X, groups))
    fairness = float("nan")
    if groups is not None:
        groups_array = _aligned(groups, X, "sensitive_features")
        values = list(pd.unique(groups_array))
        if len(values) != 2:
            raise ValueError("Equal Opportunity requires exactly two sensitive groups.")
        tprs = []
        for group in values:
            positive = (groups_array == group) & (y == 1)
            if not positive.any():
                raise ValueError("Equal Opportunity is undefined: a group has no positive observations.")
            tprs.append(float(np.mean(predictions[positive] == 1)))
        fairness = 1.0 - abs(tprs[0] - tprs[1])
    complexity, _ = effective_complexity(
        predictor, X, ranking, ec_config=ec_config, random_state=random_state, groups=groups,
    )
    return {
        METRICS.PERFORMANCE.ACCURACY: float(np.mean(predictions == y)),
        METRICS.FAIRNESS.EQUAL_OPPORTUNITY: fairness,
        METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY: complexity,
    }
