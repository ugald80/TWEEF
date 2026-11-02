"""Scientific protocol checks: original categories, fixed rankings and loss units."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from TWEEF.Evaluation import (
    effective_complexity, evaluate_metrics, frozen_ranking,
    make_tabular_preprocessor, normalize_ec_config, prediction_loss,
)
from TWEEF.PostProcessing import CalibratedEqualizedOdds


def data():
    X = pd.DataFrame({"category": ["a", "b", "c", "a"] * 12,
                      "number": [0., 1., 0., 1.] * 12}, index=np.arange(100, 148))
    y = pd.Series([0, 1, 0, 1] * 12, index=X.index)
    groups = pd.Series(["female"] * 24 + ["male"] * 24, index=X.index)
    return X, y, groups


def test_original_categories_and_loss_curves_do_not_refit_or_modify_data():
    X, y, groups = data()
    model = make_pipeline(make_tabular_preprocessor(X), DecisionTreeClassifier(random_state=0)).fit(X, y)
    before = pickle.dumps(model)
    original = X.copy(deep=True)
    ranking = frozen_ranking(model, X, y, random_state=8, groups=groups)
    score, details = effective_complexity(model, X, ranking, random_state=8, groups=groups)
    assert ranking[0] == "number"
    assert score == 0.5
    assert details["n_original_features"] == 2
    assert details["k"] == 1
    assert details["loss_units"] == "nats"
    pd.testing.assert_frame_equal(X, original)
    assert pickle.dumps(model) == before


def test_explicit_class_loss_for_svm_and_no_silent_kl_substitution():
    X, y, groups = data()
    model = make_pipeline(make_tabular_preprocessor(X), LinearSVC(random_state=0)).fit(X, y)
    with pytest.raises(ValueError, match="predict_proba"):
        effective_complexity(model, X, list(X))
    config = {"loss": "class_disagreement", "loss_threshold": 0.05}
    score, details = effective_complexity(model, X, ["number", "category"], ec_config=config)
    assert score == 0.5
    assert details["loss_units"] == "fraction"
    metrics = evaluate_metrics(model, X, y, groups, ["number", "category"], ec_config=config)
    assert metrics["equal Opportunity"] == 1


def test_identity_kl_and_threshold_units():
    values = np.array([0, 0.5, 0.8, 1], dtype=np.float32)
    assert prediction_loss(values, values, "bernoulli_kl") == 0
    assert prediction_loss([0.8], [0.6], "bernoulli_kl") == pytest.approx(0.09151622185)
    assert prediction_loss([0, 1], [1, 1], "class_disagreement") == 0.5
    with pytest.raises(ValueError, match="Unknown"):
        normalize_ec_config({"losss": "bernoulli_kl"})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normalize_ec_config({"loss": "class_disagreement", "loss_threshold": 2})


def test_inference_missing_and_unknown_categories_use_fitted_preprocessing():
    X, y, _ = data()
    model = make_pipeline(make_tabular_preprocessor(X), DecisionTreeClassifier(random_state=0)).fit(X, y)
    new = pd.DataFrame({"category": ["new", np.nan], "number": [np.nan, 0.5]})
    assert model.predict(new).shape == (2,)


def test_evaluation_rejects_misaligned_groups_and_undefined_fairness():
    X, y, groups = data()
    model = make_pipeline(make_tabular_preprocessor(X), DecisionTreeClassifier()).fit(X, y)
    with pytest.raises(ValueError, match="index"):
        evaluate_metrics(model, X, y, groups.iloc[::-1], list(X))
    invalid = groups.copy()
    invalid.loc[y == 0] = "zero-only"
    with pytest.raises(ValueError, match="two sensitive groups"):
        evaluate_metrics(model, X, y, invalid, list(X))


def fitted_post():
    _, y, groups = data()
    p = np.where(y == 1, 0.9, 0.1)
    p[:24] = np.where(y.iloc[:24] == 1, 0.7, 0.3)
    return CalibratedEqualizedOdds(random_state=29).fit(p, y, sensitive_features=groups), p, y, groups


def test_postprocessor_persists_aif360_fit_and_couples_randomness_by_row():
    pytest.importorskip("aif360")
    post, p, y, groups = fitted_post()
    saved = pickle.dumps(post)
    rng_before = np.random.get_state()
    first = post.transform(p, sensitive_features=groups, index=y.index)
    second = post.transform(p, sensitive_features=groups, index=y.index)
    np.testing.assert_array_equal(first, second)
    assert prediction_loss(first, second, "bernoulli_kl") == 0
    assert pickle.dumps(post) == saved
    rng_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(rng_before, rng_after))
    restored = pickle.loads(saved)
    np.testing.assert_array_equal(first, restored.transform(p, sensitive_features=groups, index=y.index))
    order = np.arange(len(p))[::-1]
    reordered = post.transform(p[order], sensitive_features=groups.iloc[order], index=y.index[order])
    np.testing.assert_array_equal(first, reordered[::-1])
    assert post.calibration_index_ == list(y.index)


def test_postprocessor_rejects_unknown_or_incomplete_groups():
    pytest.importorskip("aif360")
    post, p, y, groups = fitted_post()
    invalid = groups.copy()
    invalid.iloc[0] = "new"
    with pytest.raises(ValueError, match="Unknown sensitive group"):
        post.transform(p, sensitive_features=invalid, index=y.index)
    with pytest.raises(ValueError, match="both true classes"):
        CalibratedEqualizedOdds().fit(p, np.zeros(len(y)), sensitive_features=groups)
    with pytest.raises(ValueError, match="indices must match"):
        post.transform(p, sensitive_features=groups.iloc[::-1], index=y.index)
    with pytest.raises(ValueError, match="unique, stable"):
        post.transform(p, sensitive_features=groups.to_numpy(), index=[1] * len(p))
    with pytest.raises(ValueError, match="indices must match"):
        CalibratedEqualizedOdds().fit(pd.Series(p, index=y.index[::-1]), y, sensitive_features=groups)


def test_legacy_surrogate_helper_never_fits_test(monkeypatch):
    pytest.importorskip("interpret")
    from TWEEF.ProcessorInterpretability import InterpretabilityProcessing
    import importlib
    module = importlib.import_module("TWEEF.ProcessorInterpretability")
    X, y, _ = data()
    train, test = X.iloc[:32].drop(columns="category"), X.iloc[32:].drop(columns="category")
    processor = InterpretabilityProcessing()
    monkeypatch.setattr(module, "ExplainableBoostingClassifier", lambda **kwargs: DecisionTreeClassifier(random_state=0))
    explanation_indices = []
    monkeypatch.setattr(processor, "compute_attributions_expectations",
                        lambda X_train, X_rank, y_rank, model: explanation_indices.extend(X_rank.index))
    with pytest.warns(DeprecationWarning, match="legacy"):
        predictions = processor.surrogate_model_train(
            DecisionTreeClassifier(random_state=0), train, test, y.iloc[:32], y.iloc[32:],
        )
    assert explanation_indices == list(train.index)
    assert processor.explainable_classifier.tree_.n_node_samples[0] == len(train)
    assert list(predictions.index) == list(test.index)
