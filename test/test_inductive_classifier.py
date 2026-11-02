"""End-to-end invariants for the final surrogate and held-out evaluation."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

pytest.importorskip("interpret")

from TWEEF.Aggregation import MechanismResult
from TWEEF.Evaluation import effective_complexity, prediction_loss
from TWEEF.TrustworthyClassifier import TrustworthyClassifier


class MeanMechanism(BaseEstimator):
    supports_weights = True

    def fit(self, reference_metrics, *, weights):
        self.weights_ = weights.copy()
        return self

    def aggregate(self, metrics):
        return MechanismResult(sum(self.weights_[name] * value for name, value in metrics.items()),
                               score_range=(0, 1), mechanism="mean")


class ShiftScores(BaseEstimator):
    """A small fitted postprocessor makes missing inference stages observable."""
    def fit(self, scores, y, *, sensitive_features):
        self.calibration_index_ = list(y.index)
        self.groups_ = list(pd.unique(sensitive_features))
        return self

    def transform(self, scores, *, sensitive_features, index=None):
        return np.where(np.asarray(sensitive_features) == self.groups_[0], 0.1, 0.9)


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    n = 240
    X = pd.DataFrame({
        "amount": rng.normal(size=n),
        "duration": rng.normal(size=n),
        "category": np.tile(["A", "B", "C"], n // 3),
    }, index=np.arange(1000, 1000 + n))
    y = pd.Series((X.amount + 0.4 * X.duration > 0).astype(int), index=X.index, name="outcome")
    groups = pd.Series(np.tile(["female", "male"], n // 2), index=X.index)
    return X, y, groups


def classifier(**kwargs):
    options = dict(
        given_classifier=RandomForestClassifier(n_estimators=8, max_depth=3, random_state=5),
        surrogate=DecisionTreeClassifier(max_depth=2, random_state=5),
        aggregation=MeanMechanism(), reference_cv=2, random_state=5,
        disparate_impact_repair_level=0,
        effective_complexity={"repeats": 1},
    )
    options.update(kwargs)
    return TrustworthyClassifier(**options)


def fitted(data, **kwargs):
    X, y, groups = data
    result = classifier(**kwargs)
    result.fit(X.iloc[:160], y.iloc[:160], sensitive_features=groups.iloc[:160],
               validation_data=(X.iloc[160:200], y.iloc[160:200]),
               validation_sensitive_features=groups.iloc[160:200])
    return result


def test_final_model_is_surrogate_and_uses_teacher_targets(data):
    model = fitted(data, surrogate=DummyClassifier(strategy="most_frequent"))
    X, y, _ = data
    assert model.classifier_ is model.surrogate_
    assert model.classifier_ is not model.teacher_
    assert model.final_model_source_ == "surrogate"
    transformed = model.preprocessor_.transform(X.iloc[:160])
    teacher_targets = model.teacher_.predict(transformed)
    assert np.array_equal(model.surrogate_.classes_, np.unique(teacher_targets))
    report = model.evaluate(X.iloc[200:], y.iloc[200:], sensitive_features=data[2].iloc[200:])
    assert np.array_equal(report.predictions, model.predict(X.iloc[200:] ))
    assert model.classifier_ is model.surrogate_


def test_cv_provenance_disjoint_from_final_calibration_and_test(data):
    model = fitted(data, postprocessor=ShiftScores())
    training = set(data[0].index[:160])
    forbidden = set(data[0].index[160:])
    for row in model.reference_audit_:
        fit, cal, score = (set(row[key]) for key in ("fit_index", "calibration_index", "score_index"))
        assert fit and cal and score
        assert not fit & cal and not fit & score and not cal & score
        assert fit | cal | score == training
        assert not (fit | cal | score) & forbidden
    assert model.postprocessor_.calibration_index_ == list(data[0].index[160:200])
    assert model.fit_metadata_["test_used"] is False


def test_prediction_report_serialization_and_original_schema(data):
    model = fitted(data, postprocessor=ShiftScores())
    X, y, groups = (part.iloc[200:] for part in data)
    predictions = model.predict(X, sensitive_features=groups)
    base = model.surrogate_predictor_.predict(X)
    assert np.any(predictions != base)
    report = model.evaluate(X, y, sensitive_features=groups)
    assert np.array_equal(report.predictions, predictions)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict(X, sensitive_features=groups), predictions)
    assert np.array_equal(restored.predict(X.iloc[::-1], sensitive_features=groups.iloc[::-1]), predictions[::-1])
    assert np.array_equal(restored.predict(X[list(reversed(X.columns))], sensitive_features=groups), predictions)
    with pytest.raises(ValueError, match="sensitive_features"):
        restored.predict(X)
    with pytest.raises(ValueError, match="sensitive_features index"):
        restored.predictor_.predict(X, sensitive_features=groups.iloc[::-1])
    repeated = pd.concat([X.iloc[:1], X.iloc[:1]])
    with pytest.raises(ValueError, match="unique stable"):
        restored.predictor_.predict(repeated, sensitive_features=np.array(["female", "female"]))


def test_test_labels_never_change_fitted_components(data):
    model = fitted(data)
    X, y, groups = (part.iloc[200:] for part in data)
    def snapshot():
        return pickle.dumps((model.teacher_, model.surrogate_, model.preprocessor_,
                             model.reference_metrics_, model.ranking_, model.trust_estimator_))
    before = snapshot()
    model.evaluate(X, y, sensitive_features=groups)
    assert snapshot() == before
    model.evaluate(X, 1 - y, sensitive_features=groups)
    assert snapshot() == before


def test_novel_categories_and_numeric_extremes_do_not_refit_preprocessor(data):
    model = fitted(data)
    X = data[0].iloc[200:].copy()
    X["category"] = "unseen"
    X["amount"] = 100000.0
    before = pickle.dumps(model.preprocessor_)
    assert model.predict(X).shape == (40,)
    assert pickle.dumps(model.preprocessor_) == before


def test_deferred_aggregation_uses_explicit_shared_reference(data):
    model = fitted(data, aggregation=None)
    X, y, groups = (part.iloc[200:] for part in data)
    assert model.evaluate(X, y, sensitive_features=groups).trustworthiness is None
    prediction_before = model.predict(X)
    reference = model.reference_metrics_.copy()
    model.set_aggregation(MeanMechanism(), reference_metrics=reference)
    assert model.evaluate(X, y, sensitive_features=groups).trustworthiness.mechanism == "mean"
    assert np.array_equal(model.predict(X), prediction_before)
    reference.iloc[:, :] = 0
    assert not model.aggregation_reference_.equals(reference)


def test_candidate_selection_is_bounded_deterministic_and_pretest(data):
    candidates = [{"surrogate": DecisionTreeClassifier(max_depth=depth, random_state=2)} for depth in [1, 3]]
    a = fitted(data, selection_candidates=candidates)
    b = fitted(data, selection_candidates=candidates)
    assert len(a.reference_metrics_) == 4
    assert a.fit_metadata_["candidate_fit_budget"] == 4
    assert a.fit_metadata_["final_fit_count"] == 1
    assert a.selected_candidate_ == b.selected_candidate_
    pd.testing.assert_frame_equal(a.selection_report_, b.selection_report_)
    pd.testing.assert_frame_equal(a.reference_metrics_, b.reference_metrics_)


def test_candidates_share_identical_folds_when_random_state_is_none(data):
    candidates = [{"surrogate": DecisionTreeClassifier(max_depth=depth, random_state=2)} for depth in [1, 3]]
    model = fitted(data, selection_candidates=candidates, random_state=None, postprocessor=ShiftScores())
    first = [row for row in model.reference_audit_ if row["candidate"] == 0]
    second = [row for row in model.reference_audit_ if row["candidate"] == 1]
    for a, b in zip(first, second):
        for name in ("fit_index", "calibration_index", "score_index"):
            assert a[name] == b[name]


@pytest.mark.parametrize("options", [
    {"aggregation_alpha": 0.7},
    {"gwa_hesitation": 0.2},
    {"gwa_hesitation": {"Accuracy": 0.2}},
    {"trustworthiness_score_scale": "linguistic"},
])
def test_injected_mechanism_rejects_conflicting_global_configuration(options):
    model = classifier(**options)
    model.metric_weights_ = {"Accuracy": 1.0}
    with pytest.raises(ValueError, match="owns its alpha, hesitation, score scale"):
        model._make_trust_estimator(model.aggregation, pd.DataFrame({"Accuracy": [0.5, 0.9]}))


def test_large_finite_importance_weights_are_normalized_without_overflow():
    model = classifier(metric_weights={"Accuracy": 1e308, "Effective Complexity": 1e308})
    assert model._weights() == {"Accuracy": 0.5, "Effective Complexity": 0.5}


def test_performance_cost_joins_metric_names_instead_of_positions():
    model = classifier()
    model.classifier_ = DummyClassifier()
    model.metrics_pre = model._metric_list({"Accuracy": 0.8, "equal Opportunity": 0.7}, {})
    model.metrics_post = model._metric_list({"Effective Complexity": 0.1, "Accuracy": 0.6}, {})
    result = model.trust_cost_on_performance()
    assert result["metric"].tolist() == ["Accuracy"]
    assert result.iloc[0]["value"] == pytest.approx(0.25)


def test_classifier_new_configuration_can_be_cloned():
    instance = classifier(postprocessor=ShiftScores(), selection_candidates=[{"surrogate": DecisionTreeClassifier()}])
    copied = clone(instance)
    assert copied.preprocessor is None
    assert copied.surrogate is not instance.surrogate
    assert copied.postprocessor is not instance.postprocessor


def test_objective_failure_is_reported_without_switching_to_teacher(data):
    model = fitted(data, surrogate=DummyClassifier(strategy="most_frequent"), surrogate_min_agreement=1.0)
    X, y, groups = (part.iloc[200:] for part in data)
    report = model.evaluate(X, y, sensitive_features=groups)
    assert report.teacher_student_fidelity["objective_met"] is False
    assert report.metadata["fairness_objective"]["objective_met"] is True
    assert model.classifier_ is model.surrogate_
    assert model.feature_ranking_ == model.ranking_


def test_explicit_internal_calibration_is_not_named_test(data):
    X, y, groups = data
    with pytest.warns(FutureWarning, match="internal calibration"):
        model = classifier().fit(X, y, sensitive_features=groups)
    assert set(model.training_index_).isdisjoint(model.calibration_index_)
    assert model.x_test is None and model.y_test is None


def test_invalid_alignment_is_rejected(data):
    X, y, groups = data
    with pytest.raises(ValueError, match="sensitive_features index"):
        classifier().fit(X, y, sensitive_features=groups.iloc[::-1])
    with pytest.raises(ValueError, match="disjoint"):
        classifier().fit(X, y, sensitive_features=groups, validation_data=(X.iloc[:40], y.iloc[:40]),
                         validation_sensitive_features=groups.iloc[:40])


def test_actual_equalized_odds_chain_is_stable_and_ec_identity_is_zero(data):
    pytest.importorskip("aif360")
    model = fitted(data, postprocessor="calibrated_equalized_odds")
    X, y, groups = (part.iloc[200:] for part in data)
    p = model.predict_proba(X, sensitive_features=groups)[:, 1]
    q = model.predict_proba(X.copy(), sensitive_features=groups)[:, 1]
    assert prediction_loss(p, q, "bernoulli_kl") == 0
    assert model.postprocessor_.processor_ is not None
    assert np.array_equal(model.predict(X, sensitive_features=groups), (p >= 0.5).astype(int))
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict_proba(X, sensitive_features=groups)[:, 1], p)
    assert np.array_equal(model.predict_proba(X.iloc[::-1], sensitive_features=groups.iloc[::-1])[:, 1], p[::-1])
    _, details = effective_complexity(model.predictor_, X, model.ranking_,
                                     ec_config={"repeats": 1, "loss_threshold": 0}, groups=groups)
    assert details["curve"][-1]["mean_loss"] == 0
