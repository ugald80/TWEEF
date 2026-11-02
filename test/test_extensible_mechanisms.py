"""Native mechanisms are injectable without inheriting TWEEF's representation."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

from TWEEF.Aggregation import GWA, LWA, MechanismResult, SubjectiveLogic
from TWEEF.Calibration import QuartileCalibration
from TWEEF.Constants import METRICS, TRUSTWORTHINESS_SCORE_NAME
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.Metrics import MetricList


NAMES = [METRICS.PERFORMANCE.ACCURACY, METRICS.FAIRNESS.EQUAL_OPPORTUNITY]
WEIGHTS = dict(zip(NAMES, [0.8, 0.2]))


class QuadraticMechanism(BaseEstimator):
    """Illustrative external representation; no fuzzy API or calibration."""

    supports_weights = True

    def __init__(self, factor=1.0):
        self.factor = factor

    def fit(self, reference_metrics, *, weights):
        self.fit_calls_ = getattr(self, "fit_calls_", 0) + 1
        self.weights_ = dict(weights)
        self.reference_ = reference_metrics
        return self

    def aggregate(self, metrics):
        represented = {name: value ** 2 for name, value in metrics.items()}
        score = self.factor * sum(self.weights_[name] * represented[name] for name in metrics)
        return MechanismResult(
            score=score,
            score_range=None,
            higher_is_better=False,
            details={"representation": represented},
            mechanism="quadratic-test-v1",
        )


def reference():
    return pd.DataFrame([[0.0, 0.1], [0.2, 0.4], [0.8, 0.7], [1.0, 1.0]], columns=NAMES)


def metric_list(values, weights=WEIGHTS):
    result = MetricList()
    for name, value in zip(NAMES, values):
        result.add_metric(name, weights[name], goal=0.0, value=value)
    return result


@pytest.mark.parametrize("operator,mechanism", [
    ("lwa", LWA()), ("gwa", GWA()), ("sl", SubjectiveLogic()),
    ("gwa", GWA(hesitation={NAMES[0]: 0.25, NAMES[1]: 0.1})),
    ("sl", SubjectiveLogic(alpha=0.7)),
])
@pytest.mark.parametrize("scale", ["normalized", "linguistic"])
def test_builtin_objects_preserve_established_formulas(operator, mechanism, scale):
    mechanism = clone(mechanism).set_params(score_scale=scale)
    legacy = TrustworthinessEstimator(
        aggregation=operator,
        alpha=getattr(mechanism, "alpha", 0.5),
        gwa_hesitation=getattr(mechanism, "hesitation", 0.0),
        trustworthiness_score_scale=scale,
    ).fit(metrics_values=reference(), metric_weights=WEIGHTS)
    fitted = TrustworthinessEstimator(aggregation=mechanism).fit(
        metrics_values=reference(), metric_weights=WEIGHTS,
    )
    data = pd.DataFrame([[0, 0], [0.12, 0.63], [0.8, 0.9], [1, 1]], columns=NAMES)
    actual = fitted.transform(data)[TRUSTWORTHINESS_SCORE_NAME]
    expected = legacy.transform(data)[TRUSTWORTHINESS_SCORE_NAME]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    for row, value in zip(data.to_numpy(), actual):
        assert fitted.compute_metric(metric_list(row)) == pytest.approx(value, abs=1e-12)
    assert fitted.aggregation_result_.mechanism == operator
    assert fitted.aggregation_result_.score_range == ((-2, 2) if scale == "linguistic" else (0, 1))


def test_external_plugin_needs_no_fuzzy_calibration_and_preserves_raw_columns():
    data = pd.DataFrame([[0.2, 0.7], [0.9, 0.4]], columns=NAMES, index=["same", "same"])
    estimator = TrustworthinessEstimator(aggregation=QuadraticMechanism()).fit(metric_weights=WEIGHTS)
    output = estimator.transform(data[NAMES[::-1]])
    pd.testing.assert_frame_equal(output[NAMES[::-1]], data[NAMES[::-1]])
    np.testing.assert_allclose(output[TRUSTWORTHINESS_SCORE_NAME], [0.13, 0.68])
    assert estimator.calibrator_ is None
    assert estimator.thresholds_ is None
    assert estimator.structured_results_ is estimator.aggregation_results_
    assert estimator.mechanism_.fit_calls_ == 1
    assert all(not result.higher_is_better for result in estimator.structured_results_)
    assert estimator.compute_metric(metric_list([0.2, 0.7])) == pytest.approx(0.13)
    assert estimator.compute_metric(dict(zip(NAMES, [0.2, 0.7]))) == pytest.approx(0.13)
    assert estimator.mechanism_.fit_calls_ == 1


def test_plugin_accepts_constant_reference_and_nonstandard_metric_names():
    # The framework does not impose triangles, known built-in names, or a range
    # on a mechanism's output. Input metrics retain TWEEF's [0,1] contract.
    data = pd.DataFrame({"custom_metric": [0.5, 0.5]})
    estimator = TrustworthinessEstimator(aggregation=QuadraticMechanism(factor=100)).fit(
        metrics_values=data, metric_weights={"custom_metric": 1},
    )
    assert estimator.transform(data)[TRUSTWORTHINESS_SCORE_NAME].tolist() == [25, 25]


def test_constructor_and_fit_snapshots_isolate_caller_mutations():
    mechanism = QuadraticMechanism()
    source = reference()
    estimator = TrustworthinessEstimator(aggregation=mechanism, metrics_values=source)
    mechanism.factor = 100
    source.iloc[:] = 0.5
    estimator.fit(metric_weights=WEIGHTS)
    assert estimator.mechanism_ is not mechanism
    pd.testing.assert_frame_equal(estimator.reference_metrics_, reference())
    assert estimator.compute_metric(dict(zip(NAMES, [0.2, 0.7]))) == pytest.approx(0.13)
    estimator.mechanism_.reference_.iloc[:] = 0.2
    pd.testing.assert_frame_equal(estimator.reference_metrics_, reference())


def test_fitted_state_is_independent_across_estimators_and_serializable():
    mechanism = QuadraticMechanism()
    first = TrustworthinessEstimator(aggregation=mechanism).fit(metric_weights=WEIGHTS)
    second = TrustworthinessEstimator(aggregation=mechanism).fit(metric_weights=WEIGHTS)
    assert first.mechanism_ is not second.mechanism_
    first.mechanism_.factor = 2
    data = pd.DataFrame([[0.2, 0.7]], columns=NAMES)
    assert first.transform(data)[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(0.26)
    assert second.transform(data)[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(0.13)
    restored = pickle.loads(pickle.dumps(second))
    pd.testing.assert_frame_equal(restored.transform(data), second.transform(data))
    cloned = clone(second)
    with pytest.raises(NotFittedError):
        cloned.transform(data)
    cloned.fit(metric_weights=WEIGHTS)
    pd.testing.assert_frame_equal(cloned.transform(data), second.transform(data))


def test_nested_configuration_change_requires_refit():
    estimator = TrustworthinessEstimator(aggregation=QuadraticMechanism()).fit(metric_weights=WEIGHTS)
    estimator.set_params(aggregation__factor=2)
    with pytest.raises(NotFittedError):
        estimator.transform(pd.DataFrame([[0.2, 0.7]], columns=NAMES))
    estimator.fit(metric_weights=WEIGHTS)
    assert estimator.compute_metric(dict(zip(NAMES, [0.2, 0.7]))) == pytest.approx(0.26)


@pytest.mark.parametrize("arguments", [
    {"alpha": 0.7}, {"gwa_hesitation": 0.1},
    {"trustworthiness_score_scale": "linguistic"},
    {"fuzzy_thresholds": {NAMES[0]: [0, 0.25, 0.5, 0.75, 1]}},
])
def test_object_and_nondefault_global_parameters_are_rejected(arguments):
    with pytest.raises(ValueError, match="object owns"):
        TrustworthinessEstimator(aggregation=LWA(), **arguments)


@pytest.mark.parametrize("weights", [{}, {"a": 0}, {"a": -1}, {"a": np.nan}, {"a": np.inf}])
def test_invalid_weight_mapping_is_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        TrustworthinessEstimator(aggregation=QuadraticMechanism()).fit(metric_weights=weights)


def test_relative_weights_are_normalized_and_scalar_mismatch_is_rejected():
    estimator = TrustworthinessEstimator(aggregation=QuadraticMechanism()).fit(
        metric_weights=dict(zip(NAMES, [8, 2])),
    )
    assert estimator.weights_ == WEIGHTS
    with pytest.raises(ValueError, match="match the fitted"):
        estimator.compute_metric(metric_list([0.2, 0.7], dict.fromkeys(NAMES, 0.5)))


@pytest.mark.parametrize("bad_result", [
    0.5,
    MechanismResult(np.nan),
    MechanismResult(2, score_range=(0, 1)),
    MechanismResult(0.5, score_range=(1, 0)),
    MechanismResult(0.5, higher_is_better="yes"),
    MechanismResult(0.5, mechanism=""),
])
def test_invalid_plugin_results_are_rejected(bad_result):
    class InvalidMechanism(QuadraticMechanism):
        def aggregate(self, metrics):
            return bad_result

    estimator = TrustworthinessEstimator(aggregation=InvalidMechanism()).fit(metric_weights=WEIGHTS)
    with pytest.raises((ValueError, TypeError), match="MechanismResult|score|higher_is_better|identifier"):
        estimator.transform(pd.DataFrame([[0.2, 0.7]], columns=NAMES))


def test_plugin_declaring_no_weights_cannot_silently_ignore_requested_importance():
    class Unweighted(QuadraticMechanism):
        supports_weights = False

    with pytest.raises(ValueError, match="does not support"):
        TrustworthinessEstimator(aggregation=Unweighted()).fit(metric_weights=WEIGHTS)


def test_quartile_calibration_clones_and_freezes_reference_and_returned_crests():
    source = reference()
    calibration = QuartileCalibration().fit(source)
    source.iloc[:] = 0.3
    expected = calibration.thresholds_for(NAMES[0])
    returned = calibration.thresholds_for(NAMES[0])
    returned.iloc[:] = 0.1
    np.testing.assert_allclose(expected, [0, 0.15, 0.5, 0.85, 1])
    np.testing.assert_array_equal(calibration.thresholds_for(NAMES[0]), expected)
    with pytest.raises(NotFittedError):
        clone(calibration).thresholds_for(NAMES[0])
    with pytest.raises(ValueError, match="constant calibration"):
        QuartileCalibration().fit(pd.DataFrame({NAMES[0]: [0.3] * 5}))


@pytest.mark.parametrize("mechanism", [LWA(), GWA(), SubjectiveLogic()])
def test_zero_weight_reference_may_be_missing_for_builtin_objects(mechanism):
    estimator = TrustworthinessEstimator(aggregation=mechanism).fit(
        metrics_values=reference()[[NAMES[0]]], metric_weights=dict(zip(NAMES, [1, 0])),
    )
    data = pd.DataFrame([[0.7, 0.9]], columns=NAMES)
    assert np.isfinite(estimator.transform(data)[TRUSTWORTHINESS_SCORE_NAME].iloc[0])


def test_builtin_metadata_records_configuration_without_fitted_state():
    calibration = QuartileCalibration()
    estimator = TrustworthinessEstimator(aggregation=GWA(
        calibration=calibration, hesitation={NAMES[0]: 0.1},
    )).fit(metrics_values=reference(), metric_weights=WEIGHTS)
    metadata = estimator.mechanism_metadata_
    assert metadata["class_path"] == "TWEEF.Aggregation.GWA"
    assert metadata["config_source"] == "get_params"
    assert metadata["config"]["hesitation"] == {NAMES[0]: 0.1}
    assert isinstance(metadata["config"]["calibration"], QuartileCalibration)
    assert not hasattr(metadata["config"]["calibration"], "calibrator_")
    assert metadata["supports_weights"] is True
    assert metadata["seed"] is None
    assert "version" in metadata and "package_version" in metadata
    assert metadata["metric_weights"] == WEIGHTS
    estimator.mechanism_.hesitation[NAMES[0]] = 0.8
    assert metadata["config"]["hesitation"] == {NAMES[0]: 0.1}
    assert not hasattr(clone(estimator), "mechanism_metadata_")


def test_plugin_metadata_records_get_config_version_seed_and_independent_snapshot():
    class ConfiguredPlugin:
        __version__ = "experiment-1"
        supports_weights = True

        def __init__(self):
            self.random_state = 37
            self.options = {"power": [2]}

        def get_config(self):
            return {"options": self.options, "random_state": self.random_state}

        def fit(self, reference_metrics, *, weights):
            self.weights_ = weights
            return self

        def aggregate(self, metrics):
            return MechanismResult(sum(self.weights_[name] * value for name, value in metrics.items()))

    estimator = TrustworthinessEstimator(aggregation=ConfiguredPlugin()).fit(metric_weights=WEIGHTS)
    metadata = estimator.mechanism_metadata_
    assert metadata["config_source"] == "get_config"
    assert metadata["version"] == "experiment-1"
    assert metadata["version_source"] == "declared"
    assert metadata["seed"] == 37
    estimator.mechanism_.options["power"][0] = 3
    assert metadata["config"]["options"]["power"] == [2]
    assert metadata["supports_weights"] is True


def test_legacy_and_plain_plugin_metadata_are_available_without_version_invention():
    class PlainPlugin:
        def __init__(self):
            self.power = 2

        def fit(self, reference_metrics, *, weights):
            self.weights_ = weights
            return self

        def aggregate(self, metrics):
            return MechanismResult(0.5)

    plugin = TrustworthinessEstimator(aggregation=PlainPlugin()).fit(metric_weights=WEIGHTS)
    metadata = plugin.mechanism_metadata_
    assert metadata["config_source"] == "public_attributes"
    assert metadata["config"] == {"power": 2}
    assert metadata["version"] is None
    assert metadata["seed"] is None
    legacy = TrustworthinessEstimator(aggregation="sl", alpha=0.7).fit(metric_weights=WEIGHTS)
    assert legacy.mechanism_metadata_["class_path"] == "TWEEF.Aggregation.TrustAggregator"
    assert legacy.mechanism_metadata_["config"]["operator"] == "sl"
    assert legacy.mechanism_metadata_["config"]["alpha"] == 0.7
    legacy.set_params(aggregation="gwa", gwa_hesitation=0.2)
    assert legacy.mechanism_metadata_["config"]["operator"] == "gwa"
    assert legacy.mechanism_metadata_["config"]["gwa_hesitation"] == 0.2
