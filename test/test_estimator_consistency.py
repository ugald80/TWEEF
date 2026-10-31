import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

from TWEEF.Constants import METRICS, TRUSTWORTHINESS_SCORE_NAME
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.Metrics import MetricList


NAMES = [METRICS.PERFORMANCE.ACCURACY, METRICS.FAIRNESS.EQUAL_OPPORTUNITY]


def metric_list(values=None):
    metrics = MetricList()
    for index, (name, weight) in enumerate(zip(NAMES, [0.8, 0.2])):
        metrics.add_metric(name, weight, 0.8, None if values is None else values[index])
    return metrics


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
@pytest.mark.parametrize("scale", ["normalized", "linguistic"])
def test_batch_and_scalar_use_same_operator_calibration_and_weights(operator, scale):
    reference = pd.DataFrame([[0.0, 0.2], [0.2, 0.4], [0.8, 0.7], [1.0, 1.0]], columns=NAMES)
    data = pd.DataFrame([[0.34, 0.58], [0.91, 0.79]], columns=NAMES, index=["model", "model"])
    estimator = TrustworthinessEstimator(
        aggregation=operator,
        trustworthiness_score_scale=scale,
        gwa_hesitation={NAMES[0]: 0.25, NAMES[1]: 0.1},
    ).fit(metrics_values=reference, metric_list=metric_list())
    result = estimator.transform(data[NAMES[::-1]])
    expected = [estimator.compute_metric(metric_list(row)) for row in data.to_numpy()]
    np.testing.assert_allclose(result[TRUSTWORTHINESS_SCORE_NAME], expected, atol=1e-12, rtol=0)
    assert result.index.equals(data.index)
    assert len(estimator.aggregation_results_) == len(data)
    assert all(value.operator == operator for value in estimator.aggregation_results_)
    # Supplying weights must not silently select default crests.
    np.testing.assert_allclose(estimator.thresholds_[NAMES[0]], [0.0, 0.15, 0.5, 0.85, 1.0])


def test_explicit_crests_initialize_uniform_batch_weights():
    estimator = TrustworthinessEstimator().fit(
        fuzzy_functions_thresholds={name: [0, 0.25, 0.5, 0.75, 1] for name in NAMES}
    )
    result = estimator.transform(pd.DataFrame([[0.75, 0.25]], columns=NAMES))
    assert estimator.weights_ == dict.fromkeys(NAMES, 0.5)
    assert result[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(0.5)


def test_constructor_calibration_is_available_to_fit_and_stays_frozen():
    reference = pd.DataFrame([[0, 0], [0.2, 0.3], [0.7, 0.6], [1, 1]], columns=NAMES)
    estimator = TrustworthinessEstimator(metrics_values=reference).fit(metric_list=metric_list())
    expected = estimator.calibrator_.thresholds_for(NAMES[0])
    reference.iloc[:] = 0.0
    np.testing.assert_array_equal(estimator.calibrator_.thresholds_for(NAMES[0]), expected)


def test_refit_replaces_the_calibrator_used_by_both_interfaces():
    estimator = TrustworthinessEstimator().fit(metric_list=metric_list())
    first = estimator.compute_metric(metric_list([0.7, 0.7]))
    estimator.fit(
        metrics_values=pd.DataFrame([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], columns=NAMES),
        metric_list=metric_list(),
    )
    second = estimator.compute_metric(metric_list([0.7, 0.7]))
    batch = estimator.transform(pd.DataFrame([[0.7, 0.7]], columns=NAMES))
    assert second != first
    assert second == batch[TRUSTWORTHINESS_SCORE_NAME].iloc[0]


def test_transform_requires_fit_and_exact_column_names():
    data = pd.DataFrame([[0.7, 0.8]], columns=NAMES)
    with pytest.raises(NotFittedError):
        TrustworthinessEstimator().transform(data)
    estimator = TrustworthinessEstimator().fit(metric_list=metric_list())
    with pytest.raises(ValueError, match="exactly the fitted"):
        estimator.transform(data.rename(columns={NAMES[0]: METRICS.PERFORMANCE.RECALL}))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1.1, -0.1])
def test_transform_rejects_invalid_metric_values(value):
    estimator = TrustworthinessEstimator().fit(metric_list=metric_list())
    with pytest.raises(ValueError, match="finite|range|in \\[0, 1\\]"):
        estimator.transform(pd.DataFrame([[value, 0.8]], columns=NAMES))


def test_scalar_rejects_non_finite_metric_values():
    with pytest.raises(ValueError, match="finite"):
        TrustworthinessEstimator().compute_metric(metric_list([np.nan, 0.8]))


def zero_weight_metrics(values=None):
    metrics = MetricList()
    for index, (name, weight) in enumerate(zip(NAMES, [1.0, 0.0])):
        metrics.add_metric(name, weight, 0.8, None if values is None else values[index])
    return metrics


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
@pytest.mark.parametrize("calibration", ["thresholds", "reference"])
def test_zero_weight_missing_calibration_has_scalar_batch_parity(operator, calibration):
    thresholds = {NAMES[0]: [0.0, 0.25, 0.5, 0.75, 1.0]}
    reference = pd.DataFrame({NAMES[0]: [0.0, 0.25, 0.5, 0.75, 1.0]})
    arguments = {"fuzzy_thresholds": thresholds} if calibration == "thresholds" else {"metrics_values": reference}
    estimator = TrustworthinessEstimator(aggregation=operator, **arguments)
    scalar = estimator.compute_metric(zero_weight_metrics([0.8, 0.1]))
    estimator.fit(metric_list=zero_weight_metrics())
    result = estimator.transform(pd.DataFrame([[0.8, 0.99]], columns=NAMES))
    assert result[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(scalar, abs=1e-12)
    assert np.isnan(result[NAMES[1]].iloc[0])
    assert set(estimator.thresholds_) == {NAMES[0]}


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
def test_inactive_constant_reference_is_ignored_and_constructor_snapshot_is_frozen(operator):
    reference = pd.DataFrame({NAMES[0]: [0, 0.25, 0.5, 0.75, 1], NAMES[1]: [0.3] * 5})
    estimator = TrustworthinessEstimator(aggregation=operator, metrics_values=reference)
    # Before first estimation, mutate the caller's active reference to a constant.
    reference[NAMES[0]] = 0.0
    expected = estimator.compute_metric(zero_weight_metrics([0.8, 0.3]))
    estimator.fit(metric_list=zero_weight_metrics())
    actual = estimator.transform(pd.DataFrame([[0.8, 0.3]], columns=NAMES))
    assert actual[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
def test_active_constant_reference_is_still_rejected(operator):
    reference = pd.DataFrame({NAMES[0]: [0.3] * 5, NAMES[1]: [0, 0.25, 0.5, 0.75, 1]})
    estimator = TrustworthinessEstimator(aggregation=operator, metrics_values=reference)
    with pytest.raises(ValueError, match="constant calibration"):
        estimator.compute_metric(zero_weight_metrics([0.3, 0.8]))
    with pytest.raises(ValueError, match="constant calibration"):
        estimator.fit(metric_list=zero_weight_metrics())


def test_extra_unknown_calibration_metric_is_rejected_for_both_interfaces():
    reference = pd.DataFrame({
        NAMES[0]: [0.0, 1.0],
        METRICS.PERFORMANCE.PRECISION: [0.0, 1.0],
    })
    estimator = TrustworthinessEstimator(metrics_values=reference)
    with pytest.raises(ValueError, match="unknown calibration metrics"):
        estimator.compute_metric(zero_weight_metrics([0.8, 0.3]))
    with pytest.raises(ValueError, match="unknown calibration metrics"):
        estimator.fit(metric_list=zero_weight_metrics())


def test_set_params_refreshes_reference_snapshot_and_invalidates_previous_fit():
    reference = pd.DataFrame([[0, 0], [0.2, 0.2], [0.8, 0.8], [1, 1]], columns=NAMES)
    estimator = TrustworthinessEstimator(metrics_values=reference).fit(metric_list=metric_list())
    first = estimator.compute_metric(metric_list([0.7, 0.7]))
    replacement = pd.DataFrame([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], columns=NAMES)
    assert estimator.set_params(metrics_values=replacement) is estimator
    replacement.iloc[:] = 0.0
    with pytest.raises(NotFittedError):
        estimator.transform(pd.DataFrame([[0.7, 0.7]], columns=NAMES))
    # Both direct estimation and refit must use the replacement snapshot.
    second = estimator.compute_metric(metric_list([0.7, 0.7]))
    estimator.fit(metric_list=metric_list())
    batch = estimator.transform(pd.DataFrame([[0.7, 0.7]], columns=NAMES))
    assert first != second
    assert batch[TRUSTWORTHINESS_SCORE_NAME].iloc[0] == pytest.approx(second)
    np.testing.assert_allclose(estimator.thresholds_[NAMES[0]], [0, 0.075, 0.15, 0.225, 0.3])


def test_set_params_refreshes_explicit_threshold_snapshot_and_allows_refit():
    estimator = TrustworthinessEstimator().fit(metric_list=metric_list())
    replacement = {name: [0, 0.1, 0.2, 0.3, 0.4] for name in NAMES}
    estimator.set_params(fuzzy_thresholds=replacement)
    replacement[NAMES[0]][-1] = 0.0
    expected = estimator.compute_metric(metric_list([0.7, 0.7]))
    estimator.fit(metric_list=metric_list())
    assert estimator.compute_metric(metric_list([0.7, 0.7])) == expected
    np.testing.assert_allclose(estimator.thresholds_[NAMES[0]], [0, 0.1, 0.2, 0.3, 0.4])


def test_set_params_operator_preserves_calibration_and_clone_remains_unfitted():
    reference = pd.DataFrame([[0, 0], [0.2, 0.3], [0.8, 0.7], [1, 1]], columns=NAMES)
    estimator = TrustworthinessEstimator(metrics_values=reference).fit(metric_list=metric_list())
    calibrator = estimator.calibrator_
    estimator.set_params(aggregation="gwa", alpha=0.7, gwa_hesitation=0.2)
    result = estimator.transform(pd.DataFrame([[0.7, 0.7]], columns=NAMES))
    assert estimator.calibrator_ is calibrator
    assert result[TRUSTWORTHINESS_SCORE_NAME].notna().all()
    cloned = clone(estimator)
    assert cloned.get_params()["aggregation"] == "gwa"
    assert cloned.get_params()["gwa_hesitation"] == 0.2
    with pytest.raises(NotFittedError):
        cloned.transform(pd.DataFrame([[0.7, 0.7]], columns=NAMES))
    cloned.fit(metric_list=metric_list())
    assert cloned.compute_metric(metric_list([0.7, 0.7])) == estimator.compute_metric(metric_list([0.7, 0.7]))


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
def test_scalar_subset_does_not_mutate_fitted_calibrator_used_by_batch(operator):
    estimator = TrustworthinessEstimator(aggregation=operator).fit(metric_list=metric_list())
    data = pd.DataFrame([[0.7, 0.8]], columns=NAMES)
    before = estimator.transform(data)[TRUSTWORTHINESS_SCORE_NAME].iloc[0]
    calibrator = estimator.calibrator_
    estimator.compute_metric(zero_weight_metrics([0.7, 0.8]))
    assert estimator.calibrator_ is calibrator
    after = estimator.transform(data)[TRUSTWORTHINESS_SCORE_NAME].iloc[0]
    assert after == pytest.approx(before)
