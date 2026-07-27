import pandas as pd
import pytest

from TWEEF.Aggregation import (
    AGGREGATION,
    TRUSTWORTHINESS_SCORE_SCALE,
    TrustAggregator,
)
from TWEEF.Calibration import FuzzyCalibrator
from TWEEF.Constants import METRICS
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.Metrics import MetricList
from TWEEF.Validation import validate_metric_list


def _metrics():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.5, goal=0.8, value=0.85)
    metrics.add_metric(
        METRICS.FAIRNESS.STATISTICAL_PARITY, weight=0.5, goal=0.8, value=0.9
    )
    return metrics


def test_validate_metric_list_requires_weights_sum_to_one():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.6, goal=0.8)
    metrics.add_metric(METRICS.PERFORMANCE.RECALL, weight=0.6, goal=0.8)

    with pytest.raises(ValueError, match="sum to 1"):
        validate_metric_list(metrics)


def test_lwa_returns_normalized_score():
    result = TrustAggregator(operator=AGGREGATION.LWA).aggregate(_metrics())

    assert result.operator == AGGREGATION.LWA
    assert 0 <= result.score <= 1
    assert result.score > 0.5
    assert result.score == result.normalized_score
    assert -2 <= result.linguistic_score <= 2


def test_lwa_can_return_linguistic_score_scale():
    result = TrustAggregator(
        operator=AGGREGATION.LWA,
        score_scale=TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
    ).aggregate(_metrics())

    assert result.score_scale == TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC
    assert -2 <= result.score <= 2
    assert result.score == result.linguistic_score
    assert 0 <= result.normalized_score <= 1


def test_gwa_returns_normalized_score():
    result = TrustAggregator(operator=AGGREGATION.GWA).aggregate(_metrics())

    assert result.operator == AGGREGATION.GWA
    assert 0 <= result.score <= 1


def test_gwa_can_return_linguistic_score_scale():
    result = TrustAggregator(
        operator=AGGREGATION.GWA,
        score_scale=TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
    ).aggregate(_metrics())

    assert result.score_scale == TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC
    assert -2 <= result.score <= 2
    assert result.score == result.linguistic_score


def test_subjective_logic_returns_normalized_score():
    result = TrustAggregator(operator=AGGREGATION.SUBJECTIVE_LOGIC).aggregate(
        _metrics()
    )

    assert result.operator == AGGREGATION.SUBJECTIVE_LOGIC
    assert 0 <= result.score <= 1


def test_subjective_logic_can_return_linguistic_score_scale():
    result = TrustAggregator(
        operator=AGGREGATION.SUBJECTIVE_LOGIC,
        score_scale=TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
    ).aggregate(_metrics())

    assert result.score_scale == TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC
    assert -2 <= result.score <= 2
    assert result.score == result.linguistic_score


def test_data_driven_calibration_uses_metric_values():
    values = pd.DataFrame(
        {
            METRICS.PERFORMANCE.ACCURACY: [0.2, 0.4, 0.6, 0.8, 1.0],
            METRICS.FAIRNESS.STATISTICAL_PARITY: [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )
    calibrator = FuzzyCalibrator(metrics_values=values)

    thresholds = calibrator.thresholds_for(METRICS.PERFORMANCE.ACCURACY)

    assert thresholds.iloc[0] == 0.2
    assert thresholds.iloc[-1] == 1.0


def test_calibration_keeps_values_above_best_threshold_at_best_linguistic_term():
    vector = FuzzyCalibrator().membership_vector(METRICS.PERFORMANCE.ACCURACY, 0.95)

    assert vector.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_trustworthiness_estimator_uses_selected_aggregator():
    estimator = TrustworthinessEstimator(aggregation=AGGREGATION.LWA)

    score = estimator.compute_metric(_metrics())

    assert 0 <= score <= 1
    assert estimator.aggregation_result_.operator == AGGREGATION.LWA


def test_trustworthiness_estimator_can_return_linguistic_scale():
    estimator = TrustworthinessEstimator(
        aggregation=AGGREGATION.LWA,
        trustworthiness_score_scale=TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
    )

    score = estimator.compute_metric(_metrics())

    assert -2 <= score <= 2
    assert estimator.aggregation_result_.score == estimator.aggregation_result_.linguistic_score


@pytest.mark.parametrize(
    "operator",
    [AGGREGATION.LWA, AGGREGATION.GWA, AGGREGATION.SUBJECTIVE_LOGIC],
)
def test_aggregators_are_monotonic_when_metric_improves(operator):
    low = MetricList()
    low.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.5, goal=0.8, value=0.65)
    low.add_metric(
        METRICS.FAIRNESS.STATISTICAL_PARITY, weight=0.5, goal=0.8, value=0.8
    )
    high = MetricList()
    high.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.5, goal=0.8, value=0.9)
    high.add_metric(
        METRICS.FAIRNESS.STATISTICAL_PARITY, weight=0.5, goal=0.8, value=0.8
    )

    aggregator = TrustAggregator(operator=operator)

    assert aggregator.aggregate(high).score >= aggregator.aggregate(low).score
