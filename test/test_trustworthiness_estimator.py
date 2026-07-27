import pandas as pd

from TWEEF.Constants import METRICS, TRUSTWORTHINESS_SCORE_NAME
from TWEEF.Aggregation import TRUSTWORTHINESS_SCORE_SCALE
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.Metrics import MetricList


def test_transform_uses_metric_list_weights_and_normalized_score():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.5, goal=0.8)
    metrics.add_metric(
        METRICS.FAIRNESS.STATISTICAL_PARITY, weight=0.5, goal=0.8
    )
    estimator = TrustworthinessEstimator()
    estimator.fit(metric_list=metrics)
    data = pd.DataFrame(
        [
            {
                METRICS.PERFORMANCE.ACCURACY: 0.8,
                METRICS.FAIRNESS.STATISTICAL_PARITY: 0.9,
            }
        ]
    )

    result = estimator.transform(data)

    assert 0 <= result[TRUSTWORTHINESS_SCORE_NAME].iloc[0] <= 1


def test_transform_can_return_linguistic_score():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=0.5, goal=0.8)
    metrics.add_metric(
        METRICS.FAIRNESS.STATISTICAL_PARITY, weight=0.5, goal=0.8
    )
    estimator = TrustworthinessEstimator(
        trustworthiness_score_scale=TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC
    )
    estimator.fit(metric_list=metrics)
    data = pd.DataFrame(
        [
            {
                METRICS.PERFORMANCE.ACCURACY: 0.8,
                METRICS.FAIRNESS.STATISTICAL_PARITY: 0.9,
            }
        ]
    )

    result = estimator.transform(data)

    assert -2 <= result[TRUSTWORTHINESS_SCORE_NAME].iloc[0] <= 2
