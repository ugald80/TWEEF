"""Reject invalid experimental inputs before opinion/fuzzy calculations."""

import pytest

from TWEEF.Aggregation import TrustAggregator
from TWEEF.Constants import METRICS
from TWEEF.Metrics import MetricList


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "invalid"])
def test_nonfinite_or_nonnumeric_values_are_rejected(operator, value):
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, 1.0, 0.0, value)
    with pytest.raises(ValueError, match="finite"):
        TrustAggregator(operator=operator).aggregate(metrics)


def test_mutated_nan_weight_is_rejected():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, 1.0, 0.0, 0.8)
    metrics.get_metrics()[0].weight = float("nan")
    with pytest.raises(ValueError, match="finite"):
        TrustAggregator().aggregate(metrics)
