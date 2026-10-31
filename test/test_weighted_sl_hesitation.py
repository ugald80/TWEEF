"""Analytical regression tests for weighted evidence and independent hesitation."""

import numpy as np
import pytest
from scipy.special import erf, erfinv

from TWEEF.Aggregation import TrustAggregator, _weighted_evidence_fusion
from TWEEF.Calibration import FuzzyCalibrator
from TWEEF.Constants import METRICS
from TWEEF.Metrics import MetricList


ACCURACY = METRICS.PERFORMANCE.ACCURACY
FAIRNESS = METRICS.FAIRNESS.STATISTICAL_PARITY


def metrics(values=(0.8, 0.4), weights=(0.5, 0.5), reverse=False):
    items = list(zip((ACCURACY, FAIRNESS), values, weights))
    result = MetricList()
    for name, value, weight in reversed(items) if reverse else items:
        result.add_metric(name, weight, 0.0, value)
    return result


def uniform_calibrator():
    return FuzzyCalibrator(thresholds={
        ACCURACY: [0.0, 0.25, 0.5, 0.75, 1.0],
        FAIRNESS: [0.0, 0.25, 0.5, 0.75, 1.0],
    })


@pytest.mark.parametrize("weights, opinion, score", [
    ((0.8, 0.2), (0.5, 0.3, 0.2), 0.6),
    ((0.2, 0.8), (0.2, 0.6, 0.2), 0.3),
])
def test_sl_importance_weights_have_analytical_effect(weights, opinion, score):
    actual = _weighted_evidence_fusion([(0.6, 0.2, 0.2), (0.1, 0.7, 0.2)], weights)
    assert actual == pytest.approx(opinion, abs=1e-12)
    assert actual[0] + 0.5 * actual[2] == pytest.approx(score, abs=1e-12)


def test_sl_averages_evidence_not_projected_scores():
    opinion = _weighted_evidence_fusion([(0.6, 0.2, 0.2), (0.1, 0.3, 0.6)], [0.75, 0.25])
    assert opinion == pytest.approx((0.55, 0.21, 0.24), abs=1e-12)
    assert opinion[0] + 0.5 * opinion[2] == pytest.approx(0.67, abs=1e-12)


def test_sl_single_active_opinion_is_identity():
    opinion = (0.6, 0.2, 0.2)
    assert _weighted_evidence_fusion([opinion, (0.0, 1.0, 0.0)], [1.0, 0.0]) == pytest.approx(opinion)


def test_sl_all_vacuous_is_vacuous():
    assert _weighted_evidence_fusion([(0, 0, 1), (0, 0, 1)], [0.75, 0.25]) == (0, 0, 1)


def test_sl_dogmatic_limit_uses_positive_importance_weights():
    opinion = _weighted_evidence_fusion(
        [(0.9, 0.1, 0), (0.3, 0.7, 0), (0.6, 0.2, 0.2)], [0.2, 0.3, 0.5]
    )
    assert opinion == pytest.approx((0.54, 0.46, 0))


def test_sl_small_positive_uncertainty_does_not_overflow():
    opinion = _weighted_evidence_fusion([(0.6, 0.4, 1e-310), (0.1, 0.3, 0.6)], [0.5, 0.5])
    assert np.all(np.isfinite(opinion))
    assert sum(opinion) == pytest.approx(1)
    assert opinion[0] == pytest.approx(0.6)


def test_sl_integrated_score_changes_with_weights():
    aggregator = TrustAggregator(operator="sl", calibrator=uniform_calibrator())
    performance_priority = aggregator.aggregate(metrics(weights=(0.9, 0.1)))
    fairness_priority = aggregator.aggregate(metrics(weights=(0.1, 0.9)))
    assert performance_priority.score > fairness_priority.score
    assert performance_priority.details["fusion_method"] == "importance_weighted_evidence_average"
    assert sum(performance_priority.details["fused_opinion"].values()) == pytest.approx(1)


def test_gwa_single_metric_preserves_ifv_and_relative_score():
    one = MetricList()
    one.add_metric(ACCURACY, 1, 0, 0.8)
    result = TrustAggregator(operator="gwa", calibrator=uniform_calibrator(), gwa_hesitation=0.25).aggregate(one)
    assert result.details["mu"][ACCURACY] == pytest.approx(0.6)
    assert result.details["nu"][ACCURACY] == pytest.approx(0.15)
    assert result.details["hesitation"][ACCURACY] == 0.25
    assert result.details["fused_ifv"] == pytest.approx({"mu": 0.6, "nu": 0.15, "hesitation": 0.25})
    assert result.score == pytest.approx(0.8)


@pytest.mark.parametrize("count", [1, 2])
def test_gwa_total_hesitation_is_exactly_vacuous_and_neutral(count):
    selected = metrics() if count == 2 else metrics(weights=(1, 0))
    result = TrustAggregator(operator="gwa", calibrator=uniform_calibrator(), gwa_hesitation=1).aggregate(selected)
    assert result.score == 0.5
    assert result.details["fused_ifv"] == {"mu": 0.0, "nu": 0.0, "hesitation": 1.0}


def test_gwa_independent_hesitation_has_valid_mass_at_extreme_values():
    result = TrustAggregator(
        operator="gwa", calibrator=uniform_calibrator(), gwa_hesitation={ACCURACY: 0.4, FAIRNESS: 0.7}
    ).aggregate(metrics(values=(1, 0)))
    for name in (ACCURACY, FAIRNESS):
        total = sum(result.details[field][name] for field in ("mu", "nu", "hesitation"))
        assert total == pytest.approx(1)
    assert result.details["mu"][ACCURACY] == pytest.approx(0.6)
    assert result.details["nu"][FAIRNESS] == pytest.approx(0.3)
    assert sum(result.details["fused_ifv"].values()) == pytest.approx(1)


def test_gwa_zero_hesitation_preserves_previous_gaussian_formula():
    # Independent oracle: the original published product for this non-extreme pair.
    z = np.array([0.8, 0.4])
    weights = np.array([0.3, 0.7])
    mu = 1 - erfinv(np.prod(erf(1 - z) ** weights))
    nu = erfinv(np.prod(erf(1 - z) ** weights))
    result = TrustAggregator(operator="gwa", calibrator=uniform_calibrator()).aggregate(
        metrics(weights=tuple(weights))
    )
    assert result.score == pytest.approx(mu / (mu + nu), abs=1e-12)
    assert result.details["fused_ifv"]["hesitation"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
def test_aggregation_is_invariant_to_joint_permutation(operator):
    aggregator = TrustAggregator(operator=operator, calibrator=uniform_calibrator())
    assert aggregator.aggregate(metrics(weights=(0.7, 0.3))).score == pytest.approx(
        aggregator.aggregate(metrics(weights=(0.7, 0.3), reverse=True)).score, abs=1e-12
    )


@pytest.mark.parametrize("operator", ["lwa", "gwa", "sl"])
def test_zero_weight_needs_no_calibration_and_does_not_influence_score(operator):
    calibrator = FuzzyCalibrator(thresholds={ACCURACY: [0, 0.25, 0.5, 0.75, 1]})
    aggregator = TrustAggregator(operator=operator, calibrator=calibrator)
    assert aggregator.aggregate(metrics(values=(0.8, 0), weights=(1, 0))).score == pytest.approx(
        aggregator.aggregate(metrics(values=(0.8, 1), weights=(1, 0))).score
    )


@pytest.mark.parametrize("hesitation", [-0.1, 1.1, float("nan"), float("inf"), "0.5", {ACCURACY: -0.1}])
def test_hesitation_rejects_invalid_values(hesitation):
    with pytest.raises(ValueError, match="gwa_hesitation"):
        TrustAggregator(operator="gwa", gwa_hesitation=hesitation)


def test_partial_hesitation_mapping_defaults_to_zero_and_copies_input():
    configured = {ACCURACY: 0.25}
    aggregator = TrustAggregator(operator="gwa", calibrator=uniform_calibrator(), gwa_hesitation=configured)
    configured[ACCURACY] = 1
    result = aggregator.aggregate(metrics())
    assert result.details["hesitation"] == {ACCURACY: 0.25, FAIRNESS: 0.0}


def test_hesitation_unknown_name_is_an_error():
    with pytest.raises(ValueError, match="unknown gwa_hesitation"):
        TrustAggregator(operator="gwa", gwa_hesitation={"typo": 0.25}).aggregate(metrics())


@pytest.mark.parametrize("operator", ["gwa", "sl"])
def test_corrected_operators_preserve_output_scale_relation(operator):
    result = TrustAggregator(operator=operator, score_scale="linguistic").aggregate(metrics())
    assert result.score == pytest.approx(4 * result.normalized_score - 2)
    assert result.score == result.linguistic_score
