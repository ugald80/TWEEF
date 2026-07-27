from TWEEF.EstimatorInterpretability import InterpretabilityEstimator
from TWEEF.EstimatorFairness import FairnessEstimator
from TWEEF.EstimatorPerformance import PerformanceEstimator
from TWEEF.MatrixScores import MatrixScores
from TWEEF.Constants import METRICS
from TWEEF.Metrics import Metric
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pytest


def test_matrix_scores_handles_single_class_predictions():
    scores = MatrixScores()

    scores.fit([0, 0, 0], [0, 0, 0])

    assert scores.tn == 3
    assert scores.tp == 0
    assert scores.acc() == 1


def test_performance_estimator_supports_specificity_and_npv():
    estimator = PerformanceEstimator(
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
    )
    specificity = Metric(METRICS.PERFORMANCE.SPECIFICITY, weight=0.5, goal=0.8)
    npv = Metric(METRICS.PERFORMANCE.NPV, weight=0.5, goal=0.8)

    assert estimator.compute_metric(specificity) == 0.75
    assert estimator.compute_metric(npv) == 0.75


def test_fairness_estimator_matches_paper_definitions():
    estimator = FairnessEstimator(
        protected_y_test=[1, 1, 1, 0, 0, 0],
        protected_y_predict=[1, 0, 0, 1, 0, 0],
        unprotected_y_test=[1, 1, 1, 0, 0, 0],
        unprotected_y_predict=[1, 1, 0, 1, 1, 0],
    )

    assert estimator.treatment_equality() == (1 / 2) - (2 / 1)
    assert estimator.score_treatment_equality == 0
    assert estimator.equal_opportunity() == (1 / 3) - (2 / 3)
    assert estimator.score_equal_opportunity == 1 - abs((1 / 3) - (2 / 3))
    assert estimator.equalized_odds() == abs((1 / 3) - (2 / 3)) + abs((1 / 3) - (2 / 3))
    assert estimator.score_equalized_odds == 1 - (
        abs((1 / 3) - (2 / 3)) + abs((1 / 3) - (2 / 3))
    )


def test_interpretability_metrics_are_normalized():
    estimator = InterpretabilityEstimator([0.0, 0.2, 0.8], [0.0, 0.1, 0.9])
    metric = Metric(
        METRICS.INTERPRETABILITY.NON_SENSITIVITY,
        weight=1.0,
        goal=0.8,
    )

    value = estimator.compute_metric(metric)

    assert 0 <= value <= 1


def test_non_sensitivity_uses_zero_attribution_and_expectation_sets():
    estimator = InterpretabilityEstimator(
        [0.0, 0.2, -0.3, 0.0],
        [0.0, 0.0, 0.4, 0.0],
    )
    metric = Metric(
        METRICS.INTERPRETABILITY.NON_SENSITIVITY,
        weight=1.0,
        goal=0.8,
    )

    assert estimator.compute_metric(metric) == pytest.approx(0.75)


def test_monotonicity_penalizes_inverse_rank_correlation():
    estimator = InterpretabilityEstimator([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    metric = Metric(
        METRICS.INTERPRETABILITY.MONOTONICITY,
        weight=1.0,
        goal=0.8,
    )

    assert estimator.compute_metric(metric) == 0


def test_effective_complexity_uses_cumulative_dominant_attributions():
    estimator = InterpretabilityEstimator(
        [0.90, 0.04, 0.03, 0.03],
        [0.90, 0.04, 0.03, 0.03],
        ec_threshold=0.90,
    )
    metric = Metric(
        METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY,
        weight=1.0,
        goal=0.8,
    )

    value = estimator.compute_metric(metric)

    assert value == 0.75


def test_effective_complexity_accepts_precomputed_loss_based_score():
    estimator = InterpretabilityEstimator(
        [0.9, 0.1],
        [0.9, 0.1],
        effective_complexity_mode="loss",
        effective_complexity_score=0.5,
    )
    metric = Metric(
        METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY,
        weight=1.0,
        goal=0.8,
    )

    assert estimator.compute_metric(metric) == 0.5


def test_loss_based_effective_complexity_uses_model_restriction():
    pytest.importorskip("interpret")
    from TWEEF.ProcessorInterpretability import InterpretabilityProcessing

    X = pd.DataFrame(
        {
            "important": [0, 0, 1, 1, 0, 1],
            "irrelevant": [0, 1, 0, 1, 1, 0],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    processor = InterpretabilityProcessing(
        effective_complexity_mode="loss",
        effective_complexity_loss_threshold=0.05,
        effective_complexity_repeats=2,
        random_state=0,
    )
    processor.attributions = np.array([1.0, 0.0])

    score = processor._loss_based_effective_complexity(model, X)

    assert score == 0.5
