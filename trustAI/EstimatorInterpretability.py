

from scipy import stats

import numpy as np
from numpy import ndarray as NDArray
from trustAI.Constants import METRICS
from trustAI.Metrics import Metric

class InterpretabilityEstimator:
    """
    Classes:
        InterpretabilityEstimator: A class to estimate the interpretability of attributions using various metrics.

    Attributes:
    -----------
    attributions : NDArray
        The attributions to be evaluated.
    expectations : NDArray
        The expected values for comparison.
    ec_threshold : float, optional
        The threshold for effective complexity calculation (default is 0.05).
    score_monotonicity : float or None
        The monotonicity score, initialized as None.
    score_non_sensitivity : float or None
        The non-sensitivity score, initialized as None.
    score_effective_complexity : float or None
        The effective complexity score, initialized as None.

    Constants used:
        METRICS_INTERPRETABILITY_EFFECTIVE_COMPLEXITY: Metric for effective complexity.
        METRICS_INTERPRETABILITY_MONOTONICITY: Metric for monotonicity.
        METRICS_INTERPRETABILITY_NON_SENSITIVITY: Metric for non-sensitivity.

    Methods:
        __init__(self, attributions: NDArray, expectations: NDArray, ec_threshold=0.05):
            Initializes the InterpretabilityEstimator with attributions, expectations, and an optional effective complexity threshold.

        monotonicity(self) -> float:
            Computes the monotonicity score of the attributions. Returns a value between 0 (poor) and 1 (good).

        non_sensitivity(self, threshold=0.001) -> float:
            Computes the non-sensitivity score of the attributions. Returns a value between 0 (poor) and 1 (good).

        effective_complexity(self) -> float:
            Computes the effective complexity score of the attributions. Returns a value between 0 (good) and 1 (poor).

        interpretability(self, interpretability_metric=None) -> float:
            Computes the interpretability score based on the specified metric. Raises a ValueError for invalid metrics.
    """

    def __init__(self, attributions: NDArray, expectations: NDArray, ec_threshold=0.05):
        self.attributions = attributions
        self.expectations = expectations
        self.ec_threshold = ec_threshold

        self.score_monotonicity = self.monotonicity()
        self.score_non_sensitivity = self.non_sensitivity()
        self.score_effective_complexity = self.effective_complexity()

    def monotonicity(self):  # attribution quality. [0 malo, hasta 1 bueno]
        return abs(
            stats.spearmanr(abs(self.attributions), self.expectations).correlation
        )

    def non_sensitivity(self, threshold=0.001):  # robust [0 malo, hasta 1 bueno] ****
        attributions_indices = set(
            i for i, value in enumerate(self.attributions) if value > threshold
        )
        expectations_indices = set(
            i for i, value in enumerate(self.expectations) if value > threshold
        )
        diff = len(attributions_indices.symmetric_difference(expectations_indices))
        union = len(attributions_indices.union(expectations_indices))
        if union == 0:
            return 0
        return diff / union

    def effective_complexity(self):  # simplicity. [0 bueno, hasta 1 malo]
        sorted_attributions = abs(np.copy(self.attributions))
        sorted_attributions.sort()
        mk = sorted_attributions[sorted_attributions < self.ec_threshold].copy()
        return len(mk) / len(self.attributions)

    def compute_metric(self, interpretability_metric=Metric):
        if len(self.attributions) != len(self.expectations):
            raise ValueError("weights and scores have different lengths")

        if interpretability_metric.name == METRICS.INTERPRETABILITY.MONOTONICITY:
            interpretability_metric.value = self.score_monotonicity
        elif interpretability_metric.name == METRICS.INTERPRETABILITY.NON_SENSITIVITY:
            interpretability_metric.value = self.score_non_sensitivity
        elif interpretability_metric.name == METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY:
            interpretability_metric.value = self.score_effective_complexity
        else:
            raise ValueError(f"Invalid interpretability metric value: {interpretability_metric}")

        return interpretability_metric.value
