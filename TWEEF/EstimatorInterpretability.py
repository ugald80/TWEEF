

from scipy import stats

import numpy as np
from numpy import ndarray as NDArray
from TWEEF.Constants import METRICS
from TWEEF.Metrics import Metric

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
        Attribution coverage threshold for effective complexity calculation
        (default is 0.95).
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
            Computes the monotonicity score of the attributions. Positive
            Spearman correlation is faithful; inverse correlation is penalized.

        non_sensitivity(self, threshold=0.001) -> float:
            Computes the normalized inverse of the non-sensitivity cardinality.

        effective_complexity(self) -> float:
            Computes an interpretability score from effective complexity.
            Higher values mean fewer dominant features are needed to cover
            the target cumulative attribution mass.

        interpretability(self, interpretability_metric=None) -> float:
            Computes the interpretability score based on the specified metric. Raises a ValueError for invalid metrics.
    """

    def __init__(
        self,
        attributions: NDArray,
        expectations: NDArray,
        ec_threshold=0.95,
        effective_complexity_mode="attribution",
        effective_complexity_score=None,
    ):
        if not 0 <= ec_threshold <= 1:
            raise ValueError("ec_threshold must be in the range [0, 1]")
        if effective_complexity_mode not in {"attribution", "loss"}:
            raise ValueError(
                "effective_complexity_mode must be 'attribution' or 'loss'"
            )
        self.attributions = np.asarray(attributions, dtype=float)
        self.expectations = np.asarray(expectations, dtype=float)
        self.ec_threshold = ec_threshold
        self.effective_complexity_mode = effective_complexity_mode
        self.precomputed_effective_complexity_score = effective_complexity_score

        self.score_monotonicity = self.monotonicity()
        self.score_non_sensitivity = self.non_sensitivity()
        self.score_effective_complexity = self.effective_complexity()

    def monotonicity(self):  # attribution quality. [0 malo, hasta 1 bueno]
        correlation = stats.spearmanr(
            abs(self.attributions), self.expectations
        ).correlation
        return 0 if np.isnan(correlation) else max(correlation, 0)

    def non_sensitivity(self, threshold=0.001):
        attribution_zero_indices = set(
            i for i, value in enumerate(np.abs(self.attributions)) if value <= threshold
        )
        expectation_zero_indices = set(
            i for i, value in enumerate(np.abs(self.expectations)) if value <= threshold
        )
        diff = len(attribution_zero_indices.symmetric_difference(expectation_zero_indices))
        if len(self.attributions) == 0:
            return 0
        return 1 - (diff / len(self.attributions))

    def effective_complexity(self):
        """Return the selected effective-complexity interpretability score."""
        if self.precomputed_effective_complexity_score is not None:
            return self.precomputed_effective_complexity_score
        if self.effective_complexity_mode == "loss":
            raise ValueError(
                "loss-based effective complexity requires a precomputed score"
            )

        if len(self.attributions) == 0:
            return 0
        absolute_attributions = np.abs(np.copy(self.attributions))
        total_attribution = absolute_attributions.sum()
        if total_attribution == 0 or self.ec_threshold == 0:
            return 1

        sorted_attributions = np.sort(absolute_attributions)[::-1]
        cumulative_attributions = np.cumsum(sorted_attributions)
        target_attribution = self.ec_threshold * total_attribution
        dominant_feature_count = (
            np.searchsorted(cumulative_attributions, target_attribution, side="left")
            + 1
        )
        normalized_complexity = dominant_feature_count / len(self.attributions)
        return 1 - normalized_complexity

    def compute_metric(self, interpretability_metric=Metric):
        if len(self.attributions) != len(self.expectations):
            raise ValueError("weights and scores have different lengths")

        if interpretability_metric.name == METRICS.INTERPRETABILITY.MONOTONICITY:
            interpretability_metric.value = min(max(self.score_monotonicity, 0), 1)
        elif interpretability_metric.name == METRICS.INTERPRETABILITY.NON_SENSITIVITY:
            interpretability_metric.value = min(max(self.score_non_sensitivity, 0), 1)
        elif interpretability_metric.name == METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY:
            interpretability_metric.value = min(max(self.score_effective_complexity, 0), 1)
        else:
            raise ValueError(f"Invalid interpretability metric value: {interpretability_metric}")

        return interpretability_metric.value
