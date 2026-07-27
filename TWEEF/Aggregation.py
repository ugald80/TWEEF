"""Trust aggregation operators used by TWEEF."""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfinv

from TWEEF.Calibration import FuzzyCalibrator
from TWEEF.Constants import LINGUISTIC_VALUES
from TWEEF.Metrics import MetricList
from TWEEF.Validation import validate_metric_list


class AGGREGATION:
    LWA = "lwa"
    GWA = "gwa"
    SUBJECTIVE_LOGIC = "sl"


@dataclass(frozen=True)
class AggregationResult:
    score: float
    raw_score: float
    normalized_score: float
    linguistic_score: float
    score_scale: str
    operator: str
    details: dict


class TRUSTWORTHINESS_SCORE_SCALE:
    NORMALIZED = "normalized"
    LINGUISTIC = "linguistic"


class TrustAggregator:
    def __init__(
        self,
        operator: str = AGGREGATION.LWA,
        calibrator: FuzzyCalibrator = None,
        alpha: float = 0.5,
        score_scale: str = TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
    ):
        operator = operator.lower()
        if operator not in {
            AGGREGATION.LWA,
            AGGREGATION.GWA,
            AGGREGATION.SUBJECTIVE_LOGIC,
        }:
            raise ValueError(
                "operator must be one of 'lwa', 'gwa', or 'sl'."
            )
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in the range [0, 1].")
        self.operator = operator
        self.calibrator = calibrator or FuzzyCalibrator()
        self.alpha = alpha
        self.score_scale = normalize_score_scale(score_scale)

    def aggregate(self, metrics: MetricList) -> AggregationResult:
        validate_metric_list(metrics, require_values=True)
        if self.operator == AGGREGATION.LWA:
            return self._aggregate_lwa(metrics)
        if self.operator == AGGREGATION.GWA:
            return self._aggregate_gwa(metrics)
        return self._aggregate_subjective_logic(metrics)

    def _metric_arrays(self, metrics: MetricList):
        metric_items = metrics.get_metrics()
        names = [metric.name for metric in metric_items]
        values = np.array([metric.value for metric in metric_items], dtype=float)
        weights = np.array([metric.weight for metric in metric_items], dtype=float)
        return names, values, weights

    def _aggregate_lwa(self, metrics: MetricList) -> AggregationResult:
        names, values, weights = self._metric_arrays(metrics)
        linguistic_scores = []
        memberships = {}
        for name, value in zip(names, values):
            vector = self.calibrator.membership_vector(name, value)
            memberships[name] = vector.tolist()
            linguistic_scores.append(float(LINGUISTIC_VALUES[int(np.argmax(vector))]))

        raw_score = float(np.dot(weights, linguistic_scores))
        normalized_score = _to_normalized_scale(raw_score)
        score = _select_score_scale(normalized_score, self.score_scale)
        return AggregationResult(
            score=score,
            raw_score=raw_score,
            normalized_score=normalized_score,
            linguistic_score=raw_score,
            score_scale=self.score_scale,
            operator=self.operator,
            details={
                "linguistic_scores": dict(zip(names, linguistic_scores)),
                "memberships": memberships,
            },
        )

    def _aggregate_gwa(self, metrics: MetricList) -> AggregationResult:
        names, values, weights = self._metric_arrays(metrics)
        mu = []
        nu = []
        for name, value in zip(names, values):
            vector = self.calibrator.membership_vector(name, value)
            favorable, unfavorable = _intuitionistic_evidence(vector)
            mu.append(favorable)
            nu.append(unfavorable)

        mu = np.clip(np.array(mu, dtype=float), 0.0, 1.0 - np.finfo(float).eps)
        nu = np.clip(np.array(nu, dtype=float), 0.0, 1.0 - np.finfo(float).eps)
        normalizer = np.prod(np.power(erf(1.0), weights - (1.0 / len(weights))))
        normalizer = max(float(normalizer), np.finfo(float).eps)
        mu_argument = np.prod(np.power(erf(1.0 - mu), weights)) / normalizer
        nu_argument = np.prod(np.power(erf(nu), weights)) / normalizer
        mu_argument = np.clip(mu_argument, -1.0 + 1e-12, 1.0 - 1e-12)
        nu_argument = np.clip(nu_argument, -1.0 + 1e-12, 1.0 - 1e-12)
        mu_agg = 1.0 - erfinv(mu_argument)
        nu_agg = erfinv(nu_argument)
        mu_agg = float(np.clip(mu_agg, 0.0, 1.0))
        nu_agg = float(np.clip(nu_agg, 0.0, 1.0))
        denominator = mu_agg + nu_agg
        normalized_score = 0.5 if denominator == 0 else mu_agg / denominator
        normalized_score = float(np.clip(normalized_score, 0.0, 1.0))
        linguistic_score = _to_linguistic_scale(normalized_score)
        score = _select_score_scale(normalized_score, self.score_scale)
        return AggregationResult(
            score=score,
            raw_score=normalized_score,
            normalized_score=normalized_score,
            linguistic_score=linguistic_score,
            score_scale=self.score_scale,
            operator=self.operator,
            details={"mu": dict(zip(names, mu)), "nu": dict(zip(names, nu))},
        )

    def _aggregate_subjective_logic(self, metrics: MetricList) -> AggregationResult:
        names, values, weights = self._metric_arrays(metrics)
        opinions = []
        for name, value in zip(names, values):
            thresholds = self.calibrator.thresholds_for(name)
            uncertainty = _normalized_interquartile(thresholds)
            belief = np.clip(value - uncertainty / 2.0, 0.0, 1.0)
            disbelief = np.clip(1.0 - value - uncertainty / 2.0, 0.0, 1.0)
            opinion = _normalize_opinion((belief, disbelief, uncertainty))
            opinions.append(opinion)

        ordered_indices = np.argsort(-weights)
        belief, disbelief, uncertainty = opinions[int(ordered_indices[0])]
        for index in ordered_indices[1:]:
            opinion = opinions[int(index)]
            belief, disbelief, uncertainty = _consensus(
                (belief, disbelief, uncertainty), opinion
            )
        weighted_expectation = float(
            np.dot(weights, [b + self.alpha * u for b, _, u in opinions])
        )
        consensus_expectation = belief + self.alpha * uncertainty
        normalized_score = float(np.clip(consensus_expectation, 0.0, 1.0))
        linguistic_score = _to_linguistic_scale(normalized_score)
        score = _select_score_scale(normalized_score, self.score_scale)
        return AggregationResult(
            score=score,
            raw_score=normalized_score,
            normalized_score=normalized_score,
            linguistic_score=linguistic_score,
            score_scale=self.score_scale,
            operator=self.operator,
            details={
                "opinions": {
                    name: {"belief": b, "disbelief": d, "uncertainty": u}
                    for name, (b, d, u) in zip(names, opinions)
                },
                "fused_opinion": {
                    "belief": belief,
                    "disbelief": disbelief,
                    "uncertainty": uncertainty,
                },
                "weighted_expectation": weighted_expectation,
            },
        )


def _normalized_interquartile(thresholds) -> float:
    values = np.array(thresholds, dtype=float)
    metric_range = values.max() - values.min()
    if metric_range == 0:
        return 1.0
    return float(np.clip((values[3] - values[1]) / metric_range, 0.0, 1.0))


def _intuitionistic_evidence(membership_vector):
    values_0_1 = (np.array(LINGUISTIC_VALUES, dtype=float) + 2.0) / 4.0
    membership = np.asarray(membership_vector, dtype=float)
    favorable = float(np.dot(membership, values_0_1))
    return favorable, 1.0 - favorable


def _normalize_opinion(opinion):
    belief, disbelief, uncertainty = opinion
    total = belief + disbelief + uncertainty
    if total == 0:
        return 0.0, 0.0, 1.0
    return belief / total, disbelief / total, uncertainty / total


def _consensus(left, right):
    b1, d1, u1 = left
    b2, d2, u2 = right
    k = u1 + u2 - (u1 * u2)
    if k == 0:
        return 0.0, 0.0, 1.0
    belief = ((b1 * u2) + (b2 * u1)) / k
    disbelief = ((d1 * u2) + (d2 * u1)) / k
    uncertainty = (u1 * u2) / k
    return belief, disbelief, uncertainty


def normalize_score_scale(score_scale: str) -> str:
    aliases = {
        TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED: TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        "0_1": TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        "0-1": TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        "[0,1]": TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC: TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
        "-2_2": TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
        "-2-2": TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
        "[-2,2]": TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC,
    }
    try:
        return aliases[score_scale.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "score_scale must be 'normalized' ([0,1]) or 'linguistic' ([-2,2])."
        ) from exc


def _to_normalized_scale(linguistic_score: float) -> float:
    return float(np.clip((linguistic_score + 2.0) / 4.0, 0.0, 1.0))


def _to_linguistic_scale(normalized_score: float) -> float:
    return float(np.clip((4.0 * normalized_score) - 2.0, -2.0, 2.0))


def _select_score_scale(normalized_score: float, score_scale: str) -> float:
    if score_scale == TRUSTWORTHINESS_SCORE_SCALE.LINGUISTIC:
        return _to_linguistic_scale(normalized_score)
    return float(np.clip(normalized_score, 0.0, 1.0))
