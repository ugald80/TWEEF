"""Trust aggregation operators used by TWEEF."""

from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Real

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
        gwa_hesitation=0.0,
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
        if not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in the range [0, 1].")
        self.operator = operator
        self.calibrator = calibrator or FuzzyCalibrator()
        self.alpha = alpha
        self.score_scale = normalize_score_scale(score_scale)
        # Copy mappings so later caller mutations cannot change a configured operator.
        self.gwa_hesitation = _validate_hesitation(gwa_hesitation)

    def aggregate(self, metrics: MetricList) -> AggregationResult:
        validate_metric_list(metrics, require_values=True)
        if self.operator == AGGREGATION.GWA and isinstance(self.gwa_hesitation, Mapping):
            unknown_names = set(self.gwa_hesitation).difference(metrics.get_names())
            if unknown_names:
                raise ValueError(f"unknown gwa_hesitation metrics: {sorted(unknown_names)!r}")
        if self.operator == AGGREGATION.LWA:
            return self._aggregate_lwa(metrics)
        if self.operator == AGGREGATION.GWA:
            return self._aggregate_gwa(metrics)
        return self._aggregate_subjective_logic(metrics)

    def _metric_arrays(self, metrics: MetricList):
        # Zero importance means no evidence, including no calibration requirement.
        metric_items = [metric for metric in metrics.get_metrics() if metric.weight > 0]
        names = [metric.name for metric in metric_items]
        values = np.array([metric.value for metric in metric_items], dtype=float)
        weights = np.array([metric.weight for metric in metric_items], dtype=float)
        weights = weights / weights.sum()
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
        hesitation = []
        for name, value in zip(names, values):
            vector = self.calibrator.membership_vector(name, value)
            favorable, unfavorable = _intuitionistic_evidence(vector)
            h = (
                self.gwa_hesitation.get(name, 0.0)
                if isinstance(self.gwa_hesitation, Mapping)
                else self.gwa_hesitation
            )
            hesitation.append(h)
            mu.append((1.0 - h) * favorable)
            nu.append((1.0 - h) * unfavorable)

        mu = np.asarray(mu, dtype=float)
        nu = np.asarray(nu, dtype=float)
        # For normalized weights the published normalization product is erf(1)^0.
        # Evaluate products in log space and treat zero factors analytically.
        mu_argument = _weighted_product(erf(1.0 - mu), weights)
        nu_argument = _weighted_product(erf(nu), weights)
        mu_agg = float(1.0 - erfinv(mu_argument))
        nu_agg = float(erfinv(nu_argument))
        tolerance = 1e-12
        if (
            not np.isfinite(mu_agg + nu_agg)
            or min(mu_agg, nu_agg) < -tolerance
            or mu_agg + nu_agg > 1.0 + tolerance
        ):
            raise ArithmeticError("Gaussian aggregation produced an invalid IFV.")
        # Only remove floating-point noise after checking the mathematical domain.
        mu_agg = float(np.clip(mu_agg, 0.0, 1.0))
        nu_agg = float(np.clip(nu_agg, 0.0, 1.0 - mu_agg))
        if all(h == 1.0 for h in hesitation):
            mu_agg, nu_agg = 0.0, 0.0
        pi_agg = max(0.0, 1.0 - mu_agg - nu_agg)
        denominator = mu_agg + nu_agg
        normalized_score = 0.5 if denominator <= tolerance else mu_agg / denominator
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
            details={
                "mu": dict(zip(names, mu.tolist())),
                "nu": dict(zip(names, nu.tolist())),
                "hesitation": dict(zip(names, hesitation)),
                "weights": dict(zip(names, weights.tolist())),
                "fused_ifv": {"mu": mu_agg, "nu": nu_agg, "hesitation": pi_agg},
                "hesitation_mapping": "proportional_evidence",
            },
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

        belief, disbelief, uncertainty = _weighted_evidence_fusion(opinions, weights)
        weighted_expectation = float(
            np.dot(weights, [b + self.alpha * u for b, _, u in opinions])
        )
        fused_expectation = belief + self.alpha * uncertainty
        normalized_score = float(np.clip(fused_expectation, 0.0, 1.0))
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
                "weights": dict(zip(names, weights.tolist())),
                "fusion_method": "importance_weighted_evidence_average",
                "evidence_prior_weight": 2.0,
            },
        )


def _validate_hesitation(hesitation):
    """Validate a scalar or partial per-metric hesitation mapping."""
    if isinstance(hesitation, Mapping):
        if not all(isinstance(name, str) for name in hesitation):
            raise ValueError("gwa_hesitation mapping keys must be metric names.")
        return {name: _validate_hesitation_value(value) for name, value in hesitation.items()}
    return _validate_hesitation_value(hesitation)


def _validate_hesitation_value(value):
    if not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gwa_hesitation values must be finite and in [0, 1].")
    return float(value)


def _weighted_product(values, weights):
    if np.any(values == 0):
        return 0.0
    return float(np.exp(np.dot(weights, np.log(values))))


def _weighted_evidence_fusion(opinions, weights):
    """Average Dirichlet evidence with externally assigned importance weights.

    For prior weight W=2, r_i=W*b_i/u_i and s_i=W*d_i/u_i.
    Average r and s with normalized importance weights and map back to an
    opinion. This is an importance-weighted extension of averaging fusion,
    rather than cumulative consensus or confidence-weighted belief fusion.
    Dogmatic opinions (u=0) use the common-rate zero-uncertainty limit: they
    dominate finite evidence and are averaged with renormalized importance.
    """
    opinions = np.asarray(opinions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        opinions.ndim != 2
        or opinions.shape != (weights.size, 3)
        or weights.ndim != 1
        or not np.all(np.isfinite(opinions))
        or not np.all(np.isfinite(weights))
        or np.any(opinions < 0)
        or np.any(weights < 0)
        or not np.allclose(opinions.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("weighted evidence fusion requires valid opinions and normalized weights.")
    active = weights > 0
    opinions, weights = opinions[active], weights[active]
    weights = weights / weights.sum()
    dogmatic = opinions[:, 2] == 0
    if np.any(dogmatic):
        dogmatic_weights = weights[dogmatic] / weights[dogmatic].sum()
        belief, disbelief = np.dot(dogmatic_weights, opinions[dogmatic, :2])
        return float(belief), float(disbelief), 0.0

    # Equivalent to averaging evidence, scaled by min(u) to avoid division
    # overflow for opinions with very small positive uncertainty.
    minimum_uncertainty = opinions[:, 2].min()
    scaled_evidence = weights * (minimum_uncertainty / opinions[:, 2])
    denominator = scaled_evidence.sum()
    belief, disbelief = np.dot(scaled_evidence, opinions[:, :2]) / denominator
    uncertainty = minimum_uncertainty / denominator
    return float(belief), float(disbelief), float(uncertainty)


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
