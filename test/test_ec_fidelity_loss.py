"""Effective Complexity measures changes, not the reference model's entropy."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("interpret")

from TWEEF.ProcessorInterpretability import InterpretabilityProcessing


class ProbabilityFromColumn:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.asarray(X["signal"])
        return np.column_stack([1 - p, p])


class TwoFeatureProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = .1 + .35 * (np.asarray(X["first"]) + np.asarray(X["second"]))
        return np.column_stack([1 - p, p])


class LabelsFromColumn:
    def predict(self, X):
        return np.asarray(X["signal"])


@pytest.mark.parametrize("probability", [0., .01, .2, .5, .8, .99, 1.])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_identical_probabilities_have_exactly_zero_loss(probability, dtype):
    X = pd.DataFrame({"signal": np.array([probability] * 3, dtype=dtype)})
    assert InterpretabilityProcessing()._prediction_loss(ProbabilityFromColumn(), X, X.copy()) == 0.


@pytest.mark.parametrize("restricted,expected", [(.79, .00030499293160169386),
                                                (.6, .09151622184943568)])
def test_probability_fidelity_matches_analytic_bernoulli_kl(restricted, expected):
    original = pd.DataFrame({"signal": [.8, .8]})
    perturbed = pd.DataFrame({"signal": [restricted, restricted]})
    value = InterpretabilityProcessing()._prediction_loss(ProbabilityFromColumn(), original, perturbed)
    assert value == pytest.approx(expected, abs=1e-14)


def test_opposite_probability_endpoints_have_finite_positive_loss():
    original = pd.DataFrame({"signal": np.array([0., 1.], dtype=np.float32)})
    perturbed = pd.DataFrame({"signal": np.array([1., 0.], dtype=np.float32)})
    value = InterpretabilityProcessing()._prediction_loss(ProbabilityFromColumn(), original, perturbed)
    assert np.isfinite(value)
    assert value > 30.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_probabilities_are_rejected(value):
    original = pd.DataFrame({"signal": [.8]})
    perturbed = pd.DataFrame({"signal": [value]})
    with pytest.raises(ValueError, match="probabilities"):
        InterpretabilityProcessing()._prediction_loss(ProbabilityFromColumn(), original, perturbed)


@pytest.mark.parametrize("threshold", [-.01, float("nan"), float("inf"), "invalid"])
def test_loss_tolerance_must_be_finite_and_nonnegative(threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        InterpretabilityProcessing(effective_complexity_loss_threshold=threshold)


def test_uncertain_model_can_use_only_its_relevant_feature_at_zero_tolerance():
    X = pd.DataFrame({"signal": [.2, .8] * 6, "irrelevant": [0, 1, 2] * 4})
    processor = InterpretabilityProcessing(
        effective_complexity_mode="loss", effective_complexity_loss_threshold=0.,
        effective_complexity_repeats=3, random_state=0,
    )
    processor.attributions = np.array([1., 0.])
    assert processor._loss_based_effective_complexity(ProbabilityFromColumn(), X) == .5


def test_model_that_needs_all_features_legitimately_has_zero_interpretability_score():
    X = pd.DataFrame({"first": [0, 1] * 6, "second": [0] * 6 + [1] * 6})
    processor = InterpretabilityProcessing(
        effective_complexity_mode="loss", effective_complexity_loss_threshold=0.,
        effective_complexity_repeats=3, random_state=0,
    )
    processor.attributions = np.array([1., 0.])
    losses = []
    original_loss = processor._prediction_loss

    def record_loss(*args):
        loss = original_loss(*args)
        losses.append(loss)
        return loss

    processor._prediction_loss = record_loss
    assert processor._loss_based_effective_complexity(TwoFeatureProbabilityModel(), X) == 0.
    assert np.mean(losses[:3]) > 0
    assert losses[-3:] == [0., 0., 0.]


def test_equal_to_tolerance_is_accepted(monkeypatch):
    processor = InterpretabilityProcessing(effective_complexity_loss_threshold=.05)
    processor.attributions = np.array([1., 0.])
    monkeypatch.setattr(processor, "_prediction_loss", lambda *args: .05)
    X = pd.DataFrame({"first": [0, 1], "second": [1, 0]})
    assert processor._loss_based_effective_complexity(LabelsFromColumn(), X) == .5


def test_hard_label_fidelity_remains_fraction_of_changed_classes():
    original = pd.DataFrame({"signal": [0, 0, 1, 1]})
    perturbed = pd.DataFrame({"signal": [0, 1, 1, 1]})
    processor = InterpretabilityProcessing()
    assert processor._prediction_loss(LabelsFromColumn(), original, original.copy()) == 0.
    assert processor._prediction_loss(LabelsFromColumn(), original, perturbed) == .25
