"""Persisted calibrated equalized-odds with reproducible inference.

The AIF360 fit is retained. Its score replacement step uses a local draw keyed
by (seed, original row index, group), rather than changing global NumPy state.
This couples the randomization between original and perturbed EC observations.
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class CalibratedEqualizedOdds(BaseEstimator):
    def __init__(self, random_state=0, cost_constraint="fnr", threshold=0.5):
        self.random_state = random_state
        self.cost_constraint = cost_constraint
        self.threshold = threshold

    @staticmethod
    def _scores(scores):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (
            (scores < 0).any() or (scores > 1).any()
        ):
            raise ValueError("Postprocessing requires a nonempty vector of probabilities in [0, 1].")
        return scores

    def fit(self, scores, y, *, sensitive_features):
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

        score_index = scores.index if isinstance(scores, pd.Series) else None
        if score_index is not None:
            for values in (y, sensitive_features):
                if isinstance(values, pd.Series) and not score_index.equals(values.index):
                    raise ValueError("Calibration score, label and group indices must match.")
        scores = self._scores(scores)
        groups = np.asarray(sensitive_features)
        labels = np.asarray(y)
        if groups.ndim != 1 or labels.ndim != 1 or len(groups) != len(scores) or len(labels) != len(scores):
            raise ValueError("Calibration labels and groups must align with probabilities.")
        if pd.isna(groups).any() or not np.isin(labels, [0, 1]).all():
            raise ValueError("Calibration requires complete groups and binary labels 0 and 1.")
        if isinstance(y, pd.Series) and isinstance(sensitive_features, pd.Series):
            if not y.index.equals(sensitive_features.index):
                raise ValueError("Calibration label and group indices must match.")
        self.groups_ = sorted(pd.unique(groups), key=repr)
        if len(self.groups_) != 2:
            raise ValueError("Calibrated equalized odds requires exactly two groups.")
        if self.cost_constraint not in {"fnr", "fpr", "weighted"}:
            raise ValueError("cost_constraint must be fnr, fpr or weighted.")
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        for value in self.groups_:
            if len(np.unique(labels[groups == value])) != 2:
                raise ValueError("Each calibration group must contain both true classes.")
        code = (groups == self.groups_[1]).astype(float)
        frame = pd.DataFrame({"group": code, "target": labels.astype(float)})
        truth = BinaryLabelDataset(df=frame, label_names=["target"],
                                  protected_attribute_names=["group"])
        predicted = truth.copy(deepcopy=True)
        predicted.scores = scores[:, None]
        predicted.labels = (scores >= self.threshold).astype(float)[:, None]
        self.processor_ = CalibratedEqOddsPostprocessing(
            privileged_groups=[{"group": 1.0}], unprivileged_groups=[{"group": 0.0}],
            cost_constraint=self.cost_constraint, seed=self.random_state,
        ).fit(truth, predicted)
        raw_rates = np.asarray([self.processor_.unpriv_mix_rate, self.processor_.priv_mix_rate], dtype=float)
        # AIF360's comparison random<=rate already saturates outside [0,1].
        # Preserve that effective rule and expose infeasibility, never label it
        # as a successfully attained equalized-cost objective.
        if not np.isfinite(raw_rates).all():
            raise ValueError("Equalized-odds calibration has an undefined mixing rate.")
        self.attainable_ = bool(((raw_rates >= 0) & (raw_rates <= 1)).all())
        self.raw_mix_rates_ = raw_rates
        self.mix_rates_ = np.clip(raw_rates, 0, 1)
        self.base_rates_ = np.asarray([self.processor_.base_rate_unpriv, self.processor_.base_rate_priv])
        self.seed_ = 0 if self.random_state is None else int(self.random_state)
        self.calibration_index_ = (list(y.index) if isinstance(y, pd.Series) else list(range(len(y))))
        self.diagnostics_ = {
            "attainable": self.attainable_, "raw_mix_rates": raw_rates.tolist(),
            "effective_mix_rates": self.mix_rates_.tolist(), "groups": list(self.groups_),
            "cost_constraint": self.cost_constraint,
            "randomization": "blake2b(seed, row_index, group); shared across EC perturbations",
        }
        return self

    def transform(self, scores, *, sensitive_features, index=None):
        check_is_fitted(self, "processor_")
        score_index = scores.index if isinstance(scores, pd.Series) else None
        if index is None:
            index = score_index
            if index is None and isinstance(sensitive_features, pd.Series):
                index = sensitive_features.index
        if index is not None:
            explicit_index = pd.Index(index)
            if not explicit_index.is_unique:
                raise ValueError("Postprocessed inference requires unique, stable observation indices.")
            for values in (scores, sensitive_features):
                if isinstance(values, pd.Series) and not explicit_index.equals(values.index):
                    raise ValueError("Inference score and group indices must match the observation index.")
        scores = self._scores(scores).copy()
        groups = np.asarray(sensitive_features)
        if groups.ndim != 1 or len(groups) != len(scores) or pd.isna(groups).any():
            raise ValueError("Inference requires complete sensitive_features aligned with scores.")
        if not np.isin(groups, self.groups_).all():
            raise ValueError("Unknown sensitive group at inference.")
        indices = list(range(len(scores))) if index is None else list(index)
        if len(indices) != len(scores):
            raise ValueError("Inference indices must align with scores.")
        for position, (key, group) in enumerate(zip(indices, groups)):
            group_number = self.groups_.index(group)
            payload = repr((self.seed_, key, group)).encode("utf-8")
            number = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
            draw = number / 2**64
            if draw < self.mix_rates_[group_number]:
                scores[position] = self.base_rates_[group_number]
        return scores
