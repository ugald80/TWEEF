"""Fuzzy calibration utilities for TWEEF aggregation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from TWEEF.Constants import FUZZY_THRESHOLD_PEAKS, get_metric_thresholds


DEFAULT_CALIBRATION_QUANTILES = (0.25, 0.5, 0.75)


class QuartileCalibration(BaseEstimator):
    """Fitted, immutable-reference calibration at min/P25/P50/P75/max.

    A constant active reference is rejected explicitly. A caller may supply a
    different, independently obtained reference; calibration never invents data
    or falls back to evaluation observations. With no reference, the published
    per-metric TWEEF defaults are used, preserving the legacy API.
    """

    def __init__(self, constant_policy="error"):
        self.constant_policy = constant_policy

    def fit(self, reference_metrics=None, *, metric_names=None):
        if self.constant_policy != "error":
            raise ValueError("constant_policy must be 'error'; supply an explicit valid reference instead.")
        calibrator = FuzzyCalibrator(metrics_values=reference_metrics)
        names = list(metric_names) if metric_names is not None else (
            list(reference_metrics.columns) if reference_metrics is not None else []
        )
        self.thresholds_ = {name: calibrator.thresholds_for(name) for name in names}
        self.calibrator_ = FuzzyCalibrator(thresholds=self.thresholds_) if names else calibrator
        self.reference_metrics_ = (
            reference_metrics.copy(deep=True) if reference_metrics is not None else None
        )
        return self

    def thresholds_for(self, metric_name):
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.thresholds_for(metric_name)

    def membership_vector(self, metric_name, value):
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.membership_vector(metric_name, value)


class FuzzyCalibrator:
    def __init__(self, thresholds=None, metrics_values=None):
        if thresholds is not None and metrics_values is not None:
            raise ValueError("thresholds and metrics_values are mutually exclusive.")
        self.metrics_values = None
        self.thresholds = self._normalize_thresholds(thresholds)
        if metrics_values is not None:
            if not isinstance(metrics_values, pd.DataFrame):
                raise TypeError("metrics_values must be a pandas.DataFrame.")
            if metrics_values.empty or metrics_values.columns.has_duplicates:
                raise ValueError("metrics_values must have rows and unique metric columns.")
            try:
                self.metrics_values = metrics_values.astype(float).copy(deep=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("metrics_values must contain numeric values.") from exc
            if not np.isfinite(self.metrics_values.to_numpy()).all():
                raise ValueError("metrics_values must contain only finite values.")
            self.thresholds = {
                name: self._validate_thresholds(
                    self.metrics_values[name].quantile(
                        [0.0, *DEFAULT_CALIBRATION_QUANTILES, 1.0],
                        interpolation="linear",
                    ).to_numpy(),
                    name,
                )
                for name in self.metrics_values.columns
            }

    def thresholds_for(self, metric_name: str):
        if self.thresholds is not None:
            if metric_name not in self.thresholds:
                raise ValueError(f"missing fuzzy thresholds for metric {metric_name!r}.")
            return self.thresholds[metric_name].copy()

        thresholds = get_metric_thresholds(metric_name)
        if thresholds is None:
            raise ValueError(f"missing default thresholds for metric {metric_name!r}.")
        return self._validate_thresholds(thresholds, metric_name)

    def membership_vector(self, metric_name: str, value: float) -> np.ndarray:
        thresholds = self.thresholds_for(metric_name)
        return membership_vector(value, thresholds)

    @classmethod
    def _normalize_thresholds(cls, thresholds):
        if thresholds is None:
            return None
        if isinstance(thresholds, pd.DataFrame):
            if thresholds.empty or thresholds.columns.has_duplicates or thresholds.index.has_duplicates:
                raise ValueError("thresholds must have rows and unique metric and crest labels.")
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.columns):
                return {
                    index: cls._validate_thresholds(row[FUZZY_THRESHOLD_PEAKS], index)
                    for index, row in thresholds.iterrows()
                }
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.index):
                return {
                    column: cls._validate_thresholds(thresholds[column], column)
                    for column in thresholds.columns
                }
            raise ValueError(
                "thresholds DataFrame must have fuzzy peaks as columns or index."
            )
        if not hasattr(thresholds, "items") or not len(thresholds):
            raise ValueError("thresholds must contain at least one metric.")
        return {
            metric_name: cls._validate_thresholds(values, metric_name)
            for metric_name, values in thresholds.items()
        }

    @staticmethod
    def _validate_thresholds(values, metric_name):
        try:
            peaks = pd.Series(values, index=FUZZY_THRESHOLD_PEAKS, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {metric_name!r} requires five numeric fuzzy crests.") from exc
        if not np.isfinite(peaks.to_numpy()).all():
            raise ValueError(f"fuzzy thresholds for {metric_name!r} must be finite.")
        if not peaks.is_monotonic_increasing:
            raise ValueError(f"fuzzy thresholds for {metric_name!r} must be monotonic increasing.")
        if peaks.nunique() < 2:
            raise ValueError(
                f"constant calibration for metric {metric_name!r}: "
                "fuzzy thresholds must contain at least two distinct values."
            )
        return peaks


def triangular_membership(value, low, peak, high):
    if low == peak and value == peak:
        return 1.0
    if peak == high and value == peak:
        return 1.0
    if value <= low or value >= high:
        return 0.0
    if value <= peak:
        return float((value - low) / (peak - low))
    return float((high - value) / (high - peak))


def membership_vector(value: float, thresholds) -> np.ndarray:
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError("metric value must be a finite scalar.")
    peaks = FuzzyCalibrator._validate_thresholds(thresholds, "membership")

    if value <= peaks.iloc[0]:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    if value >= peaks.iloc[-1]:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    ranges = [
        (peaks.iloc[0], peaks.iloc[0], peaks.iloc[1]),
        (peaks.iloc[0], peaks.iloc[1], peaks.iloc[2]),
        (peaks.iloc[1], peaks.iloc[2], peaks.iloc[3]),
        (peaks.iloc[2], peaks.iloc[3], peaks.iloc[4]),
        (peaks.iloc[3], peaks.iloc[4], peaks.iloc[4]),
    ]
    memberships = np.array(
        [triangular_membership(value, low, peak, high) for low, peak, high in ranges],
        dtype=float,
    )
    total = memberships.sum()
    if total > 0:
        memberships = memberships / total
    return memberships
