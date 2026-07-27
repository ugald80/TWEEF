"""Fuzzy calibration utilities for TWEEF aggregation."""

import numpy as np
import pandas as pd

from TWEEF.Constants import FUZZY_THRESHOLD_PEAKS, get_metric_thresholds


DEFAULT_CALIBRATION_QUANTILES = (0.2, 0.4, 0.6, 0.8)


class FuzzyCalibrator:
    def __init__(self, thresholds=None, metrics_values=None):
        if thresholds is not None and metrics_values is not None:
            raise ValueError("thresholds and metrics_values are mutually exclusive.")
        self.thresholds = self._normalize_thresholds(thresholds)
        self.metrics_values = metrics_values

    def thresholds_for(self, metric_name: str):
        if self.thresholds is not None:
            if metric_name not in self.thresholds:
                raise ValueError(f"missing fuzzy thresholds for metric {metric_name!r}.")
            return self.thresholds[metric_name]

        if self.metrics_values is not None:
            if metric_name not in self.metrics_values.columns:
                raise ValueError(f"metrics_values does not contain {metric_name!r}.")
            values = self.metrics_values[metric_name].dropna()
            if values.empty:
                raise ValueError(f"metrics_values column {metric_name!r} is empty.")
            peaks = [
                values.min(),
                values.quantile(DEFAULT_CALIBRATION_QUANTILES[0]),
                values.quantile(DEFAULT_CALIBRATION_QUANTILES[1]),
                values.quantile(DEFAULT_CALIBRATION_QUANTILES[2]),
                values.max(),
            ]
            return pd.Series(peaks, index=FUZZY_THRESHOLD_PEAKS, dtype=float)

        thresholds = get_metric_thresholds(metric_name)
        if thresholds is None:
            raise ValueError(f"missing default thresholds for metric {metric_name!r}.")
        return pd.Series(thresholds, index=FUZZY_THRESHOLD_PEAKS, dtype=float)

    def membership_vector(self, metric_name: str, value: float) -> np.ndarray:
        thresholds = self.thresholds_for(metric_name)
        return membership_vector(value, thresholds)

    @staticmethod
    def _normalize_thresholds(thresholds):
        if thresholds is None:
            return None
        if isinstance(thresholds, pd.DataFrame):
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.columns):
                return {
                    index: pd.Series(row[FUZZY_THRESHOLD_PEAKS], dtype=float)
                    for index, row in thresholds.iterrows()
                }
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.index):
                return {
                    column: pd.Series(thresholds[column], dtype=float)
                    for column in thresholds.columns
                }
            raise ValueError(
                "thresholds DataFrame must have fuzzy peaks as columns or index."
            )
        return {
            metric_name: pd.Series(values, index=FUZZY_THRESHOLD_PEAKS, dtype=float)
            for metric_name, values in thresholds.items()
        }


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
    peaks = pd.Series(thresholds, index=FUZZY_THRESHOLD_PEAKS, dtype=float)
    if not peaks.is_monotonic_increasing:
        raise ValueError("fuzzy thresholds must be monotonic increasing.")
    if peaks.nunique() < 2:
        raise ValueError("fuzzy thresholds must contain at least two distinct values.")

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
