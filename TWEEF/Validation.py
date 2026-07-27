"""Validation helpers for TWEEF inputs."""

from collections import Counter
from math import isclose

import pandas as pd

from TWEEF.Constants import get_metrics_of, METRICS
from TWEEF.Metrics import MetricList


def validate_metric_list(metrics: MetricList, require_values: bool = False) -> None:
    if metrics is None:
        raise ValueError("metrics must be provided.")
    if not isinstance(metrics, MetricList):
        raise TypeError("metrics must be an instance of MetricList.")
    if not metrics.get_metrics():
        raise ValueError("metrics must contain at least one metric.")

    names = metrics.get_names()
    duplicated_names = [name for name, count in Counter(names).items() if count > 1]
    if duplicated_names:
        raise ValueError(f"duplicated metrics are not allowed: {duplicated_names}")

    allowed_metrics = get_metrics_of(METRICS)
    unknown_names = [name for name in names if name not in allowed_metrics]
    if unknown_names:
        raise ValueError(f"unknown metric names: {unknown_names}")

    weights = metrics.get_weights()
    if any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("metric weights must be in the range [0, 1].")
    if not isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("metric weights must sum to 1.")

    if require_values:
        for metric in metrics.get_metrics():
            if metric.value is None:
                raise ValueError(f"metric {metric.name!r} has no computed value.")
            if metric.value < 0 or metric.value > 1:
                raise ValueError(f"metric {metric.name!r} value must be in [0, 1].")


def validate_groups(X: pd.DataFrame, protected_groups, unprotected_groups) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame for group validation.")
    if not protected_groups:
        raise ValueError("protected_groups must contain at least one group.")
    if not unprotected_groups:
        raise ValueError("unprotected_groups must contain at least one group.")

    for group_name, groups in {
        "protected_groups": protected_groups,
        "unprotected_groups": unprotected_groups,
    }.items():
        for group in groups:
            for column, values in group.items():
                if column not in X.columns:
                    raise ValueError(
                        f"{group_name} references missing column {column!r}."
                    )
                if X[column].isna().any():
                    raise ValueError(
                        f"{group_name} column {column!r} contains missing values."
                    )
                missing_values = set(values).difference(set(X[column].unique()))
                if missing_values:
                    raise ValueError(
                        f"{group_name} column {column!r} does not contain values "
                        f"{sorted(missing_values)!r}."
                    )


def drop_invalid_rows(X: pd.DataFrame, y: pd.Series, protected_groups):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")
    elif y.name is None:
        y = y.rename("target")

    protected_columns = {
        column for group in protected_groups or [] for column in group.keys()
    }
    missing_columns = [column for column in protected_columns if column not in X.columns]
    if missing_columns:
        raise ValueError(f"protected group columns are missing: {missing_columns}")

    valid_mask = y.notna()
    for column in protected_columns:
        valid_mask &= X[column].notna()

    return X.loc[valid_mask].copy(), y.loc[valid_mask].copy()
