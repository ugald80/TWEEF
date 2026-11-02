"""Executable example of the final-surrogate API and a user-supplied mechanism.

Run from the checkout: python test/experiments/example_final_surrogate.py
For built-in SL, supply the training-only pooled reference from the benchmark:
  python test/experiments/example_final_surrogate.py --operator sl \
    --reference test/experiments/results/german_final_surrogate/reference_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from test.experiments.german_aggregation import load_german, make_split, make_preprocessor, METRICS
from TWEEF.Aggregation import MechanismResult, SubjectiveLogic
from TWEEF.Calibration import QuartileCalibration
from TWEEF.TrustworthyClassifier import TrustworthyClassifier


class SquaredMetricMean:
    """Illustrative complete mechanism; its representation belongs to the plugin."""

    supports_weights = True

    def fit(self, reference_metrics, *, weights):
        values = np.asarray(list(weights.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("Weights must be finite, nonnegative and have positive sum")
        self.weights_ = {name: float(weight / values.sum()) for name, weight in weights.items()}
        return self

    def aggregate(self, metrics):
        represented = {name: float(metrics[name]) ** 2 for name in self.weights_}
        return MechanismResult(score=sum(self.weights_[name] * value for name, value in represented.items()),
            score_range=(0., 1.), higher_is_better=True, mechanism="squared_metric_mean",
            details={"represented_metrics": represented})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["custom", "sl"], default="custom")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args(argv)
    X, y, groups = load_german()
    split = make_split(y, groups, seed=0)
    train, calibration, test = (split[name] for name in ("train", "calibration", "test"))
    workflow = TrustworthyClassifier(
        given_classifier=RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0),
        surrogate=ExplainableBoostingClassifier(interactions=0, max_rounds=args.max_rounds,
                                                outer_bags=1, n_jobs=1, random_state=0),
        preprocessor=make_preprocessor(X),
        aggregation=None,
        metric_weights=dict(zip(METRICS, [.4, .3, .3])),
        effective_complexity={"loss": "bernoulli_kl", "loss_threshold": .05,
                              "feature_unit": "original", "ranking_method": "permutation", "repeats": 3},
        reference_cv=3,
        postprocessor="calibrated_equalized_odds" if args.postprocess else None,
        random_state=0,
    )
    workflow.fit(X.iloc[train], y.iloc[train], sensitive_features=groups.iloc[train],
                 validation_data=(X.iloc[calibration], y.iloc[calibration]),
                 validation_sensitive_features=groups.iloc[calibration])
    reference = workflow.reference_metrics_
    if args.reference is not None:
        supplied = pd.read_csv(args.reference)
        if "seed" in supplied:
            supplied = supplied[supplied.seed.eq(0)]
        reference = supplied[METRICS]
    operator = SquaredMetricMean() if args.operator == "custom" else SubjectiveLogic(
        calibration=QuartileCalibration(), alpha=.5)
    workflow.set_aggregation(operator, reference_metrics=reference)
    predictions = workflow.predict(X.iloc[test], sensitive_features=groups.iloc[test])
    report = workflow.evaluate(X.iloc[test], y.iloc[test], sensitive_features=groups.iloc[test])
    if not np.array_equal(predictions, np.asarray(report.predictions)):
        raise AssertionError("The public predictor and evaluation must use the same final pipeline")
    print("Final predictor:", type(workflow.surrogate_).__name__)
    print("Metrics:", report.metrics)
    print("Trustworthiness:", report.trustworthiness.score)
    print("Teacher/student fidelity:", report.teacher_student_fidelity)
    return workflow, report


if __name__ == "__main__":
    main()
