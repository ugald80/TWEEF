"""Run the TWEEF experiments described in TWEEF.pdf.

The paper evaluates three binary-classification datasets:

- German Credit: Precision, Treatment Equality, Effective Complexity.
- COMPAS: Recall, Statistical Parity, Effective Complexity.
- Adult Income: F1-Score, Equal Opportunity, Monotonicity.

By default this script runs a small reproducible sample so it is practical during
development. Use ``--full`` to run each dataset with all available rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tweef")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TWEEF.Constants import METRICS
from TWEEF.Logging import TWEEFLogger
from TWEEF.Metrics import MetricList
from TWEEF.UtilsFairnessGroup import FairnessGroup


DATASETS = ROOT / "test" / "datasets"


def require_optional_dependencies() -> None:
    missing = [
        module
        for module in ("aif360", "interpret")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise SystemExit(
            "Missing optional dependencies: "
            + ", ".join(missing)
            + '. Install them with: python3 -m pip install -e ".[all]"'
        )


def scale_frame(X: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def encode_frame(
    df: pd.DataFrame,
    label: pd.Series,
    protected_column: str,
    protected_value,
) -> tuple[pd.DataFrame, pd.Series, str]:
    X = df.copy()
    protected_feature = f"{protected_column}_{protected_value}"
    protected_values = (X[protected_column] == protected_value).astype(float)
    X = pd.get_dummies(X, drop_first=False)
    if protected_feature not in X.columns:
        X[protected_feature] = protected_values
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return scale_frame(X), label.astype(int), protected_feature


def german_credit(full: bool, sample_size: int, random_state: int):
    df = pd.read_csv(DATASETS / "german.csv")
    df = df[df["Personal status and sex"].isin(["A91", "A92", "A94"])].copy()
    y = df.pop("Credit Score").map({1: 1, 2: 0})
    df["Telephone"] = df["Telephone"].map({"A191": 0, "A192": 1})
    df["foreign worker"] = df["foreign worker"].map({"A201": 1, "A202": 0})
    df["Personal status and sex"] = df["Personal status and sex"].map(
        {"A91": "male", "A92": "female", "A94": "male"}
    )
    X, y, protected_feature = encode_frame(
        df, y, "Personal status and sex", "female"
    )
    X, y = sample(X, y, full, sample_size, random_state)
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.PRECISION, 1 / 3, 0.8)
    metrics.add_metric(METRICS.FAIRNESS.TREATMENT_EQUALITY, 1 / 3, 0.8)
    metrics.add_metric(METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY, 1 / 3, 0.8)
    return ExperimentConfig(
        "german",
        X,
        y,
        FairnessGroup([{protected_feature: [1.0]}]),
        FairnessGroup([{protected_feature: [0.0]}]),
        [1.0],
        metrics,
    )


def compas(full: bool, sample_size: int, random_state: int):
    df = pd.read_csv(DATASETS / "compas.csv")
    df = df[df["is_recid"].isin([0, 1])].copy()
    if "days_b_screening_arrest" in df:
        df = df[
            df["days_b_screening_arrest"].between(-30, 30)
            | df["days_b_screening_arrest"].isna()
        ].copy()
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    y = 1 - df.pop("is_recid").astype(int)
    drop_columns = [
        column
        for column in df.columns
        if column.endswith("_date")
        or column.endswith("_in")
        or column.endswith("_out")
        or column
        in {
            "id",
            "name",
            "first",
            "last",
            "dob",
            "c_case_number",
            "r_case_number",
            "vr_case_number",
        }
    ]
    df = df.drop(columns=drop_columns, errors="ignore")
    X, y, protected_feature = encode_frame(df, y, "race", "African-American")
    X, y = sample(X, y, full, sample_size, random_state)
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.RECALL, 1 / 3, 0.8)
    metrics.add_metric(METRICS.FAIRNESS.STATISTICAL_PARITY, 1 / 3, 0.8)
    metrics.add_metric(METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY, 1 / 3, 0.8)
    return ExperimentConfig(
        "compas",
        X,
        y,
        FairnessGroup([{protected_feature: [1.0]}]),
        FairnessGroup([{protected_feature: [0.0]}]),
        [1.0],
        metrics,
    )


def adult(full: bool, sample_size: int, random_state: int):
    df = pd.read_csv(DATASETS / "adult.csv")
    df = df.replace("?", pd.NA).dropna().copy()
    y = df.pop("income").astype(str).str.contains(">50K").astype(int)
    X, y, protected_feature = encode_frame(df, y, "sex", "Female")
    X, y = sample(X, y, full, sample_size, random_state)
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.F1_SCORE, 1 / 3, 0.8)
    metrics.add_metric(METRICS.FAIRNESS.EQUAL_OPPORTUNITY, 1 / 3, 0.8)
    metrics.add_metric(METRICS.INTERPRETABILITY.MONOTONICITY, 1 / 3, 0.8)
    return ExperimentConfig(
        "adult",
        X,
        y,
        FairnessGroup([{protected_feature: [1.0]}]),
        FairnessGroup([{protected_feature: [0.0]}]),
        [1.0],
        metrics,
    )


class ExperimentConfig:
    def __init__(
        self,
        name,
        X,
        y,
        protected_groups,
        unprotected_groups,
        protected_classes,
        metrics,
    ):
        self.name = name
        self.X = X
        self.y = y
        self.protected_groups = protected_groups
        self.unprotected_groups = unprotected_groups
        self.protected_classes = protected_classes
        self.metrics = metrics


def sample(
    X: pd.DataFrame,
    y: pd.Series,
    full: bool,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if full or len(X) <= sample_size:
        return X, y
    sampled = X.assign(__target=y).groupby("__target", group_keys=False).sample(
        n=max(1, sample_size // y.nunique()),
        random_state=random_state,
        replace=False,
    )
    y_sampled = sampled.pop("__target")
    y_sampled.name = y.name
    return sampled, y_sampled


def run_experiment(config: ExperimentConfig, args):
    from TWEEF.TrustworthyClassifier import TrustworthyClassifier

    print(f"\n=== {config.name.upper()} ===")
    print(f"rows={len(config.X)} features={config.X.shape[1]}")
    model = SVC(kernel=args.kernel, probability=True, random_state=args.random_state)
    classifier = TrustworthyClassifier(
        given_classifier=model,
        protected_groups=config.protected_groups,
        unprotected_groups=config.unprotected_groups,
        protected_classes=config.protected_classes,
        metrics=config.metrics,
        aggregation=args.aggregation,
        trustworthiness_score_scale=args.trustworthiness_score_scale,
        test_size=args.test_size,
        random_state=args.random_state,
        fairness_threshold=args.fairness_threshold,
        fairness_max_iter=args.fairness_max_iter,
        surrogate_min_r2=args.surrogate_min_r2,
        surrogate_max_iter=args.surrogate_max_iter,
        surrogate_interactions=parse_interactions(args.surrogate_interactions),
        surrogate_max_rounds=args.surrogate_max_rounds,
        effective_complexity_mode=args.effective_complexity_mode,
        effective_complexity_threshold=args.effective_complexity_threshold,
        effective_complexity_loss_threshold=args.effective_complexity_loss_threshold,
        effective_complexity_repeats=args.effective_complexity_repeats,
    )
    classifier.fit(config.X, config.y)
    print("\nFinal summary")
    print(classifier.final_report_.to_string(index=False))
    print("\nTradeoffs")
    print(classifier.tradeoffs_.to_string(index=False))
    return classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["german", "compas", "adult", "all"],
        default="all",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--aggregation", choices=["lwa", "gwa", "sl"], default="lwa")
    parser.add_argument(
        "--trustworthiness-score-scale",
        choices=["normalized", "linguistic", "0_1", "-2_2"],
        default="normalized",
        help="Use normalized for [0,1] or linguistic for [-2,2].",
    )
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--fairness-threshold", type=float, default=0.9)
    parser.add_argument("--fairness-max-iter", type=int, default=5)
    parser.add_argument("--surrogate-min-r2", type=float, default=0.8)
    parser.add_argument("--surrogate-max-iter", type=int, default=5)
    parser.add_argument("--surrogate-interactions", default=0)
    parser.add_argument("--surrogate-max-rounds", type=int, default=1000)
    parser.add_argument(
        "--effective-complexity-mode",
        choices=["attribution", "loss"],
        default="attribution",
    )
    parser.add_argument("--effective-complexity-threshold", type=float, default=0.95)
    parser.add_argument(
        "--effective-complexity-loss-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--effective-complexity-repeats", type=int, default=3)
    return parser.parse_args()


def parse_interactions(value):
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args = parse_args()
    TWEEFLogger.configure(level="INFO")
    require_optional_dependencies()
    builders = {
        "german": german_credit,
        "compas": compas,
        "adult": adult,
    }
    selected = builders.keys() if args.dataset == "all" else [args.dataset]
    for name in selected:
        config = builders[name](args.full, args.sample_size, args.random_state)
        run_experiment(config, args)


if __name__ == "__main__":
    main()
