"""Reproducible, paired German benchmark for the paper's eight models.

Train each model once per seed, cache metric matrices, then execute the real legacy
and corrected TWEEF modules in isolated subprocesses. The enhancement/surrogate
pipeline is intentionally absent: every aggregation comparison has identical input
metrics. Run ``python test/experiments/german_aggregation.py --help`` for usage.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tweef-german")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "test/datasets/german.csv"
METRICS = ["Accuracy", "equal Opportunity", "Effective Complexity"]
PEAKS = ["peak_-2", "peak_-1", "peak_0", "peak_1", "peak_2"]


class BinaryLinearRegression(ClassifierMixin, BaseEstimator):
    """Paper's LinearRegression with an explicit binary threshold, not a probability."""
    def __init__(self, threshold=0.5, fit_intercept=True):
        self.threshold = threshold
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.regressor_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "regressor_")
        return (self.regressor_.predict(X) >= self.threshold).astype(int)


def json_default(value):
    if isinstance(value, (np.ndarray, pd.Index, pd.Series)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def load_german(path=DATASET):
    frame = pd.read_csv(path)
    status = frame["Personal status and sex"]
    gender = status.map({"A91": "male", "A92": "female", "A93": "male",
                         "A94": "male", "A95": "female"})
    if gender.isna().any():
        raise ValueError("Unknown German personal-status/sex code")
    target = frame.pop("Credit Score").map({1: 1, 2: 0})
    if target.isna().any() or frame.isna().any().any():
        raise ValueError("German data contain missing or unsupported values")
    return frame, target.astype(int), gender.rename("gender")


def make_split(y, groups, seed):
    strata = y.astype(str) + ":" + groups
    train, rest = train_test_split(np.arange(len(y)), test_size=0.4,
                                  stratify=strata, random_state=seed)
    calibration, test = train_test_split(rest, test_size=0.5,
                                         stratify=strata.iloc[rest], random_state=seed)
    return {"train": train, "calibration": calibration, "test": test}


def make_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [name for name in X if name not in categorical]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", MinMaxScaler(), numeric),
    ], verbose_feature_names_out=True)


def models(seed):
    from xgboost import XGBClassifier
    return {
        "DT": DecisionTreeClassifier(criterion="gini", splitter="best", random_state=seed),
        "RF": RandomForestClassifier(criterion="gini", n_estimators=100,
                                     n_jobs=1, random_state=seed),
        "ET": ExtraTreesClassifier(criterion="gini", n_estimators=100,
                                   n_jobs=1, random_state=seed),
        "LR": BinaryLinearRegression(fit_intercept=True, threshold=0.5),
        "L-SVM": LinearSVC(C=1., penalty="l2", loss="squared_hinge", dual="auto",
                           max_iter=20000, random_state=seed),
        "AB": AdaBoostClassifier(algorithm="SAMME.R", learning_rate=1.,
                                 n_estimators=50, random_state=seed),
        "GB": GradientBoostingClassifier(loss="log_loss", learning_rate=0.1,
                                         n_estimators=100, criterion="friedman_mse",
                                         random_state=seed),
        "XGB": XGBClassifier(booster="gbtree", learning_rate=0.3, n_estimators=100,
                             objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", n_jobs=1, random_state=seed),
    }


def model_metrics(model, X, y, groups, seed):
    # Metric matrices are evaluated once with this checkout's EC loss, then
    # shared with all aggregation workers. Historical EC remains in its own run.
    from TWEEF.EstimatorPerformance import PerformanceEstimator
    from TWEEF.EstimatorFairness import FairnessEstimator
    from TWEEF.ProcessorInterpretability import InterpretabilityProcessing
    predictions = model.predict(X)
    protected = groups.to_numpy() == "female"
    performance = PerformanceEstimator(y, predictions).score_accuracy
    fairness = FairnessEstimator(y.to_numpy()[protected], predictions[protected],
                                y.to_numpy()[~protected], predictions[~protected]).score_equal_opportunity
    processor = InterpretabilityProcessing(effective_complexity_mode="loss",
                                          effective_complexity_loss_threshold=0.05,
                                          effective_complexity_repeats=3,
                                          random_state=seed)
    if hasattr(model, "feature_importances_"):
        attributions = model.feature_importances_
        attribution_source = "native_feature_importances"
    else:
        attributions = permutation_importance(model, X, y, n_repeats=5,
                                               random_state=0).importances_mean
        attribution_source = "permutation_importance_accuracy_5_repeats_seed_0"
    processor.attributions = np.asarray(attributions, dtype=float)
    entropy = None
    probabilities = processor._positive_probabilities(model, X)
    if probabilities is not None:
        entropy = probability_entropy(probabilities)
    # Entropy remains an audit diagnostic. It is not a lower bound on KL and
    # must never short-circuit the corrected effective-complexity calculation.
    complexity = processor._loss_based_effective_complexity(model, X)
    values = {METRICS[0]: float(performance), METRICS[1]: float(fairness),
              METRICS[2]: float(complexity)}
    if not all(np.isfinite(list(values.values()))):
        raise ValueError("Nonfinite model metric")
    return values, {"predictions": predictions.tolist(), "attributions": attributions.tolist(),
                    "attribution_source": attribution_source, "reference_entropy": entropy,
                    "ec_entropy_short_circuit": False,
                    "ec_fidelity_loss": "bernoulli_kl" if probabilities is not None else "class_disagreement"}


def probability_entropy(probabilities):
    """Finite diagnostic entropy, including float32 probabilities at zero/one."""
    from TWEEF.Constants import EPSILON
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probability entropy requires finite values in [0, 1].")
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return float(np.mean(-p * np.log(p) - (1 - p) * np.log(1 - p)))


def train_seed(seed, output, metric_protocol_sha256):
    X, y, gender = load_german()
    split = make_split(y, gender, seed)
    preprocessor = make_preprocessor(X)
    preprocessor.fit(X.iloc[split["train"]])
    names = preprocessor.get_feature_names_out()
    transformed = {
        part: pd.DataFrame(preprocessor.transform(X.iloc[indices]), columns=names,
                           index=X.index[indices]) for part, indices in split.items()
    }
    payload = {"seed": seed, "metric_protocol_sha256": metric_protocol_sha256,
               "splits": split, "feature_names": names,
               "calibration": {}, "test": {}, "models": {},
               "preprocessing": {"categories": [v.tolist() for v in preprocessor.named_transformers_["categorical"].categories_],
                                 "numeric_data_min": preprocessor.named_transformers_["numeric"].data_min_.tolist(),
                                 "numeric_data_max": preprocessor.named_transformers_["numeric"].data_max_.tolist()}}
    for name, model in models(seed).items():
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(transformed["train"], y.iloc[split["train"]])
            parameters = model.get_params(deep=False)
            # XGBoost exposes np.nan as its default missing-value sentinel.
            parameters = {key: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
                          for key, value in parameters.items()}
            info = {"parameters": parameters}
            for part in ("calibration", "test"):
                metric_values, diagnostics = model_metrics(model, transformed[part],
                                                          y.iloc[split[part]], gender.iloc[split[part]], seed)
                payload[part][name] = metric_values
                info[part] = diagnostics
            info["warnings"] = sorted(set(str(item.message) for item in captured))
            info["seconds"] = time.monotonic() - started
            payload["models"][name] = info
        print(f"seed={seed} model={name} metrics={payload['test'][name]} seconds={info['seconds']:.2f}", flush=True)
    write_json(output / f"metrics_seed_{seed}.json", payload)
    return payload


def configurations():
    weights = [("uniform", [1 / 3] * 3)]
    for i in range(3):
        weights.append((f"only_{i}", [float(j == i) for j in range(3)]))
        for step in range(1, 10):
            value = step / 10
            weights.append((f"weight_{i}_{value:.1f}", [value if j == i else (1 - value) / 2 for j in range(3)]))
    configs = []
    for name, values in weights:
        for operator in ("lwa", "gwa", "sl"):
            for scale in ("normalized", "linguistic"):
                configs.append({"name": name, "weights": values, "operator": operator,
                                "scale": scale, "hesitation": 0.0})
    for hesitation in (0.1, 0.25, 0.5, 0.75, 1., dict(zip(METRICS, [0., 0.15, 0.1]))):
        for scale in ("normalized", "linguistic"):
            configs.append({"name": f"hesitation_{hesitation}", "weights": [1 / 3] * 3,
                            "operator": "gwa", "scale": scale, "hesitation": hesitation})
    return configs


def metric_protocol(environment):
    """Fingerprint metric computation so stale pre-correction caches cannot leak in."""
    sources = [Path(__file__).resolve(), *[ROOT / "TWEEF" / name for name in (
        "ProcessorInterpretability.py", "EstimatorPerformance.py", "EstimatorFairness.py",
        "MatrixScores.py", "Constants.py")]]
    protocol = {
        "version": "bernoulli_kl_disagreement_v1",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "environment": environment,
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources},
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, digest


def validate_output_protocol(output, digest):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("metric_protocol_sha256") != digest:
            raise ValueError("Output contains a different metric protocol. Use a new --output directory; historical results were preserved.")


def validate_metric_cache(payload, seed, digest):
    if payload.get("seed") != seed or payload.get("metric_protocol_sha256") != digest:
        raise ValueError(f"Incompatible metric cache for seed {seed}; use a new --output directory or retrain compatible metrics.")


def evaluate_worker(args):
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
    from TWEEF.Metrics import MetricList
    import TWEEF
    module_path = Path(TWEEF.__file__).resolve()
    if code_root not in module_path.parents:
        raise RuntimeError(f"Wrong TWEEF import: {module_path}, expected {code_root}")
    output = Path(args.output)
    configs = configurations()
    failures, rows = [], []
    detail_path = output / f"details_{args.implementation}.jsonl.gz"
    with gzip.open(detail_path, "wt") as details:
        for seed in args.seeds:
            payload = json.loads((output / f"metrics_seed_{seed}.json").read_text())
            calibration = pd.DataFrame.from_dict(payload["calibration"], orient="index")[METRICS]
            test = pd.DataFrame.from_dict(payload["test"], orient="index")[METRICS]
            thresholds = calibration.quantile([0, .25, .5, .75, 1], interpolation="linear")
            thresholds.index = PEAKS
            for config in configs:
                if args.implementation != "corrected" and config["hesitation"] != 0.0:
                    continue  # Legacy has no hesitation parameter; those configs are new-only.
                try:
                    metric_list = MetricList()
                    for metric_name, weight in zip(METRICS, config["weights"]):
                        metric_list.add_metric(metric_name, weight, goal=0.)
                    kwargs = {"aggregation": config["operator"],
                              "trustworthiness_score_scale": config["scale"]}
                    if args.implementation == "legacy_native":
                        kwargs["metrics_values"] = calibration
                    elif args.implementation == "legacy_quartiles":
                        kwargs["fuzzy_thresholds"] = thresholds
                    else:
                        kwargs["metrics_values"] = calibration
                        kwargs["gwa_hesitation"] = config["hesitation"]
                    estimator = TrustworthinessEstimator(**kwargs)
                    if args.implementation == "corrected":
                        estimator.fit(metrics_values=calibration, metric_list=metric_list)
                    elif args.implementation == "legacy_native":
                        estimator.fit(metrics_values=calibration)
                        estimator.weights = dict(zip(METRICS, config["weights"]))
                    else:
                        estimator.fit(fuzzy_functions_thresholds=thresholds)
                        estimator.weights = dict(zip(METRICS, config["weights"]))
                    # Legacy fit has no weights argument. The explicit assignment
                    # above injects exactly the requested weights into that path.
                    batch = estimator.transform(test)
                    for model_name, values in test.iterrows():
                        metrics = MetricList()
                        for metric_name, weight in zip(METRICS, config["weights"]):
                            metrics.add_metric(metric_name, weight, goal=0., value=float(values[metric_name]))
                        scalar = float(estimator.compute_metric(metrics))
                        vectorized = float(batch.loc[model_name, "Trustworthiness"])
                        common = {"implementation": args.implementation, "seed": seed,
                                  "model": model_name, "configuration": config["name"],
                                  "operator": config["operator"], "scale": config["scale"],
                                  "weights": json.dumps(config["weights"]),
                                  "hesitation": json.dumps(config["hesitation"]),
                                  "compute_metric": scalar, "transform": vectorized,
                                  "interface_gap": abs(scalar - vectorized)}
                        rows.append(common)
                        aggregation_result = estimator.aggregation_result_
                        details.write(json.dumps({**common, "result": aggregation_result},
                                                 default=json_default, allow_nan=False) + "\n")
                except Exception as error:
                    failures.append({"implementation": args.implementation, "seed": seed,
                                     "config": config, "error": f"{type(error).__name__}: {error}"})
    pd.DataFrame(rows).to_csv(output / f"scores_{args.implementation}.csv", index=False)
    write_json(output / f"failures_{args.implementation}.json", failures)
    write_json(output / f"import_{args.implementation}.json", {"module": module_path, "code_root": code_root})
    print(f"{args.implementation}: {len(rows)} rows, {len(failures)} failures", flush=True)
    return 1 if failures else 0


def summarize(output):
    frames = [pd.read_csv(output / f"scores_{name}.csv")
              for name in ("legacy_native", "legacy_quartiles", "corrected")]
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(output / "scores.csv", index=False)
    uniform = all_scores[(all_scores.configuration == "uniform") & (all_scores.scale == "normalized")]
    summary = uniform.groupby(["implementation", "model", "operator"])[["compute_metric", "transform", "interface_gap"]].agg(["mean", "std", "max"])
    summary.to_csv(output / "summary_uniform.csv")
    gaps = all_scores.groupby(["implementation", "operator"])["interface_gap"].agg(["mean", "max"])
    gaps.to_csv(output / "interface_gaps.csv")
    corrected = uniform[uniform.implementation == "corrected"]
    comparisons = []
    for legacy in ("legacy_native", "legacy_quartiles"):
        before = uniform[uniform.implementation == legacy]
        paired = corrected.merge(before, on=["seed", "model", "operator"], suffixes=("_new", "_old"))
        paired["delta"] = paired.compute_metric_new - paired.compute_metric_old
        paired["baseline"] = legacy
        comparisons.append(paired[["seed", "model", "operator", "baseline", "compute_metric_old", "compute_metric_new", "delta"]])
    pd.concat(comparisons).to_csv(output / "paired_differences.csv", index=False)
    ranks = uniform.copy()
    ranks["rank"] = ranks.groupby(["implementation", "seed", "operator"])["compute_metric"].rank(ascending=False, method="average")
    ranks.to_csv(output / "rankings_uniform.csv", index=False)
    calibration_peaks = {}
    raw_rows = []
    for file in sorted(output.glob("metrics_seed_*.json")):
        payload = json.loads(file.read_text())
        calibration = pd.DataFrame.from_dict(payload["calibration"], orient="index")[METRICS]
        peaks = calibration.quantile([0, .25, .5, .75, 1], interpolation="linear")
        peaks.index = PEAKS
        calibration_peaks[str(payload["seed"])] = peaks.to_dict()
        for part in ("calibration", "test"):
            for name, values in payload[part].items():
                raw_rows.append({"seed": payload["seed"], "partition": part, "model": name, **values})
    write_json(output / "quartile_peaks.json", calibration_peaks)
    pd.DataFrame(raw_rows).to_csv(output / "raw_metrics.csv", index=False)
    print("Uniform normalized scores (mean across seeds):", flush=True)
    print(uniform.pivot_table(index="model", columns=["implementation", "operator"], values="compute_metric").round(5).to_string(), flush=True)
    print("Maximum interface gaps:", gaps["max"].to_dict(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output", type=Path, default=ROOT / "test/experiments/results/german_ec_kl")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--reuse-metrics", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--code-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=["legacy_native", "legacy_quartiles", "corrected"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return evaluate_worker(args)
    if args.baseline_root is None:
        parser.error("--baseline-root must point to a checkout of the unmodified baseline")
    sys.path.insert(0, str(ROOT))
    environment = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "interpret-core")}
    protocol, protocol_digest = metric_protocol(environment)
    # Validate before writing even the manifest, preserving historical results.
    validate_output_protocol(args.output, protocol_digest)
    if args.reuse_metrics:
        for seed in args.seeds:
            cache = args.output / f"metrics_seed_{seed}.json"
            if cache.exists():
                validate_metric_cache(json.loads(cache.read_text()), seed, protocol_digest)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_commit = "a36fe25ffec66c6b1ea52841e8955cb621ca48de"
    manifest = {"seeds": args.seeds, "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                "dataset_rows": len(load_german()[0]), "split": "60/20/20 joint target/gender stratification",
                "metric_names": METRICS, "ec_mode": "loss-based fidelity: Bernoulli KL / class disagreement", "ec_loss_threshold": 0.05,
                "ec_repeats": 3, "environment": environment, "python": sys.version,
                "metric_protocol": protocol, "metric_protocol_sha256": protocol_digest,
                "ec_loss_units": {"probabilistic": "nats", "hard_labels": "disagreement fraction"},
                "baseline_root": args.baseline_root.resolve(), "baseline_commit": baseline_commit,
                "current_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "current_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest(),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [*sorted((ROOT / "TWEEF").glob("*.py")), Path(__file__).resolve()]},
                "legacy_transform_weights": "Injected into .weights because legacy fit does not accept them",
                "loss_caveat": "EC measures fidelity to original predictions, not accuracy loss against true labels. KL 0.05 is measured in nats, not five percentage points of accuracy; hard-label disagreement uses a different scale."}
    write_json(args.output / "manifest.json", manifest)
    for seed in args.seeds:
        cache = args.output / f"metrics_seed_{seed}.json"
        if args.reuse_metrics and cache.exists():
            print(f"Reusing metrics seed={seed}", flush=True)
        else:
            try:
                train_seed(seed, args.output, protocol_digest)
            except Exception as error:
                write_json(args.output / "training_failure.json", {"seed": seed, "error": f"{type(error).__name__}: {error}"})
                raise
    failed = False
    for implementation, code_root in (("legacy_native", args.baseline_root),
                                      ("legacy_quartiles", args.baseline_root), ("corrected", ROOT)):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--code-root", str(code_root.resolve()),
                   "--implementation", implementation, "--output", str(args.output.resolve()),
                   "--seeds", *map(str, args.seeds)]
        result = subprocess.run(command, cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root.resolve())})
        failed |= result.returncode != 0
    summarize(args.output)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
