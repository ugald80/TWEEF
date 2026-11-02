"""German benchmark: a surrogate is the final predictor; mechanisms are plugins.

Two experiments share cached fitted candidates: fixed-predictor mechanism comparison
and CV-only selection of a complete predictor by each mechanism. Test data enter
only after all per-seed decisions are frozen. Historical results are read, not edited.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tweef-final-surrogate-matplotlib")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.german_aggregation import DATASET, METRICS, load_german, make_preprocessor, make_split, models

PROTOCOL = "german-final-surrogate-original-features-v1"
MODEL_ORDER = ["DT", "RF", "ET", "LR", "L-SVM", "AB", "GB", "XGB"]
DEFAULT_WEIGHTS = dict.fromkeys(METRICS, 1 / 3)
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "interpret-core", "xgboost", "aif360"]


def json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, (np.ndarray, pd.Index, pd.Series)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default, allow_nan=False).encode()).hexdigest()


def source_fingerprint():
    files = sorted((ROOT / "TWEEF").glob("*.py")) + [Path(__file__).resolve(), ROOT / "experiments/german_aggregation.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def build_protocol(args):
    X, y, groups = load_german()
    return {
        "protocol": PROTOCOL,
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "dataset_rows": len(X),
        "original_feature_count": X.shape[1],
        "metric_names": METRICS,
        "seeds": args.seeds,
        "models": args.models,
        "reference_cv": args.reference_cv,
        "candidate_rounds": args.candidate_rounds,
        "postprocessor": args.postprocessor,
        "ec": {"loss": "bernoulli_kl", "loss_threshold": .05, "feature_unit": "original",
               "ranking_method": "permutation", "repeats": args.ec_repeats},
        "comparison_ec": {"loss": "class_disagreement", "loss_threshold": .05},
        "student": {"type": "ExplainableBoostingClassifier", "interactions": 0,
                    "outer_bags": 1, "n_jobs": 1, "candidate_max_rounds": args.candidate_rounds},
        "split": {"train": .6, "calibration": .2, "test": .2,
                  "stratification": "target x raw gender", "test_use": "evaluation only"},
        "reference": "pooled inner-CV held-out final-predictor metrics of all candidates and teachers within each seed",
        "selection": "mean mechanism score across each candidate's CV metric rows; ties choose lower max_rounds",
        "sensitive_groups": groups.value_counts().to_dict(),
        "environment": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DYLD_FALLBACK_LIBRARY_PATH")},
        "source_sha256": source_fingerprint(),
    }


def prepare_output(output, protocol):
    """Refuse incompatible caches before modifying the existing manifest."""
    output = Path(output)
    expected = digest(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("protocol_sha256") != expected or saved.get("protocol") != protocol:
            raise ValueError("Output contains a different protocol/source/environment; choose another output directory")
        return saved
    if output.exists() and any(output.iterdir()):
        raise ValueError("A nonempty output directory without a matching manifest cannot be reused")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"protocol_sha256": expected, "protocol": protocol, "source_commit_at_start": commit,
                "source_tree_dirty_at_start": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "source_identity": "source_sha256 is authoritative for uncommitted implementation files",
                "historical_results_modified": False}
    write_json(manifest_path, manifest)
    return manifest


def mechanism(name, hesitation=0.0):
    from TWEEF.Aggregation import LWA, GWA, SubjectiveLogic
    from TWEEF.Calibration import QuartileCalibration
    if name == "lwa":
        return LWA(calibration=QuartileCalibration())
    if name == "gwa":
        return GWA(calibration=QuartileCalibration(), hesitation=hesitation)
    if name == "sl":
        return SubjectiveLogic(calibration=QuartileCalibration(), alpha=.5)
    raise ValueError(f"Unknown mechanism: {name}")


def fitted_estimator(name, reference, weights=None, hesitation=0.0):
    from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
    return TrustworthinessEstimator(aggregation=mechanism(name, hesitation)).fit(
        metrics_values=reference, metric_weights=weights or DEFAULT_WEIGHTS)


def mechanism_scores(estimator, metrics):
    from TWEEF.Constants import TRUSTWORTHINESS_SCORE_NAME
    return estimator.transform(metrics)[TRUSTWORTHINESS_SCORE_NAME].to_numpy(dtype=float)


def select_by_cv(cv_frames, estimator):
    """Selection cannot inspect test metrics; ties prefer smaller training budgets."""
    scores = {int(rounds): float(mechanism_scores(estimator, frame).mean()) for rounds, frame in cv_frames.items()}
    result = estimator.aggregation_result_
    direction = getattr(result, "higher_is_better", True)
    chosen = min(scores, key=lambda rounds: ((-1 if direction else 1) * scores[rounds], rounds))
    return chosen, scores


def evaluate_one(predictor, X, y, groups, ranking, ec, seed):
    from TWEEF.Evaluation import evaluate_metrics
    return evaluate_metrics(predictor, X, y, groups, ranking, ec_config=ec, random_state=seed)


def train_seed(seed, args, protocol_sha256):
    from interpret.glassbox import ExplainableBoostingClassifier
    from TWEEF.TrustworthyClassifier import TrustworthyClassifier
    from TWEEF.Evaluation import frozen_ranking

    X, y, groups = load_german()
    split = make_split(y, groups, seed)
    train, calibration, test = (split[name] for name in ("train", "calibration", "test"))
    Xt, yt, gt = X.iloc[train], y.iloc[train], groups.iloc[train]
    Xc, yc, gc = X.iloc[calibration], y.iloc[calibration], groups.iloc[calibration]
    Xq, yq, gq = X.iloc[test], y.iloc[test], groups.iloc[test]
    ec = {"loss": "bernoulli_kl", "loss_threshold": .05, "feature_unit": "original",
          "ranking_method": "permutation", "repeats": args.ec_repeats}
    label_ec = dict(ec, loss="class_disagreement")
    fitted, reference_rows, cases = {}, [], {}
    teacher_models = models(seed)
    for name in args.models:
        for rounds in args.candidate_rounds:
            started = time.monotonic()
            student = ExplainableBoostingClassifier(interactions=0, max_rounds=rounds, outer_bags=1,
                                                    n_jobs=1, random_state=seed)
            workflow = TrustworthyClassifier(given_classifier=teacher_models[name], surrogate=student,
                preprocessor=make_preprocessor(X), aggregation=None,
                metric_weights=DEFAULT_WEIGHTS, effective_complexity=ec,
                reference_cv=args.reference_cv, random_state=seed,
                postprocessor=None if args.postprocessor == "none" else args.postprocessor)
            workflow.fit(Xt, yt, sensitive_features=gt, validation_data=(Xc, yc),
                         validation_sensitive_features=gc)
            reference = workflow.reference_metrics_.loc[:, METRICS].copy()
            fitted[(name, rounds)] = workflow
            for fold, row in reference.iterrows():
                reference_rows.append({"model": name, "candidate_rounds": rounds, "fold": int(fold), **row.to_dict()})
            cases[(name, rounds)] = {"fit_seconds": time.monotonic() - started,
                "teacher_parameters": teacher_models[name].get_params(deep=False),
                "student_parameters": student.get_params(deep=False),
                "reference_metrics": reference.to_dict(orient="records"),
                "feature_ranking": list(workflow.ranking_),
                "reference_provenance": getattr(workflow, "reference_audit_", None),
                "transformed_features": workflow.preprocessor_.get_feature_names_out().tolist()}
            print(f"seed={seed} teacher={name} rounds={rounds}: fit + {len(reference)} CV reference rows", flush=True)

    reference_frame = pd.DataFrame(reference_rows)
    pooled = reference_frame[METRICS]
    # Decisions are frozen here, before the first call involving Xq or yq.
    selections = []
    for operator in ("lwa", "gwa", "sl"):
        estimator = fitted_estimator(operator, pooled)
        for name in args.models:
            cv_frames = {rounds: pd.DataFrame(cases[(name, rounds)]["reference_metrics"])[METRICS]
                         for rounds in args.candidate_rounds}
            chosen, scores = select_by_cv(cv_frames, estimator)
            selections.append({"seed": seed, "model": name, "operator": operator,
                               "selected_rounds": chosen, "cv_scores": scores})

    teacher_reports = {}
    for (name, rounds), workflow in fitted.items():
        started = time.monotonic()
        final = workflow.evaluate(Xq, yq, sensitive_features=gq)
        final_labels = evaluate_one(workflow.predictor_, Xq, yq, gq, workflow.ranking_, label_ec, seed)
        base_kl = evaluate_one(workflow.surrogate_predictor_, Xq, yq, gq, workflow.ranking_, ec, seed)
        base_labels = evaluate_one(workflow.surrogate_predictor_, Xq, yq, gq, workflow.ranking_, label_ec, seed)
        if name not in teacher_reports:
            teacher_ranking = frozen_ranking(workflow.teacher_predictor_, Xt, yt, random_state=seed)
            teacher_labels = evaluate_one(workflow.teacher_predictor_, Xq, yq, gq, teacher_ranking, label_ec, seed)
            teacher_reports[name] = {"metrics_labels": teacher_labels, "feature_ranking": teacher_ranking,
                                    "predictions": workflow.teacher_predictor_.predict(Xq).tolist()}
        cases[(name, rounds)].update({
            "final_metrics_kl": final.metrics,
            "final_metrics_labels": final_labels,
            "surrogate_metrics_kl": base_kl,
            "surrogate_metrics_labels": base_labels,
            "fidelity": final.teacher_student_fidelity,
            "metadata": final.metadata,
            "predictions": {"final": np.asarray(final.predictions).tolist(),
                            "surrogate": workflow.surrogate_predictor_.predict(Xq).tolist()},
            "final_probabilities": workflow.predict_proba(Xq, sensitive_features=gq).tolist(),
            "evaluation_seconds": time.monotonic() - started,
        })
    return {"seed": seed, "protocol_sha256": protocol_sha256,
            "splits": {name: indices.tolist() for name, indices in split.items()},
            "reference": reference_rows, "selections": selections, "teachers": teacher_reports,
            "cases": [{"model": name, "candidate_rounds": rounds, **case} for (name, rounds), case in cases.items()],
            "decisions_frozen_before_test": True}


def weight_configurations():
    yield "uniform", DEFAULT_WEIGHTS
    for metric in METRICS:
        yield "only_" + metric, {name: float(name == metric) for name in METRICS}
    for values in itertools.product((.1, .5, .9), repeat=3):
        total = sum(values)
        yield "weights_" + "_".join(map(str, values)), dict(zip(METRICS, (v / total for v in values)))


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    protocol = manifest["protocol"]
    metric_rows, score_rows, selection_rows, reference_rows, fidelity_rows = [], [], [], [], []
    for seed in protocol["seeds"]:
        payload = json.loads((output / f"seed_{seed}.json").read_text())
        if payload["protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError(f"Incompatible cached seed {seed}")
        refs = pd.DataFrame(payload["reference"])
        reference_rows.extend({"seed": seed, **row} for row in payload["reference"])
        cases = {(case["model"], case["candidate_rounds"]): case for case in payload["cases"]}
        highest = max(protocol["candidate_rounds"])
        fixed = pd.DataFrame([cases[(model, highest)]["final_metrics_kl"] for model in protocol["models"]],
                             index=protocol["models"])[METRICS]
        for model in protocol["models"]:
            case = cases[(model, highest)]
            for label, loss, metric in [
                ("teacher", "class_disagreement", payload["teachers"][model]["metrics_labels"]),
                ("surrogate", "bernoulli_kl", case["surrogate_metrics_kl"]),
                ("surrogate", "class_disagreement", case["surrogate_metrics_labels"]),
                ("final", "bernoulli_kl", case["final_metrics_kl"]),
                ("final", "class_disagreement", case["final_metrics_labels"]),
            ]:
                metric_rows.append({"seed": seed, "model": model, "predictor": label, "ec_loss": loss, **metric})
            fidelity_rows.append({"seed": seed, "model": model, "fit_seconds": case["fit_seconds"],
                                  "evaluation_seconds": case["evaluation_seconds"], **case["fidelity"]})
        for configuration, weights in weight_configurations():
            for operator in ("lwa", "gwa", "sl"):
                estimator = fitted_estimator(operator, refs[METRICS], weights)
                values = mechanism_scores(estimator, fixed)
                for model, score in zip(fixed.index, values):
                    score_rows.append({"seed": seed, "model": model, "operator": operator,
                                       "configuration": configuration, "hesitation": 0.0, "score": score})
        for hesitation in (.1, .25, .5, .75, 1.0):
            estimator = fitted_estimator("gwa", refs[METRICS], hesitation=hesitation)
            for model, score in zip(fixed.index, mechanism_scores(estimator, fixed)):
                score_rows.append({"seed": seed, "model": model, "operator": "gwa",
                                   "configuration": "hesitation_sweep", "hesitation": hesitation, "score": score})
        for decision in payload["selections"]:
            chosen = decision["selected_rounds"]
            case = cases[(decision["model"], chosen)]
            estimator = fitted_estimator(decision["operator"], refs[METRICS])
            score = mechanism_scores(estimator, pd.DataFrame([case["final_metrics_kl"]])[METRICS])[0]
            selection_rows.append({"seed": seed, "model": decision["model"], "operator": decision["operator"],
                                   "selected_rounds": chosen, "cv_score": decision["cv_scores"][str(chosen)],
                                   "test_score": score, **case["final_metrics_kl"]})
    metrics, scores, selections = map(pd.DataFrame, (metric_rows, score_rows, selection_rows))
    metrics.to_csv(output / "raw_metrics.csv", index=False)
    scores.to_csv(output / "mechanism_scores.csv", index=False)
    selections.to_csv(output / "full_flow_selections.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(output / "reference_metrics.csv", index=False)
    pd.DataFrame(fidelity_rows).to_csv(output / "fidelity.csv", index=False)
    metric_summary = metrics.groupby(["model", "predictor", "ec_loss"], sort=False)[METRICS].agg(["mean", "std"])
    metric_summary.columns = [f"{name}_{statistic}" for name, statistic in metric_summary.columns]
    metric_summary.reset_index().to_csv(output / "metrics_by_model.csv", index=False)
    uniform = scores[scores.configuration.eq("uniform")]
    uniform.groupby(["model", "operator"]).score.agg(["mean", "std"]).reset_index().to_csv(output / "mechanisms_by_model.csv", index=False)
    rank = uniform.copy()
    rank["rank"] = rank.groupby(["seed", "operator"]).score.rank(method="average", ascending=False)
    rank.to_csv(output / "mechanism_rankings.csv", index=False)
    agreement = []
    for (seed, configuration), frame in scores[scores.configuration.ne("hesitation_sweep")].groupby(["seed", "configuration"]):
        pivot = frame.pivot(index="model", columns="operator", values="score")
        for left, right in itertools.combinations(("lwa", "gwa", "sl"), 2):
            correlation = (pivot[left].rank().corr(pivot[right].rank())
                           if pivot[left].nunique() > 1 and pivot[right].nunique() > 1 else float("nan"))
            agreement.append({"seed": seed, "configuration": configuration, "operator_a": left, "operator_b": right,
                              "spearman": correlation, "ties_a": len(pivot) - pivot[left].nunique(),
                              "ties_b": len(pivot) - pivot[right].nunique()})
    pd.DataFrame(agreement).to_csv(output / "rank_agreement.csv", index=False)
    protocol_comparison(output, metrics, protocol)
    plot_results(output, metrics, uniform)
    write_json(output / "summary.json", {"protocol": PROTOCOL, "seeds": protocol["seeds"], "models": protocol["models"],
        "fitted_final_candidates": len(protocol["seeds"]) * len(protocol["models"]) * len(protocol["candidate_rounds"]),
        "reference_rows": len(reference_rows), "score_rows": len(scores), "full_flow_decisions": len(selections),
        "interpretation": "Descriptive paired repeated splits on one dataset. Higher native mechanism scores do not establish greater real trustworthiness.",
        "historical_ec_comparability": "EC perturbation unit, ranking protocol, final predictor and loss mode differ; historical score changes cannot be attributed to aggregation alone."})


def protocol_comparison(output, metrics, protocol):
    rows = [
        ("Final predictor", "Original teacher only", "EBM surrogate plus persisted group postprocessing"),
        ("EC variables", "61 encoded columns", "20 original variables"),
        ("EC ranking", "Native trees; permutation on evaluation labels for LR/SVM", "Permutation on training only, frozen before test"),
        ("EC loss", "KL if probabilities; disagreement otherwise", "Explicit KL for final predictors; common disagreement for teacher/student comparison"),
        ("Mechanism reference", "Calibration partition", "Pooled training-only inner-CV metrics"),
        ("Comparison", "Aggregation on original-model metrics", "Fixed final-predictor estimation and CV-only candidate selection"),
    ]
    pd.DataFrame(rows, columns=["aspect", "historical_german_ec_kl", "new_protocol"]).to_csv(output / "protocol_comparison.csv", index=False)
    historical = ROOT / "experiments/results/german_ec_kl"
    audit = []
    for seed in protocol["seeds"]:
        path = historical / f"metrics_seed_{seed}.json"
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        new = json.loads((output / f"seed_{seed}.json").read_text())
        for model in protocol["models"]:
            before = old["models"][model]["test"]["predictions"]
            after = new["teachers"][model]["predictions"]
            audit.append({"seed": seed, "model": model, "teacher_predictions_identical": before == after,
                          "split_indices_identical": old["splits"] == new["splits"]})
    pd.DataFrame(audit).to_csv(output / "historical_teacher_audit.csv", index=False)
    old_path = historical / "raw_metrics.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old.partition.eq("test")][["seed", "model", "Accuracy", "equal Opportunity"]]
        new = metrics[metrics.predictor.eq("final") & metrics.ec_loss.eq("bernoulli_kl")]
        pairs = old.merge(new, on=["seed", "model"], suffixes=("_historical_teacher", "_new_final"), validate="one_to_one")
        pairs["accuracy_delta"] = pairs.Accuracy_new_final - pairs.Accuracy_historical_teacher
        pairs["equal_opportunity_delta"] = pairs["equal Opportunity_new_final"] - pairs["equal Opportunity_historical_teacher"]
        pairs.to_csv(output / "historical_predictive_comparison.csv", index=False)


def plot_results(output, metrics, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [name for name in MODEL_ORDER if name in set(metrics.model)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, metric in zip(axes[:2], METRICS[:2]):
        for predictor, loss, offset, color in [("teacher", "class_disagreement", -.12, "#bd6d29"),
                                               ("final", "bernoulli_kl", .12, "#2467a4")]:
            frame = metrics[metrics.predictor.eq(predictor) & metrics.ec_loss.eq(loss)].groupby("model")[metric].agg(["mean", "std"]).reindex(names)
            ax.errorbar(frame["mean"], np.arange(len(names)) + offset, xerr=frame["std"].fillna(0),
                        fmt="o", capsize=2, color=color, label=predictor)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlim(0, 1.05)
        ax.set_title(metric)
        ax.grid(axis="x", color=".9")
    for operator, color in [("lwa", "#b06b29"), ("gwa", "#438b5c"), ("sl", "#2467a4")]:
        frame = scores[scores.operator.eq(operator)].groupby("model").score.mean().reindex(names)
        axes[2].plot(range(len(names)), frame, "o-", label=operator.upper(), color=color)
    axes[2].set_xticks(range(len(names)), names, rotation=40)
    axes[2].set_title("Native mechanism scores (fixed predictor)")
    axes[2].set_ylim(-.05, 1.05)
    axes[2].legend()
    axes[0].legend()
    fig.suptitle("German: teacher and final surrogate pipeline; mean ± SD across repeated splits")
    fig.tight_layout()
    fig.savefig(output / "german_final_surrogate.png", dpi=220, facecolor="white")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/german_final_surrogate")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--candidate-rounds", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--reference-cv", type=int, default=5)
    parser.add_argument("--ec-repeats", type=int, default=3)
    parser.add_argument("--postprocessor", choices=["none", "calibrated_equalized_odds"], default="calibrated_equalized_odds")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Seeds and models must be unique")
    if min(args.candidate_rounds) <= 0 or args.reference_cv < 2 or args.ec_repeats < 1:
        parser.error("Positive training budgets/repeats and at least two CV folds are required")
    args.candidate_rounds = sorted(set(args.candidate_rounds))
    return args


def main(argv=None):
    args = parse_args(argv)
    protocol = build_protocol(args)
    manifest = prepare_output(args.output, protocol)
    for seed in args.seeds:
        path = args.output / f"seed_{seed}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("protocol_sha256") != manifest["protocol_sha256"] or cached.get("seed") != seed:
                raise ValueError(f"Incompatible seed cache: {path}")
            print(f"seed={seed}: using verified cache", flush=True)
        else:
            write_json(path, train_seed(seed, args, manifest["protocol_sha256"]))
    summarize(args.output)
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
