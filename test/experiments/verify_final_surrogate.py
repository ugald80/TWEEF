"""Audit a completed German final-surrogate experiment without retraining.

Run with --verify to write an audit report, including failures. Existing original
verification.json files are preserved. Use --archived-source after a layout-only
move to verify original source and evidence from Git and write a separate
relocation_verification.json. Source hashes,
cached metrics, CV choices, principal exported tables and prediction consistency
are checked. EC perturbations cannot be replayed without fitted models; this
auditor checks the saved loss curves and their declared original-variable unit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.experiments.german_final_surrogate import (
    DEFAULT_WEIGHTS, METRICS, MODEL_ORDER, PACKAGES, PROTOCOL, DATASET,
    digest, fitted_estimator, load_german, mechanism_scores, select_by_cv,
    source_fingerprint, weight_configurations, write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source_hashes(root, recorded):
    require(isinstance(recorded, dict) and bool(recorded), "Missing source fingerprint")
    for name, expected in recorded.items():
        path = (Path(root) / name).resolve()
        require(path.is_relative_to(Path(root).resolve()), f"Source path escapes repository: {name}")
        require(path.is_file(), f"Missing source file: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Source hash mismatch: {name}")


def _git_blob(root, commit, name):
    require(bool(re.fullmatch(r"[0-9a-f]{40,64}", commit)), "Invalid recorded Git commit")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Archived path escapes repository: {name}")
    return subprocess.run(["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root,
                          capture_output=True, check=True).stdout


def validate_archived_source_hashes(root, recorded, source_commit):
    """Validate executed Git blobs and the unchanged computational package."""
    require(isinstance(recorded, dict) and bool(recorded), "Missing archived source fingerprint")
    root = Path(root).resolve()
    current_core = {str(path.relative_to(root)) for path in (root / "TWEEF").glob("*.py")}
    archived_core = {name for name in recorded if name.startswith("TWEEF/")}
    require(current_core == archived_core, "Current core source inventory differs from the archived experiment")
    for name, expected in recorded.items():
        saved = _git_blob(root, source_commit, name)
        require(hashlib.sha256(saved).hexdigest() == expected, f"Archived Git source hash mismatch: {name}")
        if name.startswith("TWEEF/"):
            path = (root / name).resolve()
            require(path.is_relative_to(root), f"Source path escapes repository: {name}")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Current core source differs from archived source: {name}")


def validate_relocated_helpers(root, source_commit):
    """Only import/layout paths may change in the audit's numerical helpers."""
    for old_name, functions in (
        ("experiments/german_final_surrogate.py", (
            "mechanism", "fitted_estimator", "mechanism_scores", "select_by_cv", "weight_configurations",
        )),
        ("experiments/german_aggregation.py", ("load_german",)),
    ):
        original = ast.parse(_git_blob(root, source_commit, old_name).decode("utf-8"))
        current = ast.parse((Path(root) / "test" / old_name).read_text())
        for name in functions:
            before = next((node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
            after = next((node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
            require(before is not None and after is not None, f"Missing archived recomputation helper: {name}")
            require(ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False),
                    f"Recomputation helper changed beyond relocation: {name}")


def relocated_artifact_path(root, output, recorded_name):
    relative = Path(recorded_name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Recorded artifact path escapes its directory: {recorded_name}")
    if relative.parts[:2] == ("experiments", "results"):
        base = Path(root).resolve()
        path = (base / "test" / relative).resolve()
    else:
        base = Path(output).resolve()
        path = (base / relative).resolve()
    require(path.is_relative_to(base), f"Recorded artifact path escapes its directory: {recorded_name}")
    return path


def validate_recorded_artifacts(root, output, recorded):
    require(isinstance(recorded, dict) and bool(recorded), "Original artifact hashes are missing")
    for name, expected in recorded.items():
        path = relocated_artifact_path(root, output, name)
        require(path.is_file(), f"Missing relocated artifact: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Relocated artifact hash mismatch: {name}")


def original_verification_commit(root, source_commit, recorded_path):
    """Choose the first original evidence commit, rather than a later rewrite."""
    require(bool(re.fullmatch(r"[0-9a-f]{40,64}", source_commit)), "Invalid recorded Git commit")
    commits = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", f"{source_commit}..HEAD", "--", recorded_path],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for commit in commits:
        try:
            _git_blob(root, commit, recorded_path)
        except subprocess.CalledProcessError:
            continue
        return commit
    raise ValueError("Original verification evidence was not found in Git history")


def audit_original_evidence(audit, manifest):
    recorded_path = "experiments/results/german_final_surrogate/verification.json"
    commit = original_verification_commit(ROOT, manifest["source_commit_at_start"], recorded_path)
    original = _git_blob(ROOT, commit, recorded_path)
    require(audit.read("verification.json") == original, "Original verification.json differs from its first Git evidence")
    evidence = json.loads(original)
    require(evidence.get("passed") is True, "Original verification did not pass")
    original_verifier = _git_blob(ROOT, commit, "experiments/verify_final_surrogate.py")
    require(hashlib.sha256(original_verifier).hexdigest() == evidence["verifier_sha256"],
            "Original verifier does not match its recorded Git evidence")
    validate_recorded_artifacts(ROOT, audit.output, evidence["artifacts_sha256"])
    audit.archived_evidence = {"commit": commit, "verification_path": recorded_path,
                               "verification_sha256": hashlib.sha256(original).hexdigest()}


def validate_predictions(case, n_rows=200):
    probabilities = np.asarray(case["final_probabilities"], dtype=float)
    labels = np.asarray(case["predictions"]["final"])
    base = np.asarray(case["predictions"]["surrogate"])
    require(probabilities.shape == (n_rows, 2), "Final probabilities must have shape (test rows, 2)")
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Final probabilities must be finite and in [0,1]")
    require(np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12), "Probability rows do not sum to one")
    for values in (labels, base):
        require(values.shape == (n_rows,) and np.isin(values, [0, 1]).all(), "Invalid binary prediction vector")
    require(np.array_equal(labels, (probabilities[:, 1] >= 0.5).astype(int)),
            "Final labels disagree with the deployed 0.5 probability threshold")


def validate_provenance(audit, train_indices, folds, original_names):
    train = set(train_indices)
    require(isinstance(audit, list) and len(audit) == folds, "Missing inner-CV provenance rows")
    scored = []
    for fold, row in enumerate(audit):
        require(row["candidate"] == 0 and row["fold"] == fold, "Unexpected candidate/fold provenance identity")
        parts = [row[name] for name in ("fit_index", "calibration_index", "score_index")]
        require(all(len(values) == len(set(values)) for values in parts), "Duplicate observations within a CV partition")
        fit, calibration, score = map(set, parts)
        require(bool(fit) and bool(calibration) and bool(score), "CV fitting/calibration/scoring must be nonempty")
        require(not (fit & calibration or fit & score or calibration & score), "Inner-CV partitions overlap")
        require(fit | calibration | score == train, "Inner CV contains held-out/test rows or omits training rows")
        require(len(row["ranking"]) == len(original_names) and set(row["ranking"]) == set(original_names),
                "Inner ranking is not a permutation of original variables")
        scored.extend(row["score_index"])
    require(len(scored) == len(train) and set(scored) == train, "Inner score folds must cover training exactly once")


def finite_metrics(values):
    require(set(values) == set(METRICS), "Unexpected metric schema")
    array = np.asarray([values[name] for name in METRICS], dtype=float)
    require(np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(), "Metrics must be finite and in [0,1]")


def check_ranking(ranking, names):
    require(len(ranking) == 20 and set(ranking) == set(names), "Ranking is not a permutation of the 20 original variables")


def check_ec(case, protocol, names):
    metadata = case["metadata"]
    require(metadata["final_predictor"] == "surrogate" and metadata["test_used"] is False,
            "Final predictor or test-use declaration is invalid")
    require(metadata["effective_complexity"] == protocol["ec"], "EC protocol differs between cached cases")
    detail = metadata["effective_complexity_details"]
    require(detail["n_original_features"] == 20 and detail["loss"] == "bernoulli_kl" and detail["loss_units"] == "nats",
            "EC must use original variables and KL in nats")
    require(detail["loss_threshold"] == protocol["ec"]["loss_threshold"], "EC tolerance changed")
    check_ranking(detail["ranking"], names)
    require(detail["ranking"] == case["feature_ranking"], "Evaluation ranking differs from the frozen ranking")
    k = detail["k"]
    require(isinstance(k, int) and 1 <= k <= 20, "Invalid selected EC feature count")
    require(np.isclose(case["final_metrics_kl"][METRICS[2]], 1 - k / 20, rtol=0, atol=1e-12),
            "EC score does not match selected original-variable count")
    curve = detail["curve"]
    require([row["k"] for row in curve] == list(range(1, k + 1)), "Incomplete EC loss curve")
    for number, row in enumerate(curve, 1):
        losses = np.asarray(row["losses"], dtype=float)
        require(losses.shape == (protocol["ec"]["repeats"],) and np.isfinite(losses).all() and (losses >= 0).all(),
                "Invalid repeated EC losses")
        require(np.isclose(losses.mean(), row["mean_loss"], rtol=0, atol=1e-12), "EC repeated-loss mean mismatch")
        threshold = detail["loss_threshold"]
        tolerance = 4 * np.finfo(float).eps * max(abs(row["mean_loss"]), abs(threshold))
        require((row["mean_loss"] <= threshold + tolerance) == (number == k),
                "EC did not select the first feature count meeting the tolerance")


class Audit:
    def __init__(self, output, *, archived_source=False):
        self.output = Path(output).resolve()
        self.checks = []
        self.artifacts = {}
        self.archived_source = archived_source
        self.archived_evidence = None

    def run(self, name, function):
        try:
            value = function()
            self.checks.append({"name": name, "passed": True})
            return value
        except Exception as error:
            self.checks.append({"name": name, "passed": False,
                                "error": f"{type(error).__name__}: {error}"})
            return None

    def read(self, name):
        path = self.output / name
        content = path.read_bytes()
        self.artifacts[name] = hashlib.sha256(content).hexdigest()
        return content

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        self.read(name)
        return pd.read_csv(self.output / name)

    def result(self):
        return {
            "passed": bool(self.checks) and all(check["passed"] for check in self.checks),
            "output": str(self.output),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checks": self.checks, "artifacts_sha256": self.artifacts,
            "source_mode": "archived" if self.archived_source else "current",
            "archived_evidence": self.archived_evidence,
            "limits": "Recomputes aggregation/CV selection and saved prediction consistency; does not refit models or regenerate EC perturbation predictions.",
        }


def frame_matches(actual, expected, keys):
    require(set(actual.columns) == set(expected.columns), "Exported table columns differ from cached results")
    require(len(actual) == len(expected), "Exported table row count differs from cached results")
    require(not actual.duplicated(keys).any(), "Exported table has duplicate identities")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[right.columns], right, check_dtype=False,
                                  check_exact=False, rtol=0, atol=2e-12)


def audit_sources(audit, manifest):
    protocol = manifest["protocol"]
    require(manifest["protocol_sha256"] == digest(protocol), "Protocol digest mismatch")
    source_commit = manifest["source_commit_at_start"]
    if audit.archived_source:
        validate_archived_source_hashes(ROOT, protocol["source_sha256"], source_commit)
        validate_relocated_helpers(ROOT, source_commit)
    else:
        require(protocol["source_sha256"] == source_fingerprint(), "Current source fingerprint differs from the experiment")
        validate_source_hashes(ROOT, protocol["source_sha256"])
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT)
    require(ancestor.returncode == 0, "HEAD does not descend from the recorded source commit")
    for name, expected in protocol["source_sha256"].items():
        saved = subprocess.run(["git", "show", f"{source_commit}:{name}"], cwd=ROOT,
                               capture_output=True, check=True).stdout
        require(hashlib.sha256(saved).hexdigest() == expected, f"Source commit does not identify the executed file: {name}")
    require(manifest["source_tree_dirty_at_start"] is False, "The agreed main run must begin from committed source")
    require(manifest["historical_results_modified"] is False, "Historical results modification was declared")
    require(protocol["environment"] == {name: importlib.metadata.version(name) for name in PACKAGES},
            "Verifier environment differs from the recorded package versions")


def audit_protocol(protocol):
    require(protocol["protocol"] == PROTOCOL, "Wrong experiment protocol")
    require(protocol["seeds"] == list(range(10)) and protocol["models"] == MODEL_ORDER, "The full 10-seed/eight-model experiment is required")
    require(protocol["candidate_rounds"] == [100, 500] and protocol["reference_cv"] == 5, "Wrong candidate/CV budget")
    require(protocol["postprocessor"] == "calibrated_equalized_odds", "Main experiment must evaluate the persisted postprocessed predictor")
    require(protocol["dataset_rows"] == 1000 and protocol["original_feature_count"] == 20, "Wrong dataset dimensions")
    require(protocol["dataset_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest(), "Dataset hash mismatch")
    require(protocol["metric_names"] == METRICS and protocol["ec"] == {
        "loss": "bernoulli_kl", "loss_threshold": .05, "feature_unit": "original",
        "ranking_method": "permutation", "repeats": 3,
    }, "Wrong agreed main EC metric protocol")
    require(protocol["comparison_ec"] == {"loss": "class_disagreement", "loss_threshold": .05},
            "Wrong common teacher/student comparison loss")


def audit_seed(payload, manifest, seed, X, y, groups):
    protocol = manifest["protocol"]
    require(payload["seed"] == seed and payload["protocol_sha256"] == manifest["protocol_sha256"], "Incompatible cached seed/source fingerprint")
    require(payload["decisions_frozen_before_test"] is True, "Selection was not declared frozen before test")
    splits = payload["splits"]
    require(set(splits) == {"train", "calibration", "test"}, "Wrong split names")
    require([len(splits[name]) for name in ("train", "calibration", "test")] == [600, 200, 200], "Wrong 60/20/20 split sizes")
    all_rows = sum((splits[name] for name in ("train", "calibration", "test")), [])
    require(len(set(all_rows)) == 1000 and set(all_rows) == set(range(1000)), "Splits overlap or omit German observations")
    expected_cases = set(itertools.product(MODEL_ORDER, protocol["candidate_rounds"]))
    cases = {(case["model"], case["candidate_rounds"]): case for case in payload["cases"]}
    require(len(payload["cases"]) == len(cases) == 16 and set(cases) == expected_cases, "Missing or duplicate fitted candidate cases")
    references = pd.DataFrame(payload["reference"])
    require(len(references) == 80 and not references.duplicated(["model", "candidate_rounds", "fold"]).any(), "Invalid pooled reference rows")
    require(set(payload["teachers"]) == set(MODEL_ORDER), "Teacher audit is incomplete")
    test = splits["test"]
    truth = y.iloc[test].to_numpy()
    sensitive = groups.iloc[test].to_numpy()
    for model, teacher in payload["teachers"].items():
        check_ranking(teacher["feature_ranking"], X.columns)
        finite_metrics(teacher["metrics_labels"])
        labels = np.asarray(teacher["predictions"])
        require(labels.shape == (200,) and np.isin(labels, [0, 1]).all(), f"Invalid teacher predictions: {model}")
    for (model, rounds), case in cases.items():
        parameters = case["student_parameters"]
        require(all(parameters[name] == expected for name, expected in (
            ("interactions", 0), ("max_rounds", rounds), ("outer_bags", 1), ("n_jobs", 1), ("random_state", seed),
        )), "Student configuration differs from the declared candidate budget")
        validate_predictions(case)
        check_ranking(case["feature_ranking"], X.columns)
        validate_provenance(case["reference_provenance"], splits["train"], 5, X.columns)
        for field in ("final_metrics_kl", "final_metrics_labels", "surrogate_metrics_kl", "surrogate_metrics_labels"):
            finite_metrics(case[field])
        check_ec(case, protocol, X.columns)
        require(len(case["reference_metrics"]) == 5, "Wrong per-candidate reference count")
        for values in case["reference_metrics"]:
            finite_metrics(values)
        rows = references[references.model.eq(model) & references.candidate_rounds.eq(rounds)].sort_values("fold")
        require(rows.fold.tolist() == list(range(5)), "Candidate reference fold identities are invalid")
        np.testing.assert_allclose(rows[METRICS], pd.DataFrame(case["reference_metrics"])[METRICS], rtol=0, atol=1e-12)
        labels = np.asarray(case["predictions"]["final"])
        base = np.asarray(case["predictions"]["surrogate"])
        for predictions, field in ((labels, "final_metrics_kl"), (base, "surrogate_metrics_kl")):
            require(np.isclose(case[field][METRICS[0]], np.mean(predictions == truth), rtol=0, atol=1e-12), "Cached accuracy does not match predictions")
            tpr = [np.mean(predictions[(sensitive == group) & (truth == 1)] == 1) for group in pd.unique(sensitive)]
            require(np.isclose(case[field][METRICS[1]], 1 - abs(tpr[0] - tpr[1]), rtol=0, atol=1e-12), "Cached Equal Opportunity does not match predictions")
        teacher = np.asarray(payload["teachers"][model]["predictions"])
        require(np.isclose(case["fidelity"]["teacher_student_class_agreement"], np.mean(teacher == base), rtol=0, atol=1e-12), "Teacher/student agreement mismatch")
        require(np.isclose(case["fidelity"]["teacher_final_class_agreement"], np.mean(teacher == labels), rtol=0, atol=1e-12), "Teacher/final agreement mismatch")
    choices = {(row["model"], row["operator"]): row for row in payload["selections"]}
    require(len(payload["selections"]) == len(choices) == 24 and set(choices) == set(itertools.product(MODEL_ORDER, ("lwa", "gwa", "sl"))), "Invalid selection identities")
    for operator in ("lwa", "gwa", "sl"):
        estimator = fitted_estimator(operator, references[METRICS])
        for model in MODEL_ORDER:
            frames = {rounds: pd.DataFrame(cases[(model, rounds)]["reference_metrics"])[METRICS] for rounds in (100, 500)}
            chosen, scores = select_by_cv(frames, estimator)
            recorded = choices[(model, operator)]
            require(recorded["seed"] == seed and recorded["selected_rounds"] == chosen, "Saved decision differs from CV-only recomputation")
            require(set(recorded["cv_scores"]) == {"100", "500"}, "CV choice omits a candidate score")
            for rounds, score in scores.items():
                require(np.isclose(recorded["cv_scores"][str(rounds)], score, rtol=0, atol=1e-12), "Saved CV score differs from recomputation")
    return payload


def audit_exports(audit, payloads):
    expected_scores, expected_choices, expected_refs, expected_metrics = [], [], [], []
    for payload in payloads:
        seed = payload["seed"]
        cases = {(row["model"], row["candidate_rounds"]): row for row in payload["cases"]}
        refs = pd.DataFrame(payload["reference"])
        expected_refs.extend({"seed": seed, **row} for row in payload["reference"])
        fixed = pd.DataFrame([cases[(model, 500)]["final_metrics_kl"] for model in MODEL_ORDER], index=MODEL_ORDER)[METRICS]
        for configuration, weights in weight_configurations():
            for operator in ("lwa", "gwa", "sl"):
                values = mechanism_scores(fitted_estimator(operator, refs[METRICS], weights), fixed)
                expected_scores.extend({"seed": seed, "model": model, "operator": operator,
                    "configuration": configuration, "hesitation": 0.0, "score": score}
                    for model, score in zip(MODEL_ORDER, values))
        for hesitation in (.1, .25, .5, .75, 1.0):
            values = mechanism_scores(fitted_estimator("gwa", refs[METRICS], hesitation=hesitation), fixed)
            expected_scores.extend({"seed": seed, "model": model, "operator": "gwa",
                "configuration": "hesitation_sweep", "hesitation": hesitation, "score": score}
                for model, score in zip(MODEL_ORDER, values))
        for row in payload["selections"]:
            case = cases[(row["model"], row["selected_rounds"])]
            score = mechanism_scores(fitted_estimator(row["operator"], refs[METRICS]), pd.DataFrame([case["final_metrics_kl"]])[METRICS])[0]
            expected_choices.append({"seed": seed, "model": row["model"], "operator": row["operator"],
                "selected_rounds": row["selected_rounds"], "cv_score": row["cv_scores"][str(row["selected_rounds"])],
                "test_score": score, **case["final_metrics_kl"]})
        for model in MODEL_ORDER:
            case = cases[(model, 500)]
            for predictor, loss, metrics in (
                ("teacher", "class_disagreement", payload["teachers"][model]["metrics_labels"]),
                ("surrogate", "bernoulli_kl", case["surrogate_metrics_kl"]),
                ("surrogate", "class_disagreement", case["surrogate_metrics_labels"]),
                ("final", "bernoulli_kl", case["final_metrics_kl"]),
                ("final", "class_disagreement", case["final_metrics_labels"]),
            ):
                expected_metrics.append({"seed": seed, "model": model, "predictor": predictor, "ec_loss": loss, **metrics})
    for filename, expected, keys, count in (
        ("mechanism_scores.csv", expected_scores, ["seed", "model", "operator", "configuration", "hesitation"], 7840),
        ("full_flow_selections.csv", expected_choices, ["seed", "model", "operator"], 240),
        ("reference_metrics.csv", expected_refs, ["seed", "model", "candidate_rounds", "fold"], 800),
        ("raw_metrics.csv", expected_metrics, ["seed", "model", "predictor", "ec_loss"], 400),
    ):
        actual = audit.csv(filename)
        require(len(actual) == count, f"Wrong completed experiment row count: {filename}")
        frame_matches(actual, pd.DataFrame(expected), keys)
    scores = pd.DataFrame(expected_scores)
    vacuous = scores[scores.hesitation.eq(1)]
    require(len(vacuous) == 80 and np.allclose(vacuous.score, 0.5, rtol=0, atol=1e-12), "Full GWA hesitation is not neutral")
    for filename, count in (("fidelity.csv", 80), ("metrics_by_model.csv", 40), ("mechanisms_by_model.csv", 24),
                            ("mechanism_rankings.csv", 240), ("rank_agreement.csv", 930),
                            ("protocol_comparison.csv", 6), ("historical_predictive_comparison.csv", 80)):
        require(len(audit.csv(filename)) == count, f"Wrong table count: {filename}")
    require(len(audit.read("german_final_surrogate.png")) > 1000, "Missing result figure")


def audit_historical(audit, payloads):
    table = audit.csv("historical_teacher_audit.csv")
    require(len(table) == 80 and not table.duplicated(["seed", "model"]).any(), "Historical teacher audit is incomplete")
    require(table.teacher_predictions_identical.eq(True).all() and table.split_indices_identical.eq(True).all(), "Historical teachers or splits changed")
    for payload in payloads:
        path = ROOT / "test/experiments/results/german_ec_kl" / f"metrics_seed_{payload['seed']}.json"
        original = path.read_bytes()
        audit.artifacts[str(path.relative_to(ROOT))] = hashlib.sha256(original).hexdigest()
        historical = json.loads(original)
        require(historical["splits"] == payload["splits"], "Historical split cache differs")
        for model in MODEL_ORDER:
            require(historical["models"][model]["test"]["predictions"] == payload["teachers"][model]["predictions"], "Historical teacher prediction cache differs")


def verify(output, *, archived_source=False):
    audit = Audit(output, archived_source=archived_source)
    summary = audit.run("completed_summary", lambda: audit.json("summary.json"))
    manifest = audit.run("manifest", lambda: audit.json("manifest.json"))
    if summary is None or manifest is None:
        return audit.result()
    protocol = manifest["protocol"]
    audit.run("complete_main_protocol", lambda: audit_protocol(protocol))
    if archived_source:
        audit.run("original_git_verification_and_artifact_identity", lambda: audit_original_evidence(audit, manifest))
    audit.run("source_identity_and_environment", lambda: audit_sources(audit, manifest))
    X, y, groups = load_german()
    payloads = []
    for seed in range(10):
        payload = audit.run(f"seed_{seed}_cache_splits_provenance_predictions_ec_cv", lambda seed=seed:
                           audit_seed(audit.json(f"seed_{seed}.json"), manifest, seed, X, y, groups))
        if payload is not None:
            payloads.append(payload)
    def completed_counts():
        require(summary["protocol"] == PROTOCOL and summary["seeds"] == list(range(10)) and summary["models"] == MODEL_ORDER, "Summary identity mismatch")
        for field, expected in (("fitted_final_candidates", 160), ("reference_rows", 800), ("score_rows", 7840), ("full_flow_decisions", 240)):
            require(summary[field] == expected, f"Summary count mismatch: {field}")
        require(len(payloads) == 10, "Not all seed caches passed validation")
    audit.run("completed_main_counts", completed_counts)
    audit.run("exported_tables_and_mechanism_recomputation", lambda: audit_exports(audit, payloads))
    audit.run("historical_teacher_and_split_identity", lambda: audit_historical(audit, payloads))
    if archived_source:
        audit.run("source_still_matches_after_audit", lambda: (
            validate_archived_source_hashes(ROOT, protocol["source_sha256"], manifest["source_commit_at_start"]),
            validate_relocated_helpers(ROOT, manifest["source_commit_at_start"]),
        ))
    else:
        audit.run("source_still_matches_after_audit", lambda: validate_source_hashes(ROOT, protocol["source_sha256"]))
    return audit.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test/experiments/results/german_final_surrogate")
    parser.add_argument("--verify", action="store_true", help="Write an audit report without replacing original verification evidence")
    parser.add_argument("--archived-source", action="store_true",
                        help="Audit relocated results against original Git source and evidence; write relocation_verification.json")
    args = parser.parse_args(argv)
    try:
        result = verify(args.output, archived_source=args.archived_source)
    except Exception as error:
        audit = Audit(args.output, archived_source=args.archived_source)
        audit.run("audit_setup", lambda: (_ for _ in ()).throw(error))
        result = audit.result()
    if args.verify:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.archived_source:
            filename = "relocation_verification.json"
        elif (args.output / "verification.json").exists():
            filename = "verification_current.json"
        else:
            filename = "verification.json"
        write_json(args.output / filename, result)
    failed = [check for check in result["checks"] if not check["passed"]]
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "failures": failed}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
