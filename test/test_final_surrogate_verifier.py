"""Tamper detection without running or rewriting an experiment."""

from copy import deepcopy
import hashlib
import json
import importlib

import pytest

from test.experiments.verify_final_surrogate import (
    main, validate_predictions, validate_provenance, validate_source_hashes,
    validate_archived_source_hashes, validate_recorded_artifacts,
    validate_relocated_helpers,
)

verifier = importlib.import_module("test.experiments.verify_final_surrogate")


def test_source_hash_audit_detects_changed_content_and_path_escape(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("value = 1\n")
    recorded = {"source.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_source_hashes(tmp_path, recorded)
    path.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        validate_source_hashes(tmp_path, recorded)
    with pytest.raises(ValueError, match="escapes repository"):
        validate_source_hashes(tmp_path, {"../other.py": "unused"})


def test_predictions_cannot_disagree_with_saved_deployment_probabilities():
    case = {"final_probabilities": [[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]],
            "predictions": {"final": [0, 1, 1], "surrogate": [0, 0, 1]}}
    validate_predictions(case, n_rows=3)
    changed = deepcopy(case)
    changed["predictions"]["final"][1] = 0
    with pytest.raises(ValueError, match="probability threshold"):
        validate_predictions(changed, n_rows=3)
    changed = deepcopy(case)
    changed["final_probabilities"][0][0] = 0.7
    with pytest.raises(ValueError, match="sum to one"):
        validate_predictions(changed, n_rows=3)


def test_provenance_detects_training_leakage_and_failed_verification_is_saved(tmp_path):
    rows = [
        {"candidate": 0, "fold": 0, "fit_index": [3, 4], "calibration_index": [5],
         "score_index": [0, 1, 2], "ranking": ["a", "b"]},
        {"candidate": 0, "fold": 1, "fit_index": [0, 1], "calibration_index": [2],
         "score_index": [3, 4, 5], "ranking": ["b", "a"]},
    ]
    validate_provenance(rows, list(range(6)), 2, ["a", "b"])
    leaked = deepcopy(rows)
    leaked[0]["fit_index"].append(2)
    with pytest.raises(ValueError, match="overlap"):
        validate_provenance(leaked, list(range(6)), 2, ["a", "b"])
    leaked = deepcopy(rows)
    leaked[0]["fit_index"][0] = 99
    with pytest.raises(ValueError, match="held-out/test"):
        validate_provenance(leaked, list(range(6)), 2, ["a", "b"])
    # Missing main output is a nonzero failure with an auditable record, never
    # an empty success report or a silently accepted partial experiment.
    assert main(["--output", str(tmp_path), "--verify"]) == 1
    saved = json.loads((tmp_path / "verification.json").read_text())
    assert saved["passed"] is False
    assert len(saved["verifier_sha256"]) == 64
    assert any(not check["passed"] and "summary.json" in check["error"] for check in saved["checks"])


def test_archived_source_requires_original_git_bytes_and_unchanged_current_core(tmp_path, monkeypatch):
    core = tmp_path / "TWEEF" / "Evaluation.py"
    core.parent.mkdir()
    core.write_bytes(b"VALUE = 1\n")
    blobs = {"TWEEF/Evaluation.py": core.read_bytes(), "experiments/runner.py": b"old layout runner\n"}
    recorded = {name: hashlib.sha256(value).hexdigest() for name, value in blobs.items()}
    monkeypatch.setattr(verifier, "_git_blob", lambda root, commit, name: blobs[name])
    validate_archived_source_hashes(tmp_path, recorded, "a" * 40)
    core.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="Current core source differs"):
        validate_archived_source_hashes(tmp_path, recorded, "a" * 40)
    core.write_bytes(blobs["TWEEF/Evaluation.py"])
    blobs["experiments/runner.py"] = b"changed archived runner\n"
    with pytest.raises(ValueError, match="Archived Git source hash mismatch"):
        validate_archived_source_hashes(tmp_path, recorded, "a" * 40)


def test_relocated_artifact_hashes_preserve_legacy_names_and_reject_traversal(tmp_path):
    output = tmp_path / "test/experiments/results/german_final_surrogate"
    historical = tmp_path / "test/experiments/results/german_ec_kl/metrics_seed_0.json"
    output.mkdir(parents=True)
    historical.parent.mkdir(parents=True)
    (output / "seed_0.json").write_bytes(b"final")
    historical.write_bytes(b"historical")
    recorded = {"seed_0.json": hashlib.sha256(b"final").hexdigest(),
                "experiments/results/german_ec_kl/metrics_seed_0.json": hashlib.sha256(b"historical").hexdigest()}
    validate_recorded_artifacts(tmp_path, output, recorded)
    historical.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Relocated artifact hash mismatch"):
        validate_recorded_artifacts(tmp_path, output, recorded)
    with pytest.raises(ValueError, match="escapes"):
        validate_recorded_artifacts(tmp_path, output, {"../seed.json": "unused"})


def test_archived_verification_is_anchored_to_first_git_evidence(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    seed = b"original cached predictions"
    original_verifier = b"original audit code"
    original = json.dumps({"passed": True,
        "verifier_sha256": hashlib.sha256(original_verifier).hexdigest(),
        "artifacts_sha256": {"seed_0.json": hashlib.sha256(seed).hexdigest()}}).encode()
    (output / "seed_0.json").write_bytes(seed)
    (output / "verification.json").write_bytes(original)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "original_verification_commit", lambda *args: "b" * 40)
    monkeypatch.setattr(verifier, "_git_blob", lambda root, commit, name:
                        original_verifier if name.endswith(".py") else original)
    audit = verifier.Audit(output, archived_source=True)
    verifier.audit_original_evidence(audit, {"source_commit_at_start": "a" * 40})
    assert audit.archived_evidence["commit"] == "b" * 40
    (output / "verification.json").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="differs from its first Git evidence"):
        verifier.audit_original_evidence(audit, {"source_commit_at_start": "a" * 40})


def test_archived_recomputation_helpers_can_only_change_layout(tmp_path, monkeypatch):
    functions = ("mechanism", "fitted_estimator", "mechanism_scores", "select_by_cv", "weight_configurations")
    runner = "\n".join(f"def {name}():\n    return 1\n" for name in functions)
    loader = "def load_german():\n    return 1\n"
    directory = tmp_path / "test/experiments"
    directory.mkdir(parents=True)
    (directory / "german_final_surrogate.py").write_text("ROOT = 'relocated'\n" + runner)
    (directory / "german_aggregation.py").write_text("ROOT = 'relocated'\n" + loader)
    monkeypatch.setattr(verifier, "_git_blob", lambda root, commit, name:
                        runner.encode() if name.endswith("german_final_surrogate.py") else loader.encode())
    validate_relocated_helpers(tmp_path, "a" * 40)
    (directory / "german_final_surrogate.py").write_text(runner.replace("return 1", "return 2", 1))
    with pytest.raises(ValueError, match="changed beyond relocation"):
        validate_relocated_helpers(tmp_path, "a" * 40)


def test_explicit_archived_cli_never_overwrites_original_verification(tmp_path, monkeypatch):
    recorded_modes = []
    def fake_verify(output, *, archived_source=False):
        recorded_modes.append(archived_source)
        return {"passed": True, "checks": [{"passed": True}],
                "source_mode": "archived" if archived_source else "current"}
    monkeypatch.setattr(verifier, "verify", fake_verify)
    original = tmp_path / "verification.json"
    original.write_bytes(b"immutable original evidence\n")
    assert main(["--output", str(tmp_path), "--verify", "--archived-source"]) == 0
    assert original.read_bytes() == b"immutable original evidence\n"
    assert json.loads((tmp_path / "relocation_verification.json").read_text())["source_mode"] == "archived"
    assert main(["--output", str(tmp_path), "--verify"]) == 0
    assert original.read_bytes() == b"immutable original evidence\n"
    assert (tmp_path / "verification_current.json").exists()
    assert recorded_modes == [True, False]
