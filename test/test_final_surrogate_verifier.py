"""Tamper detection without running or rewriting an experiment."""

from copy import deepcopy
import hashlib
import json

import pytest

from experiments.verify_final_surrogate import (
    main, validate_predictions, validate_provenance, validate_source_hashes,
)


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
