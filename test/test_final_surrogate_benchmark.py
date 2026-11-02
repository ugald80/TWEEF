"""Protocol checks independent of the expensive German fitting run."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test.experiments.german_final_surrogate import (
    METRICS, parse_args, prepare_output, select_by_cv, weight_configurations,
    snapshot_parameters, write_json,
)


class LinearScoringEstimator:
    def __init__(self, higher=True):
        self.aggregation_result_ = SimpleNamespace(higher_is_better=higher)

    def transform(self, metrics):
        from TWEEF.Constants import TRUSTWORTHINESS_SCORE_NAME
        return pd.DataFrame({TRUSTWORTHINESS_SCORE_NAME: metrics.iloc[:, 0]}, index=metrics.index)


def test_candidate_selection_only_uses_cv_and_obeys_direction_and_tie_budget():
    frames = {100: pd.DataFrame({"metric": [.2, .4]}),
              500: pd.DataFrame({"metric": [.7, .8]})}
    assert select_by_cv(frames, LinearScoringEstimator())[0] == 500
    assert select_by_cv(frames, LinearScoringEstimator(higher=False))[0] == 100
    frames[500] = frames[100].copy()
    assert select_by_cv(frames, LinearScoringEstimator())[0] == 100


def test_incompatible_protocol_does_not_overwrite_existing_manifest(tmp_path):
    saved = {"protocol_sha256": "saved", "protocol": {"source": "old"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(saved))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="different protocol"):
        prepare_output(tmp_path, {"source": "changed"})
    assert path.read_bytes() == original


def test_nonempty_unmanifested_output_is_not_reused(tmp_path):
    path = tmp_path / "seed_0.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="nonempty"):
        prepare_output(tmp_path, {"protocol": "new"})
    assert path.read_text() == "{}"
    assert not (tmp_path / "manifest.json").exists()


def test_weight_sweep_has_normalized_finite_weights_and_one_hot_controls():
    configurations = list(weight_configurations())
    assert len(configurations) == 31
    assert len({name for name, _ in configurations}) == 31
    for name, weights in configurations:
        assert set(weights) == set(METRICS)
        values = np.array(list(weights.values()))
        assert np.isfinite(values).all() and (values >= 0).all()
        assert values.sum() == pytest.approx(1)
    assert sum(sum(value > 0 for value in weights.values()) == 1 for _, weights in configurations) == 3


def test_cli_defaults_are_the_agreed_full_experiment_and_reject_duplicate_seeds():
    args = parse_args([])
    assert args.seeds == list(range(10))
    assert len(args.models) == 8
    assert args.candidate_rounds == [100, 500]
    assert args.reference_cv == 5
    assert args.postprocessor == "calibrated_equalized_odds"
    with pytest.raises(SystemExit):
        parse_args(["--seeds", "0", "0"])


def test_parameter_snapshot_encodes_xgboost_missing_nan_and_nested_numpy_values():
    parameters = {"missing": np.nan, "nested": (np.float32(np.inf), np.array([1., -np.inf])),
                  "random_state": np.int64(7), "ordinary": None}
    encoded = snapshot_parameters(parameters)
    assert encoded == {"missing": {"__float__": "nan"},
                       "nested": [{"__float__": "inf"}, [1., {"__float__": "-inf"}]],
                       "random_state": 7, "ordinary": None}
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    assert np.isnan(parameters["missing"])


def test_metric_nan_remains_an_error_in_strict_json(tmp_path):
    path = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="Out of range float"):
        write_json(path, {"metrics": {"Accuracy": np.nan}})
    assert not path.exists()
