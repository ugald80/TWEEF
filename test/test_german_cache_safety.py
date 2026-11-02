"""Metric-definition changes must not silently reuse or overwrite old results."""
import json

import numpy as np
import pytest

from test.experiments.german_aggregation import probability_entropy, validate_metric_cache, validate_output_protocol


def test_historical_manifest_is_preserved_on_protocol_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    original = json.dumps({"ec_mode": "old absolute cross-entropy"})
    path.write_text(original)
    with pytest.raises(ValueError, match="historical results were preserved"):
        validate_output_protocol(tmp_path, "new-kl-fingerprint")
    assert path.read_text() == original


def test_matching_protocol_can_be_reused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"metric_protocol_sha256": "same"}))
    validate_output_protocol(tmp_path, "same")
    validate_metric_cache({"seed": 3, "metric_protocol_sha256": "same"}, 3, "same")


@pytest.mark.parametrize("payload", [
    {"seed": 3},
    {"seed": 3, "metric_protocol_sha256": "old"},
    {"seed": 4, "metric_protocol_sha256": "new"},
])
def test_stale_or_wrong_seed_cache_is_rejected(payload):
    with pytest.raises(ValueError, match="Incompatible metric cache"):
        validate_metric_cache(payload, 3, "new")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_entropy_diagnostic_is_finite_at_probability_endpoints(dtype):
    value = probability_entropy(np.array([0, 1, .5], dtype=dtype))
    assert np.isfinite(value)
    assert value == pytest.approx(np.log(2) / 3, abs=1e-12)
