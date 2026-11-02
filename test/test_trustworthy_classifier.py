"""Opt-in German integration using a real EBM and persisted fairness inference."""
import os
import pickle

import numpy as np
import pytest

pytest.importorskip("aif360")
pytest.importorskip("interpret")

if os.environ.get("TWEEF_RUN_INTEGRATION") != "1":
    pytest.skip("Set TWEEF_RUN_INTEGRATION=1 for the real German/EBM integration.", allow_module_level=True)


def test_real_german_final_surrogate_can_be_deployed():
    from test.experiments.example_final_surrogate import main
    from test.experiments.german_aggregation import load_german

    workflow, report = main(["--max-rounds", "50", "--postprocess"])
    X, _, groups = load_german()
    indices = report.predictions.index
    restored = pickle.loads(pickle.dumps(workflow))
    predictions = restored.predict(X.loc[indices], sensitive_features=groups.loc[indices])
    probabilities = restored.predict_proba(X.loc[indices], sensitive_features=groups.loc[indices])
    np.testing.assert_array_equal(predictions, report.predictions)
    np.testing.assert_array_equal(predictions, (probabilities[:, 1] >= 0.5).astype(int))
    assert workflow.classifier_ is workflow.surrogate_
    assert workflow.final_model_source_ == "surrogate"
    assert report.metadata["test_used"] is False
    assert report.trustworthiness.mechanism == "squared_metric_mean"
    assert len(report.predictions) == 200
