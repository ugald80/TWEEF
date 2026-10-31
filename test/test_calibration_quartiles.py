import numpy as np
import pandas as pd
import pytest

from TWEEF.Calibration import FuzzyCalibrator
from TWEEF.Constants import FUZZY_THRESHOLD_PEAKS, METRICS


ACCURACY = METRICS.PERFORMANCE.ACCURACY


def test_reference_calibration_uses_linear_quartile_crests():
    reference = pd.DataFrame({ACCURACY: [0.0, 0.2, 0.8, 1.0]})
    crests = FuzzyCalibrator(metrics_values=reference).thresholds_for(ACCURACY)
    assert crests.index.tolist() == FUZZY_THRESHOLD_PEAKS
    np.testing.assert_allclose(crests, [0.0, 0.15, 0.5, 0.85, 1.0])


def test_reference_and_returned_thresholds_cannot_mutate_fitted_calibration():
    reference = pd.DataFrame({ACCURACY: [0.0, 0.2, 0.8, 1.0]})
    calibrator = FuzzyCalibrator(metrics_values=reference)
    original = calibrator.thresholds_for(ACCURACY)
    reference.iloc[:] = 0.4
    returned_crests = calibrator.thresholds_for(ACCURACY)
    returned_crests.iloc[:] = 0.6
    np.testing.assert_array_equal(calibrator.thresholds_for(ACCURACY), original)


def test_explicit_thresholds_are_frozen_and_accept_both_frame_orientations():
    reference = pd.DataFrame({ACCURACY: [0.0, 0.2, 0.5, 0.8, 1.0]}, index=FUZZY_THRESHOLD_PEAKS)
    calibrators = [FuzzyCalibrator(thresholds=reference), FuzzyCalibrator(thresholds=reference.T)]
    reference.iloc[:] = 0.0
    for calibrator in calibrators:
        np.testing.assert_allclose(calibrator.thresholds_for(ACCURACY), [0.0, 0.2, 0.5, 0.8, 1.0])


def test_repeated_crests_preserve_a_normalized_membership_partition():
    calibrator = FuzzyCalibrator(thresholds={ACCURACY: [0.0, 0.0, 0.5, 1.0, 1.0]})
    for value in np.linspace(0, 1, 21):
        vector = calibrator.membership_vector(ACCURACY, value)
        assert np.isfinite(vector).all()
        assert np.all((vector >= 0) & (vector <= 1))
        assert vector.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("reference", [[0.4, 0.4], [0.0, np.nan], [0.0, np.inf]])
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError, match="constant calibration|finite"):
        FuzzyCalibrator(metrics_values=pd.DataFrame({ACCURACY: reference}))


def test_empty_reference_and_missing_metric_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        FuzzyCalibrator(metrics_values=pd.DataFrame(columns=[ACCURACY]))
    calibrator = FuzzyCalibrator(thresholds={ACCURACY: [0, 0.2, 0.5, 0.8, 1]})
    with pytest.raises(ValueError, match="missing fuzzy thresholds"):
        calibrator.thresholds_for(METRICS.PERFORMANCE.RECALL)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_membership_rejects_non_finite_metric_value(value):
    with pytest.raises(ValueError, match="finite"):
        FuzzyCalibrator().membership_vector(ACCURACY, value)
