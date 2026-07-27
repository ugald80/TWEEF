import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

pytest.importorskip("aif360")
pytest.importorskip("interpret")

from TWEEF.Constants import METRICS
from TWEEF.Metrics import MetricList
from TWEEF.TrustworthyClassifier import TrustworthyClassifier
from TWEEF.UtilsFairnessGroup import FairnessGroup


def _metrics():
    metrics = MetricList()
    metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=1.0, goal=0.8)
    return metrics


def _classifier():
    return TrustworthyClassifier(
        given_classifier=DummyClassifier(strategy="most_frequent"),
        protected_groups=FairnessGroup([{"group": [0]}]),
        unprotected_groups=FairnessGroup([{"group": [1]}]),
        protected_classes=[1],
        metrics=_metrics(),
        test_size=0.25,
        random_state=0,
    )


def test_trustworthy_classifier_exposes_constructor_params():
    classifier = _classifier()

    params = classifier.get_params()

    assert params["given_classifier"] is classifier.given_classifier
    assert params["test_size"] == 0.25
    assert "x_train" not in params
    assert "y_train" not in params


def test_trustworthy_classifier_can_be_cloned():
    classifier = _classifier()

    cloned = clone(classifier)

    assert isinstance(cloned, TrustworthyClassifier)
    assert cloned is not classifier
    assert cloned.given_classifier is not classifier.given_classifier
    assert cloned.get_params()["test_size"] == classifier.get_params()["test_size"]


def test_trustworthy_classifier_rejects_legacy_training_data_constructor_args():
    with pytest.raises(TypeError):
        TrustworthyClassifier(
            given_classifier=DummyClassifier(),
            x_train=None,
            protected_groups=FairnessGroup([{"group": [0]}]),
            unprotected_groups=FairnessGroup([{"group": [1]}]),
            protected_classes=[1],
            metrics=_metrics(),
        )
