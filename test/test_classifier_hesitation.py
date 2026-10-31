"""The public classifier forwards hesitation into its trust assessment."""

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

pytest.importorskip("aif360")
pytest.importorskip("interpret")

from TWEEF.Constants import METRICS
from TWEEF.Metrics import MetricList
from TWEEF.TrustworthyClassifier import TrustworthyClassifier


def test_classifier_clone_and_assessment_preserve_hesitation():
    metrics = MetricList()
    names = [METRICS.PERFORMANCE.ACCURACY,
             METRICS.FAIRNESS.EQUAL_OPPORTUNITY,
             METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY]
    hesitation = dict(zip(names, [0.2, 0.3, 0.4]))
    for name in names:
        metrics.add_metric(name, 1 / 3, goal=0.8)
    classifier = TrustworthyClassifier(
        given_classifier=DecisionTreeClassifier(random_state=0),
        protected_groups=[{"group": [0]}],
        unprotected_groups=[{"group": [1]}],
        protected_classes=[1], metrics=metrics, aggregation="gwa",
        gwa_hesitation=hesitation,
    )
    classifier = clone(classifier)
    assert classifier.get_params()["gwa_hesitation"] == hesitation
    classifier.trustworthiness_score_scale_ = "normalized"
    train = pd.DataFrame({"feature": [0., 1., 0., 1.] * 2,
                          "group": [0, 0, 1, 1] * 2})
    y_train = pd.Series([0, 1, 0, 1] * 2, name="target")
    test = pd.DataFrame({"feature": [0., 1., 0., 1.], "group": [0, 0, 1, 1]})
    y_test = pd.Series([0, 1, 0, 1], name="target")
    estimator, *_ = classifier._TrustworthyClassifier__compute_trust(
        classifier.given_classifier, metrics.copy(), train, test, y_train, y_test,
    )
    assert estimator.aggregation_result_.operator == "gwa"
    assert estimator.aggregation_result_.details["hesitation"] == hesitation
