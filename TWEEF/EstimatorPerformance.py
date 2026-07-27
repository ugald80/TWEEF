from sklearn.base import BaseEstimator, ClassifierMixin
from TWEEF.MatrixScores import MatrixScores
from TWEEF.Constants import METRICS
from TWEEF.Metrics import Metric


class PerformanceEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, y_true, y_pred):
        """
        Create a PerformanceEstimator to compute confusion matrix scores.

        Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

        Returns:
        self : object
            Returns the instance itself.
        """
        self.general_matrix_scores = MatrixScores()
        self.general_matrix_scores.fit(y_true, y_pred)

        self.score_accuracy  = self.accuracy()
        self.score_recall    = self.recall()
        self.score_precision = self.precision()
        self.score_f1_score  = self.f1_score()
        self.score_specificity = self.specificity()
        self.score_npv = self.negative_predictive_value()


    def accuracy(self):
        """Accuracy metric."""
        return self.general_matrix_scores.acc()

    def recall(self):
        """Recall metric."""
        return self.general_matrix_scores.tpr()

    def precision(self):
        """Precision metric."""
        return self.general_matrix_scores.ppv()

    def f1_score(self):
        """F1 Score metric."""
        p = self.precision()
        r = self.recall()
        return 2 * ((p * r) / (p + r)) if (p + r) != 0 else 0

    def specificity(self):
        """Specificity metric."""
        return self.general_matrix_scores.tnr()

    def negative_predictive_value(self):
        """Negative predictive value metric."""
        return self.general_matrix_scores.npv()

    def compute_metric(self, performance_metric: Metric):
        """
        Get the specified performance metric.
        """
        if self.general_matrix_scores is None:
            raise ValueError(
                "You must fit the estimator before calculating performance metrics."
            )

        if performance_metric.name == METRICS.PERFORMANCE.ACCURACY:
            performance_metric.value = self.score_accuracy
        elif performance_metric.name == METRICS.PERFORMANCE.RECALL:
            performance_metric.value = self.score_recall
        elif performance_metric.name == METRICS.PERFORMANCE.PRECISION:
            performance_metric.value = self.score_precision
        elif performance_metric.name == METRICS.PERFORMANCE.F1_SCORE:
            performance_metric.value = self.score_f1_score
        elif performance_metric.name == METRICS.PERFORMANCE.SPECIFICITY:
            performance_metric.value = self.score_specificity
        elif performance_metric.name == METRICS.PERFORMANCE.NPV:
            performance_metric.value = self.score_npv
        else:
            raise ValueError("Invalid performance metric specified.")
        return performance_metric.value
