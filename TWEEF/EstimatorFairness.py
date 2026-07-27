from numpy import ndarray as NDArray
from TWEEF.MatrixScores import MatrixScores
from TWEEF.Constants import METRICS
from TWEEF.Metrics import Metric


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0 if numerator == 0 else float("inf")
    return numerator / denominator


def _bounded_fairness_score(disparity):
    return max(0, 1 - abs(disparity))


class FairnessEstimator():
    """
    A class to estimate various fairness metrics for protected and unprotected groups.
    Attributes:
    -----------
    protected_matrix_scores : MatrixScores
        An instance of MatrixScores for the protected group.
    unprotected_matrix_scores : MatrixScores
        An instance of MatrixScores for the unprotected group.
    score_overall_accuracy_equality : float or None
        Cached score for overall accuracy equality.
    score_statistical_parity : float or None
        Cached score for statistical parity.
    score_treatment_equality : float or None
        Cached score for treatment equality.
    score_equal_opportunity : float or None
        Cached score for equal opportunity.
    score_equalized_odds : float or None
        Cached score for equalized odds.
    score_predictive_parity : float or None
        Cached score for predictive parity.
    Methods:
    --------
    __init__(protected_y_test: NDArray, protected_y_predict: NDArray, unprotected_y_test: NDArray, unprotected_y_predict: NDArray)
        Initializes the FairnessEstimator with test and prediction data for protected and unprotected groups.
    overall_accuracy_equality()
        Computes the overall accuracy equality metric.
    statistical_parity()
        Computes the statistical parity metric.
    treatment_equality()
        Computes the treatment equality metric.
    equal_opportunity()
        Computes the equal opportunity metric.
    equalized_odds()
        Computes the equalized odds metric.
    predictive_parity()
        Computes the predictive parity metric.
    fairness(fairness_metric=None)
        Computes the specified fairness metric.
    """
    def __init__(self, protected_y_test: NDArray, protected_y_predict: NDArray, unprotected_y_test: NDArray, unprotected_y_predict: NDArray):
        """
        Initializes the FairnessEstimator with test and prediction data for protected and unprotected groups.
        Parameters:
        -----------
        protected_y_test : NDArray
            The ground truth labels for the protected group.
        protected_y_predict : NDArray
            The predicted labels for the protected group.
        unprotected_y_test : NDArray
            The ground truth labels for the unprotected group.
        unprotected_y_predict : NDArray
            The predicted labels for the unprotected group.
        """

        self.protected_matrix_scores   = MatrixScores()
        self.unprotected_matrix_scores = MatrixScores()
        
        self.protected_matrix_scores.fit(protected_y_test, protected_y_predict)
        self.unprotected_matrix_scores.fit(unprotected_y_test, unprotected_y_predict)
        
        self.score_statistical_parity = _bounded_fairness_score(
            self.statistical_parity()
        )
        self.score_treatment_equality = _bounded_fairness_score(
            self.treatment_equality()
        )
        self.score_equal_opportunity = _bounded_fairness_score(
            self.equal_opportunity()
        )
        self.score_equalized_odds = max(0, 1 - self.equalized_odds())
        self.score_predictive_parity = _bounded_fairness_score(
            self.predictive_parity()
        )
        self.score_overall_accuracy_equality = _bounded_fairness_score(
            self.overall_accuracy_equality()
        )
        

    def overall_accuracy_equality(self):
        """
        Computes the overall accuracy equality metric.
        Returns:
        --------
        float
            The difference in accuracy between the protected and unprotected groups.
            P (ˆY=Y |A=0) = P (ˆY=Y |A=1). Accuracy 
        """
        return self.protected_matrix_scores.acc() - self.unprotected_matrix_scores.acc()

    def statistical_parity(self):
        """
        Computes the statistical parity metric.
        Returns:
        --------
        float
            The difference in positive prediction rates between the protected and unprotected groups.
            P (ˆY |A=0) = P ( ˆY |A=1)
        """
        return self.protected_matrix_scores.pr()  - self.unprotected_matrix_scores.pr()

    def treatment_equality(self):
        """
        Computes the treatment equality metric.
        Returns:
        --------
        float
            Difference between the false-positive to false-negative ratios
            of the protected and unprotected groups.
        """
        protected_ratio = _safe_ratio(
            self.protected_matrix_scores.fp,
            self.protected_matrix_scores.fn,
        )
        unprotected_ratio = _safe_ratio(
            self.unprotected_matrix_scores.fp,
            self.unprotected_matrix_scores.fn,
        )
        return protected_ratio - unprotected_ratio

    def equal_opportunity(self):
        """
        Computes the equal opportunity metric.
        Returns:
        --------
        float
            The difference in true positive rates between the protected and unprotected groups.
        """
        return self.protected_matrix_scores.tpr() - self.unprotected_matrix_scores.tpr()

    def equalized_odds(self):
        """
        Computes the equalized odds metric.
        Returns:
        --------
        float
            Sum of absolute differences in true positive rates and false positive rates.
        """
        tpr_difference = abs(
            self.protected_matrix_scores.tpr()
            - self.unprotected_matrix_scores.tpr()
        )
        fpr_difference = abs(
            self.protected_matrix_scores.fpr()
            - self.unprotected_matrix_scores.fpr()
        )
        return tpr_difference + fpr_difference

    def predictive_parity(self):
        """
        Computes the predictive parity metric.
        Returns:
        --------
        float
            The difference in positive predictive values between the protected and unprotected groups.
            P(ˆY=0 |A=0) = P(ˆY=0 |A=1)
        """
        return self.protected_matrix_scores.ppv() - self.unprotected_matrix_scores.ppv()

    def compute_metric(self, fairness_metric: Metric):
        """
        Computes the specified fairness metric.
        Parameters:
        -----------
        fairness_metric : str or None
            The fairness metric to compute. Must be one of the predefined metric constants.
        Returns:
        --------
        float
            The computed fairness metric.
        Raises:
        -------
        ValueError
            If an invalid fairness metric is provided.
        """

        if fairness_metric.name == METRICS.FAIRNESS.TREATMENT_EQUALITY:
            fairness_metric.value = self.score_treatment_equality
        elif fairness_metric.name == METRICS.FAIRNESS.EQUALIZED_ODDS:
            fairness_metric.value = self.score_equalized_odds
        elif fairness_metric.name == METRICS.FAIRNESS.EQUAL_OPPORTUNITY:
            fairness_metric.value = self.score_equal_opportunity
        elif fairness_metric.name == METRICS.FAIRNESS.STATISTICAL_PARITY:
            fairness_metric.value = self.score_statistical_parity
        else:
            raise ValueError('Invalid value for fairness metric')

        return fairness_metric.value
