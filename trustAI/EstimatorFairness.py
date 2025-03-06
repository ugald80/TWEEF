from numpy import ndarray as NDArray
from trustAI.MatrixScores import MatrixScores
from trustAI.Constants import METRICS
from trustAI.Metrics import Metric


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
        
        self.score_statistical_parity        = 1 - abs( self.statistical_parity() )
        self.score_treatment_equality        = 1 - abs( self.treatment_equality() )
        self.score_equal_opportunity         = 1 - abs( self.equal_opportunity() )
        self.score_equalized_odds            = 1 - abs( self.equalized_odds() )
        self.score_predictive_parity         = 1 - abs( self.predictive_parity() )
        self.score_overall_accuracy_equality = 1 - abs( self.overall_accuracy_equality())
        

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
            The treatment equality metric, which is the minimum of the absolute difference in false positive rates and false negative rates between the protected and unprotected groups, capped at 1.
            P(ˆY=1|A=0, Y=y) = P(ˆY=1|A=1, Y=y), y ∈{0,1}
        """
        te = 1
        if (self.protected_matrix_scores.fn + self.unprotected_matrix_scores.fn) > (self.protected_matrix_scores.fp + self.unprotected_matrix_scores.fp) and (self.protected_matrix_scores.fn > 0) and ( self.unprotected_matrix_scores.fn > 0):
            # print(f"1.  {self.protected_matrix_scores.fp} / {self.protected_matrix_scores.fn} - {self.unprotected_matrix_scores.fp} / {self.unprotected_matrix_scores.fn}")
            te = (self.protected_matrix_scores.fp / self.protected_matrix_scores.fn) - (self.unprotected_matrix_scores.fp / self.unprotected_matrix_scores.fn)
        elif  (self.protected_matrix_scores.fp > 0) and ( self.unprotected_matrix_scores.fp > 0):
            # print(f"2.  {self.protected_matrix_scores.fn} / {self.protected_matrix_scores.fp} - {self.unprotected_matrix_scores.fn} / {self.unprotected_matrix_scores.fp}")
            te = (self.protected_matrix_scores.fn / self.protected_matrix_scores.fp) - (self.unprotected_matrix_scores.fn / self.unprotected_matrix_scores.fp)
        return min([abs(te),1])

    def equal_opportunity(self):
        """
        Computes the equal opportunity metric.
        Returns:
        --------
        float
            The difference in false negative rates between the protected and unprotected groups.
            P(ˆY=1|A=0, Y=1) = P(ˆY=1|A=1, Y=1)
        """
        return self.protected_matrix_scores.fnr() - self.unprotected_matrix_scores.fnr()

    def equalized_odds(self):
        """
        Computes the equalized odds metric.
        Returns:
        --------
        float
            The difference in false positive rates between the protected and unprotected groups.
            P(𝑌ˆ = 1|A = a,𝑌 = 𝑦) = P(𝑌ˆ = 1|A = a′,𝑌 = 𝑦)  ∀a,a′ ∈ A, 𝑦∈{0,1}
        """
        return self.protected_matrix_scores.fpr() - self.unprotected_matrix_scores.fpr()

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
