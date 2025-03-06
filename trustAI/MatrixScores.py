"""
matrix scores module for performance and fairness metrics
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics

class MatrixScores(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for confusion matrix scores.
    This class computes and provides various metrics based on confusion matrix values.

    Attributes:
    ----------
    tn : int
        True negatives count.
    fp : int
        False positives count.
    fn : int
        False negatives count.
    tp : int
        True positives count.

    Methods:
    -------
    fit(y_true, y_pred):
        Fits the model with the true and predicted labels to compute confusion matrix scores.
    ppv():
        Computes the positive predictive value (precision).
    tpr():
        Computes the true positive rate (recall).
    fpr():
        Computes the false positive rate.
    fnr():
        Computes the false negative rate.
    pr():
        Computes the positive rate.
    acc():
        Computes the accuracy.
    te_q():
        Computes the Trade-off Equality (TEq).
    score(X, y, sample_weight=None):
        Returns accuracy as the default metric for scikit-learn compatibility.
    """

    def __init__(self):
        """
        Initializes the MatrixScores with default confusion matrix values.
        """
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def fit(self, y_true, y_pred):
        """
        Fit the model with the true and predicted labels to compute confusion matrix scores.
        
        Parameters:
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated targets as returned by a classifier.
        
        Returns:
        -------
        self : object
            Returns the instance itself.
        
        Raises:
        ------
        ValueError
            If the input arrays are empty.
        """
        
        # Validate input
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have equal length.")        
        elif len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays must not be empty.")

        # Compute confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp
        return self

    def ppv(self):
        """
        Positive predictive value (precision).
        
        Returns:
        -------
        float
            The precision score.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0

    def tpr(self):
        """
        True positive rate (recall).
        
        Returns:
        -------
        float
            The recall score.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0

    def fpr(self):
        """
        False positive rate.
        
        Returns:
        -------
        float
            The false positive rate.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) != 0 else 0

    def fnr(self):
        """
        False negative rate.
        
        Returns:
        -------
        float
            The false negative rate.
        """
        return self.fn / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0

    def pr(self):
        """
        Positive rate.
        
        Returns:
        -------
        float
            The positive rate.
        """
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.fp) / total if total != 0 else 0

    def acc(self):
        """
        Accuracy.
        
        Returns:
        -------
        float
            The accuracy score.
        """
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total != 0 else 0

    def te_q(self):
        """
        Trade-off Equality (TEq).
        
        Returns:
        -------
        float
            The trade-off equality score.
        """
        if self.fp == 0 or self.fn == 0:
            return 0
        return self.fp / self.fn if self.fp < self.fn else self.fn / self.fp

    def score(self, X, y, sample_weight=None):
        """
        Scikit-learn compatibility: Returns accuracy as the default metric.
        
        Parameters:
        ----------
        X : array-like
            The input data.
        y : array-like
            The target values.
        sample_weight : array-like, optional
            Sample weights.
        
        Returns:
        -------
        float
            The accuracy score.
        """
        return self.acc()

