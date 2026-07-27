import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics import ClassificationMetric

from TWEEF import UtilsFairnessGroup
from TWEEF.ProcessorFairness import BiasMitigator
from TWEEF.UtilsFairnessGroup import FairnessGroupUtils, FairnessGroup
from TWEEF.Logging import get_logger


logger = get_logger(__name__)


class BiasMitigatorPostProcessing(BiasMitigator):
    """
        Sklearn-compatible class for postprocessing bias mitigation using Equalized Odds
        
        Attributes:
        ----------
        relabeled_y : pd.Series
            The transformed labels after applying equalized odds postprocessing.
        equalized_odds : float
            The Equalized Odds metric calculated after postprocessing.
        group_utils : FairnessGroupUtils
            Utility class for handling fairness-related group operations.
        cost_constraint : str
            Cost constraint ('fnr', 'fpr', 'weighted').
            
    """

    def __init__(self, cost_constraint="fnr"):
        """
        Initializes the BiasMitigatorPostProcessing class.
        
        Parameters:
        ----------
        cost_constraint : str, optional
            Cost constraint ('fnr', 'fpr', 'weighted'), by default "fnr".
            A cost_constraint='fnr' will optimize generalized false negative rates,
            A cost_constraint='fpr' will optimize generalized false positive rates,
            And a cost_constraint='weighted' will optimize a weighted combination of both
        """
        super().__init__()
        self.relabeled_y = None
        self.equalized_odds = None
        self.cost_constraint = cost_constraint
        self.group_utils = FairnessGroupUtils()

    def postprocess_equalize_odds(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_name: str,
        protectedGroups: UtilsFairnessGroup,
        unprotectedGroups: UtilsFairnessGroup,
        protectedClasses: list,
        prediction_scores=None,
    ):
        """
        Apply postprocessing with Equalized Odds to mitigate bias.

        Parameters:
        x_train (NDArray): Training feature data.
        X_test (NDArray): Testing feature data.
        y_train (NDArray): Training labels.
        y_true (NDArray): Ground-truth testing labels.
        y_pred (NDArray): Predicted testing labels to postprocess.
        label_name (str): Name of the label column.
        protectedGroups (FairnessGroup): Protected groups.
        unprotectedGroups (FairnessGroup): Unprotected groups.
        protectedClasses (list): Favorable classes.
        cost_constraint (str): Cost constraint ('fnr', 'fpr', 'weighted').

        Returns:
        Series: Transformed labels.
        """
        
        logger.info("Set up datasets for equalized odds postprocessing")
        protected_keys, protected_values = self.group_utils.unparse_groups(protectedGroups)
        y_true = pd.Series(y_true, index=X_test.index, name=label_name).astype(float)
        y_pred = pd.Series(y_pred, index=X_test.index, name=label_name).astype(float)

        aif_dataset_test_true = self.generate_aif360(X_test, y_true, label_name, protected_keys, protected_values, protectedClasses)
        aif_dataset_test_pred = self.generate_aif360(X_test, y_pred, label_name, protected_keys, protected_values, protectedClasses)

        if prediction_scores is None:
            scores = y_pred.to_numpy(dtype=float)
        else:
            scores = np.asarray(prediction_scores, dtype=float).reshape(-1)
        aif_dataset_test_pred.scores = scores.reshape(-1, 1)

        # Split dataset for training and validation
        dataset_orig_test, dataset_orig_valid = aif_dataset_test_true.split([0.5], shuffle=False)
        dataset_pred_test, dataset_pred_valid = aif_dataset_test_pred.split([0.5], shuffle=False)

        # Equalized Odds Postprocessing
        logger.info("Learn equalized odds postprocessing parameters")
        cpp = CalibratedEqOddsPostprocessing(
            privileged_groups=protectedGroups,
            unprivileged_groups=unprotectedGroups,
            cost_constraint=self.cost_constraint,
        )
        cpp.fit(dataset_orig_valid, dataset_pred_valid)
        dataset_transf_test_pred = cpp.predict(dataset_pred_test)
        dataset_transf_valid_pred = cpp.predict(dataset_pred_valid)
        
        # Metrics and Equalized Odds calculation
        logger.info("Calculate postprocessing metrics and equalized odds")
        cm_transf_test = ClassificationMetric(
            dataset_orig_test,
            dataset_transf_test_pred,
            unprotectedGroups,
            protectedGroups,
        )
        array = np.concatenate(
            (
                dataset_transf_test_pred.labels.ravel(),
                dataset_transf_valid_pred.labels.ravel(),
            )
        )
        self.relabeled_y = pd.Series(
            array.astype(float),
            name=label_name,
            index=y_true.index
        )
        self.equalized_odds = cm_transf_test.generalized_false_negative_rate()

        logger.info("Calibrated equalized odds value: %s", self.equalized_odds)
        return self.relabeled_y

















    def postprocess_equalize_odds_2(
        self,
        x_train          : pd.DataFrame,
        x_test           : pd.DataFrame,
        y_train          : pd.Series,
        y_test           : pd.Series,
        label_name       : str,
        protected_groups  : FairnessGroup,
        unprotected_groups: FairnessGroup,
        protected_classes : list,
        class_thresh     : float = 0.5
    ):
        """
        Apply postprocessing with Equalized Odds to mitigate bias.
        
        Parameters:
        ----------
        X_train : pd.DataFrame
            Training feature data.
        X_test : pd.DataFrame
            Testing feature data.
        y_train : pd.Series
            Training labels.
        y_test : pd.Series
            Testing labels.
        label_name : str
            Name of the label column.
        protectedGroups : FairnessGroup
            Protected groups.
        unprotectedGroups : FairnessGroup
            Unprotected groups.
        protectedClasses : list
            Favorable classes.
        class_thresh : float
            Classification threshold, by default 0.5.
        """
        protected_keys, protected_values = self.group_utils.unparse_groups(protected_groups)
        aif_dataset_train = self.generate_aif360(x_train, y_train, label_name, protected_keys, protected_values, protected_classes)
        aif_dataset_test  = self.generate_aif360(x_test , y_test , label_name, protected_keys, protected_values, protected_classes)
        privileged_groups   = protected_groups
        unprivileged_groups = unprotected_groups

        dataset_orig_train = aif_dataset_train
        dataset_orig_vt    = aif_dataset_test

        # Learn parameters to equalize odds and apply to create a new dataset
        logger.info("Learn equalized odds postprocessing parameters")
        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                            unprivileged_groups = unprivileged_groups,
                                            cost_constraint=self.cost_constraint
                                            )

        #dataset_orig_train, dataset_orig_vt = dataset_orig.split([test_size], shuffle=False)
        dataset_orig_test, dataset_orig_valid = dataset_orig_vt.split([0.5], shuffle=False, seed=101)

        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

        # dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        # dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

        # Logistic regression classifier and predictions for training data
        logger.info("Train logistic regression model")
        scale_orig = StandardScaler()
        x_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(x_train, y_train)

        fav_idx = np.where(logistic_regression_model.classes_ == dataset_orig_train.favorable_label)[0][0]
        y_train_pred_prob = logistic_regression_model.predict_proba(x_train)[:,fav_idx]

        # Prediction probs for validation and testing data
        X_valid = scale_orig.transform(dataset_orig_valid.features)
        y_valid_pred_prob = logistic_regression_model.predict_proba(X_valid)[:,fav_idx]

        x_test = scale_orig.transform(dataset_orig_test.features)
        y_test_pred_prob = logistic_regression_model.predict_proba(x_test)[:,fav_idx]

        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
        dataset_orig_test_pred.scores  = y_test_pred_prob.reshape(-1,1)

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred

        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred

        #Metrics before eq odds
        #cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        #cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        # Perform eq odds
        logger.info("Apply equalized odds")
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        dataset_transf_test_pred  = cpp.predict(dataset_orig_test_pred)
        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid, dataset_orig_valid_pred)

        logger.info("Obtain postprocessing metrics")
        #Metrics after eq odds
        cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        #cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        logger.info("Store postprocessing results")
        # save results
        relabeled_y_test      = pd.Series(dataset_transf_test_pred.labels , name=label_name)
        relabeled_y_valid     = pd.Series(dataset_transf_valid_pred.labels, name=label_name)

        self.relabeled_y      = pd.Series([relabeled_y_test.astype(float), relabeled_y_valid.astype(float)], index=y_test.index) #, ignore_index=True)
        self.equalized_odds = 0.5*np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) + 0.5*np.abs(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
        logger.info("Calibrated equalized odds value: %s", self.equalized_odds)

        # Testing: Check if the rates for validation data has gone down
        assert np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) < np.abs(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
        
        return self.relabeled_y
