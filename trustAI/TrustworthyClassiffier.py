import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from trustAI.Constants import METRICS, get_metrics_of
from trustAI.Metrics import MetricList
from trustAI.ProcessorFairnessPreProcessing import BiasMitigatorPreProcessing
from trustAI.ProcessorFairnessPostProcessing import BiasMitigatorPostProcessing
from trustAI.ProcessorInterpretability import InterpretabilityProcessing
from trustAI.EstimatorTrustworthiness import TrustworthinessEstimator
from trustAI.EstimatorFairness import FairnessEstimator
from trustAI.EstimatorPerformance import PerformanceEstimator
from trustAI.EstimatorInterpretability import InterpretabilityEstimator
from trustAI.UtilsFairnessGroup import FairnessGroupUtils, FairnessGroup


class TrustworthyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        given_classifier: ClassifierMixin,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        protected_groups: FairnessGroup,
        unprotected_groups: FairnessGroup,
        protected_classes: list,
        metrics: MetricList,
        metrics_values: pd.DataFrame = None,
        fuzzy_thresholds: pd.DataFrame = None,
        disparate_impact_repair_level: float = 1.0,
        equalize_odds_cost_constraint: str = "fnr",
        equalize_odds_class_thresh: float = 0.5,
        **kwargs,
    ):
        if metrics_values is not None and fuzzy_thresholds is not None:
            raise ValueError(
                "Both metrics_values and fuzzy_thresholds cannot be provided at the same time. Please provide only one of them, or none, as default values available in Constants.FUZZY_MEMBERSHIP_THRESHOLDS will be used."
            )

        self.given_algorithm = given_classifier
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.protected_groups = protected_groups
        self.unprotected_groups = unprotected_groups
        self.protected_classes = protected_classes
        # Metrics
        self.metrics_pre = metrics.copy()
        self.metrics_post = metrics.copy()
        self.kwargs = kwargs
        # Fuzzy thresholds for triangular membership functions generation
        self.metrics_values = metrics_values
        self.fuzzy_thresholds = fuzzy_thresholds
        # Bias Mitigation parameters
        self.disparate_impact_repair_level = disparate_impact_repair_level
        self.equalize_odds_cost_constraint = equalize_odds_cost_constraint
        self.equalize_odds_class_thresh = equalize_odds_class_thresh

        # Fairness: pre-proccessing
        self.transformed_x_train = None
        self.transformed_x_test = None
        self.transformed_y_train = None
        self.transformed_y_test = None
        self.disparate_impact = None
        # Fairness: post-proccessing
        self.relabeled_y_test = None
        # Explainability: surrogate explanaible model
        self.explainer = None
        self.y_predicted_ex = None
        self.surrogate_r2_error = None
        # Trustworthiness estimators for diagnosis and final results
        self.trust_estimator_pre = None
        self.trust_estimator_post = None
        self.fairness_estimator_pre = None
        self.fairness_estimator_post = None
        self.performance_estimator_pre = None
        self.performance_estimator_post = None
        self.interpretability_estimator_pre = None
        self.interpretability_estimator_post = None

    def get_metrics_weights(self, metrics: MetricList):
        """
        Returns the weights of the metrics.

        Parameters:
        -----------
        metrics (MetricList): The metrics to get the weights from.

        Returns:
        --------
        list
            A list of the weights of each metric.
        """
        if metrics is None:
            raise ValueError(
                "The metrics are not defined. Please define the metrics before training the model."
            )
        if not isinstance(metrics, MetricList):
            raise ValueError("The metrics must be an instance of MetricList.")
        return metrics.get_weights()

    def __compute_trust(
        self,
        ml_model: ClassifierMixin,
        metrics: MetricList,
        x_train,
        x_test,
        y_train,
        y_test,
    ):
        print("   compute trust metrics:")
        print("      . compute trust metrics: train the model", type(ml_model))
        ml_model.fit(x_train, y_train)

        print("      . compute trust metrics: get predictions from the model")
        y_predicted = ml_model.predict(x_test)
        y_predicted = pd.Series(
            y_predicted.astype(float), index=y_test.index, name=y_test.name
        )

        print("      . compute trust metrics: fit performance estimator")
        performance_estimator = PerformanceEstimator(y_test, y_predicted)

        print("      . compute trust metrics: fit fairness estimator")
        protected_group_utils = FairnessGroupUtils()
        protected_y_predict, unprotected_y_predict = (
            protected_group_utils.split_labels_by_group(
                x_test, y_predicted, self.protected_groups
            )
        )
        protected_y_test, unprotected_y_test = (
            protected_group_utils.split_labels_by_group(
                x_test, y_test, self.protected_groups
            )
        )
        fairness_estimator = FairnessEstimator(
            protected_y_test,
            protected_y_predict,
            unprotected_y_test,
            unprotected_y_predict,
        )

        print("      . compute trust metrics: fit interpretability estimator")
        ep = InterpretabilityProcessing()
        ep.compute_attributions_expectations(
            self.x_train,
            self.x_test,
            # self.y_train,
            self.y_test,
            self.given_algorithm,
        )
        interpretability_estimator = InterpretabilityEstimator(
            ep.attributions, ep.expectations
        )

        print("      . compute trust metrics: compute metrics values")
        for metric in metrics.get_metrics():
            if metric.name in get_metrics_of(METRICS.PERFORMANCE):
                performance_estimator.compute_metric(metric)
            elif metric.name in get_metrics_of(METRICS.FAIRNESS):
                fairness_estimator.compute_metric(metric)
            elif metric.name in get_metrics_of(METRICS.INTERPRETABILITY):
                interpretability_estimator.compute_metric(metric)
            else:
                raise ValueError(f"Unknown metric name: {metric.name}")

        print("      . compute trust metrics: compute trustworthiness score")
        trust_estimator = TrustworthinessEstimator()
        # metrics_matrix = metrics.to_dataframe()
        # trust_estimator.fit(metrics_matrix)
        # trust_estimator.transform(metrics_matrix, self.get_metrics_weights(metrics))
        trust_estimator.compute_metric(metrics)
        return (
            trust_estimator,
            performance_estimator,
            fairness_estimator,
            interpretability_estimator,
        )

    def __preprocessing_fairness(self):
        print("      . pre-processing fairness: create a BiasMitigator")
        mitigator = BiasMitigatorPreProcessing()
        print(
            "      . pre-processing fairness: apply remove_disparate_impact on original data"
        )
        self.disparate_impact = mitigator.preprocess_disparate_impact_removal(
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.y_test.name,  # label name
            self.protected_groups,
            self.unprotected_groups,
            self.protected_classes,
            self.disparate_impact_repair_level,
        )
        print("      . pre-processing fairness: store transformed data")
        self.transformed_x_train = mitigator.transformed_x_train
        self.transformed_x_test = mitigator.transformed_x_test
        self.transformed_y_train = mitigator.transformed_y_train
        self.transformed_y_test = mitigator.transformed_y_test
        self.disparate_impact = mitigator.disparate_impact

    def __surrogate_model_train(self):
        """
        Train a surrogate model for explanations generation
        """
        self.explainer = InterpretabilityProcessing()
        self.y_predicted_ex = self.explainer.surrogate_model_train(
            self.given_algorithm,
            self.transformed_x_train,
            self.transformed_x_test,
            self.transformed_y_train,
            self.transformed_y_test,
        )
        self.surrogate_r2_error = self.explainer.surrogate_r2_error

    def __postprocessing_fairness(self, y_pred: pd.Series):
        print("   post-processing fairness: create a BiasMitigator")
        mitigator = BiasMitigatorPostProcessing(
            cost_constraint=self.equalize_odds_cost_constraint
        )
        print(
            "   post-processing fairness: apply equalize_odds for prediction labels mitigation"
        )
        self.relabeled_y_test = mitigator.postprocess_equalize_odds(
            self.transformed_x_train,
            self.transformed_x_test,
            self.transformed_y_train,
            y_pred,
            y_pred.name,
            self.protected_groups,
            self.unprotected_groups,
            self.protected_classes,
            # self.equalize_odds_class_thresh
        )

    def __fairness_validation(self):
        # self.y_predicted_ex must be computed from surogate model training.
        print("fairness_validation")
        max_iterations = 10
        fairness_threshold = 0.9
        iteration = 0
        actual_fairness = 0
        # initialize the relabeled_y_test with the predictions from the explanaible surogate model
        self.relabeled_y_test = self.y_predicted_ex

        while iteration < max_iterations and actual_fairness < fairness_threshold:
            print("  . validate iterations: iteration ", iteration)
            # this will update self.relabeled_y_test applying equalize_odds
            self.__postprocessing_fairness(self.relabeled_y_test)
            (
                self.trust_estimator_post,
                self.performance_estimator_post,
                self.fairness_estimator_post,
                self.interpretability_estimator_post,
            ) = self.__compute_trust(
                self.explainer.explainable_classifier,
                self.metrics_post,
                self.transformed_x_train,
                self.transformed_x_test,
                self.transformed_y_train,
                self.relabeled_y_test,
            )
            print("  . validate iterations: estimate fairness")
            fairness_metrics_values = [
                metric.value
                for metric in self.metrics_post.get_metrics()
                if metric.name in get_metrics_of(METRICS.FAIRNESS)
            ]
            print(
                "  . validate iterations: fairness score values: ",
                fairness_metrics_values,
            )
            actual_fairness = min(fairness_metrics_values)
            print(
                "  . validate iterations: min fairness metric value ", actual_fairness
            )
            iteration = iteration + 1

        if iteration == max_iterations:
            print(
                "  . validate iterations: post-processing fairness assessment did not converge"
            )

        print("fairness validation terminated!")

    def fit(self):
        """
        Trains the trustworthy classifier as a surrogate model that learns from a given model.

        This function performs the following steps:
        1. Compute trustworthiness metrics for the given model.
        2. Apply pre-processing fairness algorithm.
        3. Train the surrogate model.
        4. Apply post-processing fairness validation.
        5. Compute trustworthiness metrics for the surrogate model.

        Returns:
        --------
        ExplainableBoostingClassifier
            The trained explainable classifier
        """
        print(
            "\n### STAGE 1: TRUSTWORTHINESS DIAGNOSIS ========================================="
        )
        print("   1.1: Compute initial trustworthiness metrics")
        (
            self.trust_estimator_pre,
            self.performance_estimator_pre,
            self.fairness_estimator_pre,
            self.interpretability_estimator_pre,
        ) = self.__compute_trust(
            self.given_algorithm,
            self.metrics_pre,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        )
        print("   1.2: apply Preprocessing Fairness algorithm")
        self.__preprocessing_fairness()

        print(
            "\n### STAGE 2: TRAINING GIVEN MODEL AND SURROGATE MODEL ========================================="
        )
        self.__surrogate_model_train()

        print(
            "\n### STAGE 3: FAIRNESS POSTPROCESSING VALIDATION ========================================="
        )
        self.__fairness_validation()

        return self.explainer.explainable_classifier

    def predict(self, x: pd.DataFrame):
        """
        Predicts the labels for the given data, using the previously trained explainable classifier.

        Parameters:
        -----------
        x (DataFrame): The input data.

        Returns:
        --------
        Series
            The predicted labels.
        """
        if self.explainer is None:
            raise ValueError(
                "The explainer model is not trained yet. Please train the model before making predictions."
            )
        if self.explainer.explainable_classifier is None:
            raise ValueError(
                "The explainer model does not have a classifier. Please train the explainer model before making predictions."
            )

        check_is_fitted(self.explainer.explainable_classifier)
        return self.explainer.explainable_classifier.predict(x)

    # #OUTPUTS =====================================================================
    def trustworthiness_diagnosis(self):
        report = self.metrics_pre.to_dataframe()
        report["trustworthiness"] = self.trust_estimator_pre.trustworthiness_score
        return report

    def trustworthiness_metrics(self):
        report = self.metrics_post.to_dataframe()
        report["trustworthiness"] = self.trust_estimator_post.trustworthiness_score
        return report

    def unbiased_dataset(self):
        data = pd.concat([self.transformed_x_train, self.transformed_x_test])
        data[self.transformed_y_train.name] = pd.concat(
            [self.transformed_y_train, self.relabeled_y_test]
        )
        return data
