import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from TWEEF.Constants import METRICS, get_metrics_of
from TWEEF.Metrics import MetricList
from TWEEF.Aggregation import (
    AGGREGATION,
    TRUSTWORTHINESS_SCORE_SCALE,
    normalize_score_scale,
)
from TWEEF.ProcessorFairnessPreProcessing import BiasMitigatorPreProcessing
from TWEEF.ProcessorFairnessPostProcessing import BiasMitigatorPostProcessing
from TWEEF.ProcessorInterpretability import InterpretabilityProcessing
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.EstimatorFairness import FairnessEstimator
from TWEEF.EstimatorPerformance import PerformanceEstimator
from TWEEF.EstimatorInterpretability import InterpretabilityEstimator
from TWEEF.UtilsFairnessGroup import FairnessGroupUtils, FairnessGroup
from TWEEF.Validation import drop_invalid_rows, validate_groups, validate_metric_list
from TWEEF.Logging import get_logger


logger = get_logger(__name__)


class TrustworthyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        given_classifier: ClassifierMixin,
        protected_groups: FairnessGroup = None,
        unprotected_groups: FairnessGroup = None,
        protected_classes: list = None,
        metrics: MetricList = None,
        metrics_values: pd.DataFrame = None,
        fuzzy_thresholds: pd.DataFrame = None,
        disparate_impact_repair_level: float = 1.0,
        equalize_odds_cost_constraint: str = "fnr",
        equalize_odds_class_thresh: float = 0.5,
        test_size: float = 0.3,
        random_state: int = None,
        stratify: bool = True,
        aggregation: str = AGGREGATION.LWA,
        aggregation_alpha: float = 0.5,
        trustworthiness_score_scale: str = TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        fairness_threshold: float = 0.9,
        fairness_max_iter: int = 100,
        surrogate_min_r2: float = 0.8,
        surrogate_max_iter: int = 100,
        surrogate_interactions=0,
        surrogate_max_rounds=None,
        effective_complexity_mode="attribution",
        effective_complexity_threshold=0.95,
        effective_complexity_loss_threshold=0.05,
        effective_complexity_repeats=3,
        gwa_hesitation=0.0,
    ):
        self.given_classifier = given_classifier
        self.protected_groups = protected_groups
        self.unprotected_groups = unprotected_groups
        self.protected_classes = protected_classes
        self.metrics = metrics
        self.metrics_values = metrics_values
        self.fuzzy_thresholds = fuzzy_thresholds
        self.disparate_impact_repair_level = disparate_impact_repair_level
        self.equalize_odds_cost_constraint = equalize_odds_cost_constraint
        self.equalize_odds_class_thresh = equalize_odds_class_thresh
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.aggregation = aggregation
        self.aggregation_alpha = aggregation_alpha
        self.trustworthiness_score_scale = trustworthiness_score_scale
        self.fairness_threshold = fairness_threshold
        self.fairness_max_iter = fairness_max_iter
        self.surrogate_min_r2 = surrogate_min_r2
        self.surrogate_max_iter = surrogate_max_iter
        self.surrogate_interactions = surrogate_interactions
        self.surrogate_max_rounds = surrogate_max_rounds
        self.effective_complexity_mode = effective_complexity_mode
        self.effective_complexity_threshold = effective_complexity_threshold
        self.effective_complexity_loss_threshold = effective_complexity_loss_threshold
        self.effective_complexity_repeats = effective_complexity_repeats
        self.gwa_hesitation = gwa_hesitation

    def _reset_runtime_state(self):
        fitted_attributes = [
            "classifier_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
            "diagnosis_",
            "final_report_",
            "tradeoffs_",
            "component_scores_",
            "trustworthiness_score_",
            "trustworthiness_score_scale_",
            "given_algorithm_",
            "optimization_algorithm_",
            "is_fitted_",
        ]
        for attribute in fitted_attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.transformed_x_train = None
        self.transformed_x_test = None
        self.transformed_y_train = None
        self.transformed_y_test = None
        self.fairness_processor_pre= None
        self.fairness_processor_post = None
        self.relabeled_y_test = None
        self.explainer = None
        self.y_predicted_ex = None
        self.surrogate_r2_error = None
        self.metrics_pre = None
        self.metrics_post = None
        self.trust_estimator_pre = None
        self.trust_estimator_post = None
        self.fairness_estimator_pre = None
        self.fairness_estimator_post = None
        self.performance_estimator_pre = None
        self.performance_estimator_post = None
        self.interpretability_estimator_pre = None
        self.interpretability_estimator_post = None
        self.final_model_source_ = None

    @staticmethod
    def __series_like(values, index, name):
        array = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
        return pd.Series(array.reshape(-1).astype(float), index=index, name=name)

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
        y_predicted=None,
        fit_model=True,
    ):
        logger.info("Compute trust metrics")
        if fit_model:
            logger.info("Train model: %s", type(ml_model).__name__)
            ml_model.fit(x_train, y_train)

        if y_predicted is None:
            logger.info("Get predictions from model")
            y_predicted = ml_model.predict(x_test)
            y_predicted = self.__series_like(y_predicted, y_test.index, y_test.name)
        else:
            logger.info("Use provided predictions")
            y_predicted = self.__series_like(y_predicted, y_test.index, y_test.name)

        logger.info("Fit performance estimator")
        performance_estimator = PerformanceEstimator(y_test, y_predicted)

        logger.info("Fit fairness estimator")
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

        logger.info("Fit interpretability estimator")
        ep = InterpretabilityProcessing(
            ec_threshold=self.effective_complexity_threshold,
            effective_complexity_mode=self.effective_complexity_mode,
            effective_complexity_loss_threshold=(
                self.effective_complexity_loss_threshold
            ),
            effective_complexity_repeats=self.effective_complexity_repeats,
            random_state=self.random_state or 0,
        )
        ep.compute_attributions_expectations(
            x_train,
            x_test,
            y_test,
            ml_model,
        )
        interpretability_estimator = InterpretabilityEstimator(
            ep.attributions,
            ep.expectations,
            ec_threshold=self.effective_complexity_threshold,
            effective_complexity_mode=self.effective_complexity_mode,
            effective_complexity_score=ep.effective_complexity_score,
        )

        logger.info("Compute metric values")
        for metric in metrics.get_metrics():
            if metric.name in get_metrics_of(METRICS.PERFORMANCE):
                performance_estimator.compute_metric(metric)
            elif metric.name in get_metrics_of(METRICS.FAIRNESS):
                fairness_estimator.compute_metric(metric)
            elif metric.name in get_metrics_of(METRICS.INTERPRETABILITY):
                interpretability_estimator.compute_metric(metric)
            else:
                raise ValueError(f"Unknown metric name: {metric.name}")

        logger.info("Compute trustworthiness score")
        trust_estimator = TrustworthinessEstimator(
            aggregation=self.aggregation,
            gwa_hesitation=self.gwa_hesitation,
            fuzzy_thresholds=self.fuzzy_thresholds,
            metrics_values=self.metrics_values,
            alpha=self.aggregation_alpha,
            trustworthiness_score_scale=normalize_score_scale(
                self.trustworthiness_score_scale_
            ),
        )
        trust_estimator.compute_metric(metrics)
        validate_metric_list(metrics, require_values=True)
        return (
            trust_estimator,
            performance_estimator,
            fairness_estimator,
            interpretability_estimator,
        )

    def __preprocessing_fairness(self):
        logger.info("Create preprocessing bias mitigator")
        mitigator = BiasMitigatorPreProcessing()
        logger.info("Apply disparate impact removal on original data")
        self.fairness_processor_pre= mitigator.preprocess_disparate_impact_removal(
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
        logger.info("Store transformed preprocessing data")
        self.transformed_x_train = mitigator.transformed_x_train
        self.transformed_x_test = mitigator.transformed_x_test
        self.transformed_y_train = mitigator.transformed_y_train
        self.transformed_y_test = mitigator.transformed_y_test
        self.fairness_processor_pre= mitigator

    def __surrogate_model_train(self):
        """
        Train a surrogate model for explanations generation
        """
        self.explainer = InterpretabilityProcessing(
            min_r2=self.surrogate_min_r2,
            max_iterations=self.surrogate_max_iter,
            ec_threshold=self.effective_complexity_threshold,
            effective_complexity_mode=self.effective_complexity_mode,
            effective_complexity_loss_threshold=(
                self.effective_complexity_loss_threshold
            ),
            effective_complexity_repeats=self.effective_complexity_repeats,
            random_state=self.random_state or 0,
            ebm_interactions=self.surrogate_interactions,
            ebm_max_rounds=self.surrogate_max_rounds,
        )
        self.y_predicted_ex = self.explainer.surrogate_model_train(
            self.optimization_algorithm_,
            self.transformed_x_train,
            self.transformed_x_test,
            self.transformed_y_train,
            self.transformed_y_test,
        )
        self.surrogate_r2_error = self.explainer.surrogate_r2_error

    def __postprocessing_fairness(self, y_pred: pd.Series):
        logger.info("Create postprocessing bias mitigator")
        mitigator = BiasMitigatorPostProcessing(
            cost_constraint=self.equalize_odds_cost_constraint
        )
        prediction_scores = None
        if hasattr(self.explainer.explainable_classifier, "predict_proba"):
            classes = list(self.explainer.explainable_classifier.classes_)
            favorable_label = self.protected_classes[0]
            if favorable_label in classes:
                prediction_scores = self.explainer.explainable_classifier.predict_proba(
                    self.transformed_x_test
                )[:, classes.index(favorable_label)]
        logger.info("Apply equalized odds postprocessing")
        self.relabeled_y_test = mitigator.postprocess_equalize_odds(
            self.transformed_x_train,
            self.transformed_x_test,
            self.transformed_y_train,
            self.transformed_y_test,
            y_pred,
            y_pred.name,
            self.protected_groups,
            self.unprotected_groups,
            self.protected_classes,
            prediction_scores=prediction_scores,
            # self.equalize_odds_class_thresh
        )
        self.fairness_processor_post = mitigator

    def __capture_post_state(self):
        return {
            "relabeled_y_test": self.relabeled_y_test.copy(),
            "trust_estimator_post": self.trust_estimator_post,
            "performance_estimator_post": self.performance_estimator_post,
            "fairness_estimator_post": self.fairness_estimator_post,
            "interpretability_estimator_post": self.interpretability_estimator_post,
            "metrics_post": self.metrics_post.copy(),
            "fairness_processor_post": self.fairness_processor_post,
        }

    def __restore_post_state(self, state):
        self.relabeled_y_test = state["relabeled_y_test"]
        self.trust_estimator_post = state["trust_estimator_post"]
        self.performance_estimator_post = state["performance_estimator_post"]
        self.fairness_estimator_post = state["fairness_estimator_post"]
        self.interpretability_estimator_post = state["interpretability_estimator_post"]
        self.metrics_post = state["metrics_post"]
        self.fairness_processor_post = state["fairness_processor_post"]

    def __fairness_validation(self):
        # self.y_predicted_ex must be computed from surogate model training.
        logger.info("Start fairness validation")
        max_iterations = self.fairness_max_iter
        fairness_threshold = self.fairness_threshold
        iteration = 0
        actual_fairness = 0
        best_state = None
        best_score = float("-inf")
        # initialize the relabeled_y_test with the predictions from the explanaible surogate model
        self.relabeled_y_test = self.y_predicted_ex

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
            self.transformed_y_test,
            y_predicted=self.relabeled_y_test,
            fit_model=False,
        )
        best_state = self.__capture_post_state()
        best_score = self.trust_estimator_post.trustworthiness_score
        fairness_metrics_values = [
            metric.value
            for metric in self.metrics_post.get_metrics()
            if metric.name in get_metrics_of(METRICS.FAIRNESS)
        ]
        actual_fairness = min(fairness_metrics_values) if fairness_metrics_values else 1
        logger.info(
            "Fairness validation baseline: minimum fairness=%s, trustworthiness=%s",
            actual_fairness,
            best_score,
        )

        while iteration < max_iterations and actual_fairness < fairness_threshold:
            logger.info("Fairness validation iteration %s", iteration)
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
                self.transformed_y_test,
                y_predicted=self.relabeled_y_test,
                fit_model=False,
            )
            logger.info("Estimate fairness after postprocessing")
            fairness_metrics_values = [
                metric.value
                for metric in self.metrics_post.get_metrics()
                if metric.name in get_metrics_of(METRICS.FAIRNESS)
            ]
            logger.info("Fairness score values: %s", fairness_metrics_values)
            actual_fairness = min(fairness_metrics_values)
            trust_score = self.trust_estimator_post.trustworthiness_score
            logger.info("Candidate trustworthiness score: %s", trust_score)
            if trust_score > best_score:
                best_score = trust_score
                best_state = self.__capture_post_state()
                logger.info("Accepted new best postprocessing candidate")
            logger.info("Minimum fairness metric value: %s", actual_fairness)
            iteration = iteration + 1

        if iteration == max_iterations and actual_fairness < fairness_threshold:
            logger.warning("Postprocessing fairness assessment did not converge")

        if best_state is not None:
            self.__restore_post_state(best_state)
            logger.info("Restored best postprocessing candidate: trustworthiness=%s", best_score)

        logger.info("Fairness validation terminated")

    def __retain_diagnosis_as_final(self):
        logger.warning(
            "Best TWEEF candidate trustworthiness=%s is below diagnosis=%s. "
            "Retaining diagnosis model as final result.",
            self.trust_estimator_post.trustworthiness_score,
            self.trust_estimator_pre.trustworthiness_score,
        )
        self.metrics_post = self.metrics_pre.copy()
        self.trust_estimator_post = self.trust_estimator_pre
        self.performance_estimator_post = self.performance_estimator_pre
        self.fairness_estimator_post = self.fairness_estimator_pre
        self.interpretability_estimator_post = self.interpretability_estimator_pre
        self.fairness_processor_post = None
        self.classifier_ = self.given_algorithm_
        self.final_model_source_ = "diagnosis"
        self.relabeled_y_test = self.__series_like(
            self.given_algorithm_.predict(self.x_test),
            self.y_test.index,
            self.y_test.name,
        )

    def __prepare_training_data(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y must be provided together.")

        X, y = drop_invalid_rows(X, y, self.protected_groups)
        validate_groups(X, self.protected_groups, self.unprotected_groups)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.classes_ = np.unique(y)

        stratify_values = y if self.stratify else None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=stratify_values,
            random_state=self.random_state,
        )

    def fit(self, X, y):
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
        self
            The fitted trustworthy classifier.
        """
        self._reset_runtime_state()
        if self.metrics_values is not None and self.fuzzy_thresholds is not None:
            raise ValueError(
                "metrics_values and fuzzy_thresholds cannot be provided together."
            )
        self.trustworthiness_score_scale_ = normalize_score_scale(
            self.trustworthiness_score_scale
        )
        self.given_algorithm_ = clone(self.given_classifier)
        self.optimization_algorithm_ = clone(self.given_classifier)

        logger.info("Stage 1: trustworthiness diagnosis")
        self.__prepare_training_data(X, y)
        if self.metrics is None:
            raise ValueError("metrics must be provided before fitting the classifier.")
        validate_metric_list(self.metrics)
        validate_groups(self.x_train, self.protected_groups, self.unprotected_groups)
        validate_groups(self.x_test, self.protected_groups, self.unprotected_groups)
        self.metrics_pre = self.metrics.copy()
        self.metrics_post = self.metrics.copy()

        logger.info("Compute initial trustworthiness metrics")
        (
            self.trust_estimator_pre,
            self.performance_estimator_pre,
            self.fairness_estimator_pre,
            self.interpretability_estimator_pre,
        ) = self.__compute_trust(
            self.given_algorithm_,
            self.metrics_pre,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        )
        logger.info("Apply preprocessing fairness algorithm")
        self.__preprocessing_fairness()

        logger.info("Stage 2: train given model and surrogate model")
        self.__surrogate_model_train()

        logger.info("Stage 3: fairness postprocessing validation")
        self.__fairness_validation()

        self.classifier_ = self.explainer.explainable_classifier
        self.final_model_source_ = "tweef"
        if (
            self.trust_estimator_pre is not None
            and self.trust_estimator_post.trustworthiness_score
            < self.trust_estimator_pre.trustworthiness_score
        ):
            self.__retain_diagnosis_as_final()
        self.component_scores_ = self._component_scores()
        self.diagnosis_ = self.trustworthiness_diagnosis()
        self.final_report_ = self.component_scores_
        self.tradeoffs_ = self.dimension_tradeoffs()
        self.trustworthiness_score_ = self.trust_estimator_post.trustworthiness_score
        self.is_fitted_ = True
        return self

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
        check_is_fitted(self, "classifier_")
        check_is_fitted(self.classifier_)
        return self.classifier_.predict(x)

    def predict_proba(self, x: pd.DataFrame):
        """
        Predicts class probabilities when the final classifier supports them.
        """
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(
                f"{type(self.classifier_).__name__} does not support predict_proba."
            )
        return self.classifier_.predict_proba(x)

    def score(self, X, y, sample_weight=None):
        """
        Returns accuracy for scikit-learn compatibility.
        """
        check_is_fitted(self, "classifier_")
        y_predicted = self.predict(X)
        return accuracy_score(y, y_predicted, sample_weight=sample_weight)

    # #OUTPUTS =====================================================================

    def unbiased_dataset(self):
        """
        Returns the unbiased dataset after applying the pre-processing fairness algorithm.
        """
        check_is_fitted(self, "classifier_")
        data = pd.concat([self.transformed_x_train, self.transformed_x_test])
        data[self.transformed_y_train.name] = pd.concat(
            [self.transformed_y_train, self.relabeled_y_test]
        )
        return data

    def trustworthiness_diagnosis(self):
        """
        Returns the trustworthiness metrics for the given classifier
        """
        check_is_fitted(self, "classifier_")
        report = self.metrics_pre.to_dataframe()
        report["trustworthiness"] = self.trust_estimator_pre.trustworthiness_score
        report["trustworthiness_scale"] = (
            self.trust_estimator_pre.trustworthiness_score_scale
        )
        report["aggregation"] = self.trust_estimator_pre.aggregation
        return report

    def trustworthiness_metrics(self):
        """
        Returns the final trustworthiness metrics for the trustworthy classifier
        """
        check_is_fitted(self, "classifier_")
        report = self.metrics_post.to_dataframe()
        report["trustworthiness"] = self.trust_estimator_post.trustworthiness_score
        report["trustworthiness_scale"] = (
            self.trust_estimator_post.trustworthiness_score_scale
        )
        report["aggregation"] = self.trust_estimator_post.aggregation
        return report

    def _component_scores(self):
        check_is_fitted(self, "classifier_")
        rows = []
        for stage, metrics in (
            ("diagnosis", self.metrics_pre),
            ("final", self.metrics_post),
        ):
            for metric in metrics.get_metrics():
                if metric.name in get_metrics_of(METRICS.PERFORMANCE):
                    dimension = "performance"
                elif metric.name in get_metrics_of(METRICS.FAIRNESS):
                    dimension = "fairness"
                else:
                    dimension = "interpretability"
                rows.append(
                    {
                        "stage": stage,
                        "dimension": dimension,
                        "metric": metric.name,
                        "value": metric.value,
                        "weight": metric.weight,
                        "goal": metric.goal,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _metric_dimension(metric_name):
        if metric_name in get_metrics_of(METRICS.PERFORMANCE):
            return "performance"
        if metric_name in get_metrics_of(METRICS.FAIRNESS):
            return "fairness"
        return "interpretability"

    def _dimension_scores(self, metrics):
        rows = []
        for metric in metrics.get_metrics():
            rows.append(
                {
                    "dimension": self._metric_dimension(metric.name),
                    "value": metric.value,
                    "weight": metric.weight,
                }
            )
        metric_frame = pd.DataFrame(rows)
        scores = {}
        for dimension, frame in metric_frame.groupby("dimension"):
            values = frame["value"].astype(float)
            weights = frame["weight"].astype(float)
            weight_sum = weights.sum()
            if weight_sum > 0:
                scores[dimension] = float(np.average(values, weights=weights))
            else:
                scores[dimension] = float(values.mean())
        return scores

    @staticmethod
    def _relative_change(initial, final):
        if initial == 0:
            return np.nan
        return (final - initial) / abs(initial)

    @staticmethod
    def _relative_cost(initial, final):
        if initial == 0:
            return np.nan
        return max(0.0, (initial - final) / abs(initial))

    @staticmethod
    def _variation_ratio(relative_delta, relative_delta_reference):
        if relative_delta_reference == 0 or np.isnan(relative_delta_reference):
            return np.nan
        return relative_delta / abs(relative_delta_reference)

    @staticmethod
    def _tradeoff_label(delta_dimension, delta_reference):
        if delta_dimension > 0 and delta_reference < 0:
            return "gain-loss"
        if delta_dimension < 0 and delta_reference > 0:
            return "loss-gain"
        if delta_dimension > 0 and delta_reference > 0:
            return "joint-gain"
        if delta_dimension < 0 and delta_reference < 0:
            return "joint-loss"
        return "neutral"

    def dimension_tradeoffs(self):
        """
        Returns directional trade-offs between trust dimensions.

        Dimension scores are weighted averages of the metrics belonging to each
        dimension. Trade-off costs follow the same relative-decrease idea used by
        price-of-fairness style analyses. Each row reads as: how `dimension`
        changed with respect to `reference_dimension`.
        """
        check_is_fitted(self, "classifier_")
        diagnosis_scores = self._dimension_scores(self.metrics_pre)
        final_scores = self._dimension_scores(self.metrics_post)
        dimension_order = ["performance", "fairness", "interpretability"]
        dimensions = [
            dimension
            for dimension in dimension_order
            if dimension in diagnosis_scores or dimension in final_scores
        ]

        rows = []
        for dimension, reference_dimension in permutations(dimensions, 2):
            diagnosis = diagnosis_scores.get(dimension, np.nan)
            final = final_scores.get(dimension, np.nan)
            reference_diagnosis = diagnosis_scores.get(reference_dimension, np.nan)
            reference_final = final_scores.get(reference_dimension, np.nan)
            delta = final - diagnosis
            reference_delta = reference_final - reference_diagnosis
            relative_delta = self._relative_change(diagnosis, final)
            reference_relative_delta = self._relative_change(
                reference_diagnosis,
                reference_final,
            )
            rows.append(
                {
                    "tradeoff": f"{dimension}_vs_{reference_dimension}",
                    "dimension": dimension,
                    "diagnosis": diagnosis,
                    "final": final,
                    "delta": delta,
                    "relative_delta": relative_delta,
                    "reference_dimension": reference_dimension,
                    "reference_diagnosis": reference_diagnosis,
                    "reference_final": reference_final,
                    "reference_delta": reference_delta,
                    "reference_relative_delta": reference_relative_delta,
                    "variation_ratio": self._variation_ratio(
                        relative_delta,
                        reference_relative_delta,
                    ),
                    "relative_cost": self._relative_cost(
                        diagnosis,
                        final,
                    ),
                    "relation": self._tradeoff_label(delta, reference_delta),
                }
            )
        return pd.DataFrame(rows)

    def tradeoff_fairness(self):
        """
        Returns the fairness metrics for the pre-processing and post-processing fairness algorithms.
        """
        check_is_fitted(self, "classifier_")
        equalized_odds = None
        if self.fairness_processor_post is not None:
            equalized_odds = self.fairness_processor_post.equalized_odds
        return pd.DataFrame(
            [{
                "disparate_impact": self.fairness_processor_pre.disparate_impact,
                "equalized_odds"  : equalized_odds,
            }]
        )

    def trust_cost_on_performance(self):
        """
        Returns the performance decrease caused by fairness mitigation and interpretable surrogate model, for each performance metric. Computed as:
        cost_on_performance = (initial_performance − final_performance) / initial_performance
        """
        check_is_fitted(self, "classifier_")
        performance_metrics = []

        for metric_pre, metric_post in zip(self.metrics_pre.metrics, self.metrics_post.metrics):
            if metric_pre.name in get_metrics_of(METRICS.PERFORMANCE):
                value = 0
                if metric_pre.value != 0:
                    value = (metric_pre.value - metric_post.value)/metric_pre.value
                performance_metrics.append({
                    "metric": metric_pre.name,
                    "value": value
                })
        return pd.DataFrame(performance_metrics)
