import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from interpret.glassbox import ExplainableBoostingClassifier

from TWEEF.Constants import EPSILON
from TWEEF.Logging import get_logger


logger = get_logger(__name__)

class InterpretabilityProcessing:
    """
    A class for assessing and enhancing model interpretability using surrogate models
    and feature attribution metrics.

    Attributes:
        explainable_classifier: The surrogate explainable model.
        surrogate_r2_error: R2 error value for the surrogate model.
        min_r2: Minimum R2 error value required for accept surrogate model training.
        ec_threshold: Threshold for effective complexity.
    """

    def __init__(
        self,
        min_r2=0.8,
        ec_threshold=0.95,
        effective_complexity_mode="attribution",
        effective_complexity_loss_threshold=0.05,
        effective_complexity_repeats=3,
        random_state=0,
        max_iterations=100,
        ebm_interactions=0,
        ebm_max_rounds=None,
    ):
        if effective_complexity_mode not in {"attribution", "loss"}:
            raise ValueError(
                "effective_complexity_mode must be 'attribution' or 'loss'"
            )
        self.explainable_classifier = None
        self.surrogate_r2_error = None
        self.attributions = None
        self.expectations = None
        self.effective_complexity_score = None
        self.ec_threshold = ec_threshold
        self.effective_complexity_mode = effective_complexity_mode
        self.effective_complexity_loss_threshold = effective_complexity_loss_threshold
        self.effective_complexity_repeats = effective_complexity_repeats
        self.random_state = random_state
        self.min_r2 = min_r2
        self.max_iterations = max_iterations
        self.ebm_interactions = ebm_interactions
        self.ebm_max_rounds = ebm_max_rounds

    def _fi(self, x_i, trained_exp_model: ClassifierMixin):
        """
        Predicts the output for a specific feature by fixing other features.

        Parameters:
        x_i: Input sample with fixed feature values.

        Returns:
        Prediction for the specific feature.
        """
        if trained_exp_model is not None and hasattr(trained_exp_model, "fit"):
            prediction = trained_exp_model.predict([x_i])[0]
            if(isinstance(prediction, np.str_)): # ExplanaibleBoostingClassifier returns string predictions
                prediction = float(prediction)
            return prediction
        else:
            raise RuntimeError("Explainable classifier is not fitted or initialized!")

    def _L(self, y_i, fi):
        """
        Computes the log loss for a single sample.

        Parameters:
        y_i (float): True label.
        fi (float): Predicted value for the label.

        Returns:
        Log loss for the prediction.
        """
        fi = np.clip(fi, EPSILON, 1 - EPSILON)
        return -(y_i * np.log(fi) + (1 - y_i) * np.log(1 - fi))

    def _expectations(self, 
        trained_exp_model: ClassifierMixin, 
        X: pd.DataFrame, 
        y: pd.Series, 
        p: np.ndarray):
        """
        Computes expectation values based on feature attribution metrics.

        Parameters:
        trained_exp_model (ClassifierMixin): Trained explainable model.

        Returns:
        Expectations for each feature.
        """
        E = np.zeros((len(X), len(self.attributions)))
        for i, x_i_original in enumerate(X):
            for j in range(len(self.attributions)):
                x_i = np.copy(x_i_original)
                x_i[j] = 0
                E[i, j] = self._L(y[i], self._fi(x_i, trained_exp_model)) * p[i]
        return E.mean(axis=0)

    def _positive_probabilities(self, trained_exp_model, X: pd.DataFrame):
        if hasattr(trained_exp_model, "predict_proba"):
            probabilities = trained_exp_model.predict_proba(X)
            if probabilities.shape[1] > 1:
                classes = getattr(trained_exp_model, "classes_", [0, 1])
                positive_index = (
                    list(classes).index(1)
                    if 1 in list(classes)
                    else probabilities.shape[1] - 1
                )
                return probabilities[:, positive_index]
            classes = getattr(trained_exp_model, "classes_", [0])
            return (
                np.ones(len(X))
                if list(classes)[0] == 1
                else np.zeros(len(X))
            )
        return None

    def _prediction_loss(self, trained_exp_model, X_reference, X_restricted):
        reference_probabilities = self._positive_probabilities(
            trained_exp_model, X_reference
        )
        restricted_probabilities = self._positive_probabilities(
            trained_exp_model, X_restricted
        )
        if reference_probabilities is not None and restricted_probabilities is not None:
            restricted_probabilities = np.clip(
                restricted_probabilities, EPSILON, 1 - EPSILON
            )
            reference_probabilities = np.clip(
                reference_probabilities, EPSILON, 1 - EPSILON
            )
            return np.mean(
                -(
                    reference_probabilities * np.log(restricted_probabilities)
                    + (1 - reference_probabilities)
                    * np.log(1 - restricted_probabilities)
                )
            )

        reference_predictions = trained_exp_model.predict(X_reference)
        restricted_predictions = trained_exp_model.predict(X_restricted)
        return np.mean(reference_predictions != restricted_predictions)

    def _loss_based_effective_complexity(self, trained_exp_model, X: pd.DataFrame):
        if len(self.attributions) == 0:
            return 0

        ordered_feature_indices = np.argsort(np.abs(self.attributions))[::-1]
        rng = np.random.default_rng(self.random_state)
        repeats = max(1, int(self.effective_complexity_repeats))

        for k in range(1, len(ordered_feature_indices) + 1):
            important_indices = set(ordered_feature_indices[:k])
            non_important_indices = [
                index
                for index in range(len(ordered_feature_indices))
                if index not in important_indices
            ]
            losses = []
            for _ in range(repeats):
                restricted = X.copy()
                for index in non_important_indices:
                    column = restricted.columns[index]
                    restricted[column] = rng.permutation(X[column].to_numpy())
                losses.append(
                    self._prediction_loss(trained_exp_model, X, restricted)
                )
            if float(np.mean(losses)) < self.effective_complexity_loss_threshold:
                return 1 - (k / len(ordered_feature_indices))

        return 0

    def compute_attributions_expectations(
        self, 
        x_train: pd.DataFrame, 
        x_test: pd.DataFrame, 
        y_test: pd.Series, 
        trained_exp_model: ClassifierMixin
    ):
        """
        Estimates model interpretability using feature attribution metrics.

        Parameters:
        x_train (DataFrame): Training data.
        x_test (DataFrame): Testing data.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
        trained_exp_model (ClassifierMixin): Trained explainable model.
        """
        check_is_fitted(trained_exp_model)

        logger.info("Extract attributions")
        if isinstance(trained_exp_model, ExplainableBoostingClassifier):
            self.attributions = trained_exp_model.term_importances()[
                : len(x_train.columns)
            ]
        elif hasattr(trained_exp_model, "feature_importances_"):
            self.attributions = trained_exp_model.feature_importances_
        else:
            logger.warning("No native attributions detected. Using permutation importance")
            result = permutation_importance(
                trained_exp_model,
                x_test,
                y_test,
                n_repeats=5,
                random_state=0,
            )
            self.attributions = result.importances_mean

        logger.info("Calculate expectations")
        if hasattr(trained_exp_model, "predict_proba"):
            probabilities = trained_exp_model.predict_proba(x_test)
            if probabilities.shape[1] > 1:
                classes = getattr(trained_exp_model, "classes_", [0, 1])
                positive_index = (
                    list(classes).index(1) if 1 in list(classes) else probabilities.shape[1] - 1
                )
                pX = probabilities[:, positive_index]
            else:
                classes = getattr(trained_exp_model, "classes_", [0])
                pX = np.ones(len(x_test)) if list(classes)[0] == 1 else np.zeros(len(x_test))
        else:
            pX = trained_exp_model.predict(x_test).astype(float)
        self.expectations = self._expectations(trained_exp_model, x_test.to_numpy(), y_test.to_numpy(), pX)
        if self.effective_complexity_mode == "loss":
            logger.info("Calculate loss-based effective complexity")
            self.effective_complexity_score = self._loss_based_effective_complexity(
                trained_exp_model,
                x_test,
            )
        else:
            self.effective_complexity_score = None



    def surrogate_model_train(self, given_algorithm, x_train, x_test, y_train, y_test):
        """
        Trains a surrogate model to approximate the predictions of a given algorithm.

        Parameters:
        given_algorithm: The base model to approximate.
        x_train (DataFrame): Training data.
        x_test (DataFrame): Testing data.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.

        Returns:
        y_predicted_ex (Series): Predictions from the surrogate model.
        """
        self.surrogate_r2_error = 0
        max_iterations = self.max_iterations
        iteration = 0
        logger.info("Start surrogate model training iterations")

        ebm_kwargs = {
            "feature_names": x_test.columns,
            "interactions": self.ebm_interactions,
            "n_jobs": 1,
        }
        if self.ebm_max_rounds is not None:
            ebm_kwargs["max_rounds"] = self.ebm_max_rounds
        self.explainable_classifier = ExplainableBoostingClassifier(**ebm_kwargs)

        while iteration < max_iterations and self.surrogate_r2_error < self.min_r2:
            logger.info("Surrogate training iteration %s: train given algorithm", iteration + 1)
            given_algorithm.fit(x_train, y_train)

            logger.info("Predict with given algorithm")
            y_predicted = given_algorithm.predict(x_test)

            logger.info("Train surrogate model")
            self.explainable_classifier.fit(x_test, y_predicted)

            logger.info("Predict with surrogate model")
            y_predicted_ex = self.explainable_classifier.predict(x_test)

            logger.info("Calculate surrogate R2")
            self.surrogate_r2_error = r2_score(
                y_predicted.astype(np.float64), y_predicted_ex.astype(np.float64)
            )
            iteration += 1

        if self.surrogate_r2_error < self.min_r2:
            logger.warning(
                "Surrogate model did not reach minimum R2 threshold: %s",
                self.min_r2,
            )

        logger.info("Surrogate model training completed")
        logger.info("Extract attributions and expectations")
        
        self.compute_attributions_expectations(x_train, x_test, y_test, self.explainable_classifier)

        return pd.Series(y_predicted_ex, index=y_test.index, name=y_test.name)
