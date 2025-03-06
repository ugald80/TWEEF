import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from interpret.glassbox import ExplainableBoostingClassifier

from trustAI.Constants import EPSILON

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

    def __init__(self, min_r2=0.8, ec_threshold=0.05):
        self.explainable_classifier = None
        self.surrogate_r2_error = None
        self.attributions = None
        self.expectations = None
        self.ec_threshold = ec_threshold
        self.min_r2 = min_r2

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
        for e_i, i in zip(E, np.arange(len(E))):            
            for x_j, y_j, p_j, e_j, j in zip(X, y, p, e_i, np.arange(len(e_i))):
                x_i = np.copy(x_j)
                x_i[j] = 0
                E[i, j] = self._L(y_j, self._fi(x_i, trained_exp_model)) * p_j
        return E.mean(axis=0)

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

        print("         . Extracting attributions...")
        if isinstance(trained_exp_model, ExplainableBoostingClassifier):
            self.attributions = trained_exp_model.term_importances()[
                : len(x_train.columns)
            ]
        elif hasattr(trained_exp_model, "feature_importances_"):
            self.attributions = trained_exp_model.feature_importances_
        else:
            print("      ...No interpretable model detected. Attributions set to None.")
            self.attributions = None
            return

        print("         . Calculating expectations...")
        pX = trained_exp_model.predict_proba(x_test)[:, 1]
        self.expectations = self._expectations(trained_exp_model, x_test.to_numpy(), y_test.to_numpy(), pX)



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
        max_iterations = 3
        iteration = 0
        print("   Surrogate model training: Starting iterations...")

        self.explainable_classifier = ExplainableBoostingClassifier(
            feature_names=x_test.columns
        )

        while iteration < max_iterations and self.surrogate_r2_error < self.min_r2:
            print(f"      Iteration {iteration + 1}: Training given algorithm...")
            given_algorithm.fit(x_train, y_train)

            print("      Predicting with given algorithm...")
            y_predicted = given_algorithm.predict(x_test)

            print("      Training surrogate model...")
            self.explainable_classifier.fit(x_test, y_predicted)

            print("      Predicting with surrogate model...")
            y_predicted_ex = self.explainable_classifier.predict(x_test)

            print("      Calculating R2 error...")
            self.surrogate_r2_error = r2_score(
                y_predicted.astype(np.float64), y_predicted_ex.astype(np.float64)
            )
            iteration += 1

        if self.surrogate_r2_error < self.min_r2:
            print(f"   Surrogate model did not reach the minimum R2 threshold of {self.min_r2}.")

        print("   Surrogate model training completed.")
        print("   Extract attributions and expectations...")
        
        self.compute_attributions_expectations(x_train, x_test, y_test, self.explainable_classifier)

        return pd.Series(y_predicted_ex, index=y_test.index, name=y_test.name)
