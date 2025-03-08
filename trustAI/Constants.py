"""
Constants for TrustAI module.

This module defines various constants used for performance metrics, fairness metrics,
interpretability metrics, linguistic values and labels, and other configurations.

Constants:
    METRICS_PERFORMANCE_ACCURACY (str): Accuracy performance metric.
    METRICS_PERFORMANCE_RECALL (str): Recall performance metric.
    METRICS_PERFORMANCE_PRECISION (str): Precision performance metric.
    METRICS_PERFORMANCE_F1_SCORE (str): F1 Score performance metric.

    METRICS_FAIRNESS_TREATMENT_EQUALITY (str): Treatment equality fairness metric.
    METRICS_FAIRNESS_EQUALIZED_ODDS (str): Equalized odds fairness metric.
    METRICS_FAIRNESS_EQUAL_OPPORTUNITY (str): Equal opportunity fairness metric.
    METRICS_FAIRNESS_STATISTICAL_PARITY (str): Statistical parity fairness metric.

    METRICS_INTERPRETABILITY_MONOTONICITY (str): Monotonicity interpretability metric.
    METRICS_INTERPRETABILITY_NON_SENSITIVITY (str): Non-sensitivity interpretability metric.
    METRICS_INTERPRETABILITY_EFFECTIVE_COMPLEXITY (str): Effective complexity interpretability metric.

    LINGUISTIC_VALUES (list of int): List of linguistic values ranging from -2 to 2.
    LINGUISTIC_LABELS (list of str): List of linguistic labels corresponding to the linguistic values.
    TRUSTWORTHINESS_SCORE_NAME (str): Name of the trustworthiness score.

    EPSILON (float): A small constant value used for numerical stability.
"""

LINGUISTIC_VALUES = [-2, -1, 0, 1, 2]
LINGUISTIC_LABELS = ["Untrustworthy", "Slightly Trustworthy", "Neutral", "Trustworthy", "Very Trustworthy"]
TRUSTWORTHINESS_SCORE_NAME = "Trustworthiness"
EPSILON = 1e-15

class METRICS:
    class PERFORMANCE:
        ACCURACY  = "Accuracy"
        RECALL    = "Recall"
        PRECISION = "Precision"
        F1_SCORE  = "F1Score"
    class FAIRNESS:
        TREATMENT_EQUALITY = "Treatment Equality"
        EQUALIZED_ODDS     = "Equalized Odds"
        EQUAL_OPPORTUNITY  = "equal Opportunity"
        STATISTICAL_PARITY = "Statistical Parity"
    class INTERPRETABILITY:
        MONOTONICITY         = "Monotonicity"
        NON_SENSITIVITY      = "Non Sensitivity"
        EFFECTIVE_COMPLEXITY = "Effective Complexity"

def get_metrics_of(metrics_class):
    """
    Receives a metrics class and returns a list with all the metrics defined in the class.
    It can be a nested class or a top-level class (PERFORMANCE, FAIRNESS, or INTERPRETABILITY).
    """
    metrics = []
    
    for attr_name, attr_value in vars(metrics_class).items():
        if not attr_name.startswith("__"):  # Ignora atributos internos
            if isinstance(attr_value, type):  # Es una clase anidada
                metrics.extend(get_metrics_of(attr_value))
            elif isinstance(attr_value, str):  # Es una constante (métrica)
                metrics.append(attr_value)

    return metrics


# Diccionario interno para asociar métricas con sus thresholds
_THESHOLDS_MAP = {
    METRICS.PERFORMANCE.ACCURACY:  [0.5, 0.6, 0.7, 0.8, 0.9],
    METRICS.PERFORMANCE.RECALL:    [0.5, 0.6, 0.7, 0.8, 0.9],
    METRICS.PERFORMANCE.PRECISION: [0.5, 0.6, 0.7, 0.8, 0.9],
    METRICS.PERFORMANCE.F1_SCORE:  [0.5, 0.6, 0.7, 0.8, 0.9],

    METRICS.FAIRNESS.TREATMENT_EQUALITY: [0.6, 0.7, 0.8, 0.9, 0.95],
    METRICS.FAIRNESS.EQUALIZED_ODDS:     [0.6, 0.7, 0.8, 0.9, 0.95],
    METRICS.FAIRNESS.EQUAL_OPPORTUNITY:  [0.6, 0.7, 0.8, 0.9, 0.95],
    METRICS.FAIRNESS.STATISTICAL_PARITY: [0.6, 0.7, 0.8, 0.9, 0.95],

    METRICS.INTERPRETABILITY.MONOTONICITY:         [0.3, 0.4, 0.5, 0.6, 0.7],
    METRICS.INTERPRETABILITY.NON_SENSITIVITY:      [0.3, 0.4, 0.5, 0.6, 0.7],
    METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY: [0.3, 0.4, 0.5, 0.6, 0.7],
}
FUZZY_THRESHOLD_PEAKS = ["peak_-2", "peak_-1", "peak_0", "peak_1", "peak_2"]


def get_metric_thresholds(metric: str):
    """
    Devuelve los umbrales asociados a una métrica específica.

    :param metric: Nombre de la métrica desde METRICS.
    :return: Lista de thresholds o None si la métrica no existe.
    """
    return _THESHOLDS_MAP.get(metric, None)

