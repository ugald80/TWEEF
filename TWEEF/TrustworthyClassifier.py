"""Train a deployable surrogate and evaluate it only on explicit held-out data.

The original estimator is a teacher, never a fallback final predictor. Metric
references and candidate selection use internal training folds; the optional
external validation partition is reserved for final postprocessor calibration.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from numbers import Integral
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted

from TWEEF.Constants import METRICS, get_metrics_of
from TWEEF.Metrics import MetricList
from TWEEF.Aggregation import (
    AGGREGATION, TRUSTWORTHINESS_SCORE_SCALE, normalize_score_scale,
    normalize_metric_weights,
)
from TWEEF.ProcessorInterpretability import InterpretabilityProcessing
from TWEEF.EstimatorTrustworthiness import TrustworthinessEstimator
from TWEEF.EstimatorFairness import FairnessEstimator
from TWEEF.EstimatorPerformance import PerformanceEstimator
from TWEEF.EstimatorInterpretability import InterpretabilityEstimator
from TWEEF.UtilsFairnessGroup import FairnessGroupUtils, FairnessGroup
from TWEEF.Validation import validate_metric_list
from TWEEF.Logging import get_logger
from TWEEF.Predictor import Predictor
from TWEEF.Evaluation import (
    make_tabular_preprocessor, frozen_ranking, evaluate_metrics,
    effective_complexity as compute_effective_complexity, normalize_ec_config,
)

logger = get_logger(__name__)
METRIC_NAMES = [METRICS.PERFORMANCE.ACCURACY, METRICS.FAIRNESS.EQUAL_OPPORTUNITY,
                METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY]


@dataclass
class EvaluationReport:
    metrics: dict
    metrics_frame: pd.DataFrame
    trustworthiness: object
    teacher_student_fidelity: dict
    predictions: pd.Series
    metadata: dict


class TrustworthyClassifier(ClassifierMixin, BaseEstimator):
    accepts_sensitive_features = True
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
        surrogate=None,
        preprocessor=None,
        metric_weights=None,
        effective_complexity=None,
        reference_cv=5,
        postprocessor=None,
        reference_metrics=None,
        selection_candidates=None,
        surrogate_min_agreement=0.8,
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

        self.surrogate = surrogate
        self.preprocessor = preprocessor
        self.metric_weights = metric_weights
        self.effective_complexity = effective_complexity
        self.reference_cv = reference_cv
        self.postprocessor = postprocessor
        self.reference_metrics = reference_metrics
        self.selection_candidates = selection_candidates
        self.surrogate_min_agreement = surrogate_min_agreement

    def _reset_runtime_state(self):
        for name in list(self.__dict__):
            if name.endswith("_"):
                delattr(self, name)
        self.x_test = self.y_test = None
        self.fairness_processor_pre = self.fairness_processor_post = None
        self.metrics_pre = self.metrics_post = None
        self.trust_estimator_pre = self.trust_estimator_post = None

    @staticmethod
    def __series_like(values, index, name):
        return pd.Series(np.asarray(values).reshape(-1).astype(float), index=index, name=name)

    @staticmethod
    def _xy(X, y):
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise ValueError("X must be a nonempty DataFrame with unique original feature names.")
        if not X.index.is_unique:
            raise ValueError("Observation indices must be unique and stable for reproducible inference.")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("y index must match X, including its order.")
        values = np.asarray(y)
        if values.ndim != 1 or len(values) != len(X) or not np.isin(values, [0, 1]).all():
            raise ValueError("This protocol requires complete binary labels 0 and 1 aligned with X.")
        return X.copy(), pd.Series(values.astype(int), index=X.index, name=getattr(y, "name", None) or "target")

    def _groups(self, X, sensitive_features=None):
        if sensitive_features is None and self.protected_groups:
            mask = np.zeros(len(X), dtype=bool)
            for group in self.protected_groups:
                for name, values in group.items():
                    if name not in X:
                        raise ValueError(f"Missing protected group column {name!r}.")
                    mask |= X[name].isin(values).to_numpy()
            sensitive_features = pd.Series(np.where(mask, "protected", "unprotected"), index=X.index)
        if sensitive_features is None:
            return None
        if isinstance(sensitive_features, pd.Series) and not sensitive_features.index.equals(X.index):
            raise ValueError("sensitive_features index must match X, including its order.")
        values = np.asarray(sensitive_features)
        if values.ndim != 1 or len(values) != len(X) or pd.isna(values).any():
            raise ValueError("sensitive_features must be complete, one-dimensional and aligned with X.")
        return pd.Series(values, index=X.index, name="sensitive_features")

    def _weights(self):
        if self.metric_weights is not None and self.metrics is not None:
            raise ValueError("metric_weights and legacy metrics are mutually exclusive.")
        if self.metric_weights is not None:
            weights = dict(self.metric_weights)
        elif self.metrics is not None:
            validate_metric_list(self.metrics)
            weights = dict(zip(self.metrics.get_names(), self.metrics.get_weights()))
        else:
            weights = dict.fromkeys(METRIC_NAMES, 1.0 / 3)
        if not weights or set(weights).difference(METRIC_NAMES):
            raise ValueError(f"The main pipeline supports these metrics: {METRIC_NAMES!r}.")
        try:
            values = np.asarray(list(weights.values()), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("metric weights must be finite and nonnegative.") from exc
        if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
            raise ValueError("metric weights must be finite, nonnegative and have positive sum.")
        return normalize_metric_weights(weights)

    @staticmethod
    def _strata(y, groups):
        return y.astype(str) if groups is None else y.astype(str) + ":" + groups.astype(str)

    def _split(self, X, y, groups, size, seed):
        try:
            fit, calibration = train_test_split(
                np.arange(len(X)), test_size=size, random_state=seed,
                stratify=self._strata(y, groups),
            )
        except ValueError as exc:
            raise ValueError("Insufficient observations to stratify calibration by class and sensitive group.") from exc
        return (X.iloc[fit], y.iloc[fit], None if groups is None else groups.iloc[fit],
                X.iloc[calibration], y.iloc[calibration], None if groups is None else groups.iloc[calibration])

    def _new_surrogate(self, override=None):
        specified = self.surrogate if override is None else override
        if specified is not None:
            return clone(specified)
        from interpret.glassbox import ExplainableBoostingClassifier
        parameters = dict(interactions=self.surrogate_interactions, random_state=self.random_state, n_jobs=1)
        if self.surrogate_max_rounds is not None:
            parameters["max_rounds"] = self.surrogate_max_rounds
        return ExplainableBoostingClassifier(**parameters)

    def _new_postprocessor(self, specified):
        if specified is None:
            return None
        if isinstance(specified, str):
            if specified != "calibrated_equalized_odds":
                raise ValueError("Unknown postprocessor; use calibrated_equalized_odds or an adapter.")
            from TWEEF.PostProcessing import CalibratedEqualizedOdds
            return CalibratedEqualizedOdds(
                random_state=self.random_state, cost_constraint=self.equalize_odds_cost_constraint,
                threshold=self.equalize_odds_class_thresh,
            )
        if not callable(getattr(specified, "fit", None)) or not callable(getattr(specified, "transform", None)):
            raise TypeError("A postprocessor must implement fit and transform.")
        return deepcopy(specified)

    def _fit_predictors(self, X, y, groups, calibration, candidate):
        specified_preprocessor = candidate.get("preprocessor", self.preprocessor)
        preprocessor = (make_tabular_preprocessor(X) if specified_preprocessor is None
                        else clone(specified_preprocessor))
        transformed = preprocessor.fit_transform(X, y)
        teacher = clone(self.given_classifier).fit(transformed, y)
        targets = np.asarray(teacher.predict(transformed)).reshape(-1)
        if not np.isin(targets, [0, 1]).all():
            raise ValueError("The teacher must predict binary labels 0 and 1; provide a classifier adapter.")
        surrogate = self._new_surrogate(candidate.get("surrogate"))
        surrogate.fit(transformed, targets.astype(int))
        teacher_predictor = Predictor(preprocessor, teacher)
        base_predictor = Predictor(preprocessor, surrogate)
        postprocessor = self._new_postprocessor(candidate.get("postprocessor", self.postprocessor))
        if postprocessor is not None:
            if calibration is None:
                raise ValueError("A separate calibration partition is required for postprocessing.")
            X_cal, y_cal, groups_cal = calibration
            if groups_cal is None:
                raise ValueError("Postprocessing requires sensitive_features on calibration data.")
            postprocessor.fit(base_predictor.predict_proba(X_cal)[:, 1], y_cal,
                              sensitive_features=groups_cal)
        predictor = Predictor(preprocessor, surrogate, postprocessor,
                              threshold=getattr(postprocessor, "threshold", self.equalize_odds_class_thresh))
        return teacher_predictor, base_predictor, predictor

    def _candidate_configurations(self):
        if self.selection_candidates is None:
            return [{}]
        if not isinstance(self.selection_candidates, (list, tuple)) or not self.selection_candidates:
            raise ValueError("selection_candidates must be a nonempty sequence of configuration mappings.")
        result = []
        allowed = {"surrogate", "preprocessor", "postprocessor"}
        for candidate in self.selection_candidates:
            if not hasattr(candidate, "items") or set(candidate).difference(allowed):
                raise ValueError(f"Candidate overrides must be mappings with keys in {sorted(allowed)!r}.")
            result.append(dict(candidate))
        return result

    def _make_trust_estimator(self, mechanism, reference):
        if mechanism is None:
            return None
        # Forward the legacy options even for objects so the estimator can
        # reject conflicting global settings instead of silently ignoring them.
        kwargs = dict(aggregation=mechanism, alpha=self.aggregation_alpha,
                      gwa_hesitation=self.gwa_hesitation,
                      trustworthiness_score_scale=self.trustworthiness_score_scale)
        if isinstance(mechanism, str):
            if self.fuzzy_thresholds is not None:
                kwargs["fuzzy_thresholds"] = self.fuzzy_thresholds
                reference = None
        elif self.fuzzy_thresholds is not None:
            raise ValueError("An injected mechanism owns calibration; configure thresholds on that object.")
        return TrustworthinessEstimator(**kwargs).fit(
            metrics_values=reference, metric_weights=self.metric_weights_,
        )

    @staticmethod
    def _score_frame(estimator, frame):
        if estimator is None:
            return None
        # Evaluation must not mutate the frozen preparation or a plugin's state.
        copied = deepcopy(estimator)
        copied.transform(frame)
        return copied.aggregation_results_[0]

    def fit(self, X, y, *, sensitive_features=None, validation_data=None,
            validation_sensitive_features=None):
        """Fit the teacher and final surrogate without accepting evaluation/test data.

        External validation data are used only to calibrate the final optional
        postprocessor. Selection and metric references use training folds. If no
        validation data are supplied, reserve an internal calibration partition.
        """
        self._reset_runtime_state()
        X, y = self._xy(X, y)
        groups = self._groups(X, sensitive_features)
        self.metric_weights_ = self._weights()
        for name in ("surrogate_min_agreement", "fairness_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.surrogate_min_r2 != 0.8:
            warnings.warn("surrogate_min_r2 is deprecated and ignored: classification fidelity uses "
                          "surrogate_min_agreement, measured without fitting on evaluation data.",
                          FutureWarning, stacklevel=2)
        if self.metric_weights_.get(METRICS.FAIRNESS.EQUAL_OPPORTUNITY, 0) > 0 and groups is None:
            raise ValueError("A positive fairness weight requires sensitive_features or legacy protected_groups.")
        self.ec_config_ = normalize_ec_config(self.effective_complexity or {
            "loss_threshold": self.effective_complexity_loss_threshold,
            "repeats": self.effective_complexity_repeats,
        })
        if self.disparate_impact_repair_level:
            warnings.warn(
                "Legacy DisparateImpactRemover is not applied by the inductive pipeline: it refitted on test. "
                "Inject a reusable preprocessor to request inductive preprocessing mitigation.",
                FutureWarning, stacklevel=2,
            )
        if validation_data is None:
            warnings.warn("test_size is a deprecated name for the internal calibration fraction; "
                          "no test data are consumed by fit.", FutureWarning, stacklevel=2)
            if validation_sensitive_features is not None:
                raise ValueError("validation_sensitive_features requires validation_data.")
            X, y, groups, X_cal, y_cal, groups_cal = self._split(X, y, groups, self.test_size, self.random_state)
            calibration = (X_cal, y_cal, groups_cal)
            calibration_source = "internal_training_split"
        else:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be an (X, y) pair reserved for calibration.")
            X_cal, y_cal = self._xy(*validation_data)
            if set(X_cal.columns) != set(X.columns):
                raise ValueError("Calibration features must match training features.")
            X_cal = X_cal.loc[:, X.columns]
            if X.index.intersection(X_cal.index).size:
                raise ValueError("Training and calibration must have disjoint stable observation indices.")
            groups_cal = self._groups(X_cal, validation_sensitive_features)
            calibration = (X_cal, y_cal, groups_cal)
            calibration_source = "external_calibration"
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]
        self.x_train, self.y_train = X.copy(), y.copy()
        self.training_index_ = X.index.to_numpy(copy=True)
        self.calibration_index_ = calibration[0].index.to_numpy(copy=True)
        candidates = self._candidate_configurations()
        if not isinstance(self.reference_cv, Integral) or self.reference_cv < 2:
            raise ValueError("reference_cv must be an integer of at least 2.")
        strata = self._strata(y, groups)
        if strata.value_counts().min() < self.reference_cv:
            raise ValueError("Each class/group stratum needs at least reference_cv observations.")
        splitter = StratifiedKFold(self.reference_cv, shuffle=True, random_state=self.random_state)
        # Materialize once: with random_state=None, separate split() calls would
        # otherwise compare candidates on different observations.
        folds = list(splitter.split(X, strata))
        rows, audit = [], []
        for candidate_number, candidate in enumerate(candidates):
            for fold, (fit_index, score_index) in enumerate(folds):
                X_fit, y_fit = X.iloc[fit_index], y.iloc[fit_index]
                groups_fit = None if groups is None else groups.iloc[fit_index]
                fold_calibration = None
                if candidate.get("postprocessor", self.postprocessor) is not None:
                    X_fit, y_fit, groups_fit, X_inner_cal, y_inner_cal, groups_inner_cal = self._split(
                        X_fit, y_fit, groups_fit, 0.25, (self.random_state or 0) + fold,
                    )
                    fold_calibration = (X_inner_cal, y_inner_cal, groups_inner_cal)
                _, _, predictor = self._fit_predictors(X_fit, y_fit, groups_fit, fold_calibration, candidate)
                ranking = frozen_ranking(predictor, X_fit, y_fit, groups=groups_fit,
                                         random_state=self.random_state or 0)
                X_score, y_score = X.iloc[score_index], y.iloc[score_index]
                groups_score = None if groups is None else groups.iloc[score_index]
                values = evaluate_metrics(predictor, X_score, y_score, groups_score, ranking,
                                          ec_config=self.ec_config_, random_state=self.random_state or 0)
                rows.append({name: values[name] for name in self.metric_weights_})
                audit.append({"candidate": candidate_number, "fold": fold,
                              "fit_index": X_fit.index.tolist(),
                              "calibration_index": [] if fold_calibration is None else fold_calibration[0].index.tolist(),
                              "score_index": X_score.index.tolist(), "ranking": ranking})
        self.reference_metrics_ = pd.DataFrame(rows)
        self.reference_audit_ = audit
        configured_reference = self.reference_metrics
        if configured_reference is not None and self.metrics_values is not None:
            raise ValueError("reference_metrics and metrics_values are aliases; provide only one.")
        if configured_reference is None:
            configured_reference = self.metrics_values
        reference = self.reference_metrics_ if configured_reference is None else configured_reference
        self.trust_estimator_ = self._make_trust_estimator(self.aggregation, reference)
        self.aggregation_reference_ = reference.copy(deep=True)
        selection_rows = []
        if len(candidates) > 1 and self.trust_estimator_ is None:
            raise ValueError("Candidate selection requires a mechanism; select externally when aggregation=None.")
        selected = 0
        for number in range(len(candidates)):
            candidate_rows = self.reference_metrics_.iloc[number * self.reference_cv:(number + 1) * self.reference_cv]
            if self.trust_estimator_ is None:
                scores = []
                utility = None
            else:
                results = [self._score_frame(self.trust_estimator_, candidate_rows.iloc[[i]])
                           for i in range(len(candidate_rows))]
                scores = [result.score for result in results]
                higher = getattr(results[0], "higher_is_better", True)
                utility = float(np.mean(scores)) * (1 if higher else -1)
            selection_rows.append({"candidate": number, "fold_scores": scores, "utility": utility})
        if len(candidates) > 1:
            selected = max(range(len(candidates)), key=lambda i: selection_rows[i]["utility"])
        self.selected_candidate_ = selected
        self.selection_report_ = pd.DataFrame(selection_rows)
        teacher, base, final = self._fit_predictors(X, y, groups, calibration, candidates[selected])
        self.teacher_predictor_, self.surrogate_predictor_, self.predictor_ = teacher, base, final
        self.preprocessor_ = final.preprocessor
        self.teacher_, self.surrogate_ = teacher.model, final.model
        self.postprocessor_ = final.postprocessor
        self.classifier_ = self.surrogate_
        self.given_algorithm_ = self.teacher_
        self.final_model_source_ = "surrogate"
        self.fairness_processor_post = self.postprocessor_
        # Prepare the final ranking only after the complete prediction chain exists.
        self.ranking_ = frozen_ranking(final, X, y, groups=groups, random_state=self.random_state or 0)
        self.feature_ranking_ = list(self.ranking_)
        self.teacher_ranking_ = frozen_ranking(teacher, X, y, groups=groups, random_state=self.random_state or 0)
        self.fit_metadata_ = {
            "calibration_source": calibration_source,
            "reference_source": "training_internal_cv" if configured_reference is None else "explicit_reference",
            "reference_cv": self.reference_cv, "candidate_count": len(candidates),
            "candidate_fit_budget": len(candidates) * self.reference_cv,
            "final_fit_count": 1, "selection_tie_policy": "first_candidate",
            "final_predictor": "surrogate", "test_used": False,
            "legacy_dir_applied": False, "effective_complexity": deepcopy(self.ec_config_),
            "postprocessing": deepcopy(getattr(self.postprocessor_, "diagnostics_", None)),
            "metric_weights": deepcopy(self.metric_weights_),
        }
        self.is_fitted_ = True
        return self

    def set_aggregation(self, mechanism, *, reference_metrics=None):
        """Prepare a mechanism after predictor fitting, e.g. with a shared reference."""
        check_is_fitted(self, "predictor_")
        reference = self.reference_metrics_ if reference_metrics is None else reference_metrics
        estimator = self._make_trust_estimator(mechanism, reference)
        self.aggregation = mechanism
        self.trust_estimator_ = estimator
        self.aggregation_reference_ = None if reference is None else reference.copy(deep=True)
        return self

    def predict(self, X, *, sensitive_features=None):
        check_is_fitted(self, "predictor_")
        return self.predictor_.predict(X, sensitive_features=self._groups(X, sensitive_features))

    def predict_proba(self, X, *, sensitive_features=None):
        check_is_fitted(self, "predictor_")
        return self.predictor_.predict_proba(X, sensitive_features=self._groups(X, sensitive_features))

    def score(self, X, y, sample_weight=None, *, sensitive_features=None):
        return accuracy_score(y, self.predict(X, sensitive_features=sensitive_features), sample_weight=sample_weight)

    @staticmethod
    def _metric_list(values, weights):
        result = MetricList()
        for name, value in values.items():
            result.add_metric(name, weights.get(name, 0.0), goal=0.0, value=value)
        return result

    def evaluate(self, X, y, *, sensitive_features=None):
        """Evaluate the frozen final chain; never fit or select on these data."""
        check_is_fitted(self, "predictor_")
        X, y = self._xy(X, y)
        groups = self._groups(X, sensitive_features)
        values = evaluate_metrics(self.predictor_, X, y, groups, self.ranking_,
                                  ec_config=self.ec_config_, random_state=self.random_state or 0)
        requested = {name: values[name] for name in self.metric_weights_}
        metrics_frame = pd.DataFrame([requested])
        result = self._score_frame(self.trust_estimator_, metrics_frame)
        teacher_predictions = self.teacher_predictor_.predict(X)
        base_predictions = self.surrogate_predictor_.predict(X)
        final_predictions = self.predictor_.predict(X, sensitive_features=groups)
        fidelity = {
            "teacher_student_class_agreement": float(np.mean(teacher_predictions == base_predictions)),
            "teacher_final_class_agreement": float(np.mean(teacher_predictions == final_predictions)),
            "teacher_accuracy": float(np.mean(teacher_predictions == y.to_numpy())),
            "student_before_postprocessing_accuracy": float(np.mean(base_predictions == y.to_numpy())),
            "final_accuracy": float(np.mean(final_predictions == y.to_numpy())),
        }
        fidelity["minimum_class_agreement"] = self.surrogate_min_agreement
        fidelity["objective_met"] = bool(fidelity["teacher_student_class_agreement"] >= self.surrogate_min_agreement)
        fairness_value = values[METRICS.FAIRNESS.EQUAL_OPPORTUNITY]
        fairness_objective = {
            "threshold": self.fairness_threshold,
            "observed": fairness_value,
            "objective_met": None if not np.isfinite(fairness_value) else bool(fairness_value >= self.fairness_threshold),
        }
        _, ec_details = compute_effective_complexity(
            self.predictor_, X, self.ranking_, ec_config=self.ec_config_,
            random_state=self.random_state or 0, groups=groups,
        )
        report = EvaluationReport(
            metrics=values, metrics_frame=metrics_frame, trustworthiness=result,
            teacher_student_fidelity=fidelity,
            predictions=pd.Series(final_predictions, index=X.index, name="prediction"),
            metadata={**deepcopy(self.fit_metadata_), "evaluation_index": X.index.tolist(),
                      "effective_complexity_details": ec_details,
                      "teacher_student_fidelity_loss": "class_disagreement",
                      "mechanism": deepcopy(getattr(self.trust_estimator_, "mechanism_metadata_", None)),
                      "fairness_objective": fairness_objective,
                      "fidelity_objective_met": fidelity["objective_met"]},
        )
        self.last_report_ = report
        self.metrics_post = self._metric_list(requested, self.metric_weights_)
        # Teacher diagnosis includes only metrics comparable to the final chain.
        teacher_values = {METRICS.PERFORMANCE.ACCURACY: fidelity["teacher_accuracy"]}
        if groups is not None:
            tprs = [np.mean(teacher_predictions[(groups.to_numpy() == group) & (y.to_numpy() == 1)] == 1)
                    for group in pd.unique(groups)]
            teacher_values[METRICS.FAIRNESS.EQUAL_OPPORTUNITY] = 1.0 - abs(tprs[0] - tprs[1])
        self.metrics_pre = self._metric_list(teacher_values, self.metric_weights_)
        self.trustworthiness_score_ = None if result is None else result.score
        self.component_scores_ = self._component_scores()
        self.final_report_ = report
        return report

    def unbiased_dataset(self):
        raise RuntimeError("The inductive predictor does not relabel true outcomes. Use predict/evaluate explicitly.")

    def trustworthiness_diagnosis(self):
        if self.metrics_pre is None:
            raise RuntimeError("Call evaluate on held-out data before requesting a diagnosis.")
        return self.metrics_pre.to_dataframe()

    def trustworthiness_metrics(self):
        if self.metrics_post is None:
            raise RuntimeError("Call evaluate on held-out data before requesting a metrics report.")
        frame = self.metrics_post.to_dataframe()
        frame["trustworthiness"] = self.trustworthiness_score_
        return frame

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
            equalized_odds = getattr(self.fairness_processor_post, "diagnostics_", None)
        return pd.DataFrame(
            [{
                "disparate_impact": None,
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

        final_by_name = {metric.name: metric for metric in self.metrics_post.metrics}
        for metric_pre in self.metrics_pre.metrics:
            if metric_pre.name in get_metrics_of(METRICS.PERFORMANCE) and metric_pre.name in final_by_name:
                metric_post = final_by_name[metric_pre.name]
                value = 0
                if metric_pre.value != 0:
                    value = (metric_pre.value - metric_post.value)/metric_pre.value
                performance_metrics.append({
                    "metric": metric_pre.name,
                    "value": value
                })
        return pd.DataFrame(performance_metrics)
