"""Trustworthiness estimation with shared scalar and batch aggregation."""
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

from TWEEF.Constants import FUZZY_THRESHOLD_PEAKS, LINGUISTIC_VALUES, TRUSTWORTHINESS_SCORE_NAME
from TWEEF.Metrics import MetricList
from TWEEF.Aggregation import (
    AGGREGATION,
    TRUSTWORTHINESS_SCORE_SCALE,
    TrustAggregator,
    normalize_score_scale,
)
from TWEEF.Calibration import FuzzyCalibrator, membership_vector, triangular_membership
from TWEEF.Logging import get_logger
from TWEEF.Validation import validate_metric_list


logger = get_logger(__name__)


class TrustworthinessEstimator(BaseEstimator, TransformerMixin):
    """Aggregate normalized metrics using one calibration and selected operator.

    ``compute_metric`` accepts a populated MetricList without requiring fit.
    ``fit`` configures metric weights and frozen calibration for batch ``transform``.
    The metric columns returned by transform are continuous linguistic diagnostics;
    only the Trustworthiness column is the selected operator's aggregated score.
    """

    def __init__(
        self,
        verbose: bool = False,
        aggregation: str = AGGREGATION.LWA,
        fuzzy_thresholds: pd.DataFrame = None,
        metrics_values: pd.DataFrame = None,
        alpha: float = 0.5,
        trustworthiness_score_scale: str = TRUSTWORTHINESS_SCORE_SCALE.NORMALIZED,
        gwa_hesitation=0.0,
    ):
        self.verbose = verbose
        self.aggregation = aggregation
        self.fuzzy_thresholds = fuzzy_thresholds
        self.metrics_values = metrics_values
        self.alpha = alpha
        self.trustworthiness_score_scale = normalize_score_scale(trustworthiness_score_scale)
        self.gwa_hesitation = gwa_hesitation
        self.weights = None
        self.thresholds_ = None
        self.trustworthiness_score = None
        self.aggregation_result_ = None
        if fuzzy_thresholds is not None and metrics_values is not None:
            raise ValueError("thresholds and metrics_values are mutually exclusive.")
        # Metric weights are not known until fit/compute_metric. Snapshot now,
        # but validate/calibrate only metrics with positive weight at first use.
        self._configured_thresholds = deepcopy(fuzzy_thresholds)
        self._configured_reference = deepcopy(metrics_values)
        self.calibrator_ = None

    def set_params(self, **params):
        """Refresh calibration snapshots when sklearn changes their parameters.

        A different calibration source requires a new fit before transform;
        changing the aggregation operator or its parameters retains fitted crests.
        """
        calibration_changed = bool({"fuzzy_thresholds", "metrics_values"}.intersection(params))
        if calibration_changed:
            thresholds = params.get("fuzzy_thresholds", self.fuzzy_thresholds)
            reference = params.get("metrics_values", self.metrics_values)
            if thresholds is not None and reference is not None:
                raise ValueError("thresholds and metrics_values are mutually exclusive.")
        super().set_params(**params)
        if calibration_changed:
            self._configured_thresholds = deepcopy(self.fuzzy_thresholds)
            self._configured_reference = deepcopy(self.metrics_values)
            for name in (
                "weights_", "metric_goals_", "feature_names_in_",
                "n_features_in_", "aggregation_results_",
            ):
                self.__dict__.pop(name, None)
            self.weights = None
            self.thresholds_ = None
            self.calibrator_ = None
            self.aggregation_result_ = None
            self.trustworthiness_score = None
        return self

    def fit(self, fuzzy_functions_thresholds=None, metrics_values=None, metric_list=None):
        """Freeze calibration and weights for subsequent batch estimation.

        The positional threshold argument and all existing keyword arguments are
        preserved. Fit-time calibration overrides constructor calibration. Weights
        are configured independently through metric_list, or uniformly by default.
        Explicit crests and reference metric values are mutually exclusive.
        """
        if fuzzy_functions_thresholds is not None or metrics_values is not None:
            thresholds, reference = fuzzy_functions_thresholds, metrics_values
        else:
            thresholds, reference = self._configured_thresholds, self._configured_reference
        if thresholds is None and reference is None and metric_list is None:
            raise ValueError(
                "You must provide fuzzy_functions_thresholds, metrics_values, or metric_list."
            )
        if metric_list is not None:
            validate_metric_list(metric_list)
            names = metric_list.get_names()
            weights = np.asarray(metric_list.get_weights(), dtype=float)
            goals = dict(zip(names, metric_list.get_goals()))
        else:
            names = self._calibration_metric_names(thresholds, reference)
            weights = np.full(len(names), 1.0 / len(names))
            goals = {name: 0.0 for name in names}
        if not np.isfinite(weights).all():
            raise ValueError("metric weights must be finite.")
        # Validate names and default weights even when fit was given only a frame.
        configured_metrics = MetricList()
        for name, weight in zip(names, weights):
            configured_metrics.add_metric(name, float(weight), goals[name])
        validate_metric_list(configured_metrics)
        active_names = [name for name, weight in zip(names, weights) if weight > 0]
        calibrator = self._calibration_for_metrics(thresholds, reference, names, active_names)
        frozen_thresholds = {name: calibrator.thresholds_for(name) for name in active_names}
        self.calibrator_ = FuzzyCalibrator(thresholds=frozen_thresholds)
        self.thresholds_ = frozen_thresholds
        self.weights_ = dict(zip(names, weights))
        self.weights = self.weights_
        self.metric_goals_ = goals
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.aggregation_results_ = []
        self.aggregation_result_ = None
        self.trustworthiness_score = None
        return self

    @staticmethod
    def _calibration_metric_names(thresholds, reference):
        """Read calibration schema without evaluating inactive metric values."""
        if thresholds is not None and reference is not None:
            raise ValueError("thresholds and metrics_values are mutually exclusive.")
        if reference is not None:
            if not isinstance(reference, pd.DataFrame):
                raise TypeError("metrics_values must be a pandas.DataFrame.")
            if reference.empty or reference.columns.has_duplicates:
                raise ValueError("metrics_values must have rows and unique metric columns.")
            return list(reference.columns)
        if isinstance(thresholds, pd.DataFrame):
            if thresholds.empty or thresholds.index.has_duplicates or thresholds.columns.has_duplicates:
                raise ValueError("thresholds must have rows and unique metric and crest labels.")
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.columns):
                return list(thresholds.index)
            if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.index):
                return list(thresholds.columns)
            raise ValueError("thresholds DataFrame must have fuzzy peaks as columns or index.")
        if thresholds is not None:
            if not hasattr(thresholds, "items") or not len(thresholds):
                raise ValueError("thresholds must contain at least one metric.")
            return list(thresholds)
        return []

    @classmethod
    def _calibration_for_metrics(cls, thresholds, reference, names, active_names):
        """Exclude zero-weight metrics before validating their calibration.

        Missing inactive metrics are allowed. Unrecognized extra metric names
        are rejected so misspelled calibration columns cannot go unnoticed.
        """
        if thresholds is None and reference is None:
            return FuzzyCalibrator()
        supplied_names = cls._calibration_metric_names(thresholds, reference)
        unknown_names = set(supplied_names).difference(names)
        if unknown_names:
            raise ValueError(f"unknown calibration metrics: {sorted(unknown_names)!r}")
        missing_names = set(active_names).difference(supplied_names)
        if missing_names:
            raise ValueError(f"missing calibration for active metrics: {sorted(missing_names)!r}")
        if reference is not None:
            return FuzzyCalibrator(metrics_values=reference.loc[:, active_names])
        if isinstance(thresholds, pd.DataFrame):
            selected = (
                thresholds.loc[active_names]
                if set(FUZZY_THRESHOLD_PEAKS).issubset(thresholds.columns)
                else thresholds.loc[:, active_names]
            )
        else:
            selected = {name: thresholds[name] for name in active_names}
        return FuzzyCalibrator(thresholds=selected)

    def _compute_metric_thresholds(self, metrics_values, column):
        """Compatibility helper delegated to the shared quartile calibrator."""
        return FuzzyCalibrator(metrics_values=metrics_values[[column]]).thresholds_for(column).to_dict()

    def _triangular_membership(self, x, delta_low, delta_peak, delta_high):
        return triangular_membership(x, delta_low, delta_peak, delta_high)

    def _assign_linguistic_label(self, value, thresholds):
        vector = membership_vector(value, thresholds)
        return LINGUISTIC_VALUES[int(np.argmax(vector))], float(np.dot(vector, LINGUISTIC_VALUES))

    def _aggregate_metrics(self, metrics):
        validate_metric_list(metrics, require_values=True)
        if not np.isfinite(np.asarray(metrics.get_values(), dtype=float)).all():
            raise ValueError("metric values must be finite.")
        if not np.isfinite(np.asarray(metrics.get_weights(), dtype=float)).all():
            raise ValueError("metric weights must be finite.")
        if not hasattr(self, "weights_"):
            self.calibrator_ = self._calibration_for_metrics(
                self._configured_thresholds,
                self._configured_reference,
                metrics.get_names(),
                [metric.name for metric in metrics.get_metrics() if metric.weight > 0],
            )
        return TrustAggregator(
            operator=self.aggregation,
            calibrator=self.calibrator_,
            alpha=self.alpha,
            score_scale=self.trustworthiness_score_scale,
            gwa_hesitation=self.gwa_hesitation,
        ).aggregate(metrics)

    def transform(self, X):
        """Aggregate each row, preserving its index and linguistic diagnostics.

        Per-row AggregationResult objects are stored in aggregation_results_ in
        the same order as X, including when its index contains duplicate labels.
        A zero-weight metric has no calibration requirement and its linguistic
        diagnostic is NaN; its original value must still be a valid metric value.
        """
        check_is_fitted(self, ["weights_", "feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame of metric values.")
        if X.columns.has_duplicates or set(X.columns) != set(self.feature_names_in_):
            raise ValueError("X must contain exactly the fitted metric columns, without duplicates.")
        try:
            values = X.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("metric values must be numeric.") from exc
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("metric values must be finite.")
        if ((values < 0) | (values > 1)).to_numpy().any():
            raise ValueError("metric values must be in [0, 1].")
        diagnostics = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        scores = []
        results = []
        for position, (_, row) in enumerate(values.iterrows()):
            metrics = MetricList()
            for name in self.feature_names_in_:
                metrics.add_metric(
                    name, float(self.weights_[name]), self.metric_goals_[name], float(row[name]),
                )
            result = self._aggregate_metrics(metrics)
            results.append(result)
            scores.append(result.score)
            for name in X.columns:
                if self.weights_[name] == 0:
                    continue
                _, score = self._assign_linguistic_label(row[name], self.thresholds_[name])
                diagnostics.iloc[position, X.columns.get_loc(name)] = score
        diagnostics[TRUSTWORTHINESS_SCORE_NAME] = scores
        self.aggregation_results_ = results
        self.aggregation_result_ = results[-1] if results else None
        self.trustworthiness_score = results[-1].score if results else None
        return diagnostics

    def compute_metric(self, metrics: MetricList):
        """Aggregate a populated MetricList with the configured calibration."""
        self.aggregation_result_ = self._aggregate_metrics(metrics)
        self.trustworthiness_score = self.aggregation_result_.score
        if self.verbose:
            logger.debug("Final trustworthiness score=%s", self.trustworthiness_score)
        return self.trustworthiness_score

    def plot_results(self, s_values, title=None, save_path=None):
        """
        Plot radar chart of S values for each model and optionally save it as an image.

        Parameters:
        - s_values (pd.DataFrame): DataFrame with S values for each metric and model.
        - title (str, optional): Title displayed at the top of the chart.
        - save_path (str, optional): File path to save the image. Supported formats include .png, .jpg, .pdf, .svg.
        """
        import matplotlib.pyplot as plt

        # Define categories (exclude the 'Trustworthiness_Score' column)
        categories = list(s_values.columns[:-1])
        num_vars = len(categories)
        markers = ["-s","-D","-o","-^","-v","-X","-p","-*"]

        # Set up the radar chart
        start_angle = 60
        angles = np.linspace(np.radians(start_angle), np.radians(start_angle + 360), num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

        # Plot each model's S values
        for idx, (index, row) in enumerate(s_values.iterrows()):
            values = row[categories].values.tolist()
            values += values[:1]  # Repeat the first value to close the circle
            marker_style = markers[idx % len(markers)]  # Cycle through markers
            ax.plot(angles, values, marker_style, linewidth=1, markersize=6, label=index)
            ax.fill(angles, values, alpha=0.20)

        # Configure the radar chart
        ax.set_yticks(LINGUISTIC_VALUES)
        ax.set_yticklabels(LINGUISTIC_VALUES)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        if title:
            plt.title(title, size=16, color='blue', y=1.1)
        plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.33))

        # Save the plot if save_path is provided
        file_format = "jpg"
        if save_path:
            plt.savefig(f"{save_path}.{file_format}", format=file_format, dpi=600, bbox_inches="tight")
            if self.verbose:
                logger.info("Chart saved to %s.%s", save_path, file_format)

        # Show plot
        plt.show()

    def plot_trustworthiness_scores(self, s_values, title=None, save_path=None):
        """
        Plot a horizontal bar chart of the Trustworthiness Scores.

        Parameters:
        - s_values (pd.DataFrame): DataFrame containing the 'Trustworthiness_Score' column for each model.
        - save_path (str, optional): Path to save the chart image. Supported formats include .png, .jpg, .pdf, etc.
        """
        import matplotlib.pyplot as plt

        # Extract the trustworthiness scores and model identifiers
        scores = s_values[TRUSTWORTHINESS_SCORE_NAME]
        models = s_values.index

        # Create the bar chart
        plt.figure(figsize=(4, 4))
        plt.barh(models, scores, color=['#636efb' if s >= 0 else '#ef553b' for s in scores])

        # Draw a vertical line at zero for reference
        plt.axvline(0, color='gray', linewidth=0.8)

        # Set x-axis limits
        plt.xlim(-2.2, 2.2)

        # Labels and title
        if title:
            plt.title(title)
        plt.xlabel("Trustworthiness Score")
        plt.ylabel("Model")

        # Save plot if save_path is specified
        file_format = "jpg"
        if save_path:
            plt.savefig(f"{save_path}.{file_format}", format=file_format, dpi=600, bbox_inches="tight")
            if self.verbose:
                logger.info("Chart saved to %s.%s", save_path, file_format)

        # Show plot
        plt.show()

    def plot_combined_charts(self, s_values, title=None, save_path=None):
        """
        Plot a radar chart of S values and a horizontal bar chart of Trustworthiness Scores
        side by side, and optionally save as an image.

        Parameters:
        - s_values (pd.DataFrame): DataFrame containing Metrics values and 'Trustworthiness_Score' for each model.
        - title (str, optional): Title displayed at the top of the charts.
        - save_path (str, optional): File path to save the image. Supported formats include .png, .jpg, .pdf, .svg.
        """
        import matplotlib.pyplot as plt

        categories = list(s_values.columns[:-1])  # Exclude 'Trustworthiness_Score'
        num_vars = len(categories)
        models = s_values.index
        scores = s_values[TRUSTWORTHINESS_SCORE_NAME]
        markers = ["-s","-D","-o","-^","-v","-X","-p","-*"]

        # Set up the figure with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1]})


        # ----- Radar Chart (Polar Axes) -----
        # Convert ax1 to a polar subplot
        ax1 = plt.subplot(121, polar=True)
        start_angle = 60
        angles = np.linspace(np.radians(start_angle), np.radians(start_angle + 360), num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        for idx, (index, row) in enumerate(s_values.iterrows()):
            values = row[categories].values.tolist()
            values += values[:1]  # Repeat the first value to close the circle
            marker_style = markers[idx % len(markers)]  # Cycle through markers
            ax1.plot(angles, values, marker_style, linewidth=1, markersize=6, label=index)
            ax1.fill(angles, values, alpha=0.20)


        # Configure the radar chart
        ax1.set_yticks([-2,-1,-0,1,2])
        ax1.set_yticklabels(['-2','-1','-0','1','2'])
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        #ax1.set_title("Radar Chart of S Values", size=16, color='blue')
        ax1.legend(loc='center left', bbox_to_anchor=(0.8, 0.33))

        # ----- Horizontal Bar Chart -----
        # Plot on ax2
        ax2.barh(models, scores, color=['#636efb' if s >= 0 else '#ef553b' for s in scores])
        ax2.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at 0 for reference
        ax2.set_xlim(-2.2, 2.2)  # Set x-axis limits

        # Labels and title for bar chart
        ax2.set_xlabel(TRUSTWORTHINESS_SCORE_NAME)
        ax2.set_ylabel("Model")

        # Set main title if provided
        if title:
            fig.suptitle(title, size=18, y=1.05)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if save_path is provided
        file_format = "jpg"
        if save_path:
            plt.savefig(f"{save_path}.{file_format}", format=file_format, dpi=600, bbox_inches="tight")
            if self.verbose:
                logger.info("Combined chart saved to %s.%s", save_path, file_format)

        # Show the plot
        plt.show()
