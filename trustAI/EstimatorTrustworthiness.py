""" EstimatorTrustworthiness module containing the TrustworthinessEstimator class. """
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trustAI.Constants import LINGUISTIC_VALUES, LINGUISTIC_LABELS, TRUSTWORTHINESS_SCORE_NAME
from trustAI.Metrics import MetricList

class TrustworthinessEstimator(BaseEstimator, TransformerMixin):
    """
    TrustworthinessEstimator is a scikit-learn compatible transformer that evaluates the trustworthiness of models based on various metrics. 
    It uses fuzzy logic to assign linguistic labels to metric values and calculates a trustworthiness score for each model.
    Attributes:
        weights (dict): A dictionary of weights for each metric. Default is an empty dictionary.
        verbose (bool): If True, prints detailed information during operations. Default is False.
        thresholds_ (dict): A dictionary storing the fuzzy triangle peaks (thresholds) for each metric.
    Methods:
        __init__(self, weights=None, verbose=False):
            Initializes the TrustworthinessEstimator with optional weights and verbosity.
        fit(self, X, y=None):
            Fits the model by calculating fuzzy triangle peaks (thresholds) for each metric.
                X (pd.DataFrame): DataFrame containing metrics for each model.
                y: Not used, included to match scikit-learn conventions.
                self: Fitted estimator.
        _fuzzy_triangle_peaks(self, df, column):
            Defines fuzzy triangle peaks based on quantiles for each metric.
                df (pd.DataFrame): DataFrame containing the metric.
                column (str): Column name of the metric.
                thresholds (dict): Dictionary of threshold values.
        _triangular_membership(self, x, delta_low, delta_peak, delta_high):
            Calculates triangular membership for a given value.
                x (float): Value to calculate membership for.
                delta_low (float): Start of the triangle.
                delta_peak (float): Peak of the triangle.
                delta_high (float): End of the triangle.
                result (float): Membership value.
        _assign_linguistic_label(self, value, thresholds):
            Assigns a linguistic label based on the calculated membership values.
                value (float): The metric value to evaluate.
                thresholds (dict): Dictionary of peak thresholds.
                linguistic_value (int): Assigned linguistic value.
                S (float): Weighted sum of linguistic values.
        transform(self, X):
            Transforms the data to calculate trustworthiness scores.
                X (pd.DataFrame): DataFrame containing metrics for each model.
                result (pd.DataFrame): Original DataFrame with an additional column for trustworthiness score.
        plot_results(self, s_values, title=None, save_path=None):
            Plots a radar chart of S values for each model and optionally saves it as an image.
                s_values (pd.DataFrame): DataFrame with S values for each metric and model.
                title (str, optional): Title displayed at the top of the chart.
                save_path (str, optional): File path to save the image. Supported formats include .png, .jpg, .pdf, .svg.
        plot_trustworthiness_scores(self, s_values, title=None, save_path=None):
            Plots a horizontal bar chart of the Trustworthiness Scores.
                s_values (pd.DataFrame): DataFrame containing the 'Trustworthiness_Score' column for each model.
                title (str, optional): Title displayed at the top of the chart.
                save_path (str, optional): Path to save the chart image. Supported formats include .png, .jpg, .pdf, etc.
        plot_combined_charts(self, s_values, title=None, save_path=None):
            Plots a radar chart of S values and a horizontal bar chart of Trustworthiness Scores side by side, and optionally saves as an image.
                s_values (pd.DataFrame): DataFrame containing S values and 'Trustworthiness_Score' for each model.
                title (str, optional): Title displayed at the top of the charts.
                save_path (str, optional): File path to save the image. Supported formats include .png, .jpg, .pdf, .svg.
    """
    def __init__(self, weights=None, verbose=False):
        """
        Initialize the TrustworthinessEstimator with weights for each metric.

        Parameters:
        - weights (dict): A dictionary of weights for each metric, default is None.
        - verbose (bool): If True, print detailed information during operations.
        """
        self.weights = weights if weights is not None else {}
        self.verbose = verbose
        self.thresholds_ = {}

    def fit(self, X, y=None):
        """
        Fit the model by calculating fuzzy triangle peaks (thresholds) for each metric.

        Parameters:
        - X (pd.DataFrame): DataFrame containing metrics for each model.
        - y: Not used, included to match scikit-learn conventions.

        Returns:
        - self: Fitted estimator.
        """
        self.thresholds_ = {col: self._fuzzy_triangle_peaks(X, col) for col in X.columns}
        if self.verbose:
            print(f"\nFitting complete. Thresholds for each metric:\n{self.thresholds_}\n")
        return self

    def _fuzzy_triangle_peaks(self, df, column):
        """
        Define fuzzy triangle peaks based on quantiles for each metric.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the metric.
        - column (str): Column name of the metric.

        Returns:
        - thresholds (dict): Dictionary of threshold values.
        """
        thresholds = {
            "peak_-2": df[column].min(),
            "peak_-1": df[column].quantile(0.25),
            "peak_0" : df[column].quantile(0.50),
            "peak_1" : df[column].quantile(0.75),
            "peak_2" : df[column].max()
        }
        return thresholds

    def _triangular_membership(self, x, delta_low, delta_peak, delta_high):
        """
        Calculate triangular membership for a given value.

        Parameters:
        - x (float): Value to calculate membership for.
        - delta_low (float): Start of the triangle.
        - delta_peak (float): Peak of the triangle.
        - delta_high (float): End of the triangle.

        Returns:
        - result (float): Membership value.
        """
        if x <= delta_low or x >= delta_high:
            result = 0
        elif x <= delta_peak:
            result = (x - delta_low) / (delta_peak - delta_low)
        elif x <= delta_high:
            result = (delta_high - x) / (delta_high - delta_peak)
        else:
            result = 0

        if self.verbose:
            print(f"Calculating membership for x = {x} with range ({delta_low}, {delta_peak}, {delta_high}) => result = {result}")
        return result

    def _assign_linguistic_label(self, value, thresholds):
        """
        Assign a linguistic label based on the calculated membership values.

        Parameters:
        - value (float): The metric value to evaluate.
        - thresholds (dict): Dictionary of peak thresholds.

        Returns:
        - linguistic_value (int): Assigned linguistic value.
        - S (float): Weighted sum of linguistic values.
        """
        membership_values = [
            self._triangular_membership(value, thresholds["peak_-2"] - (thresholds["peak_-1"] - thresholds["peak_-2"]), thresholds["peak_-2"], thresholds["peak_-1"]),
            self._triangular_membership(value, thresholds["peak_-2"], thresholds["peak_-1"], thresholds["peak_0"]),
            self._triangular_membership(value, thresholds["peak_-1"], thresholds["peak_0"], thresholds["peak_1"]),
            self._triangular_membership(value, thresholds["peak_0"], thresholds["peak_1"], thresholds["peak_2"]),
            self._triangular_membership(value, thresholds["peak_1"], thresholds["peak_2"], thresholds["peak_2"] + (thresholds["peak_2"] - thresholds["peak_1"]))
        ]
        S = sum([a * b for a, b in zip(LINGUISTIC_VALUES, membership_values)])
        max_index = np.argmax(membership_values)
        linguistic_value = LINGUISTIC_VALUES[max_index]

        if self.verbose:
            print(f"\nComputing linguistic variable S for value {value}")
            print(f"Membership values: {membership_values}")
            print(f"Linguistic values: {LINGUISTIC_VALUES}")
            print(f"Weighted sum S = {S}, chosen label = {LINGUISTIC_LABELS[max_index]} ({linguistic_value})\n")
        return linguistic_value, S

    def transform(self, X):
        """
        Transform the data to calculate trustworthiness scores.

        Parameters:
        - X (pd.DataFrame): DataFrame containing metrics for each model.

        Returns:
        - result (pd.DataFrame): Original DataFrame with an additional column for trustworthiness score.
        """
        trustworthiness_scores = []
        s_values = pd.DataFrame(index=X.index, columns=X.columns)

        for index, row in X.iterrows():
            final_score = 0
            if self.verbose:
                print(f"\nCalculating trustworthiness for model at index {index}")

            for col in X.columns:
                thresholds = self.thresholds_[col]
                _, S = self._assign_linguistic_label(row[col], thresholds)
                s_values.at[index, col] = S
                weighted_value = self.weights[col] * S if col in self.weights else S
                final_score += weighted_value

                if self.verbose:
                    print(f"Metric: {col}, Value: {row[col]}")
                    print(f"Thresholds: {thresholds}")
                    print(f"S = {S}, Weight = {self.weights[col]}")
                    print(f"Weighted contribution to final score = {weighted_value}\n")

            trustworthiness_scores.append(final_score)
            if self.verbose:
                print(f"      Final trustworthiness score for metrics at index {index}: {final_score}\n")

        s_values[TRUSTWORTHINESS_SCORE_NAME] = trustworthiness_scores
        return s_values    

    def plot_results(self, s_values, title=None, save_path=None):
        """
        Plot radar chart of S values for each model and optionally save it as an image.

        Parameters:
        - s_values (pd.DataFrame): DataFrame with S values for each metric and model.
        - title (str, optional): Title displayed at the top of the chart.
        - save_path (str, optional): File path to save the image. Supported formats include .png, .jpg, .pdf, .svg.
        """
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
                print(f"Combined chart saved to {save_path}.{file_format}")

        # Show plot
        plt.show()

    def plot_trustworthiness_scores(self, s_values, title=None, save_path=None):
        """
        Plot a horizontal bar chart of the Trustworthiness Scores.

        Parameters:
        - s_values (pd.DataFrame): DataFrame containing the 'Trustworthiness_Score' column for each model.
        - save_path (str, optional): Path to save the chart image. Supported formats include .png, .jpg, .pdf, etc.
        """
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
            plt.savefig(f"{save_path}.{format}", format=format, dpi=600, bbox_inches="tight")
            if self.verbose:
                print(f"Combined chart saved to {save_path}.{file_format}")

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
                print(f"Combined chart saved to {save_path}.{file_format}")

        # Show the plot
        plt.show()
