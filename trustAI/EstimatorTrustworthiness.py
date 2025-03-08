""" EstimatorTrustworthiness module containing the TrustworthinessEstimator class. """
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trustAI.Constants import LINGUISTIC_VALUES, LINGUISTIC_LABELS, TRUSTWORTHINESS_SCORE_NAME, FUZZY_THRESHOLD_PEAKS, get_metric_thresholds
from trustAI.Metrics import MetricList

class TrustworthinessEstimator(BaseEstimator, TransformerMixin):
    """
    TrustworthinessEstimator is a scikit-learn compatible transformer that evaluates the trustworthiness of models based on various metrics. 
    It uses fuzzy logic to assign linguistic labels to metric values and calculates a trustworthiness score for each model.
    """
    
    def __init__(self, verbose: bool=False):
        """
        Initialize the TrustworthinessEstimator with weights for each metric.

        Parameters:
        - verbose (bool): If True, print detailed information during operations.
        """
        self.weights = None 
        self.verbose = verbose
        self.thresholds_ = None
        self.trustworthiness_score = None

    def fit(self, fuzzy_functions_thresholds: pd.DataFrame = None, metrics_values: pd.DataFrame = None, metric_list: MetricList = None):
        """
        Fit the model by setting fuzzy triangle peaks (thresholds) for each metric.
        
        
        Parameters:
        - fuzzy_functions_thresholds (pd.DataFrame): 
            DataFrame containing fuzzy triangle peaks for each metric. The DataFrame should have peaks as columns and metrics as rows.
            Example of a fuzzy_functions_thresholds DataFrame:
            |         | Metric1 | Metric2 | Metric3 | Metric4 | Metric5 |
            |---------|---------|---------|---------|---------|---------|
            | peak_-2 | 0.1     | 0.2     | 0.2     | 0.2     | 0.2     |
            | peak_-1 | 0.3     | 0.4     | 0.4     | 0.4     | 0.4     |
            | peak_0  | 0.5     | 0.6     | 0.6     | 0.6     | 0.6     |
            | peak_1  | 0.7     | 0.8     | 0.8     | 0.8     | 0.8     |
            | peak_2  | 0.9     | 1.0     | 1.0     | 1.0     | 1.0     |
            
        - metrics_values (pd.DataFrame): 
            DataFrame containing metrics values for a set of models. The DataFrame should have metrics as columns and models as rows.
            Example of a metrics_values DataFrame:
            |         | Metric1 | Metric2 | Metric3 | Metric4 | Metric5 |
            |---------|---------|---------|---------|---------|---------|
            | Model1  | 0.25    | 0.32    | 0.4     | 0.5     | 0.6     |
            | Model2  | 0.37    | 0.43    | 0.5     | 0.6     | 0.7     |
            | Model3  | 0.41    | 0.51    | 0.6     | 0.7     | 0.8     |
            | Model4  | 0.58    | 0.64    | 0.71    | 0.8     | 0.9     |
            | Model5  | 0.60    | 0.71    | 0.8     | 0.9     | 1.0     |
            
        Returns:
        - self (TrustworthinessEstimator): Fitted estimator.
        """
        if metrics_values is None and fuzzy_functions_thresholds is None and metric_list is None:
            raise ValueError("You must provide one of fuzzy_functions_thresholds or metrics_values or metric_list.")
        

        if fuzzy_functions_thresholds is not None:
            self.thresholds_ = fuzzy_functions_thresholds
        elif metric_list is not None:
            self.thresholds_ = { metric.name: get_metric_thresholds(metric.name) for metric in metric_list }
        else:
            self.thresholds_ = { col: self._compute_metric_thresholds(metrics_values, col) for col in metrics_values.columns }

        print("--------------------------------------------------------------------------------------------------------")
        print(f"\nFitting complete. Thresholds for each metric:\n{self.thresholds_}\n")
        print("--------------------------------------------------------------------------------------------------------")

        if self.verbose:
            print(f"\nFitting complete. Thresholds for each metric:\n{self.thresholds_}\n")
        return self

    def _compute_metric_thresholds(self, metrics_values: pd.DataFrame, column: str):
        """
        Define fuzzy triangle peaks based on quantiles for each metric.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the metric.
        - column (str): Column name of the metric.

        Returns:
        - thresholds (dict): Dictionary of threshold values.
        """
        thresholds = {
            FUZZY_THRESHOLD_PEAKS[0]: metrics_values[column].min(),
            FUZZY_THRESHOLD_PEAKS[1]: metrics_values[column].quantile(0.25),
            FUZZY_THRESHOLD_PEAKS[2] : metrics_values[column].quantile(0.50),
            FUZZY_THRESHOLD_PEAKS[3] : metrics_values[column].quantile(0.75),
            FUZZY_THRESHOLD_PEAKS[4] : metrics_values[column].max()
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
            self._triangular_membership(value, thresholds[FUZZY_THRESHOLD_PEAKS[0]] - (thresholds[FUZZY_THRESHOLD_PEAKS[1]] - thresholds[FUZZY_THRESHOLD_PEAKS[0]]), thresholds[FUZZY_THRESHOLD_PEAKS[0]], thresholds[FUZZY_THRESHOLD_PEAKS[1]]),
            self._triangular_membership(value, thresholds[FUZZY_THRESHOLD_PEAKS[0]], thresholds[FUZZY_THRESHOLD_PEAKS[1]], thresholds[FUZZY_THRESHOLD_PEAKS[2]]),
            self._triangular_membership(value, thresholds[FUZZY_THRESHOLD_PEAKS[1]], thresholds[FUZZY_THRESHOLD_PEAKS[2]], thresholds[FUZZY_THRESHOLD_PEAKS[3]]),
            self._triangular_membership(value, thresholds[FUZZY_THRESHOLD_PEAKS[2]], thresholds[FUZZY_THRESHOLD_PEAKS[3]], thresholds[FUZZY_THRESHOLD_PEAKS[4]]),
            self._triangular_membership(value, thresholds[FUZZY_THRESHOLD_PEAKS[3]], thresholds[FUZZY_THRESHOLD_PEAKS[4]], thresholds[FUZZY_THRESHOLD_PEAKS[4]] + (thresholds[FUZZY_THRESHOLD_PEAKS[4]] - thresholds[FUZZY_THRESHOLD_PEAKS[3]]))
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

    def compute_metric(self, metrics: MetricList):
        """
        Computes trustworthiness score.

        Parameters:
        - metrics (MetricList): List of metrics .

        Returns:
        - result float: Trustworthiness score.
        """
        self.trustworthiness_score = 0

        for metric in metrics.get_metrics():
            thresholds = pd.Series(get_metric_thresholds(metric.name), index=FUZZY_THRESHOLD_PEAKS)
            _, S = self._assign_linguistic_label(metric.value, thresholds)
            weighted_value = S * metric.weight
            self.trustworthiness_score += weighted_value

            #if self.verbose:
            print(f"Metric: {metric.name}, Value: {metric.value}")
            print(f"Thresholds:\n{thresholds}")
            print(f"Linguistic value = {S}, Weight = {metric.weight}")
            print(f"Weighted contribution to trustworthiness score = {weighted_value}\n")

        print(f"Final Trustworthiness score = {self.trustworthiness_score}\n")
        return self.trustworthiness_score

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
