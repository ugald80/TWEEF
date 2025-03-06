import numpy as np
import pandas as pd
from trustAI.Constants import METRICS, get_metrics_of

class Metric:
    def __init__(self, name: str, weight: float, goal: float=0, value: float=None):
        """
        Initializes an instance of the class with the provided parameters.
        Parameters:
            name (str): The name of the metric. Must be one of the allowed names in `METRICS` constant.
            weight (float): The weight of the metric. Must be in the range [0, 1].
            goal (float, optional): The goal of the metric. Must be in the range [0, 1]. Default is 0.
            value (float, optional): The current value of the metric. Default is None.
        Exceptions:
            ValueError: If `weight` or `goal` are not in the range [0, 1].
            ValueError: If `name` is not one of the allowed names in `METRICS`.
        Attributes:
            name (str): Stores the name of the metric.
            weight (float): Stores the weight of the metric.
            goal (float): Stores the goal of the metric.
            value (float): Stores the current value of the metric, initially None.
        """
        
        if not (0 <= weight <= 1):
            raise ValueError("weight must be in the range [0, 1]")
        
        if not (0 <= goal <= 1):
            raise ValueError("goal must be in the range [0, 1]")
        
        allowed_metrics = get_metrics_of(METRICS)
        if not (name in allowed_metrics):
            raise ValueError(f"metric name must be one of: {allowed_metrics}")
        
        self.name = name
        self.weight = weight
        self.goal = goal
        self.value = value
    
    def copy(self):
        """
        Generates a copy of this metric

        Returns:
            new_metric: a copy of this metric
        """
        return Metric(self.name, self.weight, self.goal, self.value)
            
    def __repr__(self):
        return f"Metric(name={self.name}, weight={self.weight}, goal={self.goal}, value={self.value})"


class MetricList:
    def __init__(self):
        """
        Initializes an empty list to store metrics.
        Attributes:
            metrics (list): A list to store instances of the Metric class.
        """
        self.metrics = []
    
    def add_metric(self, name: str, weight: float, goal: float, value: float=None):
        """
        Adds a metric to the this list.

        Parameters:
        ___________
        name (str): The name of the metric.
        weight (float): The weight of the metric.
        goal (float): The goal value of the metric.
        value (float, optional): The current value of the metric. Defaults to None.
        """
        metric = Metric(name, weight, goal, value)
        self.metrics.append(metric)
    
    def get_metrics(self):
        """
        Retrieve the stored metrics.
        
        Returns:
            dict: A dictionary containing the metrics.
        """
        return self.metrics
        
    def get_names(self):
        """
        Retrieves the names of all metrics.
        
        Returns:
            list: A list containing the names of all metrics.
        """
        return [metric.name for metric in self.metrics]
    
    def get_weights(self):
        """
        Retrieves the weights of all metrics.
        """
        return [metric.weight for metric in self.metrics]
    
    def get_goals(self):
        """
        Retrieves the goals of all metrics
        """
        return [metric.goal for metric in self.metrics]
    
    def get_values(self):
        """
        Retrieves the values of all metrics.
        """
        return [metric.value for metric in self.metrics]
        
    def to_dataframe(self):
        """
        Converts the metrics to a pandas DataFrame.
        """
        names = np.array(self.get_names()).reshape(-1, 1).flatten()
        values = np.array(self.get_values()).reshape(-1, 1)
        metrics_matrix = pd.DataFrame(values, index=names, columns=[0]).T
        return metrics_matrix

    def copy(self):
        """
        Copy
        Generates a copy of this object, containing all the Metrics added to this list
        Returns:
            new_metric_list: a copy of the metric list contained in this object
        """
        new_list = MetricList()
        new_list.metrics = [metric.copy() for metric in self.metrics]
        return new_list

    def __repr__(self):
        string="MetricsList:"
        for metric in self.metrics:
            string += f"\n    {metric}"
        return string
