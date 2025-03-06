from aif360.datasets import StandardDataset
import numpy as np
import pandas as pd
from typing import List

class BiasMitigator:
    """ Base class for bias mitigators. """
    def __init__(self):        
        self.protected_classes = None
        self.protected_keys = None
        self.protected_values = None

    def generate_aif360(self, 
            x: np.ndarray, 
            y: np.ndarray,
            label_name: str,
            protected_keys: List[str],
            protected_values: List[float],
            protected_classes: List[List],
        ) -> StandardDataset:
        """
        Generate an AIF360-compatible dataset.

        Parameters:
        X (NDArray): Feature array.
        y (NDArray): Label array.

        Returns:
        StandardDataset : AIF360-compatible dataset.
        """
        if not isinstance(x, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame or numpy array.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array.")

        df = pd.DataFrame(x.copy()) if isinstance(x, np.ndarray) else x.copy()
        df[label_name] = y.copy()
        
        return StandardDataset(
            df,
            label_name=label_name,
            favorable_classes=protected_classes,
            protected_attribute_names=protected_keys,
            privileged_classes=protected_values,
        )
