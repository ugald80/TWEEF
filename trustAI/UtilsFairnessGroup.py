from typing import NewType
import numpy as np
import pandas as pd
FairnessGroup = NewType('FairnessGroup', list[ dict[str, list]] )
#FairnessGroup([
#    {'sex': [0]},
#    {'gender': [0,1]}

class FairnessGroupUtils:
    """
    Utility class for generating and unparsing fairness groups.
    """
    def __init__(self):
        pass

    def generate_groups(self, keys: list[str], values: list[list]):
        """ 
        Generate a list of dictionaries with the keys and values provided.
        Parameters:
        -----------
        keys : list
            List of keys.
        values : list
            List of values.
        Returns:
        --------
        Fairness Groups
            List of FairnessGroup objects containing the keys and values provided.
        """
        if len(keys) != len(values):
            raise ValueError("the length of keys and values must be the same")
        groups = []
        for key, value in zip(keys, values):
            groups.append({key: value})
        return groups

    def unparse_groups(self, groups: FairnessGroup):
        """
        Unparse the groups into keys and values.

        Parameters:
        -----------
        groups : FairnessGroup
            List of FairnessGroup objects.

        Returns:
        --------
        keys : list
            List of keys.
        values : list
            List of values.
        """
        keys   = list()
        values = list()
        for group in groups:
            for key in group:
                keys.append(key)
                values.append(group[key])
        return keys, values

    def split_labels_by_group(self, x: pd.DataFrame, y: pd.Series, protected_groups: FairnessGroup):
        """
        Separates the y labels into protected and unprotected groups.

        Parameters:
        -----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Labels.
        protected_groups : FairnessGroup
            List of FairnessGroup objects containing protected values for the protected columns.

        Returns:
        --------
        y_protected : pd.Series
            Labels of the protected group.
        y_unprotected : pd.Series
            Labels of the unprotected group.
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if isinstance(y, np.ndarray):
            if y.ndim == 2 and y.shape[1] == 1:
                y = pd.Series(y.flatten())
            elif y.ndim == 1:
                y = pd.Series(y)
            else:
                raise ValueError("y must be a one-dimensional array or a two-dimensional array with a single column")
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be a DataFrame with a single column or a Series")
            y = y.iloc[:, 0]  # Convert single-column DataFrame to Series
        elif not isinstance(y, pd.Series):
            raise TypeError("y must be a numpy.ndarray, pandas.DataFrame, or pandas.Series")
                
                
        protected = pd.DataFrame()
        for group in protected_groups:
            for key, values in group.items():
                identified = x[x[key].isin(values)]
                protected = pd.concat([protected, identified])

        # print(f"       | protected indexes: {protected.index.tolist()}")
        y_protected   = y[y.index.isin(protected.index.tolist())]
        y_unprotected = y[~y.index.isin(protected.index.tolist())]

        return y_protected, y_unprotected
