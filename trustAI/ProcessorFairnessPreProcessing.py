from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import numpy as np
from trustAI.ProcessorFairness import BiasMitigator
from trustAI import UtilsFairnessGroup
from trustAI.UtilsFairnessGroup import FairnessGroupUtils



class BiasMitigatorPreProcessing(BiasMitigator):
    """
    Scikit-learn compatible preprocessing class for mitigating bias
    by removing disparate impact.
    """

    def __init__(self):
        super().__init__()
        self.transformed_x_train = None
        self.transformed_x_test = None
        self.transformed_y_train = None
        self.transformed_y_test = None
        self.disparate_impact = None
        self.group_utils = FairnessGroupUtils()

    def preprocess_disparate_impact_removal(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        label_name: str,
        protected_groups: UtilsFairnessGroup,
        unprotected_groups: UtilsFairnessGroup,
        protected_classes: list,
        repair_level: float = 1.0,
    ):
        """
        Removes disparate impact from the dataset.

        Parameters:
        X_train (NDArray): Training feature data.
        X_test (NDArray): Testing feature data.
        y_train (NDArray): Training labels.
        y_test (NDArray): Testing labels.
        label_name (str): Name of the label column.
        protectedGroups (FairnessGroup): Protected groups.
        unprotectedGroups (FairnessGroup): Unprotected groups.
        protectedClasses (list): List of favorable classes.
        repair_level (float): Level of bias repair (0-1).

        Returns:
        float: Disparate impact score after preprocessing.
        """

        print("        . disparate_impact: setting up protected and unprotected groups in dataset")
        protectedKey, protectedValues = self.group_utils.unparse_groups(protected_groups)
        # print(f"        . disparate_impact: protectedKey: {protectedKey}")
        # print(f"        . disparate_impact: protectedValues: {protectedValues}")
 
        print("        . disparate_impact: generating AIF360 datasets for training and testing")
        aif_dataset_train = self.generate_aif360(
            X_train, y_train, label_name, protectedKey, protectedValues, protected_classes,
        )
        aif_dataset_test = self.generate_aif360(
            X_test , y_test , label_name, protectedKey, protectedValues, protected_classes
        )

        # Normalize the data
        scaler = MinMaxScaler(copy=False)
        aif_dataset_train.features = scaler.fit_transform(aif_dataset_train.features)
        aif_dataset_test.features = scaler.transform(aif_dataset_test.features)

        # Apply Disparate Impact Remover
        di = DisparateImpactRemover(repair_level=repair_level)
        train_repd = di.fit_transform(aif_dataset_train)
        test_repd = di.fit_transform(aif_dataset_test)

        # Remove protected attribute columns from features
        X_train_trunc = train_repd.features
        X_test_trunc  = test_repd.features
        for key in protectedKey:
            index = train_repd.feature_names.index(key)
            X_train_trunc = np.delete(X_train_trunc, index, axis=1)
            X_test_trunc = np.delete(X_test_trunc, index, axis=1)

        y_train_trunc = train_repd.labels.ravel()
        y_test_trunc = test_repd.labels.ravel()

        # Train a logistic regression model for predictions
        lmod = LogisticRegression(class_weight="balanced", solver="liblinear")
        lmod.fit(X_train_trunc, y_train_trunc)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_test_trunc)

        print( "        . disparate_impact: evaluate disparate impact")
        cm = BinaryLabelDatasetMetric(
            test_repd_pred, privileged_groups=protected_groups, unprivileged_groups=unprotected_groups,
        )
        # Save results
        print( "        . disparate_impact: storing results")
        self.disparate_impact = cm.disparate_impact()
        self.transformed_x_train = pd.DataFrame(train_repd.features, columns=X_train.columns, index=X_train.index)
        self.transformed_x_test  = pd.DataFrame(test_repd.features , columns=X_test.columns , index=X_test.index )
        self.transformed_y_train = pd.Series(y_train_trunc, name=label_name, index=y_train.index)
        self.transformed_y_test  = pd.Series(y_test_trunc , name=label_name , index=y_test.index)

        return self.disparate_impact
