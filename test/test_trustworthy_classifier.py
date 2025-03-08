import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from trustAI.UtilsFairnessGroup import FairnessGroup
from trustAI.Metrics import MetricList
from trustAI.TrustworthyClassiffier import TrustworthyClassifier
from trustAI.Constants import METRICS, get_metrics_of

print("")
print("")
print("__________________________________________________________")
print("|                                                        |")
print("|                                                        |")
print("|                    Testing trustAI                     |")
print("|                                                        |")
print("|________________________________________________________|")
print("")
print(
    "For invoking TrustworthyClassifier class, we need to provide the following parameters:"
)
print("")

print("1. Dataset")

# Import Data
print("    Import Data")
dataset_german = pd.read_csv("datasets/german.csv", sep=",")
# Prepare data
print("    Prepare Data")
label_name = "Credit Score"
dataset_german = dataset_german[
    dataset_german["Personal status and sex"].isin(["A91", "A92", "A94"])
]
Y = dataset_german[label_name]
Y = Y.map({1: 1, 2: 0})

X = dataset_german.drop([label_name], axis=1)
X["Telephone"] = X["Telephone"].map({"A191": 0, "A192": 1})
X["foreign worker"] = X["foreign worker"].map({"A201": 1, "A202": 0})
X["Personal status and sex"] = X["Personal status and sex"].map(
    {"A91": "male", "A92": "female", "A94": "male"}
)
X = pd.concat(
    [
        X,
        pd.get_dummies(
            X["Status of existing checking account"],
            prefix="Status of existing checking account",
        ),
    ],
    axis=1,
)
X = pd.concat([X, pd.get_dummies(X["Credit history"], prefix="Credit history")], axis=1)
X = pd.concat([X, pd.get_dummies(X["Purpose"], prefix="Purpose")], axis=1)
X = pd.concat(
    [X, pd.get_dummies(X["Savings account/bonds"], prefix="Savings account/bonds")],
    axis=1,
)
X = pd.concat(
    [
        X,
        pd.get_dummies(
            X["Present employment since"], prefix="Present employment since"
        ),
    ],
    axis=1,
)
X = pd.concat(
    [X, pd.get_dummies(X["Personal status and sex"], prefix="Personal status and sex")],
    axis=1,
)
X = pd.concat(
    [
        X,
        pd.get_dummies(
            X["Other debtors/guarantors"], prefix="Other debtors/guarantors"
        ),
    ],
    axis=1,
)
X = pd.concat([X, pd.get_dummies(X["Property"], prefix="Property")], axis=1)
X = pd.concat(
    [X, pd.get_dummies(X["Other installment plans"], prefix="Other installment plans")],
    axis=1,
)
X = pd.concat([X, pd.get_dummies(X["Housing"], prefix="Housing")], axis=1)
X = pd.concat([X, pd.get_dummies(X["Job"], prefix="Job")], axis=1)

X = X.drop(
    [
        "Status of existing checking account",
        "Credit history",
        "Purpose",
        "Savings account/bonds",
        "Present employment since",
        "Personal status and sex",
        "Other debtors/guarantors",
        "Property",
        "Other installment plans",
        "Housing",
        "Job",
    ],
    axis=1,
)

# Train a regular model
print("    Scale data")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.3, stratify=Y, random_state=0
)

print("")
print("2. A classifier model")
DesicionTree = DecisionTreeClassifier()
print(f"    {DesicionTree}")

print("")
print("3. Fairness Groups")
protectedGroups = FairnessGroup([{"Personal status and sex_female": [0.0]}])
unprotectedGroups = FairnessGroup([{"Personal status and sex_female": [1.0]}])
protectedClasses = [1.0]  # list con los valores de las clases favorecidas
print(f"    protected groups:   {protectedGroups}")
print(f"    unprotected groups: {unprotectedGroups}")
print(f"    protected classes:  {protectedClasses}")
print("")

print("4. Metrics definition:")
metrics = MetricList()
metrics.add_metric(name=METRICS.PERFORMANCE.ACCURACY, weight=0.33, goal=0.8)
metrics.add_metric(name=METRICS.FAIRNESS.TREATMENT_EQUALITY, weight=0.33, goal=0.8)
metrics.add_metric(
    name=METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY, weight=0.34, goal=0.8
)
print(f"   {metrics.to_dataframe()}")

print(f"\n** allowed metrics: {get_metrics_of(METRICS)}\n\n")

print("\nExample: train a trustworthy classifier from a Decision Tree model\n")

print("Create a TrustworthyClassifier instance")
trustworthy_classifier = TrustworthyClassifier(
    DesicionTree,
    x_train,
    x_test,
    y_train,
    y_test,
    protectedGroups,
    unprotectedGroups,
    protectedClasses,
    metrics,
)

print("\nTrain the TrustworthyClassifier instance:")
trustworthy_classifier.fit()

print("\nThe TrustworthyClassifier is trained and ready to be used!!")
trustworthy_predictions = trustworthy_classifier.predict(x_test)

print("\nThe TrustworthyClassifier offer some usefull outputs:")

print("   1. Unbiased dataset: A dataframe with the transformed data resulting from the application of preprocessing and postprocessing bias mitigation techniques.")
print("      This can be accesed by calling the unbiased_dataset method of the TrustworthyClassifier instance.")
print("      This method returns a pandas DataFrame with a processed copy of the input dataset")

procesed_data = trustworthy_classifier.unbiased_dataset()
print(f"\n      For this example, the Decision Tree evaluation is:\n{procesed_data.head()}")

print("\n   2. Trustworthiness Diagnosis: A dataframe with the trustworthiness metrics of the given classifier.")
print("      This can be obtained by calling the trustworthiness_diagnosis() method of the TrustworthyClassifier instance.")
print("      This method returns a pandas DataFrame with the following columns:")
print("         - 'name': The name of each metric.")
print("         - 'value': The value of each metric.")

tw_diagnosis = trustworthy_classifier.trustworthiness_diagnosis()
print(f"\n      For this example, the Decision Tree evaluation is:\n{tw_diagnosis}")

print("\n   3. Trustworthiness Metrics: A dataframe with the final value of the trustworthiness metrics of the trustworthy model.")
print("      This can be obtained by calling the trustworthiness_metrics() method of the TrustworthyClassifier instance.")
print("      This method returns a pandas DataFrame with the following columns:")
print("         - 'name': The name of each metric.")
print("         - 'value': The value of each metric.")

tw_metrics = trustworthy_classifier.trustworthiness_metrics()
print(f"\n      For this example, the Decision Tree evaluation is:\n{tw_metrics}")

print("\nEnd of testing trustAI\n")
