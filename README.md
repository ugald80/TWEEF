# TWEEF

Trustworthiness Estimation and Enhancement Framework for Machine Learning Models.

TWEEF is a Python framework for estimating and improving the trustworthiness of
binary classification models. It evaluates three trust dimensions:

- Performance
- Fairness
- Interpretability

The framework computes metric values for an initial model, applies fairness and
interpretability processing, evaluates the final model, and reports both the final
trustworthiness summary and the trade-offs between trust dimensions.

## Publication

This framework was published in the following paper:

Ugalde, J., Salas, R., Torres, R., Velandia, D., Bariviera, A. F., Estevez, P. A.,
& Godoy, M. P. (2026). TWEEF: Trustworthiness Estimation and Enhancement
Framework for Machine Learning Models. *Applied Sciences*, *16*(2), 1077.
[Read the paper](https://www.mdpi.com/2076-3417/16/2/1077)

## Installation

Install TWEEF from the repository:

```bash
pip install git+https://github.com/ugald80/TWEEF.git@v0.0.3
```

For local development:

```bash
python3 -m pip install -e ".[all,dev]"
```

Core requirements:

- Python >= 3.9
- scikit-learn

Optional dependencies are grouped by capability:

```bash
pip install "tweef[fairness]"
pip install "tweef[interpretability]"
pip install "tweef[all]"
```

If an existing environment upgraded NumPy to 2.x and reports binary compatibility
errors, reinstall the project constraints:

```bash
python3 -m pip install --force-reinstall -e ".[all,dev]"
```

## User Manual

### 1. Prepare Your Data

TWEEF expects a pandas `DataFrame` for `X` and a pandas `Series` for `y`.
The protected attributes used for fairness analysis must be present as columns in
`X`.

```python
import pandas as pd

X = pd.DataFrame(...)
y = pd.Series(..., name="target")
```

TWEEF currently targets binary classification problems.

### 2. Define Fairness Groups

Use `FairnessGroup` to define protected and unprotected groups. Each group is
defined as a dictionary where the key is a column in `X` and the value is the list
of values belonging to that group.

```python
from TWEEF.UtilsFairnessGroup import FairnessGroup

protected_groups = FairnessGroup([
    {"gender_female": [1.0]},
])

unprotected_groups = FairnessGroup([
    {"gender_female": [0.0]},
])

protected_classes = [1.0]
```

### 3. Select Metrics

Metrics are grouped into performance, fairness, and interpretability dimensions.
Metric weights must sum to `1.0`.

```python
from TWEEF.Constants import METRICS
from TWEEF.Metrics import MetricList

metrics = MetricList()
metrics.add_metric(METRICS.PERFORMANCE.ACCURACY, weight=1 / 3, goal=0.8)
metrics.add_metric(METRICS.FAIRNESS.EQUAL_OPPORTUNITY, weight=1 / 3, goal=0.8)
metrics.add_metric(
    METRICS.INTERPRETABILITY.EFFECTIVE_COMPLEXITY,
    weight=1 / 3,
    goal=0.8,
)
```

Supported performance metrics include:

- `Accuracy`
- `Precision`
- `Recall`
- `F1Score`
- `Specificity`
- `Negative Predictive Value`

Supported fairness metrics include:

- `Statistical Parity`
- `Treatment Equality`
- `Equalized Odds`
- `equal Opportunity`

Supported interpretability metrics include:

- `Monotonicity`
- `Non Sensitivity`
- `Effective Complexity`

### 4. Create and Fit a TWEEF Classifier

`TrustworthyClassifier` follows the standard scikit-learn estimator contract:
configure it in the constructor, then call `fit(X, y)`.

```python
from sklearn.tree import DecisionTreeClassifier
from TWEEF import TrustworthyClassifier

model = DecisionTreeClassifier(random_state=0)

classifier = TrustworthyClassifier(
    given_classifier=model,
    protected_groups=protected_groups,
    unprotected_groups=unprotected_groups,
    protected_classes=protected_classes,
    metrics=metrics,
    aggregation="lwa",
    test_size=0.3,
    random_state=0,
)

classifier.fit(X, y)
predictions = classifier.predict(X_test)
```

The model passed in `given_classifier` can be any compatible scikit-learn
classifier.

### 5. Read the Outputs

After `fit`, TWEEF exposes the main reports as attributes:

```python
classifier.final_report_
classifier.tradeoffs_
classifier.diagnosis_
classifier.component_scores_
classifier.trustworthiness_score_
```

`final_report_` is the main summary table. It shows the diagnosis and final metric
values by stage, dimension, and metric.

```python
print(classifier.final_report_)
```

`tradeoffs_` shows directional trade-offs between trust dimensions:

```python
print(classifier.tradeoffs_)
```

The directional trade-off rows include:

- `performance_vs_fairness`
- `performance_vs_interpretability`
- `fairness_vs_performance`
- `fairness_vs_interpretability`
- `interpretability_vs_performance`
- `interpretability_vs_fairness`

Each row reports the diagnosis value, final value, absolute change, relative
change, reference-dimension change, variation ratio, relative cost, and relation
type.

`diagnosis_` keeps the initial trustworthiness assessment of the input model.
`component_scores_` keeps the detailed metric table used to build `final_report_`.
`trustworthiness_score_` stores the final aggregated trustworthiness score.

### 6. Configure Trust Aggregation

TWEEF exposes the aggregation operators described by the framework:

- `lwa`: Linguistic Weighted Average.
- `gwa`: Gaussian Weighted Aggregation.
- `sl`: Subjective Logic aggregation.

Select the operator with:

```python
classifier = TrustworthyClassifier(
    given_classifier=model,
    protected_groups=protected_groups,
    unprotected_groups=unprotected_groups,
    protected_classes=protected_classes,
    metrics=metrics,
    aggregation="gwa",
)
```

By default, TWEEF reports trustworthiness in the normalized `[0,1]` scale. For
paper-style linguistic reporting, use the `[-2,2]` scale:

```python
classifier = TrustworthyClassifier(
    given_classifier=model,
    protected_groups=protected_groups,
    unprotected_groups=unprotected_groups,
    protected_classes=protected_classes,
    metrics=metrics,
    trustworthiness_score_scale="linguistic",
)
```

Accepted values are:

- `normalized` for `[0,1]`
- `linguistic` for `[-2,2]`

### 7. Configure Effective Complexity

`Effective Complexity` supports two calculation modes:

- `attribution`: fast proxy based on the cumulative mass of feature attributions.
- `loss`: original definition based on the expected loss after restricting the
  model to the top-k attributed features.

Use the original loss-based definition with:

```python
classifier = TrustworthyClassifier(
    given_classifier=model,
    protected_groups=protected_groups,
    unprotected_groups=unprotected_groups,
    protected_classes=protected_classes,
    metrics=metrics,
    effective_complexity_mode="loss",
    effective_complexity_loss_threshold=0.05,
    effective_complexity_repeats=3,
)
```

## Logging

TWEEF uses standard Python logging. Applications can configure logs once at
startup:

```python
from TWEEF import TWEEFLogger

TWEEFLogger.configure(level="INFO")
```

The default format is:

```text
YYYY-MM-DD HH:MM:SS | LEVEL | module_or_class.function | message
```

## Paper Experiments Demo

The experiments from the published TWEEF paper can be executed from this
repository. The reference is:

Ugalde, J., Salas, R., Torres, R., Velandia, D., Bariviera, A. F., Estevez, P. A.,
& Godoy, M. P. (2026). TWEEF: Trustworthiness Estimation and Enhancement
Framework for Machine Learning Models. *Applied Sciences*, *16*(2), 1077.
[Read the paper](https://www.mdpi.com/2076-3417/16/2/1077)

The demo script reproduces the experimental setup described in that publication.

```bash
python3 demo/paper_experiments.py --dataset all
```

By default the script uses a reproducible sample for faster development runs. To
use all available rows:

```bash
python3 demo/paper_experiments.py --dataset all --full
```

To run one dataset:

```bash
python3 demo/paper_experiments.py --dataset german
python3 demo/paper_experiments.py --dataset compas
python3 demo/paper_experiments.py --dataset adult
```

To report trustworthiness in the paper's linguistic scale:

```bash
python3 demo/paper_experiments.py --dataset all --trustworthiness-score-scale linguistic
```

To use the loss-based Effective Complexity definition:

```bash
python3 demo/paper_experiments.py --dataset german --effective-complexity-mode loss
```

The demo requires the optional fairness and interpretability dependencies:

```bash
python3 -m pip install -e ".[all]"
```

## Development

Run the test suite with:

```bash
python3 -m pytest -q
```

Some integration tests are opt-in because they require datasets and optional
dependencies:

```bash
TWEEF_RUN_INTEGRATION=1 python3 -m pytest test/test_trustworthy_classifier.py
```
