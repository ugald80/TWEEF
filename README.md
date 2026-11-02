# TWEEF

Trustworthiness Estimation and Enhancement Framework for Machine Learning Models.


## Final-surrogate predictor and complete mechanisms

The primary classifier now keeps the interpretable surrogate as its final model.
The given classifier is a teacher/reference, never a fallback chosen after looking
at test performance. The persisted prediction chain includes preprocessing,
the surrogate and any configured fairness postprocessor. `predict`, `predict_proba`
and `evaluate` use that same chain; evaluation does not fit or select components.

An executable example is available in
[`test/experiments/example_final_surrogate.py`](test/experiments/example_final_surrogate.py):

```sh
.venv-paper/bin/python test/experiments/example_final_surrogate.py
```

It demonstrates an injected user mechanism and checks that report predictions
match the public predictor. The example accepts `--postprocess` and can use built-in
SL with `--operator sl --reference PATH_TO_TRAINING_ONLY_REFERENCE_CSV`.

The new usage pattern is:

```python
from interpret.glassbox import ExplainableBoostingClassifier
from TWEEF.TrustworthyClassifier import TrustworthyClassifier
from TWEEF.Aggregation import SubjectiveLogic
from TWEEF.Calibration import QuartileCalibration

workflow = TrustworthyClassifier(
    given_classifier=teacher,
    surrogate=ExplainableBoostingClassifier(interactions=0, random_state=0),
    preprocessor=preprocessor,
    aggregation=None,
    metric_weights={"Accuracy": .4, "equal Opportunity": .3,
                    "Effective Complexity": .3},
    effective_complexity={"loss": "bernoulli_kl", "loss_threshold": .05,
                          "feature_unit": "original", "ranking_method": "permutation",
                          "repeats": 3},
    reference_cv=5,
    postprocessor="calibrated_equalized_odds",  # optional; default is disabled
    random_state=0,
)
workflow.fit(X_train, y_train, sensitive_features=g_train,
             validation_data=(X_calibration, y_calibration),
             validation_sensitive_features=g_calibration)
workflow.set_aggregation(
    SubjectiveLogic(alpha=.5, calibration=QuartileCalibration()),
    reference_metrics=training_only_reference,
)
report = workflow.evaluate(X_test, y_test, sensitive_features=g_test)
y_pred = workflow.predict(X_new, sensitive_features=g_new)
```

`aggregation=None` allows predictor fitting and collection of training-only CV
metrics before selecting a shared reference. A built-in or custom mechanism can
also be supplied directly at construction. Quartile calibration rejects a constant
active reference instead of manufacturing thresholds. A common pooled reference
is useful when comparing several models; it must exclude their test data.

A complete mechanism implements `fit(reference_metrics, *, weights)` and
`aggregate(metrics)`. It owns its representation and aggregation and returns a
`MechanismResult` with its native score, declared range/direction and optional
details. TWEEF does not require linguistic membership or normalize custom scores.
The built-in `LWA`, `GWA` and `SubjectiveLogic` preserve their native formulas;
strings `"lwa"`, `"gwa"` and `"sl"` remain supported. GWA's `hesitation` is independent
of importance weights. Plugins must explicitly declare unsupported weighting;
configuration conflicts are errors rather than hidden precedence rules.

EC now perturbs original variables and calls the full stored predictor. Permuting
a categorical variable does not create invalid one-hot combinations. Permutation
rankings are fitted before test and frozen. KL is explicit and requires valid
probabilities; `class_disagreement` provides a common teacher/student assessment
when a teacher such as LinearSVC or thresholded LinearRegression lacks probabilities.
The two losses have different units and are never silently interchanged in the
new pipeline. A teacher is not calibrated merely to compute the EBM student's EC.

The legacy DisparateImpactRemover procedure is excluded from the primary flow:
it rebuilt repairs separately on each evaluation batch and did not provide a
persisted inductive transformation. Configured preprocessing must learn on
training data and transform new observations. Fairness postprocessing is fitted
on calibration data, stored, and applied at prediction time with raw sensitive
groups. True labels are never replaced by postprocessed predictions.

## Tests and experiments

All tests, experimental protocols, reports and saved results are maintained under
[`test/`](test/INDEX.md). See that index for execution commands and the experiment report.

## Migration from the earlier classifier

Prefer `metric_weights`, explicit EC configuration and explicit calibration data.
`given_classifier` names the teacher; `surrogate_` and the persisted predictor name
the final model. Read teacher/student fidelity separately from final Accuracy,
equity and EC. Methods that relied on replacing the final model with the teacher
or on fitting repairs using test data no longer have those semantics. Legacy
constructor parameters are retained where compatible, with validation or
migration warnings when necessary. The manual sections below document the older
configuration style; the executable final-surrogate example is the primary guide.

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

## Legacy configuration reference

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
- `loss`: fidelity to the original predictions after restricting the model to
  its top-k attributed features, using Bernoulli KL or class disagreement.

Use the loss-based definition with:

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
