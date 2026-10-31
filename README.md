# TWEEF

Trustworthiness Estimation and Enhancement Framework for Machine Learning Models.

## Effective Complexity fidelity correction

Loss-based Effective Complexity now measures changes from the unrestricted model.
For probabilistic predictions it uses mean Bernoulli KL divergence
`D(p || q) = H(p,q) - H(p,p)`, with natural logarithms. Original and restricted
predictions that are identical have exactly zero loss, even if their probabilities
are uncertain. For models without `predict_proba`, loss remains the fraction of
class predictions that change. Probabilities are clipped in float64 for finite
endpoint calculations. The minimum top-k subset is accepted when mean fidelity
loss is at most the configured nonnegative tolerance, allowing only relative
floating-point rounding at the comparison boundary.

The interpretation score remains `I_EC = 1 - k/N`. A zero score is still valid
when all variables are needed. The correction removes the previous entropy floor
that could force this result even with all variables retained. The `attribution`
proxy, attribution ordering, permutations, random seeds and hard-label prediction
rule are unchanged.

The default tolerance `0.05` is **0.05 nats for KL** and a **0.05 disagreement
fraction for hard labels**. These have different units. This is fidelity to the
model's predictions, rather than loss of accuracy against true labels; comparing
the meaning of tolerances across model interfaces requires care.

Historical aggregation-only results are preserved in `experiments/results/german`.
The EC correction is evaluated separately in `experiments/results/german_ec_kl`.
Metric caches include data, source and environment fingerprints; incompatible
caches or output directories are rejected before the old manifest is overwritten.

The paired ten-seed validation of this correction trained the same eight models
on the same splits and verified identical class predictions, attributions,
Accuracy and Equal Opportunity in 160 calibration/test model-partitions.

| Model | Previous mean EC score | KL/disagreement mean EC score |
|---|---:|---:|
| DT | 0.270492 | 0.270492 |
| RF | 0 | 0.873770 |
| ET | 0 | 0.616393 |
| LR | 0.031148 | 0.031148 |
| L-SVM | 0.222951 | 0.224590 |
| AB | 0 | 0.983607 |
| GB | 0 | 0.747541 |
| XGB | 0 | 0.339344 |

The small L-SVM change occurs at the inclusive tolerance boundary. EC changes
also change its empirical crests and uncertainty, so aggregated scores can change
even for a model with unchanged EC. With equal weights, corrected SL means now
span 0.650294–0.789365, compared with 0.508892–0.577071 before this EC correction;
the standard deviation across the eight model means changes from 0.029378 to
0.058984. This is descriptive dispersion, not a significance test or a claim of
better predictive performance. The KL tolerance measures preservation of probability
distributions; it does not measure human understanding or classification accuracy.

See [EC differences by model](experiments/results/german_ec_kl/ec_comparison_by_model.csv),
[paired trustworthiness differences](experiments/results/german_ec_kl/ec_score_comparison.csv)
and the [pairing audit](experiments/results/german_ec_kl/ec_correction_summary.json).
Historical artifacts store class predictions but not per-row probability vectors;
the audit verifies their stored diagnostics, allowing 2e-7 absolute rounding error
for the old float32 entropy diagnostic (observed maximum 2.73e-8).

![Paired EC correction on German Credit](experiments/results/german_ec_kl/ec_correction_comparison.png)

## Aggregation correction and German benchmark

Scalar `TrustworthinessEstimator.compute_metric()` and batch `fit()/transform()`
use the same selected operator, calibration, importance weights, and score scale.
`transform()` keeps the original row index and continuous linguistic diagnostics
in the metric columns; `Trustworthiness` contains the selected operator's score.
Per-row `AggregationResult` objects are available in `aggregation_results_`.

Empirical calibration is frozen at `[minimum, P25, P50, P75, maximum]`, using
linear quantile interpolation independently for each metric. Repeated crests
are allowed. A completely constant reference cannot identify a relative scale
and raises an explicit error instead of inventing thresholds. Explicit expert
crests and the existing fixed defaults remain supported.

Subjective Logic now averages Dirichlet evidence with the assigned metric
importance weights. With prior weight `W=2`, each non-dogmatic opinion gives
`r_i=2*b_i/u_i`, `s_i=2*d_i/u_i`; evidence is averaged with the normalized weights
and mapped back using `b=r/(2+r+s)`, `d=s/(2+r+s)`, `u=2/(2+r+s)`.
The result is projected as `b + alpha*u`. This is an importance-weighted extension
of averaging fusion, **not** the previous cumulative consensus or the standard
confidence-weighted belief fusion. Zero weights contribute no evidence. Vacuous
opinions contribute zero evidence; positive-weight dogmatic opinions dominate
finite evidence and are averaged among themselves using renormalized importance
weights (a common-rate zero-uncertainty limit).
The distinction between averaging and cumulative fusion is discussed in
[Multi-Source Fusion Operations in Subjective Logic](https://arxiv.org/abs/1805.01388).

GWA accepts `gwa_hesitation=0.0` as a scalar or a partial mapping by metric name.
For the favorable linguistic expectation `z`, the TWEEF mapping is
`mu=(1-h)*z`, `nu=(1-h)*(1-z)`, `pi=h`. Unspecified mapping entries have zero
hesitation. The default preserves the previous GWA with the same calibration.
Input and aggregated hesitations are included in `AggregationResult.details`.
Complete uncertainty returns the neutral score `0.5`; uncertainty and the
relative score `mu/(mu+nu)` are reported separately. A greater hesitation does
not necessarily decrease this relative score.
The Gaussian aggregation formula follows
[Ünver (2025)](https://doi.org/10.1007/s00357-025-09507-4); the proportional
hesitation mapping above is the explicit TWEEF adaptation.

The reproducible runner `experiments/german_aggregation.py` evaluates DT, RF, ET,
Linear Regression (binary threshold 0.5), Linear SVM, AdaBoost SAMME.R, Gradient
Boosting and XGBoost on all 1,000 German records. It uses Accuracy, Equal
Opportunity and loss-based Effective Complexity. Each seed uses identical
60/20/20 train/calibration/test partitions stratified by target and gender, with
preprocessing fitted only on training data. The enhancement/surrogate pipeline
does not participate in this aggregation benchmark.

Create a separate environment and preserve a baseline source snapshot:

```sh
python3 -m venv .venv-paper
.venv-paper/bin/python -m pip install --upgrade pip
.venv-paper/bin/python -m pip install -e . -r experiments/requirements-german.txt
mkdir -p /tmp/tweef-baseline-a36fe25
git archive a36fe25 | tar -x -C /tmp/tweef-baseline-a36fe25
```

On macOS, if XGBoost cannot locate OpenMP, the installed scikit-learn wheel's
bundled runtime can be used without a system installation:

```sh
export DYLD_FALLBACK_LIBRARY_PATH="$(.venv-paper/bin/python -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
```

Run all repetitions, reusing cached metric matrices when available:

```sh
.venv-paper/bin/python experiments/german_aggregation.py \
  --baseline-root /tmp/tweef-baseline-a36fe25 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --output experiments/results/german_ec_kl --reuse-metrics
.venv-paper/bin/python experiments/summarize_german.py \
  --input experiments/results/german_ec_kl
.venv-paper/bin/python experiments/compare_ec_correction.py \
  --before experiments/results/german --after experiments/results/german_ec_kl
```

The saved results include data/source checksums, splits, model parameters,
environment versions, metric matrices, crests, aggregation details, paired score
differences, rankings and API consistency checks. `legacy_native` executes the
unmodified baseline's P20/P40/P60 scalar calibration and quartile batch path;
`legacy_quartiles` supplies identical quartile crests to both baseline paths to
isolate operator changes. Nonuniform legacy batch weights are assigned explicitly
to its existing `.weights` attribute because its `fit()` has no weight argument.
`corrected` executes this branch with both interfaces sharing one calibrator.

The historical aggregation-only run below (source commit `785b340`) used absolute
soft-label cross-entropy for EC. Reference entropy above `0.05` made the condition
unattainable, forcing the EC score to zero. Its entropy shortcut was valid only
for that old loss and has been removed from the corrected KL runner. Historical
results remain available for the paired EC comparison.

That historical ten-seed run contains 80 trained models and 45,600 paired API result
rows, with no failed configurations. Batch/scalar scores are exactly equal in
the corrected implementation. SL responds to importance weights in all 80
seed/model cases, compared with none in either legacy scenario. With full GWA
hesitation, all 80 cases return score 0.5 and aggregated hesitation 1.

| Operator | Mean paired score change vs legacy native | Change with identical quartile calibration |
|---|---:|---:|
| LWA | -0.027083 | 0 |
| GWA, zero hesitation | -0.019092 | within 2.1e-16 per case |
| SL | +0.363882 | -0.006023 |

These are descriptive changes on the same metric inputs, not improvements in
predictive performance or evidence that an operator is superior. Most of the
large SL change comes from recalibration: the old EC crests often produce
zero uncertainty, giving zero-EC opinions dogmatic influence. Under identical
quartiles, SL's mean fusion change is much smaller. Effective Complexity was
zero in every test repetition for RF, ET, AB, GB and XGB, and in 4/10 for LR.

See the [per-model comparison](experiments/results/german/comparison_by_model.csv)
for means, standard deviations and paired differences, and the
[audit summary](experiments/results/german/audit_summary.json) for API consistency,
weight sensitivity, hesitation, rankings and metric limitations.

![German comparison: means and standard deviations across ten paired splits](experiments/results/german/german_before_after.png)

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
