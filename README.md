# TWEEF

Trustworthiness Estimation and Enhancement Framework for Machine Learning Models.


## Final-surrogate predictor and complete mechanisms

The primary classifier now keeps the interpretable surrogate as its final model.
The given classifier is a teacher/reference, never a fallback chosen after looking
at test performance. The persisted prediction chain includes preprocessing,
the surrogate and any configured fairness postprocessor. `predict`, `predict_proba`
and `evaluate` use that same chain; evaluation does not fit or select components.

An executable example is available in
[`experiments/example_final_surrogate.py`](experiments/example_final_surrogate.py):

```sh
.venv-paper/bin/python experiments/example_final_surrogate.py
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

## German: final-surrogate experiment

Run the complete new protocol in a separate directory:

```sh
.venv-paper/bin/python experiments/german_final_surrogate.py
```

On macOS, use the OpenMP environment variable described in the installation notes
below if required by the XGBoost wheel. A smaller smoke run is:

```sh
.venv-paper/bin/python experiments/german_final_surrogate.py \
  --output /tmp/tweef-final-surrogate-smoke --seeds 0 --models DT RF \
  --candidate-rounds 25 50 --reference-cv 3 --ec-repeats 1
```

The full run retains all 1,000 records, the paper's eight teacher configurations
and seeds 0–9. Each seed has the same 600/200/200 train/calibration/test split as
the earlier benchmark. Two additive EBM candidates have fixed budgets of 100 and
500 maximum rounds, one outer bag and one job; these are explicit experimental
settings, not EBM defaults. There are 160 final candidate fits plus internal CV
fits. Postprocessing is explicitly enabled for the main run.

Within each seed, training-only five-fold references from every teacher/candidate
are pooled. Fold-local fairness calibration is separate from the fold used for
metric measurement. Final fairness calibration uses the external calibration
partition. No candidate selection or reference fitting uses test.

Two analyses reuse the same fitted candidates and cached metric matrices:

1. **Fixed predictor:** the 500-round candidate is evaluated by all mechanisms with
   identical metrics/reference. Weight sweeps, GWA hesitation, rankings, ties and
   rank agreement describe mechanism behavior.
2. **Complete flow:** each mechanism selects the candidate with the best mean
   internal CV mechanism score before any test evaluation. Candidates, splits,
   references and budgets are identical; ties prefer the smaller budget.

`experiments/results/german_final_surrogate` contains the protocol/source/environment
manifest, per-seed caches, CV references, raw teacher/student/final metrics,
class-disagreement comparisons, KL metrics, fidelity, mechanism scores/rankings,
full-flow selections, predictive differences and a summary figure. Cache reuse
requires the same full protocol fingerprint and rejects incompatible output
before overwriting its manifest.

Historical results remain in `german` and `german_ec_kl`. The new experiment changes
the final predictor, EC perturbation unit, attribution fitting and calibration
reference; EC/score changes against those artifacts cannot be attributed to
aggregation alone. Accuracy/equity changes are reported with the changed protocol
made explicit. Repeated splits overlap and are descriptive evidence from one
dataset. Native mechanism scores are not interchangeable units; a higher average
score alone is not evidence of greater real trustworthiness.

## Migration from the earlier classifier

Prefer `metric_weights`, explicit EC configuration and explicit calibration data.
`given_classifier` names the teacher; `surrogate_` and the persisted predictor name
the final model. Read teacher/student fidelity separately from final Accuracy,
equity and EC. Methods that relied on replacing the final model with the teacher
or on fitting repairs using test data no longer have those semantics. Legacy
constructor parameters are retained where compatible, with validation or
migration warnings when necessary. The manual sections below document the older
configuration style; the executable final-surrogate example is the primary guide.

## Historical correction records

The following correction results describe the preserved aggregation-only protocol.

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
