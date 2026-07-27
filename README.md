# TWEEF
Trustworthiness Evaluation for Machine Learning Models

### Installation:
Install TWEEF by executing the following command:
```
pip install git+https://github.com/ugald80/TWEEF.git@v0.0.3
```

The core installation needs the following dependencies: 

Requirements
- python >= 3.9
- pip install scikit-learn

Optional dependencies are grouped by capability:

```
pip install "tweef[fairness]"
pip install "tweef[interpretability]"
pip install "tweef[all]"
```

And done! You have successfully installed TWEEF! 

### Trust aggregation
TWEEF now exposes the aggregation operators described by the framework:

- `lwa`: Linguistic Weighted Average.
- `gwa`: Gaussian Weighted Aggregation.
- `sl`: Subjective Logic aggregation.

The operator can be selected through `TrustworthyClassifier(..., aggregation="lwa")`
or directly through `TrustworthinessEstimator(aggregation="gwa")`.

By default, TWEEF reports trustworthiness in the normalized `[0,1]` scale. For
paper-style linguistic reporting, use the `[-2,2]` scale:

```python
classifier = TrustworthyClassifier(
    given_classifier=model,
    metrics=metrics,
    trustworthiness_score_scale="linguistic",
)
```

The accepted values are `normalized` for `[0,1]` and `linguistic` for `[-2,2]`.

### Logging
TWEEF uses standard Python logging. Applications can configure logs once at startup:

```python
from TWEEF import TWEEFLogger

TWEEFLogger.configure(level="INFO")
```

The default format is:

```text
YYYY-MM-DD HH:MM:SS | LEVEL | module_or_class.function | message
```

### Effective Complexity
`Effective Complexity` supports two calculation modes:

- `attribution`: fast proxy based on the cumulative mass of feature attributions.
- `loss`: original definition based on the expected loss after restricting the
  model to the top-k attributed features.

Select the mode in Python:

```python
classifier = TrustworthyClassifier(
    given_classifier=model,
    metrics=metrics,
    effective_complexity_mode="loss",
    effective_complexity_loss_threshold=0.05,
    effective_complexity_repeats=3,
)
```

Or in the paper demo:

```bash
python3 demo/paper_experiments.py --dataset german --effective-complexity-mode loss
```

### Paper experiments demo
The published TWEEF experiments can be executed from:

```bash
python3 demo/paper_experiments.py --dataset all
```

To report trustworthiness in the paper's linguistic scale:

```bash
python3 demo/paper_experiments.py --dataset all --trustworthiness-score-scale linguistic
```

By default the script uses a reproducible sample for faster development runs. To use
all available rows:

```bash
python3 demo/paper_experiments.py --dataset all --full
```

The demo requires the optional fairness and interpretability dependencies:

```bash
python3 -m pip install -e ".[all]"
```

If an existing environment upgraded NumPy to 2.x and reports binary compatibility
errors, reinstall the project constraints:

```bash
python3 -m pip install --force-reinstall -e ".[all,dev]"
```

### Next Steps
You can learn how to use TWEEF following the test cases included in test/ directory.

### Classifier API
`TrustworthyClassifier` follows the standard scikit-learn estimator contract:

```python
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
