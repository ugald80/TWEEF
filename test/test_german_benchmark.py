"""Cheap protocol checks; training the eight models belongs to the benchmark CLI."""
import numpy as np
import pandas as pd
from experiments.german_aggregation import (
    BinaryLinearRegression, load_german, make_preprocessor, make_split,
)


def test_german_keeps_all_rows_and_sensitive_groups():
    X, y, gender = load_german()
    assert X.shape == (1000, 20)
    assert (X['Personal status and sex'] == 'A93').sum() == 548
    assert gender.value_counts().to_dict() == {'male': 690, 'female': 310}
    assert y.value_counts().to_dict() == {1: 700, 0: 300}


def test_split_is_disjoint_reproducible_and_stratified():
    _, y, gender = load_german()
    split = make_split(y, gender, seed=3)
    repeated = make_split(y, gender, seed=3)
    assert {k: len(v) for k, v in split.items()} == {'train': 600, 'calibration': 200, 'test': 200}
    assert len(set(np.concatenate(list(split.values())))) == 1000
    for part, indices in split.items():
        assert np.array_equal(indices, repeated[part])
        assert pd.crosstab(y.iloc[indices], gender.iloc[indices]).shape == (2, 2)


def test_preprocessor_only_learns_training_extrema_and_categories():
    train = pd.DataFrame({'number': [0., 1.], 'category': ['a', 'b']})
    held_out = pd.DataFrame({'number': [10.], 'category': ['unseen']})
    preprocessing = make_preprocessor(train).fit(train)
    transformed = preprocessing.transform(held_out)
    assert transformed[0, -1] == 10.  # fitting on held-out data would yield 1.
    assert transformed[0, :-1].sum() == 0


def test_linear_regression_classifier_uses_threshold_without_fake_probability():
    X = np.array([[0.], [1.], [2.], [3.]])
    model = BinaryLinearRegression().fit(X, [0, 0, 1, 1])
    assert model.predict(X).tolist() == [0, 0, 1, 1]
    assert not hasattr(model, 'predict_proba')
