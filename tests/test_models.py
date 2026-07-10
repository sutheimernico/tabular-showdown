"""Tests for the LightGBM tune/fit/predict interface, on a small synthetic dataset."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification

from tabular_showdown.models import fit_lgbm, predict_proba_positive, tune_lgbm

SEARCHED_KEYS = {
    "num_leaves",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
}

FIXED_TEST_PARAMS = {
    "objective": "binary",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
    "n_estimators": 20,
}


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """300 rows, 5 numeric features + 1 fabricated categorical column.

    Mirrors the real Adult setup (mixed numeric/category dtype columns) so
    LightGBM's native categorical handling is exercised here too.
    """
    X_num, y = make_classification(
        n_samples=300, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(5)])
    rng = np.random.default_rng(42)
    X["cat_feature"] = pd.Categorical(rng.choice(["a", "b", "c"], size=300))
    return X, pd.Series(y, name="target")


def test_tune_lgbm_returns_dict_with_searched_and_fixed_keys(synthetic_data):
    X, y = synthetic_data
    params = tune_lgbm(X, y, n_trials=2, n_folds=3, seed=42)
    assert params.keys() >= SEARCHED_KEYS
    assert params["objective"] == "binary"
    assert params["verbosity"] == -1
    assert params["n_jobs"] == -1
    assert params["seed"] == 42


def test_tune_lgbm_is_deterministic(synthetic_data):
    X, y = synthetic_data
    params_1 = tune_lgbm(X, y, n_trials=2, n_folds=3, seed=42)
    params_2 = tune_lgbm(X, y, n_trials=2, n_folds=3, seed=42)
    assert params_1 == params_2


def test_fit_lgbm_returns_fitted_classifier(synthetic_data):
    X, y = synthetic_data
    model = fit_lgbm(X, y, FIXED_TEST_PARAMS)
    assert isinstance(model, LGBMClassifier)
    assert list(model.classes_) == [0, 1]


def test_predict_proba_positive_shape_and_range(synthetic_data):
    X, y = synthetic_data
    model = fit_lgbm(X, y, FIXED_TEST_PARAMS)
    proba = predict_proba_positive(model, X)
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
