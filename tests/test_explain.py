"""Tests for M5's beyond-accuracy helpers: calibration binning, TreeSHAP for
LightGBM, and model-agnostic permutation importance.

TabPFN itself stays out of these tests (see test_curve.py for the one real
tiny TabPFN smoke test in this repo): permutation_importance_tabpfn only
needs a predict_fn callable, so a synthetic, deterministic predict_fn is
enough to exercise the real ranking logic without any model-fitting cost.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tabular_showdown.explain import (
    calibration_data,
    permutation_importance_tabpfn,
    shap_values_lgbm,
)
from tabular_showdown.models import fit_lgbm

# --- calibration_data -------------------------------------------------------


def test_calibration_data_perfectly_calibrated_input_observed_matches_predicted():
    """Hand-checked: 10 rows per predicted level, positive count == p * 10
    exactly, so observed frequency must equal the predicted level exactly."""
    proba, y_true = [], []
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        n_pos = round(p * 10)
        proba += [p] * 10
        y_true += [1] * n_pos + [0] * (10 - n_pos)

    data = calibration_data(y_true, proba, n_bins=5)

    assert len(data["bin_centers"]) == 5
    assert data["counts"] == [10, 10, 10, 10, 10]
    for predicted, observed in zip(data["mean_predicted"], data["observed_freq"], strict=True):
        assert observed == pytest.approx(predicted, abs=1e-9)


def test_calibration_data_drops_empty_bins():
    # All predictions land in the first and last of 10 bins; the 8 middle
    # bins have nothing in them and must not appear in the output.
    proba = [0.02, 0.05, 0.08, 0.95]
    y_true = [0, 0, 1, 1]
    data = calibration_data(proba=proba, y_true=y_true, n_bins=10)
    assert len(data["bin_centers"]) == 2
    assert sum(data["counts"]) == len(proba)


def test_calibration_data_returns_expected_keys():
    data = calibration_data([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7], n_bins=4)
    assert set(data.keys()) == {"bin_centers", "mean_predicted", "observed_freq", "counts"}


# --- shap_values_lgbm --------------------------------------------------------


@pytest.fixture
def synthetic_lgbm():
    X_num, y = make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)
    X = pd.DataFrame(X_num, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y, name="target")
    params = {"objective": "binary", "verbosity": -1, "n_jobs": -1, "seed": 0, "n_estimators": 30}
    model = fit_lgbm(X, y, params)
    return model, X


def test_shap_values_lgbm_one_value_per_feature(synthetic_lgbm):
    model, X = synthetic_lgbm
    result = shap_values_lgbm(model, X, sample_size=50, seed=0)
    assert result["values"].shape == (50, X.shape[1])
    assert result["feature_names"] == list(X.columns)
    assert len(result["X"]) == 50


def test_shap_values_lgbm_uses_all_rows_when_sample_size_exceeds_data(synthetic_lgbm):
    model, X = synthetic_lgbm
    result = shap_values_lgbm(model, X, sample_size=10_000, seed=0)
    assert result["values"].shape == (len(X), X.shape[1])


def test_shap_values_lgbm_is_deterministic_given_seed(synthetic_lgbm):
    model, X = synthetic_lgbm
    r1 = shap_values_lgbm(model, X, sample_size=50, seed=0)
    r2 = shap_values_lgbm(model, X, sample_size=50, seed=0)
    np.testing.assert_array_equal(r1["values"], r2["values"])


# --- permutation_importance_tabpfn -------------------------------------------


def test_permutation_importance_ranks_informative_feature_over_noise():
    """A deterministic predict_fn that only ever looks at one column: shuffling
    that column must hurt AUC, shuffling the untouched noise column must not
    move it at all -- the cleanest possible ground truth for this ranking."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"informative": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = pd.Series((X["informative"] > 0).astype(int))

    def predict_fn(X_in: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-5 * X_in["informative"].to_numpy()))

    result = permutation_importance_tabpfn(predict_fn, X, y, n_repeats=3, seed=0)
    importances = result["importances"]

    assert result["baseline_auc"] > 0.9
    assert importances["noise"]["mean_drop"] == pytest.approx(0.0, abs=1e-9)
    assert importances["informative"]["mean_drop"] > 0.1
    assert importances["informative"]["mean_drop"] > importances["noise"]["mean_drop"]


def test_permutation_importance_returns_all_columns():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    def predict_fn(X_in: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-(X_in["a"] + X_in["b"]).to_numpy()))

    result = permutation_importance_tabpfn(predict_fn, X, y, n_repeats=2, seed=0)
    assert set(result["importances"].keys()) == {"a", "b", "c"}
    assert result["n_repeats"] == 2
