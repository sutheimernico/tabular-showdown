"""Tests for M5's beyond-accuracy helpers: calibration binning, TreeSHAP for
LightGBM, model-agnostic permutation importance, and the honestly per-size
tuned LightGBM used by the calibration/SHAP figures.

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
    tuned_lgbm_for_calibration,
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


# --- tuned_lgbm_for_calibration -----------------------------------------------
# review notes B-2: the calibration/SHAP figure used to refit results/lgbm_baseline
# .json's FULL-TRAIN-tuned params (32,561 rows, 30 trials) at n_train=2000 --
# "a knowingly mis-tuned configuration". curve.run_curve tunes fresh at every
# size but never persists the winning params (_to_row keeps metrics only), so
# there is nothing to load for n_train=2000 from any committed artifact. The
# honest fix is a fresh re-tune with curve.py's own per-size recipe.


@pytest.fixture
def synthetic_calibration_split():
    X_num, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=9)
    X = pd.DataFrame(X_num, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y, name="target")
    X_train = X.iloc[:300].reset_index(drop=True)
    y_train = y.iloc[:300].reset_index(drop=True)
    X_eval = X.iloc[300:].reset_index(drop=True)
    return X_train, y_train, X_eval


# Stand-in for results/lgbm_baseline.json's real full-train-tuned params --
# the bug this guards against is refitting THESE at a much smaller n_train.
FULL_TRAIN_BASELINE_PARAMS = {
    "objective": "binary",
    "verbosity": -1,
    "n_jobs": -1,
    "num_leaves": 20,
    "learning_rate": 0.04210720687271331,
    "n_estimators": 498,
    "min_child_samples": 75,
    "feature_fraction": 0.7821301115091406,
    "bagging_fraction": 0.9993347652056921,
    "bagging_freq": 2,
    "lambda_l1": 3.733016835407741e-07,
    "lambda_l2": 7.712017102370413e-05,
    "seed": 42,
}


def test_tuned_lgbm_for_calibration_does_not_reuse_full_train_baseline_params(
    synthetic_calibration_split,
):
    X_train, y_train, X_eval = synthetic_calibration_split

    result = tuned_lgbm_for_calibration(X_train, y_train, X_eval, n_trials=2, n_folds=3, seed=0)

    assert result["params"] != FULL_TRAIN_BASELINE_PARAMS


def test_tuned_lgbm_for_calibration_params_source_names_size_and_seed_not_baseline_reuse(
    synthetic_calibration_split,
):
    X_train, y_train, X_eval = synthetic_calibration_split

    result = tuned_lgbm_for_calibration(X_train, y_train, X_eval, n_trials=2, n_folds=3, seed=0)

    assert "reused, not re-tuned" not in result["params_source"]
    assert "lgbm_baseline.json" not in result["params_source"]
    assert f"n={len(X_train)}" in result["params_source"]
    assert "seed=0" in result["params_source"]


def test_tuned_lgbm_for_calibration_returns_model_consistent_with_its_own_proba(
    synthetic_calibration_split,
):
    """The SHAP figure explains this same returned model -- its predictions
    on X_eval must be exactly the proba this function also reports, so the
    calibration numbers and the SHAP attributions describe one model."""
    X_train, y_train, X_eval = synthetic_calibration_split

    result = tuned_lgbm_for_calibration(X_train, y_train, X_eval, n_trials=2, n_folds=3, seed=0)

    assert result["proba"].shape == (len(X_eval),)
    assert np.all((result["proba"] >= 0.0) & (result["proba"] <= 1.0))
    np.testing.assert_array_equal(
        result["model"].predict_proba(X_eval)[:, 1], result["proba"]
    )
