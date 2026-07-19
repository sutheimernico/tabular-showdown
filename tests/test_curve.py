"""Tests for the learning-size curve orchestration.

Uses small synthetic data shaped like the real Adult schema (same column
names/dtypes) so the curve helpers exercise the exact code paths they'll hit
on the real run, without needing the real data files or a long Optuna
search. TabPFN gets one tiny real-weights smoke test (30 rows, well under
5s on CPU, weights already cached) -- everything else about TabPFN stays out
of the unit tests since predicting on the real 4,000-row eval set is the
expensive part, reserved for scripts/run_curve.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_showdown.curve import (
    CSV_COLUMNS,
    SEEDS_BY_SIZE,
    compute_seed_spread,
    fit_predict_lgbm_tuned,
    fit_predict_logreg,
    fit_predict_tabpfn,
    rows_to_dataframe,
    run_curve,
    subsample_train,
)
from tabular_showdown.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

REAL_CURVE_CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "learning_curve.csv"


def _make_synthetic(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """n rows shaped like the cleaned Adult frame: same column names/dtypes.

    Categorical columns get a "None" choice too, mirroring the real
    missing-value categoricals (workclass, occupation, native_country).
    """
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(18, 90, size=n) for col in NUMERIC_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        choices = rng.choice([f"{col}_a", f"{col}_b", f"{col}_c", None], size=n)
        data[col] = pd.Categorical(choices)
    X = pd.DataFrame(data)
    y = pd.Series((rng.random(n) < 0.3).astype(int), name="income")
    return X, y


@pytest.fixture
def synthetic_train() -> tuple[pd.DataFrame, pd.Series]:
    return _make_synthetic(300, seed=1)


@pytest.fixture
def synthetic_eval() -> tuple[pd.DataFrame, pd.Series]:
    return _make_synthetic(80, seed=2)


# --- subsample_train ------------------------------------------------------


def test_subsample_train_is_deterministic(synthetic_train):
    X, y = synthetic_train
    X1, y1 = subsample_train(X, y, n=50, seed=0)
    X2, y2 = subsample_train(X, y, n=50, seed=0)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_subsample_train_size_and_stratification(synthetic_train):
    X, y = synthetic_train
    X_sub, y_sub = subsample_train(X, y, n=90, seed=0)
    assert len(X_sub) == len(y_sub) == 90
    assert abs(y.mean() - y_sub.mean()) <= 0.05


def test_subsample_train_different_seeds_give_different_rows(synthetic_train):
    X, y = synthetic_train
    X1, _ = subsample_train(X, y, n=50, seed=0)
    X2, _ = subsample_train(X, y, n=50, seed=1)
    assert not X1.index.equals(X2.index)


def test_subsample_train_n_ge_len_returns_everything(synthetic_train):
    X, y = synthetic_train
    X_sub, y_sub = subsample_train(X, y, n=len(X) + 10, seed=0)
    assert len(X_sub) == len(X)


# --- uniform fit_predict_* interface ---------------------------------------


def _assert_uniform_result(result, n_expected: int) -> None:
    assert result.proba.shape == (n_expected,)
    assert np.all((result.proba >= 0.0) & (result.proba <= 1.0))
    assert result.fit_seconds > 0
    assert result.predict_seconds > 0


def test_fit_predict_logreg_contract(synthetic_train, synthetic_eval):
    X_train, y_train = synthetic_train
    X_eval, _ = synthetic_eval
    result = fit_predict_logreg(X_train, y_train, X_eval, seed=0)
    _assert_uniform_result(result, len(X_eval))


def test_fit_predict_lgbm_tuned_contract(synthetic_train, synthetic_eval):
    X_train, y_train = synthetic_train
    X_eval, _ = synthetic_eval
    result = fit_predict_lgbm_tuned(X_train, y_train, X_eval, n_trials=2, n_folds=3, seed=0)
    _assert_uniform_result(result, len(X_eval))
    assert result.meta["tuned_fresh"] is True


def test_fit_predict_lgbm_tuned_reuses_given_params_without_tuning(synthetic_train, synthetic_eval):
    X_train, y_train = synthetic_train
    X_eval, _ = synthetic_eval
    params = {"objective": "binary", "verbosity": -1, "n_jobs": -1, "n_estimators": 20, "seed": 0}
    result = fit_predict_lgbm_tuned(X_train, y_train, X_eval, params=params, seed=0)
    _assert_uniform_result(result, len(X_eval))
    assert result.meta["tuned_fresh"] is False
    assert result.meta["tune_seconds"] == 0.0


def test_fit_predict_tabpfn_contract_tiny_real_weights():
    """One tiny real TabPFN fit (30 train rows) -- fast on CPU, weights cached."""
    X_train, y_train = _make_synthetic(30, seed=3)
    X_eval, _ = _make_synthetic(10, seed=4)
    result = fit_predict_tabpfn(X_train, y_train, X_eval, seed=0)
    _assert_uniform_result(result, len(X_eval))


# --- run_curve + CSV schema ------------------------------------------------


def test_run_curve_yields_uniform_rows(synthetic_train, synthetic_eval):
    X_train, y_train = synthetic_train
    X_eval, y_eval = synthetic_eval
    rows = list(
        run_curve(
            X_train,
            y_train,
            X_eval,
            y_eval,
            sizes=[50, 100],
            seeds_by_size={50: [0, 1], 100: [0]},
            models=("lgbm", "logreg"),
            lgbm_n_trials=2,
            lgbm_n_folds=3,
        )
    )
    # 2 models x (2 seeds @ n=50 + 1 seed @ n=100) = 6 rows
    assert len(rows) == 6
    for row in rows:
        assert set(row.keys()) == set(CSV_COLUMNS)
        assert row["model"] in {"lgbm", "logreg"}
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert row["fit_s"] > 0
        assert row["predict_s"] > 0


def test_run_curve_full_size_only_runs_lgbm_and_logreg(synthetic_train, synthetic_eval):
    X_train, y_train = synthetic_train
    X_eval, y_eval = synthetic_eval
    rows = list(
        run_curve(
            X_train,
            y_train,
            X_eval,
            y_eval,
            sizes=[],
            full_size=len(X_train),
            models=("tabpfn", "lgbm", "logreg"),
            lgbm_n_trials=2,
            lgbm_n_folds=3,
        )
    )
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"lgbm", "logreg"}
    assert all(r["n_train"] == len(X_train) for r in rows)


def test_run_curve_skip_keys_resumes_without_rerunning(synthetic_train, synthetic_eval):
    """skip_keys points are neither re-run nor re-yielded (crash-resume path)."""
    X_train, y_train = synthetic_train
    X_eval, y_eval = synthetic_eval
    skip = {("logreg", 50, 0), ("lgbm", 50, 0)}
    rows = list(
        run_curve(
            X_train,
            y_train,
            X_eval,
            y_eval,
            sizes=[50],
            seeds_by_size={50: [0]},
            models=("lgbm", "logreg"),
            lgbm_n_trials=2,
            lgbm_n_folds=3,
            skip_keys=skip,
        )
    )
    # both points at (50, 0) were pre-skipped, so nothing new is produced
    assert rows == []


def test_run_curve_respects_tabpfn_safety_valve(synthetic_train, synthetic_eval):
    """A near-zero time budget must skip every size after the first tabpfn point."""
    X_train, y_train = synthetic_train
    X_eval, y_eval = synthetic_eval
    rows = list(
        run_curve(
            X_train,
            y_train,
            X_eval,
            y_eval,
            sizes=[50, 100, 150],
            seeds_by_size={50: [0], 100: [0], 150: [0]},
            models=("tabpfn",),
            tabpfn_max_seconds=0.0,
        )
    )
    tabpfn_sizes = sorted(r["n_train"] for r in rows)
    assert tabpfn_sizes == [50]


def test_rows_to_dataframe_schema():
    rows = [
        {
            "model": "logreg",
            "n_train": 100,
            "seed": 0,
            "roc_auc": 0.8,
            "accuracy": 0.7,
            "log_loss": 0.5,
            "brier": 0.2,
            "fit_s": 0.1,
            "predict_s": 0.01,
        }
    ]
    df = rows_to_dataframe(rows)
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 1


def test_seeds_by_size_matches_design_spec():
    """Guards against silently drifting from the documented seed policy."""
    assert SEEDS_BY_SIZE[200] == [0, 1, 2]
    assert SEEDS_BY_SIZE[500] == [0, 1, 2]
    assert SEEDS_BY_SIZE[1000] == [0, 1, 2]
    assert SEEDS_BY_SIZE[2000] == [0, 1, 2]
    assert SEEDS_BY_SIZE[5000] == [0, 1]
    assert SEEDS_BY_SIZE[10000] == [0]


# --- compute_seed_spread ----------------------------------------------------


def test_compute_seed_spread_basic_stats():
    df = pd.DataFrame(
        {
            "model": ["lgbm", "lgbm", "lgbm", "tabpfn", "tabpfn"],
            "n_train": [200, 200, 200, 200, 200],
            "seed": [0, 1, 2, 0, 1],
            "roc_auc": [0.80, 0.82, 0.78, 0.90, 0.92],
        }
    )
    spread = compute_seed_spread(df)
    lgbm = spread["200"]["lgbm"]
    assert lgbm["n_seeds"] == 3
    assert lgbm["min"] == pytest.approx(0.78)
    assert lgbm["max"] == pytest.approx(0.82)
    assert lgbm["mean"] == pytest.approx(0.80)
    expected_std = float(np.std([0.80, 0.82, 0.78]))  # population std (ddof=0), not sample std
    assert lgbm["std"] == pytest.approx(expected_std)

    tabpfn = spread["200"]["tabpfn"]
    assert tabpfn["n_seeds"] == 2


def test_compute_seed_spread_single_seed_size_has_zero_std_not_nan():
    df = pd.DataFrame({"model": ["lgbm"], "n_train": [10000], "seed": [0], "roc_auc": [0.92]})
    spread = compute_seed_spread(df)
    point = spread["10000"]["lgbm"]
    assert point["n_seeds"] == 1
    assert point["std"] == 0.0
    assert point["min"] == point["max"] == point["mean"] == pytest.approx(0.92)


def test_compute_seed_spread_is_json_serializable():
    df = pd.DataFrame(
        {"model": ["lgbm", "tabpfn"], "n_train": [200, 200], "seed": [0, 0], "roc_auc": [0.8, 0.9]}
    )
    spread = compute_seed_spread(df)
    json.dumps(spread)  # must not raise -- keys are strings, values are plain floats/ints


def test_compute_seed_spread_on_real_curve_csv_matches_review_b1_numbers():
    """Anchors against REVIEW.md B-1's hand-recomputed lgbm@5000 numbers:
    mean gap ~0.0015 vs tabpfn, lgbm's own seed spread ~0.0093."""
    df = pd.read_csv(REAL_CURVE_CSV_PATH)
    spread = compute_seed_spread(df)

    lgbm_5k = spread["5000"]["lgbm"]
    assert lgbm_5k["n_seeds"] == 2
    assert lgbm_5k["mean"] == pytest.approx(0.91524, abs=1e-4)
    assert lgbm_5k["min"] == pytest.approx(0.910595, abs=1e-5)
    assert lgbm_5k["max"] == pytest.approx(0.919883, abs=1e-5)
    assert (lgbm_5k["max"] - lgbm_5k["min"]) == pytest.approx(0.0093, abs=1e-4)

    tabpfn_5k = spread["5000"]["tabpfn"]
    mean_gap = lgbm_5k["mean"] - tabpfn_5k["mean"]
    assert mean_gap == pytest.approx(0.0015, abs=1e-4)


def test_compute_seed_spread_missing_model_at_size_is_absent_not_crashed():
    """TabPFN has no rows above n=5000 in the real CSV (compute valve) -- it
    must simply be absent from that size's dict, not a crash or a fake 0."""
    df = pd.read_csv(REAL_CURVE_CSV_PATH)
    spread = compute_seed_spread(df)
    assert "tabpfn" not in spread["10000"]
    assert "lgbm" in spread["10000"]
    assert spread["10000"]["lgbm"]["n_seeds"] == 1
    assert spread["10000"]["lgbm"]["std"] == 0.0
