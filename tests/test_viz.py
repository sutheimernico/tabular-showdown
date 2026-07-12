"""Tests for the house-style chart builders and the metrics-table helper.

Real assertions on structure (Figure returned, axes/bars present, table
shape), not pixel comparisons. Two tests fit the real tuned LightGBM on the
real Adult data (fast, no tuning, no TabPFN) so at least one chart per model
family is exercised against a real artifact rather than only synthetic
data; TabPFN itself stays out of these tests (see scripts/make_figures.py
for the real, slower TabPFN run that produces results/figures/*).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from tabular_showdown import explain, viz
from tabular_showdown.data import frozen_eval_set, load_adult, split_features_target
from tabular_showdown.models import fit_lgbm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LGBM_BASELINE_PATH = Path(__file__).resolve().parent.parent / "results" / "lgbm_baseline.json"
CURVE_CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "learning_curve.csv"


@pytest.fixture(scope="module")
def real_eval() -> tuple[pd.DataFrame, pd.Series]:
    _, test_df = load_adult(DATA_DIR)
    eval_df = frozen_eval_set(test_df)
    return split_features_target(eval_df)


@pytest.fixture(scope="module")
def real_lgbm_model_and_shap(real_eval):
    """Fit the real tuned LightGBM (reusing lgbm_baseline.json params, no
    fresh tuning) on the real frozen eval set's own rows -- fast, and enough
    to get a real fitted model + real TreeSHAP values for the chart test."""
    X_eval, y_eval = real_eval
    params = json.loads(LGBM_BASELINE_PATH.read_text())["params"]
    model = fit_lgbm(X_eval, y_eval, params)
    shap_result = explain.shap_values_lgbm(model, X_eval, sample_size=200, seed=0)
    return shap_result


# --- plot_calibration --------------------------------------------------------


def _synthetic_curves(seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = 300
    y = (rng.random(n) < 0.3).astype(int)
    base = np.clip(y * 0.6 + rng.random(n) * 0.4, 0.01, 0.99)
    return {
        "tabpfn": (y, base),
        "lgbm": (y, np.clip(base * 0.9, 0.01, 0.99)),
        "logreg": (y, np.clip(base * 1.05, 0.01, 0.99)),
    }


def test_plot_calibration_returns_figure_with_one_line_per_model():
    curves = _synthetic_curves()
    fig = viz.plot_calibration(curves, n_train=2000, n_eval=300)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # one Line2D per model plus the diagonal reference line
    assert len(ax.lines) == len(curves) + 1


def test_plot_calibration_title_names_lowest_brier_model():
    from tabular_showdown.metrics import classification_metrics

    curves = _synthetic_curves()
    fig = viz.plot_calibration(curves, n_train=2000, n_eval=300)

    briers = {m: classification_metrics(y, p)["brier"] for m, (y, p) in curves.items()}
    best = min(briers, key=lambda m: briers[m])
    assert viz.MODEL_LABELS[best] in fig.axes[0].get_title(loc="left")


# --- plot_timing --------------------------------------------------------


def test_plot_timing_on_real_learning_curve_csv():
    """Reuses the real committed curve timings, same as scripts/make_figures.py."""
    df = pd.read_csv(CURVE_CSV_PATH)
    point = df[(df["n_train"] == 2000) & (df["seed"] == 0)]
    timings = {
        row["model"]: {"fit_s": row["fit_s"], "predict_s": row["predict_s"]}
        for _, row in point.iterrows()
    }
    fig = viz.plot_timing(timings)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 2 bars (fit, predict) per model
    assert len(ax.patches) == 2 * len(timings)


def test_plot_timing_handles_missing_model_gracefully():
    fig = viz.plot_timing({"lgbm": {"fit_s": 1.0, "predict_s": 0.1}})
    assert isinstance(fig, Figure)


# --- plot_shap_summary (real LightGBM + real Adult data) --------------------


def test_plot_shap_summary_on_real_lgbm_baseline(real_lgbm_model_and_shap):
    shap_result = real_lgbm_model_and_shap
    fig = viz.plot_shap_summary(shap_result["values"], shap_result["feature_names"], top_n=10)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == 10
    assert len(ax.get_yticklabels()) == 10


def test_plot_shap_summary_respects_top_n_when_fewer_features(real_lgbm_model_and_shap):
    shap_result = real_lgbm_model_and_shap
    n_features = len(shap_result["feature_names"])
    fig = viz.plot_shap_summary(shap_result["values"], shap_result["feature_names"], top_n=1000)
    assert len(fig.axes[0].patches) == n_features


# --- plot_permutation_importance --------------------------------------------


def test_plot_permutation_importance_returns_figure():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = pd.Series((X["a"] > 0).astype(int))

    def predict_fn(X_in: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-5 * X_in["a"].to_numpy()))

    perm_dict = explain.permutation_importance_tabpfn(predict_fn, X, y, n_repeats=2, seed=0)
    fig = viz.plot_permutation_importance(perm_dict)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 3
    assert "a" in fig.axes[0].get_title(loc="left")  # most important feature named in the title


# --- metrics_table ------------------------------------------------------


def test_metrics_table_shape():
    results = {
        "tabpfn": {"roc_auc": 0.90, "accuracy": 0.85, "log_loss": 0.31, "brier": 0.10},
        "lgbm": {"roc_auc": 0.93, "accuracy": 0.87, "log_loss": 0.27, "brier": 0.09},
        "logreg": {"roc_auc": 0.91, "accuracy": 0.85, "log_loss": 0.32, "brier": 0.10},
    }
    table = viz.metrics_table(results)
    assert table.shape == (3, 5)  # model + 4 metrics
    assert set(table.columns) == {"model", "roc_auc", "accuracy", "log_loss", "brier"}
    assert set(table["model"]) == {"TabPFN v2", "LightGBM (tuned)", "LogReg (untuned)"}
