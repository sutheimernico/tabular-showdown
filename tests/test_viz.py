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
CURVE_META_PATH = Path(__file__).resolve().parent.parent / "results" / "learning_curve_meta.json"


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


# --- learning_curve_title (REVIEW.md B-1: noise-aware crossover wording) ----


def test_learning_curve_title_raises_when_no_lgbm_rows():
    """Guard added alongside the B1 review: with zero lgbm rows, lgbm_sizes
    would be empty and max(lgbm_sizes) would raise an opaque ValueError deep
    inside the function -- assert the clear, early guard message instead."""
    df = pd.DataFrame(
        {
            "model": ["tabpfn", "tabpfn"],
            "n_train": [1000, 1000],
            "seed": [0, 1],
            "roc_auc": [0.90, 0.91],
        }
    )
    with pytest.raises(ValueError, match="lgbm"):
        viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)


def test_learning_curve_title_on_real_csv_is_a_range_not_a_point_crossover():
    """At n=5000 the lgbm-vs-tabpfn mean gap (~0.0015) is far smaller than
    lgbm's own seed spread (~0.0093) and the two seeds disagree in sign
    (REVIEW.md B-1) -- the title must not assert a point crossover there."""
    df = pd.read_csv(CURVE_CSV_PATH)
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)
    assert "~5,000" not in title
    assert "~5k" not in title
    assert "noise" in title.lower() or "tied" in title.lower()


def test_learning_curve_title_range_bounds_come_from_the_data_on_real_csv():
    """Bracket must be [last size TabPFN leads with consistent seed sign,
    first later size that is no longer a confirmed TabPFN win] -- 2k and 5k
    for the currently committed CSV -- not the hardcoded "2k"/"10k" prose
    from REVIEW.md itself."""
    df = pd.read_csv(CURVE_CSV_PATH)
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)
    assert "2k" in title
    assert "5k" in title


def test_learning_curve_title_asserts_point_crossover_when_gap_beats_spread():
    """Positive control: a clean, seed-consistent gap larger than either
    model's own seed spread must still produce a point-crossover statement
    -- the noise guard should not swallow real signal."""
    df = pd.DataFrame(
        {
            "model": ["tabpfn", "tabpfn", "lgbm", "lgbm"],
            "n_train": [1000, 1000, 1000, 1000],
            "seed": [0, 1, 0, 1],
            "roc_auc": [0.80, 0.81, 0.90, 0.91],
        }
    )
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)
    assert "overtakes" in title
    assert "1k" in title


def test_learning_curve_title_never_catches_up_when_tabpfn_always_ahead():
    df = pd.DataFrame(
        {
            "model": ["tabpfn", "tabpfn", "lgbm", "lgbm"],
            "n_train": [1000, 1000, 1000, 1000],
            "seed": [0, 1, 0, 1],
            "roc_auc": [0.90, 0.91, 0.80, 0.81],
        }
    )
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=True)
    assert "never" in title.lower()
    assert "pretraining cap" in title.lower()


# --- plot_learning_curve (absorbed from scripts/plot_curve.py) --------------


def test_plot_learning_curve_returns_figure_with_one_line_per_model():
    df = pd.read_csv(CURVE_CSV_PATH)
    meta = json.loads(CURVE_META_PATH.read_text())
    fig = viz.plot_learning_curve(df, meta)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 4  # tabpfn, lgbm, logreg series + 1 axvline (tabpfn cutoff marker)
    assert ax.get_xscale() == "log"


def test_plot_learning_curve_title_is_on_the_axes():
    df = pd.read_csv(CURVE_CSV_PATH)
    meta = json.loads(CURVE_META_PATH.read_text())
    fig = viz.plot_learning_curve(df, meta)
    title = fig.axes[0].get_title(loc="left")
    assert title == viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)


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
