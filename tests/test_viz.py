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
from tabular_showdown.stats import paired_seed_comparison

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


def test_learning_curve_title_on_real_csv_tabpfn_never_caught_up_after_seed_expansion():
    """The 10/5-seed curve (commit 9dd7729, Task B4) changes which branch of
    learning_curve_title fires at n=5000, not just the numbers feeding it.
    REVIEW.md B-1's original 2-seed reading had lgbm's mean nose ahead of
    tabpfn's by ~0.0015 with disagreeing seed signs, which forced the
    noise-range branch ("crossover... between 2k and 5k"). With 5 seeds,
    lgbm's mean NEVER reaches tabpfn's mean at any measured size (200
    through 5000) -- there's no crossover left to bracket, not even a noisy
    one. Statistically n=5000 is still a tie (see test_stats.py's paired
    assertion), but the honest headline is "TabPFN never caught up within
    the measured range", not "somewhere between 2k and 5k". Flag for B5."""
    df = pd.read_csv(CURVE_CSV_PATH)
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)
    assert "never catches up" in title.lower()
    assert "overtakes" not in title.lower()  # no point crossover either
    assert "5k" in title


def test_learning_curve_title_never_catches_up_bound_is_derived_from_the_data_on_real_csv():
    """The bound named in the "never catches up through <n> rows" wording
    must be read off the data (the largest size where both models were
    measured, i.e. TabPFN's actual max n_train), not hardcoded -- same
    intent as the pre-expansion test this replaces (bounds come from data),
    but the shape changed: one bound now, not a two-sided bracket, since
    lgbm's mean never reaches tabpfn's mean anywhere in the 10/5-seed curve
    (see the sibling test above)."""
    df = pd.read_csv(CURVE_CSV_PATH)
    title = viz.learning_curve_title(df, tabpfn_stopped_at_pretrain_cap=False)

    tabpfn_max_n = int(df.loc[df["model"] == "tabpfn", "n_train"].max())
    assert tabpfn_max_n == 5000  # sanity: what "the largest we ran TabPFN on CPU" means today
    assert viz._fmt_n(tabpfn_max_n) in title
    assert "2k" not in title  # no two-sided bracket anymore -- confirm the old bound is gone


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


# --- verdict_markdown ---------------------------------------------------
#
# Orientation of the signed numbers (TabPFN - LightGBM, opposite of the raw
# b - a mean_diff) is pinned against the *real* committed CSV in
# tests/test_stats_table.py; these guard the label mapping and the low-n
# footnote logic on hermetic synthetic PairedResults.


def test_verdict_markdown_model_a_win_maps_to_named_verdict_and_positive_diff():
    # tabpfn (model_a) consistently ~0.10 above lgbm every seed -> raw
    # mean_diff (lgbm - tabpfn) is negative; the displayed TabPFN - LightGBM
    # diff must flip to positive, and the verdict must name TabPFN.
    a = [0.90, 0.91, 0.92, 0.905, 0.915, 0.895]
    b = [0.80, 0.813, 0.818, 0.806, 0.812, 0.799]
    res = {"1000": paired_seed_comparison(a, b, model_a="tabpfn", model_b="lgbm")}
    md = viz.verdict_markdown(res)

    assert "Mean ROC-AUC diff (TabPFN − LightGBM)" in md
    assert "| TabPFN wins |" in md
    assert "+0.0" in md  # positive gap displayed, sign flipped from raw b - a
    assert "†" not in md  # n=6, no Wilcoxon floor


def test_verdict_markdown_model_b_win_names_the_other_model():
    # Mirror image: lgbm (model_b) leads -> verdict names LightGBM.
    a = [0.80, 0.813, 0.818, 0.806, 0.812, 0.799]
    b = [0.90, 0.91, 0.92, 0.905, 0.915, 0.895]
    res = {"1000": paired_seed_comparison(a, b, model_a="tabpfn", model_b="lgbm")}
    md = viz.verdict_markdown(res)
    assert "| LightGBM wins |" in md


def test_verdict_markdown_tie_at_five_pairs_appends_floor_footnote():
    # Mixed-sign small gaps at n=5 -> CI includes 0 (tie), and n<6 triggers
    # the Wilcoxon-floor footnote.
    a = [0.913, 0.914, 0.912, 0.915, 0.913]
    b = [0.911, 0.916, 0.913, 0.910, 0.918]
    res = {"5000": paired_seed_comparison(a, b, model_a="tabpfn", model_b="lgbm")}
    md = viz.verdict_markdown(res)

    assert "| tie |" in md
    assert "†" in md
    assert "2^(1-n)" in md  # the floor footnote


def test_verdict_markdown_orders_sizes_numerically():
    # 500 vs 1000 (not 500 vs 5000): lexicographically "1000" < "500", so a
    # regression to plain sorted(str) would put 1,000 first and fail here --
    # 500/5000 would pass under either order and hide the bug.
    res = {
        "1000": paired_seed_comparison([0.9, 0.9], [0.91, 0.89], model_a="tabpfn", model_b="lgbm"),
        "500": paired_seed_comparison([0.9, 0.9], [0.91, 0.89], model_a="tabpfn", model_b="lgbm"),
    }
    md = viz.verdict_markdown(res)
    assert md.index("| 500 |") < md.index("| 1,000 |")


def test_verdict_markdown_raises_on_empty_results():
    # next(iter(...)) on an empty dict would raise a bare StopIteration --
    # surface an explicit, actionable error instead.
    with pytest.raises(ValueError, match="empty"):
        viz.verdict_markdown({})


def test_fmt_signed_shows_unsigned_zero_not_negative_zero():
    # A nonzero value that rounds to 0.0000 must not render as "−0.0000".
    assert viz._fmt_signed(-0.00001) == "0.0000"
    assert viz._fmt_signed(0.00001) == "0.0000"
    # Normal cases keep their explicit sign and typographic minus.
    assert viz._fmt_signed(0.0033) == "+0.0033"
    assert viz._fmt_signed(-0.0014) == "−0.0014"
