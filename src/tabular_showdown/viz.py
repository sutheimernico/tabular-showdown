"""House-style chart builders for the M5 "beyond accuracy" figures.

Same house style as scripts/plot_curve.py (the money chart): warm-paper
surface, ink text tokens, muted hairline grid, hidden top/right spines, one
y-axis per chart, and a title that states the finding as a sentence rather
than a generic axis label. Colors are fixed per model everywhere in this
project -- reused here unchanged so every figure in the write-up reads as
one system.

The four plot_* builders return a matplotlib Figure; metrics_table returns a
plain DataFrame for the app's side-by-side table. scripts/make_figures.py
saves the figures to results/figures/*.{png,svg}; app.py renders the same
Figure objects live via st.pyplot -- one chart implementation, not a
separate matplotlib and plotly version, which would double the maintenance
surface for a local demo that isn't kept running (see app.py's docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tabular_showdown.explain import calibration_data
from tabular_showdown.metrics import classification_metrics

MODEL_COLORS = {"tabpfn": "#2a78d6", "lgbm": "#1baf7a", "logreg": "#eda100"}
MODEL_LABELS = {"tabpfn": "TabPFN v2", "lgbm": "LightGBM (tuned)", "logreg": "LogReg (untuned)"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Side-by-side metrics table for the app: one row per model, columns
    roc_auc/accuracy/log_loss/brier (plus whatever else is in each model's
    metrics dict, e.g. fit_s/predict_s), model names in house display form.

    results: {model_key: {metric_name: value, ...}}, same model keys used
    everywhere else (tabpfn/lgbm/logreg).
    """
    rows = [{"model": MODEL_LABELS.get(m, m), **metrics} for m, metrics in results.items()]
    return pd.DataFrame(rows)


def _new_axes(figsize: tuple[float, float] = (9, 5.5)) -> tuple[Figure, Axes]:
    """One Figure/Axes pair with the shared surface + spine styling applied."""
    fig = Figure(figsize=figsize, facecolor=SURFACE)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    return fig, ax


def _title_and_subtitle(ax: Axes, title: str, subtitle: str | None) -> None:
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left", pad=24)
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=9.5,
            va="bottom",
        )


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    n_train: int,
    n_eval: int,
    dataset: str = "Adult census income",
    n_bins: int = 10,
) -> Figure:
    """Reliability diagram: mean predicted probability vs. observed frequency
    per bin, one line per model, plus the unattainable-perfection diagonal
    as a muted reference. The title names whichever model has the lowest
    Brier score on this same eval set -- computed from the data passed in,
    not asserted.

    curves: {model_key: (y_true, proba)} -- y_true is typically the same
    frozen eval labels for every model, proba is that model's prediction on
    it. n_train/n_eval/dataset are for the subtitle only (which run this is).
    """
    fig, ax = _new_axes()

    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.5, linestyle=(0, (3, 3)), zorder=1)

    briers = {}
    for model, (y_true, proba) in curves.items():
        briers[model] = classification_metrics(y_true, proba)["brier"]
        data = calibration_data(y_true, proba, n_bins=n_bins)
        # Brier score in the legend label itself -- more compact than a
        # separate summary line, and keeps the per-model number next to the
        # series it belongs to.
        label = f"{MODEL_LABELS.get(model, model)} (Brier {briers[model]:.3f})"
        ax.plot(
            data["mean_predicted"],
            data["observed_freq"],
            color=MODEL_COLORS.get(model, INK_SECONDARY),
            linewidth=2,
            marker="o",
            markersize=7,
            label=label,
            zorder=2,
        )

    best_model = min(briers, key=lambda m: briers[m])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability (per bin)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("observed positive frequency", color=INK_MUTED, fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    _title_and_subtitle(
        ax,
        f"{MODEL_LABELS.get(best_model, best_model)} is best-calibrated "
        f"(lowest Brier score: {briers[best_model]:.3f})",
        f"{dataset} · frozen {n_eval:,}-row eval set · all models trained on {n_train:,} rows",
    )
    return fig


def plot_timing(timings: dict[str, dict[str, float]]) -> Figure:
    """Fit vs. predict wall-time per model, grouped bars, one log-scaled
    y-axis (seconds) -- fit and predict costs span several orders of
    magnitude here (TabPFN fit ~0.1s vs. predict ~3-4 min; LightGBM the
    reverse), so a linear axis would flatten one side to invisible.

    timings: {model_key: {"fit_s": ..., "predict_s": ...}}.
    """
    fig, ax = _new_axes()

    models = [m for m in ("tabpfn", "lgbm", "logreg") if m in timings]
    x = np.arange(len(models))
    width = 0.35

    fit_vals = [max(timings[m]["fit_s"], 1e-3) for m in models]
    predict_vals = [max(timings[m]["predict_s"], 1e-3) for m in models]

    bar_colors = [MODEL_COLORS[m] for m in models]
    ax.bar(x - width / 2, fit_vals, width, color=bar_colors, alpha=0.55, label="fit")
    ax.bar(x + width / 2, predict_vals, width, color=bar_colors, alpha=1.0, label="predict")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("wall time, seconds (log scale)", color=INK_MUTED, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    if "tabpfn" in timings and "lgbm" in timings:
        predict_ratio = timings["tabpfn"]["predict_s"] / max(timings["lgbm"]["predict_s"], 1e-3)
        title = (
            f"TabPFN needs virtually no fit time but predicts "
            f"~{predict_ratio:,.0f}x slower than tuned LightGBM"
        )
    else:
        title = "Fit vs. predict wall-time per model"
    _title_and_subtitle(ax, title, "same trained models as the calibration figure")
    return fig


def _sorted_bar_data(
    labels: list[str], values: list[float], top_n: int
) -> tuple[list[str], list[float]]:
    order = np.argsort(values)[::-1][:top_n]
    return [labels[i] for i in order], [values[i] for i in order]


def plot_shap_summary(shap_values: np.ndarray, feature_names: list[str], top_n: int = 10) -> Figure:
    """SHAP bar chart (mean |value| per feature) for the tuned LightGBM,
    sorted descending, top_n features. This is exact TreeSHAP, not an
    approximation -- LightGBM is a tree ensemble, so shap.TreeExplainer
    reads attributions straight off the trees.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    labels, values = _sorted_bar_data(feature_names, list(mean_abs), top_n)

    fig, ax = _new_axes()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=MODEL_COLORS["lgbm"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9.5)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel("mean |SHAP value| (impact on predicted log-odds)", color=INK_MUTED, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    _title_and_subtitle(
        ax,
        f"LightGBM leans hardest on \"{labels[0]}\" (TreeSHAP, mean |value|)",
        "tuned LightGBM · same training size as the calibration figure",
    )
    return fig


def plot_permutation_importance(perm_dict: dict, top_n: int = 10) -> Figure:
    """TabPFN permutation-importance bar chart: mean ROC-AUC drop when each
    feature is shuffled, error bars = std across repeats. TabPFN has no
    native feature importances (it's not a tree ensemble), so this
    model-agnostic measure is the honest substitute -- see
    explain.permutation_importance_tabpfn.
    """
    importances = perm_dict["importances"]
    labels = list(importances.keys())
    means = [importances[k]["mean_drop"] for k in labels]
    stds = [importances[k]["std_drop"] for k in labels]

    order = np.argsort(means)[::-1][:top_n]
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = _new_axes()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds, color=MODEL_COLORS["tabpfn"], ecolor=INK_MUTED, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("mean ROC-AUC drop when shuffled", color=INK_MUTED, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    _title_and_subtitle(
        ax,
        f"TabPFN relies most on \"{labels[0]}\" "
        f"(permutation importance, baseline AUC {perm_dict['baseline_auc']:.3f})",
        f"model-agnostic permutation importance · {perm_dict['n_repeats']} repeats/feature",
    )
    return fig
