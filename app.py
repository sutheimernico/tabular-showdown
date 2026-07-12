"""M6: Streamlit demo -- TabPFN v2 vs. tuned LightGBM vs. untuned LogReg.

Chiefly a report viewer over the artifacts already produced by
scripts/run_lgbm_baseline.py, scripts/run_curve.py, and scripts/make_figures.py
(results/lgbm_baseline.json, results/learning_curve.{csv,png}, and
results/figures/*.png) -- reading committed results keeps the app fast and
avoids re-running TabPFN somewhere that has to stay responsive.

The one live-recompute feature is a training-size slider that refits LogReg
and LightGBM (both fast on CPU, cached by size) on an arbitrary subsample.
TabPFN is deliberately excluded from that slider: a single CPU predict on
the 4,000-row frozen eval set takes 1-4+ minutes at these sizes (see the
timing chart), which would make a slider unusable. Every TabPFN number in
this app comes from the pre-computed artifacts.

CSV upload is accepted but not processed: the tuned LightGBM hyperparameters
and TabPFN's categorical encoding are fitted to the Adult census schema
specifically (14 named columns), and building a second pipeline that adapts
to an arbitrary uploaded schema is out of scope for this demo (YAGNI).

Run: uv run streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from tabular_showdown import viz
from tabular_showdown.curve import fit_predict_lgbm_tuned, fit_predict_logreg, subsample_train
from tabular_showdown.data import frozen_eval_set, load_adult, split_features_target
from tabular_showdown.metrics import classification_metrics

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

st.set_page_config(page_title="Tabular Showdown: TabPFN v2 vs. tuned LightGBM", layout="wide")


@st.cache_data
def _load_eval_set():
    train_df, test_df = load_adult(DATA_DIR)
    eval_df = frozen_eval_set(test_df)
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)
    return X_train, y_train, X_eval, y_eval


@st.cache_data
def _load_curve_csv() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "learning_curve.csv")


@st.cache_data
def _load_lgbm_params() -> dict:
    return json.loads((RESULTS_DIR / "lgbm_baseline.json").read_text())["params"]


@st.cache_data
def _metrics_at_size(n_train: int) -> dict[str, dict[str, float]]:
    """Mean metrics per model at n_train, averaged over the curve's
    subsample seeds -- read straight from results/learning_curve.csv, no
    recompute. Models with no row at this size (TabPFN above n=5000) are
    simply absent, not zero-filled."""
    df = _load_curve_csv()
    point = df[df["n_train"] == n_train]
    return {
        model: {
            "roc_auc": group["roc_auc"].mean(),
            "accuracy": group["accuracy"].mean(),
            "log_loss": group["log_loss"].mean(),
            "brier": group["brier"].mean(),
        }
        for model, group in point.groupby("model")
    }


@st.cache_data
def _live_refit(n_train: int, seed: int = 0) -> dict[str, dict[str, float]]:
    """Fit LogReg + LightGBM (baseline params, no re-tuning) fresh at an
    arbitrary n_train and score on the frozen eval set. TabPFN is
    deliberately excluded -- see module docstring."""
    X_train, y_train, X_eval, y_eval = _load_eval_set()
    X_sub, y_sub = subsample_train(X_train, y_train, n=n_train, seed=seed)
    params = _load_lgbm_params()

    lgbm_result = fit_predict_lgbm_tuned(X_sub, y_sub, X_eval, params=params, seed=seed)
    logreg_result = fit_predict_logreg(X_sub, y_sub, X_eval, seed=seed)

    return {
        "lgbm": {
            **classification_metrics(y_eval, lgbm_result.proba),
            "fit_s": lgbm_result.fit_seconds,
            "predict_s": lgbm_result.predict_seconds,
        },
        "logreg": {
            **classification_metrics(y_eval, logreg_result.proba),
            "fit_s": logreg_result.fit_seconds,
            "predict_s": logreg_result.predict_seconds,
        },
    }


st.title("Tabular Showdown: TabPFN v2 vs. tuned LightGBM")
st.caption(
    "A zero-training, in-context transformer vs. a properly tuned gradient booster, "
    "benchmarked on the UCI Adult census income dataset. See README.md for the full write-up."
)

st.header("The money chart")
st.image(
    str(RESULTS_DIR / "learning_curve.png"),
    caption=(
        "TabPFN v2 (zero training) wins the small-data regime; "
        "tuned LightGBM overtakes it by ~5k rows."
    ),
)

st.header("Metrics at a glance")
curve_sizes = sorted(int(n) for n in _load_curve_csv()["n_train"].unique())
n_train = st.select_slider("Training-set size", options=curve_sizes, value=2000)
metrics_table = viz.metrics_table(_metrics_at_size(n_train))
st.dataframe(metrics_table.set_index("model"), width="stretch")
if n_train > 5000:
    st.caption(
        "TabPFN has no row here: capped at n=5000 by the CPU compute valve "
        "(see the money chart's dashed cutoff line)."
    )
st.caption(f"From results/learning_curve.csv, mean over subsample seeds at n_train={n_train:,}.")

st.header("Calibration")
st.image(str(FIGURES_DIR / "calibration.png"))
st.caption(
    "All three models trained fresh at n_train=2,000 (not tied to the slider above) -- "
    "see results/figures/figures_meta.json for the exact run."
)

st.header("Prediction-time cost")
st.image(str(FIGURES_DIR / "timing.png"))
st.caption("Same n_train=2,000 models as the calibration figure; timings read from the curve run.")

st.header("What each model leans on")
col1, col2 = st.columns(2)
with col1:
    st.image(str(FIGURES_DIR / "shap_summary.png"))
    st.caption("TreeSHAP: exact for tree ensembles like LightGBM.")
with col2:
    st.image(str(FIGURES_DIR / "permutation_importance.png"))
    st.caption(
        "Model-agnostic permutation importance (TabPFN has no native feature "
        "importances). Computed at a reduced n_train=200 to keep the "
        "1 + features x repeats predict calls fast on CPU -- see figures_meta.json."
    )

st.header("Live recompute (LogReg + LightGBM only)")
st.caption(
    "TabPFN is excluded here on purpose: a single CPU predict on the frozen eval set "
    "takes 1-4+ minutes at these sizes (see the timing chart above), which would make "
    "this slider unusable. Every TabPFN number in this app comes from the pre-computed run."
)
live_n = st.slider("n_train", min_value=100, max_value=5000, value=1000, step=100)
if st.button("Fit live at this size"):
    with st.spinner(f"Fitting LogReg + LightGBM on {live_n:,} rows..."):
        live_metrics = _live_refit(live_n)
    st.dataframe(viz.metrics_table(live_metrics).set_index("model"), width="stretch")

st.header("Upload your own CSV")
uploaded = st.file_uploader("CSV upload", type="csv")
if uploaded is not None:
    st.info(
        "Upload is accepted but not processed in this demo. The tuned LightGBM "
        "hyperparameters and TabPFN's categorical encoding are fitted specifically to "
        "the Adult census schema (14 named columns) and won't generalize to an arbitrary "
        "CSV without re-tuning the whole pipeline -- out of scope here. Every result on "
        "this page uses the built-in Adult dataset regardless of what you upload."
    )

st.header("Honest limits")
st.markdown(
    "- TabPFN is capped at n=5,000 in this benchmark by an 8-minute-per-point **CPU "
    "compute valve** -- distinct from TabPFN v2's ~10,000-row pretraining limit, which "
    "this run never reached.\n"
    "- Adult census income is a well-studied, near-\"solved\" dataset; the novelty here "
    "is the **comparison method** (learning-size curve + calibration + explanations), "
    "not state-of-the-art accuracy.\n"
    "- Every number in this app is scored on the **same frozen 4,000-row eval subsample** "
    "of adult.test -- comparisons across models are fair, but it is one split, not a "
    "cross-validated estimate.\n"
    "- LogReg is an **untuned** linear reference, included as a floor, not a serious "
    "competitor.\n"
    "- AutoGluon, planned in PLAN.md as an AutoML reference point, is **deferred**, not "
    "implemented.\n"
)
