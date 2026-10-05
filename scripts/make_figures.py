"""M5: render results/figures/{calibration,timing,shap_summary,permutation_importance}.{png,svg}.

Beyond-accuracy figures: calibration (reliability + Brier), fit/predict wall
time, SHAP for the tuned LightGBM, and permutation importance for TabPFN
(which has no native feature importances -- it's an in-context transformer,
not a tree ensemble). Reuses data.py/curve.py/models.py; the only fresh
computation here is refitting three models at a moderate size to get actual
per-row probabilities and a live TabPFN predict closure -- everything else
(hyperparameters, the money-chart timings) is read from committed artifacts.

Two different training sizes are used, both documented below, because a
single TabPFN predict call's cost sets the budget:

- Calibration + SHAP models: fit fresh at n_train=2000 (seed 0), all three
  models on the identical subsample so the reliability diagram is an
  apples-to-apples comparison. 2000 is the largest size where TabPFN's
  single predict on the full 4,000-row frozen eval set stays a few minutes
  on CPU (measured ~219s -- see results/learning_curve.csv, n_train=2000,
  seed=0). LightGBM is tuned FRESH at this size (review notes B-2: this figure
  used to reuse results/lgbm_baseline.json's full-train-tuned params, "a
  knowingly mis-tuned configuration" -- TabPFN and logreg were already
  honest per-size configs, LightGBM was not). curve.run_curve tunes fresh
  at every size on the learning-size curve too, but never persists the
  winning params, so there is nothing to load for n_train=2000 from any
  committed artifact -- explain.tuned_lgbm_for_calibration re-runs the same
  recipe (10-trial, 3-fold Optuna search, matching scripts/run_curve.py's
  LGBM_N_TRIALS/LGBM_N_FOLDS) on this exact subsample instead. logreg is the
  same untuned linear reference used everywhere else in this project.
- TabPFN permutation importance: a SEPARATE, smaller run at n_train=200 with
  a 200-row eval subsample and n_repeats=3. Permutation importance needs
  1 + n_features * n_repeats predict calls (43 for Adult's 14 features) --
  at n_train=2000 a single predict is ~219s, so 43 calls would take over
  2.5 hours on CPU. At n_train=200 a predict is ~3.4s, so the whole sweep
  finishes in a few minutes. This is honestly a smaller/rougher model than
  the calibration one; it exists to show *which* features TabPFN leans on,
  not to be the most accurate TabPFN configuration in this repo.

Fit/predict timing reuses results/learning_curve.csv's n_train=2000, seed=0
rows rather than re-measuring: those numbers already include the real cost
of getting a "tuned" LightGBM at that size (a fresh 10-trial Optuna search),
which is the honest fit-cost story -- refitting here with pre-tuned params
would understate it.
"""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import tabpfn

from tabular_showdown import explain, viz
from tabular_showdown.curve import (
    TABPFN_CATEGORICAL_PARAM_SUPPORTED,
    _encode_for_tabpfn,
    fit_predict_logreg,
    fit_predict_tabpfn,
    subsample_train,
)
from tabular_showdown.data import frozen_eval_set, load_adult, split_features_target
from tabular_showdown.metrics import classification_metrics

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CURVE_CSV_PATH = RESULTS_DIR / "learning_curve.csv"
META_PATH = FIGURES_DIR / "figures_meta.json"

CALIBRATION_N_TRAIN = 2000
CALIBRATION_SEED = 0
# Matches scripts/run_curve.py's LGBM_N_TRIALS/LGBM_N_FOLDS (and
# curve.fit_predict_lgbm_tuned's own defaults) -- the per-size tuning recipe
# this figure must match to be an honest comparison (review notes B-2).
CALIBRATION_LGBM_N_TRIALS = 10
CALIBRATION_LGBM_N_FOLDS = 3

PERMUTATION_N_TRAIN = 200
PERMUTATION_N_EVAL = 200
PERMUTATION_N_REPEATS = 3
PERMUTATION_SEED = 0


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=viz.SURFACE)
    fig.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight", facecolor=viz.SURFACE)
    print(f"wrote {FIGURES_DIR / f'{name}.png'}")
    print(f"wrote {FIGURES_DIR / f'{name}.svg'}")


def _tabpfn_predict_fn(X_train, y_train, X_eval, seed: int):
    """Fit a TabPFN classifier once, return a reusable predict_fn(X) -> proba
    plus the encoded eval frame it should be called with.

    Reuses curve._encode_for_tabpfn (the exact same ordinal encoding used by
    curve.fit_predict_tabpfn) so a live TabPFNClassifier and its fitted
    encoding stay available for repeated predict_proba calls -- required for
    permutation importance, which needs many predicts against one fit.
    """
    X_train_enc, X_eval_enc, cat_positions = _encode_for_tabpfn(X_train, X_eval)

    kwargs: dict = {"device": "cpu", "random_state": seed, "ignore_pretraining_limits": True}
    if TABPFN_CATEGORICAL_PARAM_SUPPORTED:
        kwargs["categorical_features_indices"] = cat_positions
    clf = tabpfn.TabPFNClassifier(**kwargs)
    clf.fit(X_train_enc, y_train.to_numpy())

    X_eval_enc_df = pd.DataFrame(X_eval_enc, columns=list(X_train.columns), index=X_eval.index)

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        return clf.predict_proba(X.to_numpy(dtype=float))[:, 1]

    return predict_fn, X_eval_enc_df


def main() -> None:
    train_df, test_df = load_adult(DATA_DIR)
    eval_df = frozen_eval_set(test_df)
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)

    # --- calibration + SHAP: all three models fit fresh at CALIBRATION_N_TRAIN ---
    print(f"fitting calibration/SHAP models at n_train={CALIBRATION_N_TRAIN} ...")
    X_sub, y_sub = subsample_train(X_train, y_train, n=CALIBRATION_N_TRAIN, seed=CALIBRATION_SEED)

    print(
        f"tuning LightGBM fresh at n_train={CALIBRATION_N_TRAIN} "
        f"({CALIBRATION_LGBM_N_TRIALS} Optuna trials, {CALIBRATION_LGBM_N_FOLDS}-fold CV, "
        "the same per-size recipe curve.py uses -- review notes B-2) ..."
    )
    lgbm_result = explain.tuned_lgbm_for_calibration(
        X_sub,
        y_sub,
        X_eval,
        n_trials=CALIBRATION_LGBM_N_TRIALS,
        n_folds=CALIBRATION_LGBM_N_FOLDS,
        seed=CALIBRATION_SEED,
    )
    lgbm_model = lgbm_result["model"]
    lgbm_proba = lgbm_result["proba"]

    logreg_result = fit_predict_logreg(X_sub, y_sub, X_eval, seed=CALIBRATION_SEED)

    print("fitting + predicting TabPFN (this is the slow step, ~3-4 min on CPU) ...")
    tabpfn_result = fit_predict_tabpfn(X_sub, y_sub, X_eval, seed=CALIBRATION_SEED)
    print(
        f"  tabpfn fit={tabpfn_result.fit_seconds:.1f}s "
        f"predict={tabpfn_result.predict_seconds:.1f}s"
    )

    curves = {
        "tabpfn": (y_eval, tabpfn_result.proba),
        "lgbm": (y_eval, lgbm_proba),
        "logreg": (y_eval, logreg_result.proba),
    }
    curve_metrics = {m: classification_metrics(y, p) for m, (y, p) in curves.items()}
    briers = {m: metrics["brier"] for m, metrics in curve_metrics.items()}
    log_losses = {m: metrics["log_loss"] for m, metrics in curve_metrics.items()}

    fig = viz.plot_calibration(curves, n_train=CALIBRATION_N_TRAIN, n_eval=len(X_eval))
    _save(fig, "calibration")

    # --- timing: reuse the already-run learning-size curve, same n_train/seed ---
    print("building timing chart from results/learning_curve.csv ...")
    curve_df = pd.read_csv(CURVE_CSV_PATH)
    same_n = curve_df["n_train"] == CALIBRATION_N_TRAIN
    same_seed = curve_df["seed"] == CALIBRATION_SEED
    point = curve_df[same_n & same_seed]
    timings = {
        row["model"]: {"fit_s": row["fit_s"], "predict_s": row["predict_s"]}
        for _, row in point.iterrows()
    }
    fig = viz.plot_timing(timings)
    _save(fig, "timing")

    # --- SHAP: the same lgbm model as the calibration figure ---
    print("computing TreeSHAP for the tuned LightGBM ...")
    shap_result = explain.shap_values_lgbm(lgbm_model, X_eval, sample_size=500, seed=0)
    fig = viz.plot_shap_summary(shap_result["values"], shap_result["feature_names"])
    _save(fig, "shap_summary")

    # --- permutation importance: a separate, smaller TabPFN run ---
    print(
        f"TabPFN permutation importance at a reduced n_train={PERMUTATION_N_TRAIN}, "
        f"n_eval={PERMUTATION_N_EVAL} (see module docstring for why) ..."
    )
    X_perm_train, y_perm_train = subsample_train(
        X_train, y_train, n=PERMUTATION_N_TRAIN, seed=PERMUTATION_SEED
    )
    X_perm_eval, y_perm_eval = subsample_train(
        X_eval, y_eval, n=PERMUTATION_N_EVAL, seed=PERMUTATION_SEED
    )
    predict_fn, X_perm_eval_enc = _tabpfn_predict_fn(
        X_perm_train, y_perm_train, X_perm_eval, seed=PERMUTATION_SEED
    )
    perm_dict = explain.permutation_importance_tabpfn(
        predict_fn,
        X_perm_eval_enc,
        y_perm_eval,
        n_repeats=PERMUTATION_N_REPEATS,
        seed=PERMUTATION_SEED,
    )
    fig = viz.plot_permutation_importance(perm_dict)
    _save(fig, "permutation_importance")

    meta = {
        "calibration_shap": {
            "n_train": CALIBRATION_N_TRAIN,
            "seed": CALIBRATION_SEED,
            "n_eval": len(X_eval),
            "brier": briers,
            "log_loss": log_losses,
            "lgbm_params_source": lgbm_result["params_source"],
            "lgbm_params": lgbm_result["params"],
        },
        "permutation_importance": {
            "n_train": PERMUTATION_N_TRAIN,
            "n_eval": PERMUTATION_N_EVAL,
            "n_repeats": PERMUTATION_N_REPEATS,
            "seed": PERMUTATION_SEED,
            "baseline_auc": perm_dict["baseline_auc"],
            "reason_for_smaller_run": (
                "permutation importance needs 1 + n_features * n_repeats predict "
                "calls; at n_train=2000 (the calibration size) a single TabPFN "
                "predict is ~219s, making the full sweep impractical on CPU"
            ),
        },
        "timing": {
            "source": "results/learning_curve.csv",
            "n_train": CALIBRATION_N_TRAIN,
            "seed": CALIBRATION_SEED,
        },
        "versions": {
            "python": platform.python_version(),
            "tabpfn": tabpfn.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"wrote {META_PATH}")
    print()
    print("Brier scores (lower = better calibrated):", {k: round(v, 4) for k, v in briers.items()})
    print("Log loss (lower = better calibrated):", {k: round(v, 4) for k, v in log_losses.items()})


if __name__ == "__main__":
    main()
