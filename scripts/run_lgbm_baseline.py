"""M2: tuned LightGBM baseline on Adult — the real run.

Tunes on the full train set (never touches the frozen eval set), fits the
winning params on full train, then scores once on the frozen eval set.
Writes results/lgbm_baseline.json for use as the baseline reference point
in later milestones (e.g. the learning-size curve).
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tabular_showdown.data import frozen_eval_set, load_adult, split_features_target
from tabular_showdown.metrics import classification_metrics
from tabular_showdown.models import fit_lgbm, predict_proba_positive, tune_lgbm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "lgbm_baseline.json"

SEED = 42
N_TRIALS = 30
N_FOLDS = 3


def _cv_auc(X, y, params: dict, n_folds: int, seed: int) -> float:
    """Report the tuned params' mean CV ROC-AUC via the public fit/predict API.

    Uses the same StratifiedKFold(seed) scheme as tune_lgbm's internal
    objective, so this reproduces the winning trial's score without reaching
    into Optuna's study internals.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = fit_lgbm(X.iloc[train_idx], y.iloc[train_idx], params)
        proba = predict_proba_positive(model, X.iloc[val_idx])
        fold_aucs.append(roc_auc_score(y.iloc[val_idx], proba))
    return float(np.mean(fold_aucs))


def main() -> None:
    train_df, test_df = load_adult(DATA_DIR)
    eval_df = frozen_eval_set(test_df)

    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)

    tune_start = time.perf_counter()
    params = tune_lgbm(X_train, y_train, n_trials=N_TRIALS, n_folds=N_FOLDS, seed=SEED)
    tune_seconds = time.perf_counter() - tune_start

    cv_auc = _cv_auc(X_train, y_train, params, N_FOLDS, SEED)

    fit_start = time.perf_counter()
    model = fit_lgbm(X_train, y_train, params)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    proba = predict_proba_positive(model, X_eval)
    predict_seconds = time.perf_counter() - predict_start

    eval_metrics = classification_metrics(y_eval, proba)

    result = {
        "params": params,
        "cv_auc": cv_auc,
        "eval_metrics": eval_metrics,
        "timings": {
            "tune_seconds": tune_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        },
        "n_train": len(X_train),
        "n_eval": len(X_eval),
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("LightGBM baseline (Adult)")
    print(f"  n_train={result['n_train']}  n_eval={result['n_eval']}  seed={SEED}")
    print(f"  cv_auc        = {cv_auc:.4f}")
    print(f"  eval_roc_auc  = {eval_metrics['roc_auc']:.4f}")
    print(f"  eval_accuracy = {eval_metrics['accuracy']:.4f}")
    print(f"  eval_log_loss = {eval_metrics['log_loss']:.4f}")
    print(f"  eval_brier    = {eval_metrics['brier']:.4f}")
    print(f"  tune={tune_seconds:.1f}s  fit={fit_seconds:.1f}s  predict={predict_seconds:.3f}s")
    print(f"  wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
