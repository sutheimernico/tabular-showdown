"""Tuned LightGBM baseline.

Tuning (Optuna) and fitting are kept as separate functions: M4's
learning-size curve refits the same tuned hyperparameters across many
subsample sizes and must not re-run the Optuna search each time.
"""

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Params that are never tuned, always fixed for every trial and the final fit.
_FIXED_PARAMS = {
    "objective": "binary",
    "verbosity": -1,
    "n_jobs": -1,
}


def _suggest_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def _make_objective(
    X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = {**_FIXED_PARAMS, **_suggest_params(trial), "seed": seed}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_aucs = []
        for train_idx, val_idx in skf.split(X, y):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[val_idx], proba))
        return float(np.mean(fold_aucs))

    return objective


def tune_lgbm(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 30, n_folds: int = 3, seed: int = 42
) -> dict:
    """Search LightGBM hyperparameters with Optuna (TPE), scored by mean CV ROC-AUC.

    Returns the winning hyperparameters merged with the fixed params, ready
    to pass straight into fit_lgbm.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_make_objective(X, y, n_folds, seed), n_trials=n_trials)
    return {**_FIXED_PARAMS, **study.best_params, "seed": seed}


def fit_lgbm(X: pd.DataFrame, y: pd.Series, params: dict) -> LGBMClassifier:
    """Fit an LGBMClassifier on (X, y) with the given params."""
    model = LGBMClassifier(**params)
    model.fit(X, y)
    return model


def predict_proba_positive(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of the positive class for each row."""
    return model.predict_proba(X)[:, 1]
