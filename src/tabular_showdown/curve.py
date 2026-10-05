"""M3+M4: TabPFN v2 vs. tuned LightGBM vs. an untuned linear reference,
swept across training-set sizes -- the project's headline "money chart".

This module holds the reusable pieces: a stratified/seeded train-subsample
helper, one fit/predict function per model with a uniform
(proba, fit_seconds, predict_seconds) return, and run_curve(), the
orchestration generator that sweeps size x seed x model and yields one
result row per point. scripts/run_curve.py drives this on the real data and
writes the CSV/metadata; tests/test_curve.py exercises it on small synthetic
data shaped like the real Adult schema.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tabpfn import TabPFNClassifier

from tabular_showdown.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from tabular_showdown.metrics import classification_metrics
from tabular_showdown.models import fit_lgbm, predict_proba_positive, tune_lgbm

# TabPFN v2 pretraining limit -- a hard design cap, independent of the
# dynamic compute safety valve below. The cap is part of the story: TabPFN
# never runs on sizes above this, and never on the full train set.
TABPFN_ROW_CAP = 10_000

# The curve's sizes and per-size seed counts (design decision, see the project plan
# M3+M4, widened per the 2026-07-19 SOTA-upgrade plan Task B4 so the paired
# stats have power): more seeds at small sizes where subsample variance
# matters most, fewer as the subsample approaches the full train set.
# TabPFN never reaches n=10000 regardless of this list: the compute valve
# reconstructed from persisted timings skips it (5000 already exceeds the
# 8-minute budget), so the 10000 seeds apply to LGBM/LogReg only.
CURVE_SIZES = [200, 500, 1000, 2000, 5000, 10000]
SEEDS_BY_SIZE: dict[int, list[int]] = {
    200: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    500: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    1000: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    2000: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    5000: [0, 1, 2, 3, 4],
    10000: [0, 1, 2, 3, 4],
}
FULL_TRAIN_SEED = 0

CSV_COLUMNS = [
    "model",
    "n_train",
    "seed",
    "roc_auc",
    "accuracy",
    "log_loss",
    "brier",
    "fit_s",
    "predict_s",
]

# tabpfn 2.0.9 has this constructor param; checked at import time rather than
# hardcoded so the behavior (and the run metadata that records it) tracks
# whatever tabpfn version is actually installed.
TABPFN_CATEGORICAL_PARAM_SUPPORTED = "categorical_features_indices" in inspect.signature(
    TabPFNClassifier.__init__
).parameters


@dataclass
class FitPredictResult:
    """Uniform return type for every fit_predict_* helper below."""

    proba: np.ndarray
    fit_seconds: float
    predict_seconds: float
    meta: dict = field(default_factory=dict)


def subsample_train(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic, stratified subsample of size n from (X, y).

    Same stratified-by-label approach as data.frozen_eval_set, applied to
    the train side: every point on the learning-size curve sees a subsample
    with (approximately) the same class balance as the full train set, so
    quality differences track n rather than accidental label-ratio drift.
    """
    if n >= len(X):
        return X, y
    idx, _ = train_test_split(X.index, train_size=n, stratify=y, random_state=seed)
    idx = idx.sort_values()
    return X.loc[idx], y.loc[idx]


# --- per-model fit/predict helpers -------------------------------------


def _encode_for_tabpfn(
    X_train: pd.DataFrame, X_eval: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Ordinal-encode categoricals for TabPFN; numerics pass through as float.

    The encoder is fit on the train subsample only (no eval-set leakage).
    Unknown categories seen only at eval time and missing values (NaN) get
    distinct encoded values (-1 and -2) so TabPFN can tell "never seen" from
    "genuinely missing" apart, rather than lumping them into one bucket.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    cat_train = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    cat_eval = encoder.transform(X_eval[CATEGORICAL_COLUMNS])

    columns = list(X_train.columns)
    cat_positions = [columns.index(c) for c in CATEGORICAL_COLUMNS]

    def _assemble(df: pd.DataFrame, cat_encoded: np.ndarray) -> np.ndarray:
        out = df.copy()
        for i, col in enumerate(CATEGORICAL_COLUMNS):
            out[col] = cat_encoded[:, i]
        return out[columns].to_numpy(dtype=float)

    return _assemble(X_train, cat_train), _assemble(X_eval, cat_eval), cat_positions


def fit_predict_tabpfn(
    X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, seed: int = 0
) -> FitPredictResult:
    """Fit (in-context, no training) and predict with TabPFN v2 on CPU.

    ignore_pretraining_limits=True is required here: tabpfn 2.0.9 hard-refuses
    CPU inference above 1000 training rows (RuntimeError, not a warning),
    since it's slow without a GPU. That's a performance guard, not TabPFN's
    actual capability limit -- the model is pretrained for up to ~10k rows.
    We're on CPU by necessity (no GPU) and want the curve across TabPFN's real
    supported range, so we override the guard and cap ourselves at
    TABPFN_ROW_CAP. The override + the on-CPU-slowness are documented in the
    run metadata; the compute safety valve in run_curve keeps wall time sane.

    Callers (run_curve) are expected to respect TABPFN_ROW_CAP before calling.
    """
    X_train_enc, X_eval_enc, cat_positions = _encode_for_tabpfn(X_train, X_eval)

    kwargs: dict = {
        "device": "cpu",
        "random_state": seed,
        "ignore_pretraining_limits": True,
    }
    if TABPFN_CATEGORICAL_PARAM_SUPPORTED:
        kwargs["categorical_features_indices"] = cat_positions
    clf = TabPFNClassifier(**kwargs)

    fit_start = time.perf_counter()
    clf.fit(X_train_enc, y_train.to_numpy())
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    proba = clf.predict_proba(X_eval_enc)[:, 1]
    predict_seconds = time.perf_counter() - predict_start

    meta = {"categorical_features_indices_used": TABPFN_CATEGORICAL_PARAM_SUPPORTED}
    return FitPredictResult(
        proba=proba, fit_seconds=fit_seconds, predict_seconds=predict_seconds, meta=meta
    )


def fit_predict_lgbm_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    *,
    n_trials: int = 10,
    n_folds: int = 3,
    seed: int = 0,
    params: dict | None = None,
) -> FitPredictResult:
    """Tune (unless params is given) + fit + predict LightGBM.

    For sizes <= TABPFN_ROW_CAP a small fresh Optuna search runs per (size,
    seed) -- n_trials=10, 3-fold -- which is what "tuned GBDT" honestly means
    at each point of the curve. The full-train point instead passes in the
    already 30-trial-tuned params from results/lgbm_baseline.json, so
    fit_seconds there is fit-only, not tune+fit -- documented in the run
    metadata, not hidden in this function.

    fit_seconds includes the Optuna search time when tuning happens here,
    since that search is the real cost of producing a "tuned" model at this
    size (there's no separate tune_s column in the CSV schema).
    """
    tuned_fresh = params is None
    tune_seconds = 0.0
    if tuned_fresh:
        tune_start = time.perf_counter()
        params = tune_lgbm(X_train, y_train, n_trials=n_trials, n_folds=n_folds, seed=seed)
        tune_seconds = time.perf_counter() - tune_start

    fit_start = time.perf_counter()
    model = fit_lgbm(X_train, y_train, params)
    model_fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    proba = predict_proba_positive(model, X_eval)
    predict_seconds = time.perf_counter() - predict_start

    meta = {
        "tuned_fresh": tuned_fresh,
        "tune_seconds": tune_seconds,
        "model_fit_seconds": model_fit_seconds,
        "params": params,
    }
    return FitPredictResult(
        proba=proba,
        fit_seconds=tune_seconds + model_fit_seconds,
        predict_seconds=predict_seconds,
        meta=meta,
    )


def _logreg_pipeline(seed: int) -> Pipeline:
    """OneHot(handle_unknown='ignore') + StandardScaler + LogisticRegression.

    Documented as the "untuned linear reference": no hyperparameter search,
    just a sane default pipeline to show what a plain linear model gets you.
    """
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("cat", categorical, CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def fit_predict_logreg(
    X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, seed: int = 0
) -> FitPredictResult:
    """Fit + predict the untuned linear reference."""
    model = _logreg_pipeline(seed)

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    proba = model.predict_proba(X_eval)[:, 1]
    predict_seconds = time.perf_counter() - predict_start

    return FitPredictResult(proba=proba, fit_seconds=fit_seconds, predict_seconds=predict_seconds)


# --- orchestration -------------------------------------------------------


def _to_row(
    model: str, n_train: int, seed: int, y_eval: pd.Series, result: FitPredictResult
) -> dict:
    metrics = classification_metrics(y_eval, result.proba)
    return {
        "model": model,
        "n_train": n_train,
        "seed": seed,
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "fit_s": result.fit_seconds,
        "predict_s": result.predict_seconds,
    }


def rows_to_dataframe(rows: Iterable[dict]) -> pd.DataFrame:
    """Build the long-format results DataFrame with the fixed CSV column order."""
    return pd.DataFrame(list(rows), columns=CSV_COLUMNS)


def compute_seed_spread(
    df: pd.DataFrame, metric: str = "roc_auc"
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per (n_train, model) spread of `metric` across seeds: mean/std/min/max/n_seeds.

    Pure aggregation over a learning_curve.csv-shaped DataFrame (columns
    model, n_train, seed, <metric>, ...) -- no model training involved, so
    it's cheap to call from tests and from the meta-JSON writer alike.
    Generic over however many seeds a size actually has: this reads seed
    counts off the data itself rather than SEEDS_BY_SIZE, so it keeps working
    unchanged once WP-B1.3 appends more seeds at n_train=5000.

    Population std (ddof=0) so a single-seed size reports std=0.0 rather
    than NaN (JSON has no NaN literal). Keys are strings (str(n_train)) so
    the result serializes straight into learning_curve_meta.json, matching
    that file's existing seeds_by_size convention. A model missing at a
    given size (e.g. TabPFN above its row cap) is simply absent from that
    size's dict -- never a crash, never a fabricated zero.
    """
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    agg = df.groupby(["n_train", "model"])[metric].agg(
        mean="mean",
        std=lambda s: float(s.std(ddof=0)),
        min="min",
        max="max",
        n_seeds="count",
    )
    for (n_train, model), row in agg.iterrows():
        out.setdefault(str(int(n_train)), {})[model] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "n_seeds": int(row["n_seeds"]),
        }
    return out


def run_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    sizes: Sequence[int] = CURVE_SIZES,
    seeds_by_size: dict[int, list[int]] | None = None,
    full_size: int | None = None,
    full_lgbm_params: dict | None = None,
    models: Sequence[str] = ("tabpfn", "lgbm", "logreg"),
    lgbm_n_trials: int = 10,
    lgbm_n_folds: int = 3,
    tabpfn_max_seconds: float = 8 * 60,
    skip_keys: set[tuple[str, int, int]] | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> Iterator[dict]:
    """Sweep model x size x seed on the frozen eval set, yielding one row per point.

    X_eval/y_eval are never resampled -- every point scores against the same
    fixed evaluation set passed in once by the caller.

    Compute safety valve: TabPFN is hard-capped at TABPFN_ROW_CAP regardless
    of `sizes`. On top of that, the first time a TabPFN point takes longer
    than tabpfn_max_seconds, every larger size is skipped for TabPFN only --
    everything else (lgbm, logreg, smaller/equal tabpfn points already
    queued) still runs. Which sizes TabPFN actually reached is recoverable
    from the yielded rows themselves (max n_train with model == "tabpfn").

    skip_keys: (model, n_train, seed) tuples to skip -- used to resume a run
    that already wrote those rows to disk. Skipped points are not re-run and
    not re-yielded. The caller is responsible for pre-adding TabPFN sizes
    above a slow already-completed point so the valve decision survives a
    resume (this generator can't time a point it never ran).

    full_size (if given) adds one additional point at that size for lgbm and
    logreg only -- never for TabPFN, which never sees the full train set.
    full_lgbm_params, if given, is used as-is (no fresh tuning) for that
    point, e.g. reusing an already-tuned baseline.
    """
    seeds_by_size = seeds_by_size if seeds_by_size is not None else SEEDS_BY_SIZE
    skip_keys = skip_keys or set()
    tabpfn_cutoff_size: int | None = None

    def done(model: str, n: int, seed: int) -> bool:
        if (model, n, seed) in skip_keys:
            progress(f"{model:<7} n={n:>6} seed={seed}  skipped (already in results)")
            return True
        return False

    for size in sizes:
        seeds = seeds_by_size.get(size, [0])
        for seed in seeds:
            X_sub, y_sub = subsample_train(X_train, y_train, size, seed)

            if "logreg" in models and not done("logreg", size, seed):
                result = fit_predict_logreg(X_sub, y_sub, X_eval, seed=seed)
                yield _to_row("logreg", size, seed, y_eval, result)
                progress(
                    f"logreg  n={size:>6} seed={seed}  "
                    f"fit={result.fit_seconds:.2f}s predict={result.predict_seconds:.2f}s"
                )

            if "lgbm" in models and not done("lgbm", size, seed):
                result = fit_predict_lgbm_tuned(
                    X_sub, y_sub, X_eval, n_trials=lgbm_n_trials, n_folds=lgbm_n_folds, seed=seed
                )
                yield _to_row("lgbm", size, seed, y_eval, result)
                progress(
                    f"lgbm    n={size:>6} seed={seed}  "
                    f"fit={result.fit_seconds:.2f}s predict={result.predict_seconds:.2f}s"
                )

            if "tabpfn" in models and size <= TABPFN_ROW_CAP and not done("tabpfn", size, seed):
                if tabpfn_cutoff_size is not None and size > tabpfn_cutoff_size:
                    progress(
                        f"tabpfn  n={size:>6} seed={seed}  SKIPPED "
                        f"(safety valve tripped at n={tabpfn_cutoff_size})"
                    )
                else:
                    result = fit_predict_tabpfn(X_sub, y_sub, X_eval, seed=seed)
                    total = result.fit_seconds + result.predict_seconds
                    yield _to_row("tabpfn", size, seed, y_eval, result)
                    progress(
                        f"tabpfn  n={size:>6} seed={seed}  "
                        f"fit={result.fit_seconds:.2f}s predict={result.predict_seconds:.2f}s "
                        f"total={total:.1f}s"
                    )
                    if total > tabpfn_max_seconds and tabpfn_cutoff_size is None:
                        tabpfn_cutoff_size = size
                        progress(
                            f"tabpfn  safety valve: n={size} took {total:.1f}s > "
                            f"{tabpfn_max_seconds:.0f}s cap -> skipping larger sizes for tabpfn"
                        )

    if full_size:
        seed = FULL_TRAIN_SEED
        if "lgbm" in models and not done("lgbm", full_size, seed):
            result = fit_predict_lgbm_tuned(
                X_train,
                y_train,
                X_eval,
                n_trials=lgbm_n_trials,
                n_folds=lgbm_n_folds,
                seed=seed,
                params=full_lgbm_params,
            )
            yield _to_row("lgbm", full_size, seed, y_eval, result)
            progress(
                f"lgbm    n={full_size:>6} (full) seed={seed}  "
                f"fit={result.fit_seconds:.2f}s predict={result.predict_seconds:.2f}s"
            )

        if "logreg" in models and not done("logreg", full_size, seed):
            result = fit_predict_logreg(X_train, y_train, X_eval, seed=seed)
            yield _to_row("logreg", full_size, seed, y_eval, result)
            progress(
                f"logreg  n={full_size:>6} (full) seed={seed}  "
                f"fit={result.fit_seconds:.2f}s predict={result.predict_seconds:.2f}s"
            )
        # TabPFN never runs here: hard-capped at TABPFN_ROW_CAP, see docstring.
