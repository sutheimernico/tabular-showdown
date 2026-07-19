"""M3+M4: the real learning-size curve run -- the project's money chart data.

Sweeps TabPFN v2, freshly-tuned LightGBM, and an untuned linear reference
(logreg) across training-set sizes, always scoring on the same frozen
4,000-row eval set. Writes results/learning_curve.csv (long format) and
results/learning_curve_meta.json (seed policy, TabPFN cap + reason, LightGBM
tuning policy, package versions, total runtime).

TabPFN cap: v2's pretraining limit is ~10k rows, so TabPFN never runs above
10,000 and never on the full train set -- the cap is part of the story, not
an implementation shortcut. On top of that a compute safety valve skips
TabPFN's larger sizes if any point exceeds an 8-minute budget on CPU.

Crash-resilient: every finished point is appended to the CSV immediately and
flushed, and a re-run reads the CSV on startup and skips points already
present. A kill (session cleanup, OOM, ...) loses at most one point; just
launch the script again and it resumes from where it stopped.
"""

import csv
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm
import optuna
import pandas as pd
import sklearn
import tabpfn
import torch

from tabular_showdown.curve import (
    CSV_COLUMNS,
    CURVE_SIZES,
    FULL_TRAIN_SEED,
    SEEDS_BY_SIZE,
    TABPFN_CATEGORICAL_PARAM_SUPPORTED,
    TABPFN_ROW_CAP,
    compute_seed_spread,
    run_curve,
)
from tabular_showdown.data import frozen_eval_set, load_adult, split_features_target

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
BASELINE_PATH = RESULTS_DIR / "lgbm_baseline.json"
CSV_PATH = RESULTS_DIR / "learning_curve.csv"
META_PATH = RESULTS_DIR / "learning_curve_meta.json"

LGBM_N_TRIALS = 10
LGBM_N_FOLDS = 3
TABPFN_MAX_SECONDS = 8 * 60


def _load_skip_keys() -> tuple[set[tuple[str, int, int]], int]:
    """Read the CSV (if any) and build the set of already-computed points.

    Also reconstructs the TabPFN safety-valve decision from persisted timings:
    if a completed TabPFN point already exceeded the budget, every larger
    TabPFN size is pre-added to the skip set so a resume honors the valve
    even though this invocation never times that earlier point itself.
    """
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return set(), 0
    prev = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    skip = {
        (str(m), int(n), int(s))
        for m, n, s in zip(prev["model"], prev["n_train"], prev["seed"], strict=False)
    }

    tp = prev[prev["model"] == "tabpfn"]
    if not tp.empty:
        slow = tp[(tp["fit_s"] + tp["predict_s"]) > TABPFN_MAX_SECONDS]
        if not slow.empty:
            cutoff = int(slow["n_train"].min())
            for n in CURVE_SIZES:
                if cutoff < n <= TABPFN_ROW_CAP:
                    for s in SEEDS_BY_SIZE.get(n, [0]):
                        skip.add(("tabpfn", n, s))
    return skip, len(prev)


def main() -> None:
    run_start = time.perf_counter()

    train_df, test_df = load_adult(DATA_DIR)
    eval_df = frozen_eval_set(test_df)  # THE frozen eval set, built exactly once
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)

    full_lgbm_params = json.loads(BASELINE_PATH.read_text())["params"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    skip_keys, n_resumed = _load_skip_keys()

    print(f"learning-size curve: n_train_full={len(X_train)} n_eval={len(X_eval)}")
    print(f"sizes={CURVE_SIZES} + full ({len(X_train)}) for lgbm/logreg only")
    print(f"tabpfn row cap={TABPFN_ROW_CAP}, safety valve={TABPFN_MAX_SECONDS:.0f}s/point")
    if n_resumed:
        print(f"RESUMING: {n_resumed} points already in {CSV_PATH.name}, skipping those")
    print()

    def progress(msg: str) -> None:
        elapsed = time.perf_counter() - run_start
        print(f"[{elapsed:7.1f}s] {msg}", flush=True)

    # Append + flush each finished point immediately so a kill loses at most
    # one point; write the header only when starting a fresh CSV.
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            fh.flush()
        for row in run_curve(
            X_train,
            y_train,
            X_eval,
            y_eval,
            sizes=CURVE_SIZES,
            seeds_by_size=SEEDS_BY_SIZE,
            full_size=len(X_train),
            full_lgbm_params=full_lgbm_params,
            models=("tabpfn", "lgbm", "logreg"),
            lgbm_n_trials=LGBM_N_TRIALS,
            lgbm_n_folds=LGBM_N_FOLDS,
            tabpfn_max_seconds=TABPFN_MAX_SECONDS,
            skip_keys=skip_keys,
            progress=progress,
        ):
            writer.writerow(row)
            fh.flush()

    total_seconds = time.perf_counter() - run_start

    df = pd.read_csv(CSV_PATH)

    tabpfn_max_n = int(df.loc[df["model"] == "tabpfn", "n_train"].max())
    if tabpfn_max_n < TABPFN_ROW_CAP:
        tabpfn_cap_reason = (
            f"compute safety valve: a TabPFN point took longer than "
            f"{TABPFN_MAX_SECONDS:.0f}s on CPU, larger sizes were skipped"
        )
    else:
        tabpfn_cap_reason = (
            "TabPFN v2 pretraining limit (~10k rows): sizes above 10,000 and the "
            "full train set are out of scope by design, not skipped for compute"
        )

    meta = {
        "sizes": CURVE_SIZES,
        "full_size_lgbm_logreg_only": len(X_train),
        "seeds_by_size": {str(k): v for k, v in SEEDS_BY_SIZE.items()},
        "full_train_seed": FULL_TRAIN_SEED,
        "n_eval": len(X_eval),
        "eval_set": "frozen 4000-row stratified subsample of adult.test (seed 42)",
        "tabpfn": {
            "row_cap": TABPFN_ROW_CAP,
            "tabpfn_max_n": tabpfn_max_n,
            "cap_reason": tabpfn_cap_reason,
            "device": "cpu",
            "ignore_pretraining_limits": True,
            "categorical_features_indices_used": TABPFN_CATEGORICAL_PARAM_SUPPORTED,
            "categorical_encoding": (
                "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, "
                "encoded_missing_value=-2) fitted on the train subsample only"
            ),
        },
        "lgbm_tuning_policy": {
            "subsampled_sizes": f"fresh Optuna search per (size, seed): "
            f"n_trials={LGBM_N_TRIALS}, {LGBM_N_FOLDS}-fold CV",
            "full_train": "reuses the 30-trial tuned params from results/lgbm_baseline.json",
            "fit_s_note": "fit_s includes the Optuna search time where tuning ran fresh",
        },
        "logreg": "untuned linear reference: OneHot(ignore unknowns, NaN->'missing') "
        "+ StandardScaler + LogisticRegression(max_iter=1000)",
        "spread": {
            "metric": "roc_auc",
            "note": "per (n_train, model) mean/std/min/max/n_seeds across subsample "
            "seeds -- see REVIEW.md B-1/B-5: the headline crossover must not claim "
            "more precision than this spread supports",
            "by_size": compute_seed_spread(df, metric="roc_auc"),
        },
        "versions": {
            "python": platform.python_version(),
            "tabpfn": tabpfn.__version__,
            "torch": torch.__version__,
            "lightgbm": lightgbm.__version__,
            "optuna": optuna.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "total_runtime_seconds": total_seconds,
        "runtime_note": (
            "total_runtime_seconds is this invocation's wall time; the CSV was "
            "built incrementally and may span multiple resumed invocations"
        ),
        "n_points": len(df),
        "resumed_from_existing_csv": bool(n_resumed),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print()
    print(f"done in {total_seconds:.1f}s ({total_seconds / 60:.1f} min), {len(df)} rows")
    print(f"tabpfn_max_n={tabpfn_max_n}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {META_PATH}")
    print()
    print("mean ROC-AUC per (model, n_train):")
    summary = df.groupby(["model", "n_train"])["roc_auc"].mean().unstack(level=0)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
