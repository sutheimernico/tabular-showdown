"""M5: beyond-accuracy explainability -- calibration, SHAP, permutation importance.

TreeSHAP gives exact per-feature attributions for the tuned LightGBM for
free: it's a tree ensemble, so shap.TreeExplainer walks the trees directly
instead of approximating. TabPFN has no such structure to walk -- it's an
in-context transformer, not a tree -- so the honest way to explain it is
model-agnostic permutation importance: shuffle one feature at a time in the
eval set and measure how much ROC-AUC drops. Both answer "why does this
model predict what it predicts", not just "how well does it predict".

calibration_data is the third piece: it turns (y_true, proba) into the
binned (predicted, observed) pairs a reliability diagram plots. Brier score
itself already lives in metrics.classification_metrics.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score


def calibration_data(y_true, proba, n_bins: int = 10) -> dict:
    """Bin predictions into n_bins equal-width bins over [0, 1].

    For each non-empty bin, reports the bin center, the mean predicted
    probability, the observed positive frequency, and the row count -- the
    raw material for a reliability diagram. A perfectly calibrated model has
    observed_freq[b] == mean_predicted[b] for every bin. Empty bins are
    dropped rather than reported as NaN/0, since there's nothing observed
    there.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # pd.cut with include_lowest puts proba==0.0 in the first bin and
    # proba==1.0 in the last bin (both would otherwise fall outside the
    # half-open intervals cut uses by default).
    bin_idx = pd.cut(proba, bins=edges, labels=False, include_lowest=True)

    bin_centers, mean_predicted, observed_freq, counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centers.append(float((edges[b] + edges[b + 1]) / 2))
        mean_predicted.append(float(proba[mask].mean()))
        observed_freq.append(float(y_true[mask].mean()))
        counts.append(count)

    return {
        "bin_centers": bin_centers,
        "mean_predicted": mean_predicted,
        "observed_freq": observed_freq,
        "counts": counts,
    }


def shap_values_lgbm(model, X_sample: pd.DataFrame, sample_size: int = 500, seed: int = 0) -> dict:
    """TreeSHAP attributions for a fitted LightGBM classifier.

    X_sample is subsampled down to sample_size rows (default 500): TreeSHAP
    is exact but still scales with rows x trees, and 500 rows is plenty for
    a stable mean-|SHAP| ranking while staying fast on CPU.

    Returns the raw per-row, per-feature SHAP values plus the feature names
    and the rows they were computed on, so a caller can build either a bar
    chart (mean |value|) or a full beeswarm from the same object.
    """
    if len(X_sample) > sample_size:
        X_used = X_sample.sample(n=sample_size, random_state=seed)
    else:
        X_used = X_sample

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X_used)
    # Older shap versions return a [class0, class1] list for binary
    # LGBMClassifier; newer ones (installed here: 0.52) return a single
    # (n_rows, n_features) array already scoped to the positive class.
    # Normalize both to "the positive class" array.
    values = raw_values[1] if isinstance(raw_values, list) else raw_values

    return {
        "values": values,
        "feature_names": list(X_used.columns),
        "X": X_used,
    }


def permutation_importance_tabpfn(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """Model-agnostic permutation importance -- the honest way to explain a
    black-box in-context model like TabPFN, which has no native feature
    importances to read off (it's not a tree ensemble).

    predict_fn(X) -> proba is the only thing this needs from the model: a
    function that scores a feature matrix. Bind it to an already-fitted
    model (see scripts/make_figures.py for the TabPFN closure). For each
    column in X, this shuffles just that column n_repeats times, re-scores,
    and reports how far ROC-AUC drops below the unpermuted baseline -- a
    bigger drop means the model relied on that feature more.

    Keep X small and n_repeats low for TabPFN specifically: every repeat
    costs one full predict_fn call per feature, and TabPFN predict on CPU is
    the bottleneck (see curve.py's per-point timings).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    baseline_proba = predict_fn(X)
    baseline_auc = float(roc_auc_score(y, baseline_proba))

    importances: dict[str, dict] = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            proba = predict_fn(X_perm)
            drops.append(baseline_auc - roc_auc_score(y, proba))
        importances[col] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
        }

    return {"baseline_auc": baseline_auc, "importances": importances, "n_repeats": n_repeats}
