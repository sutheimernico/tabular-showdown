"""Plain sklearn classification metrics shared by every model baseline."""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def classification_metrics(y_true, proba) -> dict:
    """Compute roc_auc, accuracy (0.5 threshold), log_loss, and brier score.

    proba is the predicted probability of the positive class.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
    }
