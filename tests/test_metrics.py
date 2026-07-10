"""Tests for classification_metrics, hand-checked on tiny fixtures."""

from tabular_showdown.metrics import classification_metrics


def test_perfect_predictions_give_perfect_scores():
    y_true = [0, 0, 1, 1]
    proba = [0.0, 0.0, 1.0, 1.0]
    metrics = classification_metrics(y_true, proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == 0.0
    assert metrics["log_loss"] >= 0.0


def test_worst_predictions_give_worst_scores():
    y_true = [0, 0, 1, 1]
    proba = [1.0, 1.0, 0.0, 0.0]
    metrics = classification_metrics(y_true, proba)
    assert metrics["roc_auc"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["brier"] == 1.0


def test_threshold_is_0_5_and_ties_count_as_positive():
    y_true = [0, 1, 0, 1]
    proba = [0.5, 0.5, 0.5, 0.5]
    metrics = classification_metrics(y_true, proba)
    # every row predicted positive at the >= 0.5 threshold: 2 of 4 correct
    assert metrics["accuracy"] == 0.5


def test_returns_exactly_the_expected_keys():
    metrics = classification_metrics([0, 1], [0.1, 0.9])
    assert set(metrics.keys()) == {"roc_auc", "accuracy", "log_loss", "brier"}
