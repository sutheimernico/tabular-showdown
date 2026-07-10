# AUTOPILOT LOG — tabular-showdown

- 2026-07-10: M1 data pipeline — `src/tabular_showdown/data.py` (load_adult, frozen_eval_set, split_features_target) + `tests/test_data.py` (10 tests, pytest+ruff green).
- 2026-07-10: M2 tuned LightGBM baseline — `src/tabular_showdown/models.py` (tune_lgbm/fit_lgbm/predict_proba_positive), `src/tabular_showdown/metrics.py` (classification_metrics), `scripts/run_lgbm_baseline.py` + tests (18 tests total, pytest+ruff green); real run (30 trials, 3-fold, full train, frozen eval set): cv_auc 0.9286, eval ROC-AUC 0.9304, tune 157s / fit 1.3s / predict 0.02s → `results/lgbm_baseline.json`.
