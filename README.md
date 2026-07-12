# Tabular Showdown

> When does a model that does **no training** beat a properly tuned GBDT? The
> answer is a curve, not a bragging number.

![Learning-size curve: TabPFN v2 wins the small-data regime, tuned LightGBM overtakes it by ~5k rows](results/learning_curve.png)

**TabPFN v2 (zero training) wins the small-data regime; tuned LightGBM
overtakes it at ~5,000 rows.** Below 2,000 training rows, an in-context
transformer that has never seen this dataset beats a LightGBM model tuned
with 10 fresh Optuna trials at every point on the curve. Past ~5,000 rows,
the tuned GBDT pulls ahead and keeps extending its lead all the way to the
full 32,561-row train set (ROC-AUC 0.930 vs. TabPFN's last measured point,
0.914 at n=5,000).

This repo benchmarks **TabPFN v2** ([Hollmann et al., *Nature* 2025](https://www.nature.com/articles/s41586-024-08328-6) — "Accurate predictions on small data with a tabular foundation model"), a prior-data fitted transformer that classifies by in-context learning instead of gradient-based training, against a **tuned LightGBM** (Optuna TPE search) and an **untuned logistic regression** reference, on the UCI Adult / census income dataset. The goal isn't beating a leaderboard — Adult is a "solved" dataset — it's mapping out *where each approach wins* and being honest about the compute cost of each.

## When to use what

| | **TabPFN v2** | **LightGBM (tuned)** | **LogReg (untuned)** |
|---|---|---|---|
| Best regime here | ≤ ~2,000 rows | ≥ ~5,000 rows | never the best, but a fast floor |
| Training cost | ~0s (in-context, no gradient training) | Optuna search + fit (~1s–3min: 10-trial per-size searches up to ~49s, 30-trial full-train tune ~2.6min) | ~0s |
| Prediction cost | **slow**: ~29–755s to score 4,000 rows on CPU, growing steeply with training-set size | fast: ~0.02s to score 4,000 rows, regardless of training size | fast: ~0.01–0.02s |
| Needs tuning? | no (that's the point) | yes (Optuna, ~10–30 trials here) | no (this project never tunes it) |
| Hard caps | ~10,000-row / ~500-feature pretraining limit (Nature 2025); CPU makes it much slower before that | none inherent; scales with data | none inherent, but ceiling is lower |
| Calibration (this run, n_train=2,000) | **best**: Brier 0.101 | worst: Brier 0.109 | middle: Brier 0.105 |
| GPU required for real use? | recommended (CPU is viable but slow, see below) | no | no |
| Pick it when... | you have a small/medium labeled set and no time to tune, or need a fast baseline with no training loop | you have thousands+ rows and can afford a tuning pass, and need fast repeated inference | you need an instant, interpretable, zero-dependency floor |

## Figures

### Calibration

![Calibration: TabPFN v2 is best-calibrated at n_train=2,000, Brier 0.101](results/figures/calibration.png)

All three models are decently calibrated on Adult (a well-behaved binary
target helps), but at the n_train=2,000 spotlight size, TabPFN v2 has the
lowest Brier score. LightGBM at this size is a lightly-informed model (fit
fresh on only 2,000 rows), which shows up as the most over/under-confident
of the three in the mid-probability bins — this is a statement about *this
training size*, not a general claim that trees are worse calibrated.

### Prediction-time cost

![Timing: TabPFN needs virtually no fit time but predicts thousands of times slower than tuned LightGBM](results/figures/timing.png)

The honest compute-cost story: TabPFN's fit is a rounding error (it just
stores the training set as context) but its predict is the bottleneck —
about four orders of magnitude slower than LightGBM's predict at the same
training size. LightGBM inverts the trade-off: it pays for a tuning search
up front, then predicts almost instantly, however large the training set.
Neither is "faster" in general — it depends whether your workload is
train-once/predict-many (favors LightGBM) or predict-once-per-cold-start
(favors TabPFN, if you can tolerate the per-call latency).

### What each model leans on

<table>
<tr>
<td><img src="results/figures/shap_summary.png" alt="SHAP summary: LightGBM leans hardest on marital_status" width="100%"></td>
<td><img src="results/figures/permutation_importance.png" alt="Permutation importance: TabPFN relies most on relationship" width="100%"></td>
</tr>
</table>

Two different tools for two different model architectures, on purpose:
**TreeSHAP** is exact for LightGBM because it's a tree ensemble — the
attribution can be read straight off the trees. **TabPFN has no native
feature importances** (it's an in-context transformer, not a tree), so the
honest substitute is **model-agnostic permutation importance**: shuffle one
feature at a time in the eval set and measure the ROC-AUC drop. Both models
converge on a similar theme (marital/relationship status and
education/occupation drive Adult's income label most), which is itself a
useful cross-check that both are learning something sensible rather than
memorizing noise.

## Reproduce

```bash
uv sync

# M1-M4: data pipeline, tuned LightGBM baseline, TabPFN, the money chart
uv run python scripts/run_lgbm_baseline.py     # -> results/lgbm_baseline.json (~3 min)
uv run python scripts/run_curve.py             # -> results/learning_curve.csv (~50 min, CPU)
uv run python scripts/plot_curve.py            # -> results/learning_curve.{png,svg}

# M5: calibration, SHAP, permutation importance
uv run python scripts/make_figures.py          # -> results/figures/*.{png,svg} (~5 min, CPU)

# M6: the Streamlit demo
uv run streamlit run app.py

# gate
uv run pytest -q
uv run ruff check .
```

Every step above reads the artifacts from the previous one (`results/*.json`,
`results/*.csv`) rather than re-deriving them, so any single script can be
re-run in isolation once the earlier artifacts exist. `results/` and
`results/figures/` are committed so `plot_curve.py`, `make_figures.py`, and
`app.py` are all reproducible/inspectable without re-running the slow steps.

## Method notes

- **Frozen eval set**: every number in this project — every curve point,
  every calibration/SHAP/permutation figure — is scored on the same fixed,
  stratified 4,000-row subsample of `adult.test` (seed 42). This is what
  makes the comparisons across models and training sizes fair; it also
  means every reported metric is a single split, not a cross-validated
  estimate.
- **The learning-size curve** sweeps training-set size ∈ {200, 500, 1k, 2k,
  5k, 10k, full (32,561)}, 1–3 subsample seeds per size (more seeds at
  small sizes, where subsample variance matters most). LightGBM is
  freshly Optuna-tuned (10 trials, 3-fold CV) at every subsampled size;
  the full-train point reuses the 30-trial baseline from
  `results/lgbm_baseline.json` instead of re-tuning. LogReg is never
  tuned, anywhere.
- **Calibration + SHAP figures** (`results/figures/calibration.png`,
  `shap_summary.png`) train all three models fresh at **n_train=2,000**
  (seed 0) — the largest size where a single TabPFN predict on the full
  4,000-row eval set stays a few minutes on CPU (measured ~209–220s across
  runs). LightGBM reuses the tuned baseline params (no re-tuning) at that
  size.
- **TabPFN permutation importance** (`permutation_importance.png`) runs at
  a *separately reduced* n_train=200 with a 200-row eval subsample and 3
  repeats/feature. Permutation importance costs `1 + features × repeats`
  predict calls (43 here); at the n=2,000 calibration size that would be
  ~2.5 hours of TabPFN predicts on CPU. At n=200 it's a few minutes. This
  is honestly a smaller/rougher TabPFN fit than the calibration one — see
  `results/figures/figures_meta.json` for the exact sizes and seeds of
  every figure.
- **Timing figure** reuses the wall-clock times already recorded by the
  learning-size curve run at n_train=2,000 rather than re-measuring, since
  those numbers include the real cost of a freshly-tuned LightGBM at that
  size (a 10-trial Optuna search), which is the more honest "tuned-fit"
  story than timing a fit with pre-tuned params.

## Limits (read this before trusting a number above)

- **TabPFN is capped at n=5,000 here by an 8-minute-per-point CPU compute
  valve**, not by TabPFN v2's own ~10,000-row pretraining limit. The curve
  run skips any larger TabPFN size once one point exceeds the time budget
  — n=10,000 was never attempted for TabPFN (LightGBM and LogReg did run
  there; see `results/learning_curve.csv`). Don't read "TabPFN's line stops at 5k"
  as "TabPFN can't handle more than 5k rows"; it can, per the paper, up to
  ~10k rows / ~500 features — this repo just never gave it a GPU to do so
  quickly. `ignore_pretraining_limits=True` is required to even run TabPFN
  above 1,000 rows on CPU with this library version (2.0.9); it overrides a
  performance guard, not a hard capability ceiling.
- **Adult is a "solved" dataset.** Every model here lands in the same
  0.88–0.93 ROC-AUC band that's been published dozens of times. The point
  of this repo is the *comparison method* — the learning-size curve,
  calibration, timing, and explanation methodology — not a SOTA accuracy
  claim.
- **Single frozen eval set, no cross-validation.** Every metric in every
  figure and in the app is one split. Treat differences smaller than the
  curve's seed-to-seed band (the shaded region in the money chart) as
  noise, not signal.
- **LogReg is deliberately untuned** — it's a floor/sanity-check reference,
  not a competitor. Don't read its numbers as "what logistic regression can
  achieve"; a tuned LogReg would do better.
- **AutoGluon was planned (see `PLAN.md`) but is deferred**, not
  implemented. The three-way comparison here is TabPFN v2 vs. tuned
  LightGBM vs. untuned LogReg only.
- **CSV upload in the Streamlit app is intentionally disabled** for
  anything other than the built-in Adult dataset — the tuned LightGBM
  hyperparameters and TabPFN's categorical encoding are fitted to Adult's
  specific 14-column schema, and generalizing the pipeline to arbitrary
  uploads was out of scope (YAGNI) for this demo.

## Stack

TabPFN v2 (`tabpfn==2.0.9`, pinned — the last version whose weights download
without a Prior Labs account) · LightGBM + Optuna (TPE) · scikit-learn ·
SHAP (TreeExplainer) · permutation importance (hand-rolled, model-agnostic)
· matplotlib (house-style static figures) · Streamlit (demo app) · uv + ruff
+ pytest (tooling).

## Project layout

```
src/tabular_showdown/
  data.py      data pipeline: load_adult, frozen_eval_set, split_features_target
  models.py    tune_lgbm / fit_lgbm / predict_proba_positive
  metrics.py   classification_metrics (roc_auc, accuracy, log_loss, brier)
  curve.py     per-model fit/predict + the learning-size curve orchestration
  explain.py   calibration_data, shap_values_lgbm, permutation_importance_tabpfn
  viz.py       house-style chart builders (matplotlib) + metrics_table
scripts/
  run_lgbm_baseline.py   M2: tuned LightGBM on full train -> lgbm_baseline.json
  run_curve.py           M3+M4: the learning-size curve -> learning_curve.csv
  plot_curve.py          renders the money chart
  make_figures.py        M5: calibration/timing/SHAP/permutation figures
app.py                   M6: Streamlit demo
results/                 committed artifacts (JSON/CSV/PNG/SVG)
PLAN.md                  original milestone plan
AUTOPILOT_LOG.md         running log of what was built, when, with what real numbers
```
