# Tabular Foundation Model Showdown

> When does a model that does **no training** beat the tuned GBDT machinery?
> The answer is a curve, not a bragging number.

## Goal / Narrative
Benchmark **TabPFN v2** (in-context, zero training) against a **tuned GBDT** and an
**AutoML** baseline on a real tabular classification task — and honestly map *where each
one wins*. The headline deliverable is a single plot: model quality as a function of
training-set size. Expected story: TabPFN dominates the small-data regime, GBDT catches
up and overtakes as data grows.

## Dataset
- **Adult / Census Income** — `data/adult.data` (32,562 rows) + `data/adult.test` (16,283 rows), 14 features, binary target income `>50K` / `<=50K`.
- Source: UCI ML Repository, id 2 (open, CC BY 4.0). See `data/adult.names` for the schema. Kaggle mirrors the same dataset — we pulled UCI because no Kaggle auth is configured.
- Why this one: non-trivial (mixed categorical/numeric, class imbalance ~24% positive), large enough to **subsample down** for the learning-size curve, small enough that TabPFN is in its comfort zone at the low end.
- Verify license before any public release (per publish checklist).

## Techniques & Stack
- **TabPFN v2** (`tabpfn`, PyPI) — prior-data fitted transformer, in-context classification, no training. Nature 2025 (Prior Labs / Hutter et al.). Targets datasets up to ~10k rows / ~500 features; the row cap is itself part of the story.
- **LightGBM** + **Optuna** — properly tuned gradient boosting baseline.
- **AutoGluon (Tabular)** — AutoML reference, "what you get for free".
- Eval: scikit-learn, SHAP (GBDT importances), permutation importance (TabPFN).
- Viz + app: matplotlib/plotly, **Streamlit** (v1). React frontend = optional stretch (Nico's growth area).
- Tooling: uv + ruff + pytest.

## Milestones (verifiable)
1. **Data pipeline** — load adult.data/.test, clean (`?` → NA, strip whitespace, dtype cast), fixed train/test split reused everywhere. Test: row counts + no leakage assertion.
2. **Baselines** — LightGBM (Optuna, k-fold), AutoGluon on full train. Log ROC-AUC, accuracy, log-loss, Brier, fit+predict wall time.
3. **TabPFN v2** — run on full-ish train (respect row cap; document how the cap is handled). Same metrics.
4. **The curve** — subsample train at {200, 500, 1k, 2k, 5k, 10k, 20k, full}, evaluate all three on the *same fixed test set*, repeat over seeds. Produce the size-vs-quality plot with error bands. **This is the money chart.**
5. **Beyond accuracy** — calibration curves + Brier, prediction-time comparison, SHAP for GBDT, permutation importance for TabPFN.
6. **App** — Streamlit: upload a CSV → run all three → side-by-side metrics, calibration plot, timing, top-feature explanation.
7. **Write-up** — README with the curve, the "when to use what" table, and explicit limits.

## Deliverables
- Reproducible benchmark (`make bench` or a single script, seeded).
- The learning-size curve + calibration comparison.
- Deployed Streamlit demo.
- Honest "when TabPFN wins / when GBDT wins" decision table.

## "Krass" factor & honest framing
- TabPFN v2 is genuine 2025 cutting edge that few reviewers have used — high "reads papers, not just kernels" signal.
- The maturity signal is showing the **boundary and the failure mode** (TabPFN doesn't scale to large/wide data), not overclaiming a single AUC.

## Risks & limits
- TabPFN v2 has row/feature caps — plan the subsampling/context handling explicitly; don't present a capped run as if uncapped.
- Adult is a "solved" dataset; the novelty is the *comparison method*, not SOTA accuracy. Frame accordingly.
- Keep the test set frozen across all experiments or the curve is meaningless.

## Open / Needs Nico
- Streamlit-only v1, or invest in a React frontend for the portfolio?
- Confirm this is the lead project (my recommendation) before full bootstrap (PROJECT/LOOP/CLAUDE doc set).
