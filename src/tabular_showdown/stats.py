"""Paired seed-level significance stats for model-vs-model comparisons
(REVIEW.md B-1, research/2026-07-19-tabular-sota.md section 3).

The field's 2026 norm (Position paper, arXiv 2605.17273) is: report a named
significance test per comparison and default to "tie" unless the data proves
otherwise -- a mean-vs-mean headline with no test behind it is malpractice.
This module supplies that test for the learning-size curve: given the same
metric from the same seeds for two models, it reports the paired difference's
mean, a t-based 95% CI, the paired t-test, and the Wilcoxon signed-rank test
(the field's non-parametric standard), then applies the tie rule mechanically
so the verdict can't be eyeballed into a more exciting story than the data
supports.

Caveat worth internalizing before reading wilcoxon_p on this project's actual
seed counts: Wilcoxon signed-rank's minimum achievable two-sided p-value with
n paired, all-same-sign differences is 2**(1 - n) -- 0.0625 at n=5, 0.03125
at n=6. Below 6 seeds it structurally cannot cross the conventional 0.05
threshold, no matter how consistent the effect. That's not a bug in the test;
it's exactly why the CI-based tie rule (not wilcoxon_p) carries the verdict
here, and why PairedResult surfaces the floor explicitly via
wilcoxon_floor_note whenever n_pairs < 6 -- so a reader isn't misled by a
"non-significant" Wilcoxon result into thinking the comparison found nothing,
when in fact it couldn't have found anything by that route at this sample size.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class PairedResult:
    """Paired seed-level comparison of two models on one metric, at one size.

    Differences are computed as b - a (positive means model_b scored higher
    that seed) -- the same convention viz.py's _sign_consistent already uses
    for "does lgbm/tabpfn agree in direction across seeds". model_a/model_b
    default to the generic "a"/"b" but carry real model names when built via
    paired_from_curve; verdict itself stays the generic "tie"/"a_wins"/
    "b_wins" label -- consult model_a/model_b to know which actual model
    "a"/"b" refers to.

    wilcoxon_p is None exactly when it's undefined: every paired difference
    is exactly 0 (all seeds tied), so there's no signal left for a rank test.
    wilcoxon_floor_note is set whenever n_pairs < 6 -- see module docstring.
    """

    n_pairs: int
    mean_diff: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    t_p: float
    wilcoxon_p: float | None
    verdict: str
    model_a: str = "a"
    model_b: str = "b"
    wilcoxon_floor_note: str | None = None

    def to_dict(self) -> dict:
        """Plain, JSON-serializable dict -- for the learning_curve_meta.json writer."""
        return asdict(self)


def paired_seed_comparison(
    a: list[float],
    b: list[float],
    *,
    model_a: str = "a",
    model_b: str = "b",
) -> PairedResult:
    """Paired comparison of model_a vs. model_b on matching seeds.

    a[i]/b[i] must be the same metric (e.g. roc_auc) from the *same* shared
    seed i, in matching order -- this function only sees the paired values,
    not the seed numbers themselves; pairing seeds correctly is the caller's
    job (paired_from_curve does it for a learning_curve.csv-shaped frame).

    Works for any n_pairs >= 2 (a CI/t-test needs at least 1 degree of
    freedom); raises ValueError below that or on mismatched lengths.

    Tie-by-default rule (REVIEW.md B-1, arXiv 2605.17273): verdict is "tie"
    unless the 95% CI on the mean difference excludes 0 -- not whichever mean
    happens to be larger. A CI excluding 0 is equivalent to the two-sided
    paired t-test rejecting at alpha=0.05, so verdict never contradicts t_p
    here. wilcoxon_p is reported alongside as the field's non-parametric
    standard check, but does not itself drive verdict -- see the module
    docstring for why (the floor problem at low seed counts).
    """
    if len(a) != len(b):
        raise ValueError(
            f"a and b must be the same length (paired seeds), got {len(a)} vs {len(b)}"
        )
    n = len(a)
    if n < 2:
        raise ValueError(f"need at least 2 paired seeds for a CI/t-test, got {n}")

    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1))

    if sd == 0.0:
        # No variance for a t-test/CI to work with. When the constant is
        # exactly 0 there's also nothing left for Wilcoxon to rank -- reported
        # as None rather than trusting scipy's degenerate-case output (it
        # returns p=1.0 there, but only after an internal divide-by-zero).
        ci_low = ci_high = mean_diff
        if mean_diff == 0.0:
            t_stat, t_p = 0.0, 1.0
            wilcoxon_p = None
        else:
            # Every seed agrees on an identical nonzero gap: perfect
            # separation, i.e. infinitely significant by the t-test's own
            # logic. Wilcoxon still has ranks to work with here (no zeros).
            t_stat, t_p = math.copysign(math.inf, mean_diff), 0.0
            wilcoxon_p = float(scipy_stats.wilcoxon(diffs).pvalue)
    else:
        se = sd / math.sqrt(n)
        t_crit = float(scipy_stats.t.ppf(0.975, n - 1))
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
        t_result = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
        wilcoxon_p = float(scipy_stats.wilcoxon(diffs).pvalue)

    if ci_low > 0:
        verdict = "b_wins"
    elif ci_high < 0:
        verdict = "a_wins"
    else:
        verdict = "tie"

    wilcoxon_floor_note = None
    if n < 6:
        floor = 2.0 ** (1 - n)
        wilcoxon_floor_note = (
            f"n_pairs={n}: Wilcoxon signed-rank's minimum two-sided p-value is "
            f"{floor:.4g} here, so it cannot reach the conventional 0.05 threshold "
            "even for a perfectly consistent effect -- the verdict is carried by "
            "the CI-based tie rule above, not by wilcoxon_p."
        )

    return PairedResult(
        n_pairs=n,
        mean_diff=mean_diff,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        t_stat=float(t_stat),
        t_p=float(t_p),
        wilcoxon_p=wilcoxon_p,
        verdict=verdict,
        model_a=model_a,
        model_b=model_b,
        wilcoxon_floor_note=wilcoxon_floor_note,
    )


def _seed_pairs(
    df: pd.DataFrame, n_train: int, model_a: str, model_b: str, metric: str
) -> list[tuple[float, float]]:
    """(model_a, model_b) metric values for every seed both models share at this size.

    A seed run for only one of the two models at this size can't be paired,
    so it's dropped -- it can't speak to sign consistency either way.

    Assumes at most one row per (model, seed) at this n_train -- i.e. seeds
    are unique within a model/size in the source DataFrame. A duplicate seed
    (e.g. a resumed run that appended a point already present) would make
    `.loc[s]` return more than one value where a scalar is expected, breaking
    the float() cast below; that's a data-quality bug upstream, not something
    this helper is meant to detect.
    """
    sub = df.loc[df["n_train"] == n_train]
    a = sub.loc[sub["model"] == model_a].set_index("seed")[metric]
    b = sub.loc[sub["model"] == model_b].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    return [(float(a.loc[s]), float(b.loc[s])) for s in common]


def paired_from_curve(
    df: pd.DataFrame, model_a: str, model_b: str, metric: str = "roc_auc"
) -> dict[str, PairedResult]:
    """Paired seed-level comparison of model_a vs. model_b, per n_train size.

    Mirrors curve.compute_seed_spread's conventions: a pure aggregation over
    a learning_curve.csv-shaped DataFrame (columns model, n_train, seed,
    <metric>, ...), string keys (str(n_train)) so the result serializes
    straight into learning_curve_meta.json under e.g.
    meta["paired_comparison"]["5000"] (via {k: v.to_dict() for k, v in ...}).

    Only sizes where both models share >=2 paired seeds get a comparison --
    below that there's no CI/t-test to compute, so that size is simply absent
    from the result (never a crash, never a fabricated stat) -- same
    "absent, not fake" convention as compute_seed_spread.
    """
    out: dict[str, PairedResult] = {}
    for n_train in sorted(df["n_train"].unique()):
        pairs = _seed_pairs(df, int(n_train), model_a, model_b, metric)
        if len(pairs) < 2:
            continue
        a_vals, b_vals = zip(*pairs, strict=True)
        out[str(int(n_train))] = paired_seed_comparison(
            list(a_vals), list(b_vals), model_a=model_a, model_b=model_b
        )
    return out
