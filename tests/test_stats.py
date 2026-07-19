"""Tests for the paired seed-level model-comparison stats (REVIEW.md B-1,
research/2026-07-19-tabular-sota.md section 3): a named significance test
per n_train size, with "tie" as the verdict unless the data proves otherwise.

Hand-computed expectations where practical (diffs [0.01, 0.02, 0.03] -> mean
0.02, sd 0.01, t = 3.4641, CI = 0.02 +/- 4.3027*0.005774), plus a real-CSV
anchor at n=5000 mirroring test_curve.py's REVIEW.md B-1 numbers: only 2
paired seeds today, so the CI is huge and the verdict must be "tie".
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

from tabular_showdown.stats import (
    PairedResult,
    _seed_pairs,
    paired_from_curve,
    paired_seed_comparison,
)

REAL_CURVE_CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "learning_curve.csv"


# --- paired_seed_comparison: hand-computed ----------------------------------


def test_paired_seed_comparison_hand_computed_mean_and_ci():
    """diffs = b - a = [0.01, 0.02, 0.03]: mean 0.02, sd 0.01 (ddof=1),
    t = 0.02 / (0.01 / sqrt(3)) = 3.4641..., df=2, t_crit(0.975, 2) =
    4.302653, CI = 0.02 +/- 4.302653 * 0.0057735 = [-0.004841, 0.044841].
    """
    a = [0.50, 0.50, 0.50]
    b = [0.51, 0.52, 0.53]
    result = paired_seed_comparison(a, b)

    assert result.n_pairs == 3
    assert result.mean_diff == pytest.approx(0.02, abs=1e-9)
    assert result.t_stat == pytest.approx(3.4641016, abs=1e-5)
    assert result.ci95_low == pytest.approx(-0.0048414, abs=1e-6)
    assert result.ci95_high == pytest.approx(0.0448414, abs=1e-6)
    # CI includes 0 (t_p ~0.074 > 0.05) -- tie-by-default rule applies even
    # though the raw t-stat alone might look suggestive.
    assert result.t_p == pytest.approx(0.0741799, abs=1e-6)
    assert result.verdict == "tie"


def test_paired_seed_comparison_wilcoxon_p_matches_scipy_on_the_diffs():
    a = [0.50, 0.50, 0.50]
    b = [0.51, 0.52, 0.53]
    result = paired_seed_comparison(a, b)

    diffs = np.array(b) - np.array(a)
    expected_p = scipy_stats.wilcoxon(diffs).pvalue
    assert result.wilcoxon_p == pytest.approx(expected_p)


def test_paired_seed_comparison_mixed_zero_and_nonzero_diffs_uses_pratt():
    """scipy's default zero_method='wilcox' silently drops zero-difference
    pairs before ranking, shrinking the effective n below n_pairs -- pratt
    keeps them in the ranking instead, so wilcoxon_floor_note's floor (which
    is computed from n_pairs) stays honest. Two of six diffs here are exactly
    0; pratt and the default disagree on this data, so this also guards
    against silently reverting to scipy's default.
    """
    a = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    b = [0.85, 0.86, 0.845, 0.87, 0.85, 0.84]
    result = paired_seed_comparison(a, b)

    diffs = np.array(b) - np.array(a)
    expected_pratt = scipy_stats.wilcoxon(diffs, zero_method="pratt").pvalue
    default_wilcox = scipy_stats.wilcoxon(diffs, zero_method="wilcox").pvalue
    # sanity: this data actually distinguishes pratt from the default
    assert not math.isclose(expected_pratt, default_wilcox)

    assert result.wilcoxon_p == pytest.approx(expected_pratt)
    assert result.wilcoxon_p != pytest.approx(default_wilcox)


def test_paired_seed_comparison_default_model_labels_and_to_dict():
    result = paired_seed_comparison([0.5, 0.5, 0.5], [0.51, 0.52, 0.53])
    assert result.model_a == "a"
    assert result.model_b == "b"

    d = result.to_dict()
    assert d["n_pairs"] == 3
    assert d["verdict"] == "tie"
    json.dumps(d)  # must not raise -- JSON-serializable, no NaN/Inf leaking in


def test_paired_seed_comparison_explicit_model_names_carried_through():
    result = paired_seed_comparison(
        [0.5, 0.5, 0.5], [0.51, 0.52, 0.53], model_a="tabpfn", model_b="lgbm"
    )
    assert result.model_a == "tabpfn"
    assert result.model_b == "lgbm"
    # verdict itself stays the generic a/b label -- model_a/model_b disambiguate.
    assert result.verdict in {"tie", "a_wins", "b_wins"}


# --- degenerate cases --------------------------------------------------------


def test_paired_seed_comparison_degenerate_all_equal_case():
    """a == b at every seed: zero variance, nothing for Wilcoxon to rank."""
    a = [0.9, 0.9, 0.9]
    b = [0.9, 0.9, 0.9]
    result = paired_seed_comparison(a, b)

    assert result.mean_diff == 0.0
    assert result.ci95_low == 0.0
    assert result.ci95_high == 0.0
    assert result.t_stat == 0.0
    assert result.t_p == 1.0
    assert result.wilcoxon_p is None
    assert result.verdict == "tie"


def test_paired_seed_comparison_raises_below_two_pairs():
    with pytest.raises(ValueError, match="2"):
        paired_seed_comparison([0.9], [0.8])


def test_paired_seed_comparison_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_seed_comparison([0.9, 0.8], [0.7])


# --- verdict: CI must exclude 0, not just "the bigger mean" -----------------


def test_paired_seed_comparison_b_wins_when_ci_excludes_zero_above():
    # b consistently ~0.2 higher than a, small but not perfectly tied gaps
    # (a real Wilcoxon rank test, not a tied-differences edge case) -> CI
    # excludes 0.
    a = [0.70, 0.71, 0.69, 0.72, 0.695, 0.705]
    b = [0.902, 0.908, 0.887, 0.921, 0.894, 0.913]
    result = paired_seed_comparison(a, b)
    assert result.ci95_low > 0
    assert result.verdict == "b_wins"


def test_paired_seed_comparison_a_wins_when_ci_excludes_zero_below():
    a = [0.902, 0.908, 0.887, 0.921, 0.894, 0.913]
    b = [0.70, 0.71, 0.69, 0.72, 0.695, 0.705]
    result = paired_seed_comparison(a, b)
    assert result.ci95_high < 0
    assert result.verdict == "a_wins"


def test_paired_seed_comparison_tie_when_two_seeds_disagree_in_sign():
    """Mirrors REVIEW.md B-1's real n=5000 case: only 2 seeds, disagreeing
    sign -> a necessarily huge CI that must include 0."""
    a = [0.913924, 0.913529]  # tabpfn
    b = [0.910595, 0.919883]  # lgbm
    result = paired_seed_comparison(a, b)
    assert result.n_pairs == 2
    assert result.ci95_low < 0 < result.ci95_high
    assert result.verdict == "tie"


# --- wilcoxon floor note -----------------------------------------------------


def test_paired_seed_comparison_wilcoxon_floor_note_below_six_pairs():
    """With n=5 seeds, Wilcoxon's minimum two-sided p-value is 2**(1-5) =
    0.0625 -- it can never reach 0.05, however consistent the effect. The
    note is purely a function of n_pairs, so any n=5 data triggers it."""
    a = [0.10, 0.19, 0.31, 0.42, 0.505]
    b = [0.152, 0.263, 0.348, 0.471, 0.549]
    result = paired_seed_comparison(a, b)
    assert result.n_pairs == 5
    assert result.wilcoxon_floor_note is not None
    assert "0.0625" in result.wilcoxon_floor_note


def test_paired_seed_comparison_no_floor_note_at_six_or_more_pairs():
    a = [0.10, 0.19, 0.31, 0.42, 0.505, 0.603]
    b = [0.152, 0.263, 0.348, 0.471, 0.549, 0.658]
    result = paired_seed_comparison(a, b)
    assert result.n_pairs == 6
    assert result.wilcoxon_floor_note is None


# --- PairedResult dataclass basics -------------------------------------------


def test_paired_result_is_a_dataclass_with_expected_fields():
    result = paired_seed_comparison([0.5, 0.5, 0.5], [0.51, 0.52, 0.53])
    for field in (
        "n_pairs",
        "mean_diff",
        "ci95_low",
        "ci95_high",
        "t_stat",
        "t_p",
        "wilcoxon_p",
        "verdict",
        "model_a",
        "model_b",
        "wilcoxon_floor_note",
    ):
        assert hasattr(result, field)
    assert isinstance(result, PairedResult)
    assert not math.isnan(result.mean_diff)


# --- paired_from_curve: extraction from a learning_curve.csv-shaped df ------


def _synthetic_curve_df():
    import pandas as pd

    rows = []
    # n_train=1000: lgbm has seeds 0,1,2; tabpfn only 0,2 -> 2 paired seeds.
    for seed, lgbm_v, tabpfn_v in [(0, 0.80, 0.852), (1, 0.81, None), (2, 0.82, 0.874)]:
        rows.append({"model": "lgbm", "n_train": 1000, "seed": seed, "roc_auc": lgbm_v})
        if tabpfn_v is not None:
            rows.append({"model": "tabpfn", "n_train": 1000, "seed": seed, "roc_auc": tabpfn_v})
    # n_train=2000: only seed 0 shared -> below the >=2 threshold, must be dropped.
    rows.append({"model": "lgbm", "n_train": 2000, "seed": 0, "roc_auc": 0.90})
    rows.append({"model": "tabpfn", "n_train": 2000, "seed": 0, "roc_auc": 0.91})
    # n_train=5000: lgbm only, tabpfn absent entirely -> must be absent from result.
    rows.append({"model": "lgbm", "n_train": 5000, "seed": 0, "roc_auc": 0.92})
    return pd.DataFrame(rows)


def test_paired_from_curve_uses_only_paired_seeds():
    df = _synthetic_curve_df()
    result = paired_from_curve(df, "tabpfn", "lgbm")
    assert "1000" in result
    assert result["1000"].n_pairs == 2


def test_paired_from_curve_skips_sizes_below_two_paired_seeds():
    df = _synthetic_curve_df()
    result = paired_from_curve(df, "tabpfn", "lgbm")
    assert "2000" not in result


def test_paired_from_curve_skips_size_where_one_model_is_entirely_absent():
    df = _synthetic_curve_df()
    result = paired_from_curve(df, "tabpfn", "lgbm")
    assert "5000" not in result


def test_paired_from_curve_result_values_are_paired_result_instances():
    df = _synthetic_curve_df()
    result = paired_from_curve(df, "tabpfn", "lgbm")
    for v in result.values():
        assert isinstance(v, PairedResult)
        assert v.model_a == "tabpfn"
        assert v.model_b == "lgbm"


def test_paired_from_curve_on_real_csv_n5000_is_a_tie_with_two_pairs():
    """Anchors against REVIEW.md B-1's real numbers: at n=5000 there are only
    2 paired seeds today and they disagree in sign -- verdict must be "tie"."""
    import pandas as pd

    df = pd.read_csv(REAL_CURVE_CSV_PATH)
    result = paired_from_curve(df, "tabpfn", "lgbm")

    r5k = result["5000"]
    assert r5k.n_pairs == 2
    assert r5k.mean_diff == pytest.approx(0.0015, abs=1e-4)  # lgbm - tabpfn
    assert r5k.verdict == "tie"
    assert r5k.wilcoxon_floor_note is not None  # n=2 < 6


def test_paired_from_curve_on_real_csv_excludes_10000_tabpfn_never_ran():
    """TabPFN has no rows at n=10000 (compute valve) -- must be absent, not
    a crash or a fabricated comparison."""
    import pandas as pd

    df = pd.read_csv(REAL_CURVE_CSV_PATH)
    result = paired_from_curve(df, "tabpfn", "lgbm")
    assert "10000" not in result


def test_paired_from_curve_on_real_csv_has_the_expected_sizes():
    import pandas as pd

    df = pd.read_csv(REAL_CURVE_CSV_PATH)
    result = paired_from_curve(df, "tabpfn", "lgbm")
    assert set(result.keys()) == {"200", "500", "1000", "2000", "5000"}


# --- _seed_pairs (shared helper, moved here from viz.py) --------------------


def test_seed_pairs_shared_helper_importable_from_stats():
    df = _synthetic_curve_df()
    pairs = _seed_pairs(df, 1000, "tabpfn", "lgbm", "roc_auc")
    assert len(pairs) == 2
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
