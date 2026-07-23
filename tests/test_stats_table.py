"""Tests for the README verdict-table generator (tabular_showdown.readme_table,
wired thinly by scripts/make_stats_table.py).

Two guarantees:
  1. the table rendered from the committed results/learning_curve.csv still
     carries the pinned per-size verdicts (mirrors test_stats.py's real-CSV
     anchors -- the numbers a reader actually sees);
  2. the block committed in README.md is exactly what regeneration produces,
     so a hand-edit -- or a curve rerun not followed by
     `uv run python scripts/make_stats_table.py` -- fails here, not silently.
"""

from pathlib import Path

import pytest

from tabular_showdown.readme_table import END_MARKER, START_MARKER, inject, render_table

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "learning_curve.csv"
README_PATH = ROOT / "README.md"


def test_readme_stats_table_block_is_up_to_date():
    """Regenerating from the committed CSV must be a no-op against README --
    catches hand-edits and post-rerun drift."""
    readme = README_PATH.read_text()
    regenerated = inject(readme, render_table(CSV_PATH))
    assert regenerated == readme, (
        "README stats-table block is stale -- run: uv run python scripts/make_stats_table.py"
    )


def test_render_table_carries_the_pinned_verdicts():
    md = render_table(CSV_PATH)

    # n=5000 tie: mirrors test_stats.py's real-CSV anchor (raw mean_diff
    # lgbm-tabpfn = -0.00137, CI straddles 0). Displayed TabPFN - LightGBM
    # flips to +0.0014, and the verdict is a tie carried by the CI, with the
    # Wilcoxon-floor dagger.
    assert "| 5,000 | 5 | +0.0014 | [−0.0045, +0.0072] |" in md
    assert "Wilcoxon p=0.625 † | tie |" in md

    # small-n TabPFN win: CI excludes 0.
    assert "| 2,000 | 10 | +0.0033 | [+0.0013, +0.0053] |" in md
    assert "| TabPFN wins |" in md

    # every size with >=2 shared seeds present, none fabricated or dropped.
    for size in ("200", "500", "1,000", "2,000", "5,000"):
        assert f"| {size} |" in md
    # 10000/32561 have no paired TabPFN row -> must be absent.
    assert "| 10,000 |" not in md


def test_inject_raises_without_markers():
    with pytest.raises(ValueError, match="markers"):
        inject("readme without any markers", "table")


def test_inject_raises_when_markers_out_of_order():
    bad = f"{END_MARKER}\n{START_MARKER}"
    with pytest.raises(ValueError, match="precedes"):
        inject(bad, "table")
