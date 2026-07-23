"""Tests for scripts/make_stats_table.py -- the README verdict-table generator.

Two guarantees:
  1. the table the script renders from the committed results/learning_curve.csv
     still carries the pinned per-size verdicts (mirrors test_stats.py's
     real-CSV anchors, one surface up -- the numbers a reader actually sees);
  2. the block committed in README.md is exactly what the script regenerates,
     so a hand-edit -- or a curve rerun not followed by
     `uv run python scripts/make_stats_table.py` -- fails here instead of
     silently drifting.

scripts/ is not an importable package, so the script is loaded by path.
Its module top level only defines functions (main is __main__-guarded), so
importing it has no side effects.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "make_stats_table.py"
README_PATH = ROOT / "README.md"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_stats_table", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_stats_table = _load_script()


def test_readme_stats_table_block_is_up_to_date():
    """Regenerating from the committed CSV must be a no-op against README --
    catches hand-edits and post-rerun drift."""
    readme = README_PATH.read_text()
    regenerated = make_stats_table.inject(readme, make_stats_table.render_table())
    assert regenerated == readme, (
        "README stats-table block is stale -- run: uv run python scripts/make_stats_table.py"
    )


def test_render_table_carries_the_pinned_verdicts():
    md = make_stats_table.render_table()

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
        make_stats_table.inject("readme without any markers", "table")


def test_inject_raises_when_markers_out_of_order():
    bad = f"{make_stats_table.END}\n{make_stats_table.START}"
    with pytest.raises(ValueError, match="precedes"):
        make_stats_table.inject(bad, "table")
