"""Generate the per-size verdict table and inject it into README.md.

The logic scripts/make_stats_table.py wires together: render the table from
the committed learning curve (viz.verdict_markdown over stats.paired_from_curve),
and replace the block between the README marker comments. Kept here rather than
in the script so it is importable and unit-tested directly -- the repo keeps
scripts/ thin, pure wiring around src/ logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabular_showdown import viz
from tabular_showdown.stats import paired_from_curve

START_MARKER = "<!-- stats-table:start -->"
END_MARKER = "<!-- stats-table:end -->"


def render_table(csv_path: Path) -> str:
    """Verdict markdown for the committed curve, TabPFN vs. tuned LightGBM.

    Reads csv_path via stats.paired_from_curve (the function is the contract;
    learning_curve_meta.json's copy of these numbers is the run's record, not
    re-parsed here) and renders it with viz.verdict_markdown.
    """
    df = pd.read_csv(csv_path)
    return viz.verdict_markdown(paired_from_curve(df, "tabpfn", "lgbm"))


def inject(readme: str, table_md: str) -> str:
    """Replace the content between the markers with table_md. Refuses to guess
    where the block goes: raises if either marker is missing or out of order."""
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start == -1 or end == -1:
        raise ValueError(
            f"README markers {START_MARKER} / {END_MARKER} not found -- add them first"
        )
    if end < start:
        raise ValueError(
            f"README end marker precedes start marker ({END_MARKER} before {START_MARKER})"
        )
    return f"{readme[: start + len(START_MARKER)]}\n{table_md}\n{readme[end:]}"
