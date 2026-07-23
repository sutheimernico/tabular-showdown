"""Render the per-size paired-verdict table and inject it into README.md.

Same script -> committed-output pattern as plot_curve.py / make_figures.py:
the verdicts are computed from the committed results/learning_curve.csv via
tabular_showdown.stats.paired_from_curve -- the function is the contract, so
the learning_curve_meta.json copy of these numbers is the run's record, not
re-parsed here -- rendered by viz.verdict_markdown, printed to stdout, and
written into the block between the

    <!-- stats-table:start --> / <!-- stats-table:end -->

markers in README.md. That keeps the README table regenerable and catches
hand-edits: a test asserts re-running this script is a no-op against the
committed README.

Run: uv run python scripts/make_stats_table.py
"""

from pathlib import Path

import pandas as pd

from tabular_showdown import viz
from tabular_showdown.stats import paired_from_curve

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "learning_curve.csv"
README_PATH = ROOT / "README.md"

START = "<!-- stats-table:start -->"
END = "<!-- stats-table:end -->"


def render_table() -> str:
    """The verdict markdown for the committed curve, TabPFN vs. tuned LightGBM."""
    df = pd.read_csv(CSV_PATH)
    results = paired_from_curve(df, "tabpfn", "lgbm")
    return viz.verdict_markdown(results)


def inject(readme: str, table_md: str) -> str:
    """Replace the content between the markers with table_md. Refuses to guess
    where the block goes: raises if either marker is missing or out of order."""
    start = readme.find(START)
    end = readme.find(END)
    if start == -1 or end == -1:
        raise ValueError(f"README markers {START} / {END} not found -- add them first")
    if end < start:
        raise ValueError(f"README end marker precedes start marker ({END} before {START})")
    return f"{readme[: start + len(START)]}\n{table_md}\n{readme[end:]}"


def main() -> None:
    table_md = render_table()
    print(table_md)

    readme = README_PATH.read_text()
    updated = inject(readme, table_md)
    if updated == readme:
        print(f"\n{README_PATH.name} already up to date")
        return
    README_PATH.write_text(updated)
    print(f"\nupdated {README_PATH.name}")


if __name__ == "__main__":
    main()
