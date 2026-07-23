"""Render the per-size paired-verdict table and inject it into README.md.

Thin wiring around tabular_showdown.readme_table (same script -> committed-
output pattern as plot_curve.py): the table is computed from the committed
results/learning_curve.csv, printed to stdout, and written into the block
between the

    <!-- stats-table:start --> / <!-- stats-table:end -->

markers in README.md. Regenerable; a test asserts re-running this is a no-op
against the committed README, so hand-edits and post-rerun drift fail loudly.

Run: uv run python scripts/make_stats_table.py
"""

from pathlib import Path

from tabular_showdown.readme_table import inject, render_table

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "learning_curve.csv"
README_PATH = ROOT / "README.md"


def main() -> None:
    table_md = render_table(CSV_PATH)
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
