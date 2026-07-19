"""Render the money chart from results/learning_curve.csv.

Thin caller of tabular_showdown.viz.plot_learning_curve, which owns the
actual chart construction (styling, aggregation, and the honest,
noise-aware title -- see viz.learning_curve_title). Writes
results/learning_curve.png and .svg.
"""

import json
from pathlib import Path

import pandas as pd

from tabular_showdown import viz

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "learning_curve.csv"
META_PATH = ROOT / "results" / "learning_curve_meta.json"
PNG_PATH = ROOT / "results" / "learning_curve.png"
SVG_PATH = ROOT / "results" / "learning_curve.svg"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    meta = json.loads(META_PATH.read_text())

    fig = viz.plot_learning_curve(df, meta)

    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight", facecolor=viz.SURFACE)
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=viz.SURFACE)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")
    print(f"title: {fig.axes[0].get_title(loc='left')}")


if __name__ == "__main__":
    main()
