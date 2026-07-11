"""Render the money chart from results/learning_curve.csv.

x = training-set size (log scale), y = ROC-AUC on the frozen 4,000-row eval
set. One line per model with markers at measured points, a min-max band
across subsample seeds, direct labels at each line's right edge. The title
states the actual finding computed from the data -- where (or whether) tuned
LightGBM overtakes TabPFN -- rather than a canned claim.

Writes results/learning_curve.png and .svg.
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # file output only; no display needed (headless WSL)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "learning_curve.csv"
META_PATH = ROOT / "results" / "learning_curve_meta.json"
PNG_PATH = ROOT / "results" / "learning_curve.png"
SVG_PATH = ROOT / "results" / "learning_curve.svg"

# House chart style -- fixed per model everywhere, never reassigned.
MODEL_COLORS = {"tabpfn": "#2a78d6", "lgbm": "#1baf7a", "logreg": "#eda100"}
MODEL_LABELS = {"tabpfn": "TabPFN v2", "lgbm": "LightGBM (tuned)", "logreg": "LogReg (untuned)"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

TABPFN_PRETRAIN_CAP = 10_000


def _fmt_n(n: int) -> str:
    if n < 1000:
        return str(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"


def _finding_title(agg: pd.DataFrame, tabpfn_stopped_at_pretrain_cap: bool) -> str:
    """Derive the headline sentence from the data, honestly.

    Compares mean ROC-AUC of tabpfn vs lgbm at every size where both were
    measured; reports the first size where LightGBM catches up, or -- if it
    never does within the measured TabPFN range -- says so, and is careful
    about *why* TabPFN's line ends where it does: the 10k pretraining cap is
    a real model limit, but a shorter stop is just the CPU compute valve, not
    a quality ceiling, so we don't dress it up as "TabPFN's cap".
    """
    both = agg.loc[agg["model"].isin(["tabpfn", "lgbm"])].pivot(
        index="n_train", columns="model", values="mean"
    )
    both = both.dropna()
    overtaken = both[both["lgbm"] >= both["tabpfn"]]
    if overtaken.empty:
        last_n = _fmt_n(int(both.index.max()))
        if tabpfn_stopped_at_pretrain_cap:
            return (
                f"TabPFN wins the small-data regime and never lets tuned LightGBM "
                f"catch up, right up to its {last_n}-row pretraining cap"
            )
        return (
            f"TabPFN wins the small-data regime; tuned LightGBM never catches up "
            f"through {last_n} rows (the largest we ran TabPFN on CPU)"
        )
    cross_n = int(overtaken.index.min())
    return (
        f"TabPFN wins the small-data regime; tuned LightGBM overtakes it "
        f"by ~{_fmt_n(cross_n)} rows"
    )


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    meta = json.loads(META_PATH.read_text())

    agg = (
        df.groupby(["model", "n_train"])["roc_auc"]
        .agg(mean="mean", lo="min", hi="max")
        .reset_index()
        .sort_values("n_train")
    )

    tabpfn_max_n = int(df.loc[df["model"] == "tabpfn", "n_train"].max())
    tabpfn_stopped_at_pretrain_cap = tabpfn_max_n == TABPFN_PRETRAIN_CAP

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for model in ["logreg", "lgbm", "tabpfn"]:  # draw order: headline series on top
        color = MODEL_COLORS[model]
        series = agg.loc[agg["model"] == model]
        ax.fill_between(series["n_train"], series["lo"], series["hi"], color=color, alpha=0.15)
        ax.plot(
            series["n_train"],
            series["mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            label=MODEL_LABELS[model],
        )
        # Direct label at the right edge of each line, in the series hue.
        last = series.iloc[-1]
        ax.annotate(
            MODEL_LABELS[model],
            xy=(last["n_train"], last["mean"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=color,
        )

    ax.set_xscale("log")
    sizes = sorted(df["n_train"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([_fmt_n(n) for n in sizes])
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.minorticks_off()

    ax.set_xlabel("training rows (log scale)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("ROC-AUC (frozen eval set)", color=INK_MUTED, fontsize=10)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)

    # Mark where TabPFN's line stops, and say honestly WHY: the 10k pretraining
    # limit is a real model cap, but a shorter stop here is the CPU compute
    # valve (a point exceeded the per-point time budget), not a quality ceiling.
    if tabpfn_stopped_at_pretrain_cap:
        cap_caption = "TabPFN v2 pretraining cap "
    else:
        cap_caption = "TabPFN stops here: >8 min/point on CPU "
    ax.axvline(tabpfn_max_n, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 3)))
    # Anchor left of the line, in the empty bottom-left region, so the caption
    # never collides with the lower-right legend.
    ax.text(
        tabpfn_max_n,
        ax.get_ylim()[0] + 0.015,
        cap_caption,
        color=INK_MUTED,
        fontsize=8.5,
        va="bottom",
        ha="right",
    )

    # Leave room on the right for the direct end-of-line labels.
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax.set_title(
        _finding_title(agg, tabpfn_stopped_at_pretrain_cap),
        color=INK_PRIMARY,
        fontsize=13,
        loc="left",
        pad=24,
    )
    ax.text(
        0,
        1.02,
        f"Adult census income · frozen {meta['n_eval']:,}-row eval set · "
        "bands = min–max over subsample seeds",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=9.5,
        va="bottom",
    )

    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")
    print(f"title: {_finding_title(agg, tabpfn_stopped_at_pretrain_cap)}")


if __name__ == "__main__":
    main()
