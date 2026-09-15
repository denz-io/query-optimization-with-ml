"""The closing chart: baseline vs Smart across selectivity. Reads results/sweep.csv.

Two things to point at: Smart's curve heads toward zero as selectivity drops, and at 100%
the two curves converge rather than cross - every derived predicate is pulled up and
merged away, so there is no overhead penalty.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "sweep.csv"
PNG = ROOT / "charts" / "selectivity.png"

BASELINE = "#2a78d6"   # categorical slot 1
SMART = "#eb6834"      # categorical slot 2
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def make_chart(csv_path: Path = CSV, out: Path = PNG) -> Path:
    rows = list(csv.DictReader(csv_path.open()))
    sel = [float(r["selectivity_actual"]) * 100 for r in rows]
    base = [float(r["baseline_ms"]) for r in rows]
    smart = [float(r["smart_ms"]) for r in rows]
    rb = [int(r["rows_base"]) for r in rows]
    rs = [int(r["rows_smart"]) for r in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, facecolor="#fcfcfb",
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xscale("log")

    def series(ax, ys, color, label, fmt):
        ax.plot(sel, ys, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=2, label=label, zorder=3)
        # direct labels at both ends only
        for i in (0, len(sel) - 1):
            ax.annotate(fmt(ys[i]), (sel[i], ys[i]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=INK)

    series(ax1, base, BASELINE, "baseline: ML UDF on top of the full join", lambda v: f"{v / 1000:.1f} s")
    series(ax1, smart, SMART, "Smart: derived predicates pushed below the joins", lambda v: f"{v / 1000:.1f} s")
    ax1.set_yscale("log")
    ax1.set_ylabel("execution time (ms, log)")
    ax1.set_title("Predicted-price query, 3M apartments x 4-way join: execution time vs ML predicate selectivity",
                  loc="left", fontsize=11, color=INK, pad=14)
    ax1.legend(frameon=False, loc="lower right", fontsize=9)

    series(ax2, rb, BASELINE, "baseline", lambda v: f"{v / 1e6:.2g}M")
    series(ax2, rs, SMART, "Smart", lambda v: f"{v / 1e6:.2g}M" if v >= 1e6 else f"{v / 1e3:.0f}k")
    ax2.set_yscale("log")
    ax2.set_ylabel("rows fed into inference (log)")
    ax2.set_xlabel("selectivity of the ML predicate (% of joined rows kept, log)")
    ax2.set_xticks(sel)
    ax2.set_xticklabels([f"{s:.3g}%" for s in sel])
    ax2.minorticks_off()

    fig.text(0.01, 0.005, "min of N runs per point; both curves include the retained ML predicate. "
             "At 100% every derived predicate merges away: the curves converge, they do not cross.",
             fontsize=8, color=MUTED)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    return out


if __name__ == "__main__":
    print(make_chart())
