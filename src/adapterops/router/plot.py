"""Draw the operating curve (F11, M3, M4) from the committed report — the chart PRD §5 describes.

`router-report` has always written the curve as JSON and CSV; F11 asks for it *plotted*, and until
now nothing drew it. This reads `runs/router__operating_curve__judged.json` — the judge-labelled
report every dashboard number comes from — and needs no data, model or API.

Two panels share one y-axis: in-distribution (M3) and within-task shift (M4), never blended. The
two policies being compared carry the colour; random and the oracle are reference lines in ink.
The endpoints are F8's baselines: escalating nothing is always-cheap, escalating everything is
always-frontier, identical for every policy. The operating point the dashboard reads — confidence
at a 20% budget — is marked.

    uv run adapterops router-plot
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_FILE = REPO_ROOT / "runs" / "router__operating_curve__judged.json"
OUT_FILE = REPO_ROOT / "runs" / "router__operating_curve__judged.png"
OPERATING_BUDGET = 0.2

# Reference palette (dataviz skill), light surface. Slots 1-2 validate all-pairs; the reference
# lines are ink, not series, so they take no categorical slot.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "confidence": {"label": "Confidence (adapter log-prob)", "color": "#2a78d6", "lw": 2,
                   "marker": "o", "z": 4},
    "router_p_fail": {"label": "Learned router (DeBERTa)", "color": "#eb6834", "lw": 2,
                      "marker": "o", "z": 3},
    "oracle": {"label": "Oracle (upper bound)", "color": INK_2, "lw": 1.25, "marker": None, "z": 2},
    "random": {"label": "Random", "color": MUTED, "lw": 1.25, "marker": None, "z": 2},
}
PANELS = {
    "router_in_distribution": "In-distribution (M3)",
    "router_shift": "Within-task shift (M4)",
}


def point(curve: list[dict], policy: str, budget: float) -> dict:
    return next(r for r in curve if r["policy"] == policy and r["budget"] == budget)


def figure(report: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                         "font.size": 10})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.5, 4.6), sharey=True,
                             facecolor=SURFACE)
    for ax, (name, title) in zip(axes, PANELS.items(), strict=True):
        block = report["populations"][name]["all_tasks"]
        curve = block["curve"]
        ax.set_facecolor(SURFACE)
        ax.axhline(block["local_quality"], color=AXIS, lw=1, zorder=1)
        # Label positions are chosen from the committed curves' shapes: just above the never-escalate
        # line between the confidence curve and it, above the oracle's plateau, and under the 100%
        # endpoint, where no line runs.
        ax.annotate(f"never escalate {block['local_quality']:.3f}", (0.20, block["local_quality"]),
                    xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                    color=INK_2, fontsize=8.5)
        for policy, s in STYLE.items():
            rows = sorted((r for r in curve if r["policy"] == policy), key=lambda r: r["budget"])
            ax.plot([r["escalation_rate"] for r in rows], [r["quality"] for r in rows],
                    color=s["color"], lw=s["lw"], solid_capstyle="round", solid_joinstyle="round",
                    marker=s["marker"], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=s["label"], zorder=s["z"])

        op = point(curve, "confidence", OPERATING_BUDGET)
        ax.plot(op["escalation_rate"], op["quality"], "o", markersize=9, color=STYLE["confidence"]["color"],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"operating point · {op['escalation_rate']:.0%} escalated · {op['quality']:.3f}",
                    (op["escalation_rate"], op["quality"]), xytext=(0.30, 0.855),
                    textcoords="data", ha="left", va="center", color=INK, fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8,
                                "shrinkA": 2, "shrinkB": 6})
        ax.annotate(f"GPT-4o-mini alone {block['frontier_quality']:.3f}",
                    (1.0, block["frontier_quality"]), xytext=(0, -10), textcoords="offset points",
                    ha="right", va="top", color=INK_2, fontsize=8.5)

        ax.set_title(f"{title} · {block['pairs']:,} pairs", color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Share of (ticket, task) pairs escalated to GPT-4o-mini", color=INK_2)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
    axes[0].set_ylabel("Quality retained (share of pairs correct)", color=INK_2)
    axes[0].set_ylim(0.45, 0.88)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(STYLE), frameon=False,
               bbox_to_anchor=(0.058, 0.935), labelcolor=INK_2, fontsize=9)
    fig.suptitle("Routing operating curve — quality vs escalation, drafting graded by GPT-4o",
                 x=0.065, y=0.985, ha="left", color=INK, fontsize=12)
    fig.text(0.065, 0.015, f"Source: {IN_FILE.relative_to(REPO_ROOT)} · points are the 8 budgets "
             "swept; 0% is always-cheap, 100% always-frontier (F8)", color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.8, wspace=0.08)
    return fig


def main(in_file: Path = IN_FILE, out_file: Path = OUT_FILE) -> int:
    if not in_file.exists():
        print(f"  no {in_file} — run `adapterops router-report --judged` first")
        return 2
    fig = figure(json.loads(in_file.read_text()))
    fig.savefig(out_file, dpi=200, facecolor=SURFACE)
    try:
        shown = out_file.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_file
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
