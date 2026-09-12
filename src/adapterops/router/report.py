"""Join the local and frontier runs and produce the operating curve (F11, M3).

This is where the three inputs meet: the local run's per-pair success and confidence, the
frontier run's per-pair success, and the learned router's per-pair failure probability.
Everything before this point was careful about keeping them comparable; this module is
where being careless would cash out.

**Two invariants it enforces rather than assumes.**

`drafting`'s proxy cut is taken from the local run and applied to both arms. Cutting each
arm at its own median would give the frontier a 50% drafting success rate whatever it
actually produced, and the curve would be comparing two different bars while looking
entirely normal.

The curve is reported **twice** — over all four tasks and over the three with real labels.
D28 committed to that before any result existed. If the two readings disagree, the
disagreement is the finding: it says how much of the headline rests on the one task whose
label is a stand-in.

    uv run adapterops router-report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adapterops.router.baselines import (
    DEFAULT_BUDGETS,
    compare,
    headroom_captured,
    operating_curve,
)
from adapterops.router.scoring import PROXY_LABELLED_TASKS, drafting_cut_from, label

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
OUT_JSON = REPO_ROOT / "runs" / "router__operating_curve.json"
OUT_CSV = REPO_ROOT / "runs" / "router__operating_curve.csv"


def join(local: pd.DataFrame, frontier: pd.DataFrame,
         router_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pair, carrying both arms' success under a single set of rules."""
    cut = drafting_cut_from(local)
    local = label(local, drafting_cut=cut)

    front = frontier.rename(columns={c: c.removeprefix("frontier_")
                                     for c in frontier.columns
                                     if c.startswith("frontier_")})
    front = label(front, drafting_cut=cut)[["pair_id", "success"]]
    front = front.rename(columns={"success": "frontier_success"})

    joined = local.merge(front, on="pair_id", how="inner", validate="one_to_one")
    joined.attrs["drafting_cut"] = cut
    if router_scores is not None:
        joined = joined.merge(router_scores[["pair_id", "router_p_fail"]],
                              on="pair_id", how="left", validate="one_to_one")
    return joined


def curves(joined: pd.DataFrame, policies: tuple[str, ...],
           budgets=DEFAULT_BUDGETS) -> dict:
    """The comparison, computed twice: all tasks, and real-labelled tasks only (D28)."""
    views = {
        "all_tasks": joined,
        "real_labels_only": joined[~joined.task.isin(PROXY_LABELLED_TASKS)],
    }
    out: dict = {}
    for view, frame in views.items():
        table = compare(frame, policies=policies, budgets=budgets)
        oracle = operating_curve(frame, "oracle", budgets)
        out[view] = {
            "pairs": len(frame),
            "tasks": sorted(frame.task.unique()),
            "local_quality": round(float(frame.success.mean()), 4),
            "frontier_quality": round(float(frame.frontier_success.mean()), 4),
            "curve": table.to_dict("records"),
            "headroom_captured_at_0.20": {
                p: headroom_captured(table[table.policy == p], oracle, 0.20)
                for p in policies if p != "oracle"
            },
        }
    return out


def main() -> int:
    local_path = DATA_DIR / "scored.parquet"
    frontier_path = DATA_DIR / "frontier.parquet"
    if not local_path.exists():
        print(f"  no {local_path.relative_to(REPO_ROOT)} — the pool has not been scored "
              f"through the adapters yet (GPU session).")
        return 2
    if not frontier_path.exists():
        print(f"  no {frontier_path.relative_to(REPO_ROOT)} — run `adapterops frontier`.")
        return 2

    local = pd.read_parquet(local_path)
    local = local[local.purpose == "router"]
    frontier = pd.read_parquet(frontier_path)

    router_path = DATA_DIR / "router_eval_scored.parquet"
    router_scores = pd.read_parquet(router_path) if router_path.exists() else None
    policies = ("random", "confidence", "oracle")
    if router_scores is not None:
        policies = ("random", "confidence", "router_p_fail", "oracle")

    joined = join(local, frontier, router_scores)
    result = {
        "note": "Escalation quality is measured, not assumed — frontier_success comes "
                "from an actual GPT-4o-mini run over the same pairs (F8).",
        "drafting_cut": joined.attrs["drafting_cut"],
        "drafting_cut_note": "taken from the local run and applied to both arms, so the "
                             "two are scored against one bar (D28)",
        "pairs": len(joined),
        "views": curves(joined, policies),
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["views"]["all_tasks"]["curve"]).to_csv(OUT_CSV, index=False)

    for view, block in result["views"].items():
        print(f"\n  === {view} ({block['pairs']:,} pairs) ===")
        print(f"  local {block['local_quality']:.4f} · frontier {block['frontier_quality']:.4f}")
        table = pd.DataFrame(block["curve"])
        print(table.pivot(index="budget", columns="policy", values="quality").to_string())
        print(f"  headroom captured at 20% budget: {block['headroom_captured_at_0.20']}")
    print(f"\n  wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0
