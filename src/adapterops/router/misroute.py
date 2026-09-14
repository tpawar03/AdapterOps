"""Router-misroute records (F29, F37) — the routing decisions that cost quality or money.

Adapter failures have been persisted since Phase 2 as the hard-cases split. The router's own errors
never were: the operating curve reports what a policy's decisions add up to, not which decisions
were wrong. This persists them, tagged by component and failure type, as a **diagnostic set that is
reported and never gated** (F37) — a misroute is a router defect, and an adapter failure is a model
defect; blending the two is what PRD changelog 14 removed.

**A misroute is defined against the gain from escalating**, the same quantity the oracle ranks by
(D32): `frontier_success − success`, so +1 where escalation rescues a local failure, −1 where it
breaks a local success, 0 where it changes nothing. At the dashboard's 20% operating point:

- `harmful_escalation` — escalated, gain −1. Costs quality and money.
- `wasted_escalation` — escalated, gain 0. Costs money only.
- `missed_rescue` — kept local, gain +1. Forgoes quality the budget could have bought.

An escalated rescue, or a pair kept local that escalation would not have helped, is a correct
decision. `missed_rescue` is only a fair charge while the budget could have covered every rescue;
the output records whether it could.

**Decisions come from the curve's own code** (`baselines.escalation_scores` and its tie-break), on
the judged report's frames, and nothing is written unless three checks hold: each policy's quality
from these decisions matches the published curve; that quality equals never-escalating plus
`(rescued − harmful) / pairs`, which is the tagging checking itself; and no misroute pair is in the
hard-cases split, which D29's allocation guarantees and this re-verifies.

Two policies are recorded: the learned router, whose defects these are, and confidence, the
operating choice, so each misroute count has a comparator. Misroutes need router scores and a
frontier run, so they are produced after `router-report`, not by `regress` — no router runs in the
serving path.

    uv run adapterops router-misroutes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.router.baselines import _ordering, escalation_scores, evaluate_at_budget
from adapterops.router.report import frontier_grades, join_judged, populations

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
PUBLISHED = REPO_ROOT / "runs" / "router__operating_curve__judged.json"
HARD_FILE = REPO_ROOT / "evals" / "hard" / "hard_cases.parquet"
OUT_DIR = REPO_ROOT / "evals" / "misroutes"
OUT_FILE = OUT_DIR / "misroutes.parquet"
SUMMARY = REPO_ROOT / "runs" / "router__misroutes.json"

BUDGET = 0.20
POLICIES = ("router_p_fail", "confidence")
TYPES = ("harmful_escalation", "wasted_escalation", "missed_rescue")
KEEP = ("pair_id", "task", "side", "text", "gold", "prediction", "success", "frontier_success")


def tag(frame: pd.DataFrame, scores: pd.Series, budget: float = BUDGET) -> pd.DataFrame:
    """Each pair's decision at `budget`, its gain from escalating, and its misroute type (or None)."""
    n_escalate = round(budget * len(frame))
    escalated = np.zeros(len(frame), dtype=bool)
    if n_escalate:
        escalated[_ordering(scores, frame.pair_id)[:n_escalate]] = True
    gain = (frame.frontier_success.astype(int) - frame.success.astype(int)).to_numpy()
    kind = np.select(
        [escalated & (gain < 0), escalated & (gain == 0), ~escalated & (gain > 0)],
        list(TYPES), default="")
    failure_type = pd.Series(np.where(kind == "", None, kind), index=frame.index, dtype=object)
    return frame.assign(escalated=escalated, gain=gain, escalation_score=scores.to_numpy(),
                        failure_type=failure_type)


def summarise(tagged: pd.DataFrame) -> dict:
    """Counts per type, overall and per task, plus the identity the tags must satisfy."""
    n = len(tagged)
    rescued = int((tagged.escalated & (tagged.gain > 0)).sum())
    harmful = int((tagged.failure_type == "harmful_escalation").sum())
    local = float(tagged.success.astype(float).mean())
    realised = np.where(tagged.escalated, tagged.frontier_success, tagged.success).astype(float)
    counts = tagged.failure_type.value_counts()
    return {
        "pairs": n,
        "escalated": int(tagged.escalated.sum()),
        "rescued": rescued,
        **{t: int(counts.get(t, 0)) for t in TYPES},
        "rescues_available": int((tagged.gain > 0).sum()),
        "budget_could_cover_all_rescues": bool((tagged.gain > 0).sum() <= tagged.escalated.sum()),
        "quality": round(float(realised.mean()), 4),
        "quality_from_tags": round(local + (rescued - harmful) / n, 4),
        "per_task": {
            task: {"pairs": len(g), "escalated": int(g.escalated.sum()),
                   **{t: int((g.failure_type == t).sum()) for t in TYPES}}
            for task, g in tagged.groupby("task")
        },
    }


def check(summary: dict, published_quality: float, population: str, policy: str) -> None:
    if abs(summary["quality"] - published_quality) > 1e-4:
        msg = (f"{population}/{policy}: decisions give quality {summary['quality']}, the published "
               f"curve {published_quality} — these are not the decisions the curve measured")
        raise ValueError(msg)
    if abs(summary["quality"] - summary["quality_from_tags"]) > 1e-4:
        msg = f"{population}/{policy}: tags do not account for the quality — tagging is wrong"
        raise ValueError(msg)


def frames() -> dict[str, pd.DataFrame]:
    """The judged report's populations, rebuilt exactly as `router-report --judged` builds them."""
    local = pd.read_parquet(DATA_DIR / "scored__judged.parquet")
    local = local[local.purpose == "router"]
    router_scores = pd.concat(
        [pd.read_parquet(DATA_DIR / f"router_{n}_scored__judged.parquet")
         for n in ("eval", "shift_eval")], ignore_index=True)
    joined = join_judged(local, pd.read_parquet(DATA_DIR / "frontier.parquet"),
                         frontier_grades(), router_scores)
    return populations(joined, router_scores)


def main() -> int:
    published = json.loads(PUBLISHED.read_text())["populations"]
    hard_ids = set(pd.read_parquet(HARD_FILE).pair_id)
    records, out = [], {}
    for population, frame in frames().items():
        curve = published[population]["all_tasks"]["curve"]
        out[population] = {}
        for policy in POLICIES:
            scores = escalation_scores(frame, policy)
            tagged = tag(frame, scores)
            summary = summarise(tagged)
            quality = next(r["quality"] for r in curve
                           if r["policy"] == policy and r["budget"] == BUDGET)
            check(summary, quality, population, policy)
            if evaluate_at_budget(frame, scores, BUDGET)["quality"] != summary["quality"]:
                msg = f"{population}/{policy}: tag() and evaluate_at_budget disagree"
                raise ValueError(msg)
            out[population][policy] = summary
            wrong = tagged[tagged.failure_type.notna()]
            records.append(wrong[[*KEEP, "escalated", "gain", "escalation_score", "failure_type"]]
                           .assign(population=population, policy=policy, budget=BUDGET,
                                   component="router"))

    misroutes = pd.concat(records, ignore_index=True)
    leaked = set(misroutes.pair_id) & hard_ids
    if leaked:
        msg = f"{len(leaked)} misroute pairs are in the hard-cases split — D29's allocation broke"
        raise ValueError(msg)

    OUT_DIR.mkdir(exist_ok=True)
    misroutes.to_parquet(OUT_FILE)
    SUMMARY.write_text(json.dumps({
        "prd": "F29 router-misroute records, F37 diagnostic set — reported, never gated",
        "budget": BUDGET,
        "definition": {
            "gain": "frontier_success - success, on the judged report's labels (D38)",
            "harmful_escalation": "escalated, gain -1 — costs quality and money",
            "wasted_escalation": "escalated, gain 0 — costs money only",
            "missed_rescue": "kept local, gain +1 — forgoes quality",
        },
        "records": {"file": str(OUT_FILE.relative_to(REPO_ROOT)), "rows": len(misroutes)},
        "checks": ["quality matches runs/router__operating_curve__judged.json",
                   "quality = never-escalate + (rescued - harmful) / pairs",
                   "no misroute pair is in evals/hard/hard_cases.parquet"],
        "populations": out,
    }, indent=2) + "\n")

    for population, by_policy in out.items():
        for policy, s in by_policy.items():
            print(f"  {population:24s} {policy:14s} escalated {s['escalated']:4d} · rescued "
                  f"{s['rescued']:3d} · harmful {s['harmful_escalation']:3d} · wasted "
                  f"{s['wasted_escalation']:3d} · missed rescue {s['missed_rescue']:3d}")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)} ({len(misroutes):,} rows) and "
          f"{SUMMARY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
