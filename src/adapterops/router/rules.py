"""Rules-based routing baseline (F26) — what a person writes without a model.

PRD F26 asks for a keyword/length rule as a fourth comparison on the operating curve. Two are scored:

- `rules_length` — escalate the longest tickets first. The heuristic a support team reaches for.
- `rules_task_length` — escalate tasks in the order escalation helped them on the **train** split,
  longest ticket first within a task. One lookup table and one sort.

No keyword rule is scored. Any keyword list for four tasks would be picked by hand, and the only
thing to pick it against is the eval split.

**Not pre-registered, and not a claim.** `rules_task_length` was written after D42's per-task
allocation table showed escalation pays only on drafting. Its ordering comes from the train split
alone, but the idea came from reading eval results — so it is a reference line on the curve, not a
result that can be said to beat anything.

    uv run python -m adapterops.router.rules
"""

from __future__ import annotations

import json

import pandas as pd

from adapterops.router.baselines import compare, headroom_captured, operating_curve
from adapterops.router.cascade import (
    BUDGET,
    DATA_DIR,
    REPO_ROOT,
    SPLITS,
    check_reproduces,
    paired_bootstrap,
)

OUT_FILE = REPO_ROOT / "runs" / "router__rules.json"
RULES = ("rules_length", "rules_task_length")
RESAMPLES, SEED = 2000, 20260909
"""The bootstrap D42 pre-registered, reused so the two reports read on one scale."""


def task_order(train: pd.DataFrame) -> pd.Series:
    """Mean gain from escalating every pair of a task, best first — from the train split only."""
    gain = (train.groupby("task").frontier_success.mean()
            - train.groupby("task").success.apply(lambda s: s.astype(float).mean()))
    return gain.sort_values(ascending=False)


def rule_scores(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    length = frame.text.astype(str).str.len().astype(float)
    order = task_order(train)
    rank = frame.task.map({task: len(order) - i for i, task in enumerate(order.index)})
    # Tickets are far shorter than 1e6 characters, so length never crosses a task boundary.
    return pd.DataFrame({"rules_length": length, "rules_task_length": rank * 1e6 + length},
                        index=frame.index)


def main() -> int:
    train = pd.read_parquet(DATA_DIR / "router_train__rescue.parquet")
    order = task_order(train)
    policies = ("random", "confidence", "router_v1", *RULES, "oracle")
    out: dict = {
        "decision": "F26",
        "note": ("Reference lines, not a claim: rules_task_length was written after D42's "
                 "allocation table was read. Its task order is taken from the train split."),
        "task_order_from_train": {t: round(float(g), 4) for t, g in order.items()},
        "keyword_rule": "not scored — a hand-picked keyword list could only be tuned on eval",
        "populations": {},
    }
    for split, population in SPLITS.items():
        frame = pd.read_parquet(DATA_DIR / f"router_{split}__rescue.parquet").reset_index(drop=True)
        frame = frame.join(rule_scores(train, frame))
        v1 = pd.read_parquet(DATA_DIR / f"router_{split}_scored__judged.parquet")
        frame = frame.merge(v1[["pair_id", "router_p_fail"]].rename(
            columns={"router_p_fail": "router_v1"}), on="pair_id", how="left", validate="one_to_one")

        table = compare(frame, policies=policies)
        check_reproduces(table, population)
        oracle = operating_curve(frame, "oracle")
        confidence = (-frame.mean_logprob).to_numpy()
        out["populations"][population] = {
            "pairs": len(frame),
            "headroom_captured": {
                str(b): {p: headroom_captured(table[table.policy == p], oracle, b)
                         for p in policies if p != "oracle"} for b in (0.10, BUDGET)},
            "vs_confidence_at_0.20": {
                r: paired_bootstrap(frame, frame[r].to_numpy(), confidence, BUDGET, RESAMPLES, SEED)
                for r in RULES},
            "at_budget_0.20": table[table.budget == BUDGET].to_dict("records"),
            "curve": table.to_dict("records"),
        }
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")

    print(f"  task order from train: {out['task_order_from_train']}")
    for population, block in out["populations"].items():
        print(f"\n  === {population} ({block['pairs']} pairs) ===")
        print(f"  headroom captured at 0.20: {block['headroom_captured'][str(BUDGET)]}")
        for r, res in block["vs_confidence_at_0.20"].items():
            print(f"  {r:18s} vs confidence {res['difference']:+.4f}  95% CI {res['ci95']}")
    print(f"\n  wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
