"""Routing policies and the operating curve they are compared on (F8, F9, F11, M3).

A routing policy decides, per (ticket, task) pair, whether to answer locally or escalate
to the frontier. Quality and cost move in opposite directions, so a single number cannot
describe one — the deliverable is a curve, and the comparison is between curves.

**The x-axis is escalation budget, not a raw threshold.** Sweeping each policy's own
threshold puts the policies on incomparable axes: confidence 0.8 and router-probability
0.8 are different amounts of traffic. Sweeping the *budget* — escalate the b% of pairs the
policy ranks highest — reads every policy at the same cost and turns the comparison into
the question that matters: given that you may escalate one pair in five, which policy
picks the best fifth? Each point still records the threshold it corresponds to.

**Five policies, and three of them exist to make the learned router look worse.**

- `always-cheap` and `always-frontier` (F8) are the budget-0 and budget-1 endpoints.
- `random` is the no-information policy. A learned router that does not beat it has
  learned nothing, and this is the cheapest way to find that out.
- `confidence` (F9) is the one that matters. The adapter's own sequence log-probability is
  free at inference time and is the strongest realistic alternative to a learned router.
  D3 committed in advance to reporting the case where it wins.
- `oracle` escalates exactly the pairs escalation *helps* — where the adapter fails and
  the frontier succeeds. Unreachable, and the point: it bounds the headroom any router
  could capture, so "the router recovered 0.4 of the available gap" is sayable instead of a
  bare delta.

  It ranks by gain, not by local failure, and the difference is not academic. A first
  version escalated wherever the adapter was wrong, which is only an upper bound if the
  frontier always succeeds. It does not: on a rehearsal the confidence policy *beat* the
  "oracle" and reported capturing 1.39 of the available headroom. Escalating a pair the
  frontier also fails costs quality and buys nothing, and a bound that does not know that
  is not a bound.

**Escalation is not assumed to succeed.** `frontier_success` is a per-pair column measured
by actually running the frontier model. Where it is missing, the curve can be computed
under an explicit `assume_frontier_success` — but that assumption flatters escalation and
therefore flatters every policy that escalates, so it is recorded in the output rather
than buried in a default.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
"""Eight points; PRD F11 asks for at least five. Dense at the low end because that is the
region a cost-sensitive deployment actually operates in."""


def _ordering(scores: pd.Series, pair_ids: pd.Series) -> np.ndarray:
    """Positions of the pairs, most-escalatable first.

    Ties are broken by pair_id rather than left to the sort's own stability, because a
    policy that scores every pair identically — `oracle` on a task the adapter never
    fails, for one — would otherwise escalate whichever rows the frame happened to hold
    first, and the curve would move when the pool was re-ordered. Positional, not by
    label: the caller's index is not assumed to be a RangeIndex.
    """
    order = pd.DataFrame({"s": scores.to_numpy(), "id": pair_ids.to_numpy()})
    return order.sort_values(["s", "id"], ascending=[False, True]).index.to_numpy()


def escalation_scores(scored: pd.DataFrame, policy: str, seed: int = 20260909) -> pd.Series:
    """Per-pair propensity to escalate — higher means escalate sooner."""
    if policy == "confidence":
        if "mean_logprob" not in scored:
            msg = "confidence policy needs a mean_logprob column from the generation run"
            raise KeyError(msg)
        return -scored.mean_logprob          # least confident escalates first
    if policy == "random":
        return pd.Series(np.random.default_rng(seed).random(len(scored)), index=scored.index)
    if policy == "oracle":
        # Gain from escalating: +1 where the frontier rescues a local failure, -1 where it
        # breaks a local success, 0 where it changes nothing. Ranking by local failure
        # instead makes this a bound only if the frontier never fails.
        if "frontier_success" not in scored:
            msg = ("the oracle needs frontier_success to know which escalations help; "
                   "without it, 'escalate every local failure' is not an upper bound")
            raise KeyError(msg)
        return (scored.frontier_success.astype(float)
                - scored.success.astype(float))
    if policy in scored.columns:             # a learned router's P(failure) column
        return scored[policy]
    msg = f"unknown policy {policy!r} and no column of that name to rank on"
    raise ValueError(msg)


def evaluate_at_budget(
    scored: pd.DataFrame,
    scores: pd.Series,
    budget: float,
    assume_frontier_success: float | None = None,
) -> dict:
    """Quality and cost when the top `budget` share of pairs is escalated."""
    n_escalate = round(budget * len(scored))
    escalate = np.zeros(len(scored), dtype=bool)
    if n_escalate:
        escalate[_ordering(scores, scored.pair_id)[:n_escalate]] = True

    if "frontier_success" in scored:
        frontier = scored.frontier_success.astype(float).to_numpy()
        assumed = None
    else:
        if assume_frontier_success is None:
            msg = ("no frontier_success column and no assume_frontier_success given. "
                   "Escalation quality has to come from somewhere, and the assumption "
                   "flatters every escalating policy — state it explicitly.")
            raise ValueError(msg)
        frontier = np.full(len(scored), float(assume_frontier_success))
        assumed = float(assume_frontier_success)

    local = scored.success.astype(float).to_numpy()
    realised = np.where(escalate, frontier, local)
    score_values = scores.to_numpy()

    row = {
        "budget": budget,
        "escalation_rate": round(float(escalate.mean()), 4),
        "quality": round(float(np.mean(realised)), 4),
        "threshold": (round(float(score_values[escalate].min()), 6) if n_escalate else None),
        "assumed_frontier_success": assumed,
    }
    tasks = scored.task.to_numpy()
    for task in sorted(set(tasks)):
        pos = tasks == task
        row[f"quality__{task}"] = round(float(np.mean(realised[pos])), 4)
        row[f"escalation__{task}"] = round(float(escalate[pos].mean()), 4)
    return row


def operating_curve(
    scored: pd.DataFrame,
    policy: str,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    assume_frontier_success: float | None = None,
    seed: int = 20260909,
) -> pd.DataFrame:
    scores = escalation_scores(scored, policy, seed=seed)
    rows = [evaluate_at_budget(scored, scores, b, assume_frontier_success) for b in budgets]
    return pd.DataFrame(rows).assign(policy=policy)


def compare(
    scored: pd.DataFrame,
    policies: Sequence[str] = ("random", "confidence", "oracle"),
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    assume_frontier_success: float | None = None,
    seed: int = 20260909,
) -> pd.DataFrame:
    """All policies on one frame — F11's "one chart" in tabular form.

    `always-cheap` and `always-frontier` (F8) need no policy of their own: they are the
    budget-0 and budget-1 rows, identical for every policy, which is itself the clearest
    way to show that the endpoints are not where the argument lives.
    """
    frames = [operating_curve(scored, p, budgets, assume_frontier_success, seed)
              for p in policies]
    return pd.concat(frames, ignore_index=True)[
        ["policy", "budget", "escalation_rate", "quality", "threshold",
         "assumed_frontier_success",
         *sorted(c for c in frames[0].columns if c.startswith(("quality__", "escalation__")))]
    ]


def headroom_captured(curve: pd.DataFrame, oracle: pd.DataFrame, budget: float) -> float | None:
    """Share of the oracle's available gain that a policy actually captures at one budget.

    A bare delta over always-cheap is not interpretable: +0.03 is excellent if the oracle
    only gains 0.04 and poor if it gains 0.30. This is the number to report.
    """
    floor = float(curve.loc[curve.budget == 0.0, "quality"].iloc[0])
    got = float(curve.loc[curve.budget == budget, "quality"].iloc[0]) - floor
    available = float(oracle.loc[oracle.budget == budget, "quality"].iloc[0]) - floor
    return None if available <= 0 else round(got / available, 4)
