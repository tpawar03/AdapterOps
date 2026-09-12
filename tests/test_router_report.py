"""Guards on the join that feeds the operating curve (F11, M3).

This module runs for the first time *after* a paid GPU session, so it is tested against
synthetic runs beforehand. Two of these tests exist because the mistakes they catch produce
a chart rather than an error.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import report

TASKS = ("intent", "urgency", "pii", "drafting")


def synthetic(n_per_task: int = 60, seed: int = 3):
    """A local run the frontier beats on average, with a real confidence signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for task in TASKS:
        for i in range(n_per_task):
            fails = rng.random() < 0.3
            rows.append({
                "pair_id": f"{task}-r{i:04d}", "task": task, "purpose": "router",
                "text": f"{task} text {i}", "gold": "g",
                "exact": 0.0 if fails else 1.0,
                "span_f1": 0.4 if fails else 1.0, "span_f1_relaxed": 0.5,
                "span_precision": 0.9, "span_recall": 0.5 if fails else 1.0,
                "pred_spans": 3,
                # Drafting's proxy: failures score low, successes high.
                "proxy_token_f1": 0.10 if fails else 0.80,
                "mean_logprob": -4.0 if fails else -0.2,
                "side": "shift" if i % 3 == 0 else "in_distribution",
            })
    local = pd.DataFrame(rows)

    front = local[["pair_id", "task"]].copy()
    beats = rng.random(len(front)) < 0.8          # frontier succeeds on 80%
    front["frontier_exact"] = beats.astype(float)
    front["frontier_span_f1"] = np.where(beats, 1.0, 0.3)
    front["frontier_span_f1_relaxed"] = 0.5
    front["frontier_span_precision"] = 0.9
    front["frontier_span_recall"] = np.where(beats, 1.0, 0.4)
    front["frontier_pred_spans"] = 3
    front["frontier_proxy_token_f1"] = np.where(beats, 0.85, 0.05)
    return local, front


def test_both_arms_are_cut_at_the_same_drafting_threshold():
    """The invariant D28 rests on.

    If each arm cut at its own median, the frontier's drafting success rate would be ~0.50
    by construction regardless of what it produced — and the curve would compare two
    different bars while looking completely normal.
    """
    local, front = synthetic()
    joined = report.join(local, front)

    cut = joined.attrs["drafting_cut"]
    drafting = joined[joined.task == "drafting"]
    # The frontier's drafting success must follow from the shared cut, not from its own
    # median: recomputing by hand with that one number has to reproduce it exactly.
    expected = ((front.set_index("pair_id")
                 .loc[drafting.pair_id, "frontier_proxy_token_f1"] > 0)
                & (front.set_index("pair_id")
                   .loc[drafting.pair_id, "frontier_proxy_token_f1"] >= cut))
    assert drafting.frontier_success.to_numpy().tolist() == expected.to_numpy().tolist()
    assert drafting.frontier_success.mean() != pytest.approx(0.5, abs=0.001), (
        "frontier drafting success landed exactly at 0.5 — the hallmark of each arm "
        "cutting at its own median")


def test_join_keeps_one_row_per_pair():
    local, front = synthetic()
    joined = report.join(local, front)
    assert len(joined) == len(local)
    assert joined.pair_id.is_unique
    assert joined.frontier_success.notna().all()


def test_every_policy_is_scored_on_the_same_rows():
    """The learned router only predicts on its eval splits. Computing the curve over the
    whole pool would rank the router on missing values while ranking confidence on
    everything — two policies, two populations, one chart, no error."""
    local, front = synthetic()
    eval_ids = local[local.side == "in_distribution"].pair_id.head(40)
    router_scores = pd.DataFrame({"pair_id": eval_ids,
                                  "router_p_fail": np.linspace(0, 1, len(eval_ids))})
    joined = report.join(local, front, router_scores)
    pops = report.populations(joined, router_scores)

    assert pops, "no population selected"
    for name, frame in pops.items():
        assert frame.router_p_fail.notna().all(), (
            f"{name} contains rows the router never scored")
        assert set(frame.pair_id) <= set(eval_ids)


def test_without_a_router_the_population_says_so():
    local, front = synthetic()
    pops = report.populations(report.join(local, front), None)
    assert list(pops) == ["router_slice__no_router_yet"]


def test_report_is_computed_with_and_without_the_proxy_labelled_task():
    """D28 committed to reporting both before any result existed."""
    local, front = synthetic()
    joined = report.join(local, front)
    built = report.build_report(joined, None, ("random", "confidence", "oracle"))
    views = built["populations"]["router_slice__no_router_yet"]
    assert set(views) == {"all_tasks", "real_labels_only"}
    assert "drafting" in views["all_tasks"]["tasks"]
    assert "drafting" not in views["real_labels_only"]["tasks"]
    assert views["real_labels_only"]["pairs"] < views["all_tasks"]["pairs"]


def test_a_population_with_no_failures_reports_a_note_not_a_crash():
    """Possible in real data: a task the adapter never fails, sliced small enough."""
    local, front = synthetic()
    joined = report.join(local, front)
    joined["success"] = True
    built = report.build_report(joined, None, ("random", "confidence", "oracle"))
    view = built["populations"]["router_slice__no_router_yet"]["all_tasks"]
    assert "curve" not in view and "note" in view
