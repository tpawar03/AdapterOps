"""Guards on the Phase 3 report: D28, the same-family check, and escalation under the judge.

Synthetic grades throughout. The test worth reading is the last one: a partial grading run
must say it is partial, because a report built on 384 of 1,950 grades looks exactly like
one built on all of them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import report as jr


def make(n: int = 40, seed: int = 0, frontier_score: int | None = None):
    rng = np.random.default_rng(seed)
    ids = [f"drafting-r{i:04d}" for i in range(n)]
    proxy = rng.random(n)
    local_score = np.clip(np.rint(1 + 4 * proxy), 1, 5).astype(int)
    front = (np.full(n, frontier_score) if frontier_score is not None
             else rng.integers(1, 6, n))
    scored = pd.concat([
        pd.DataFrame({"pair_id": ids, "task": "drafting", "purpose": "router",
                      "proxy_token_f1": proxy}),
        pd.DataFrame({"pair_id": ["intent-r0000"], "task": ["intent"], "purpose": ["router"],
                      "proxy_token_f1": [np.nan]}),
    ], ignore_index=True)
    judgments = pd.concat([
        pd.DataFrame({"item_id": ["local:" + i for i in ids], "source": "local", "pair_id": ids,
                      "purpose": "router", "split": "train", "score": local_score,
                      "parsed_ok": True, "proxy_token_f1": proxy}),
        pd.DataFrame({"item_id": ["frontier:" + i for i in ids], "source": "frontier",
                      "pair_id": ids, "purpose": "router", "split": "frontier_eval",
                      "score": front, "parsed_ok": True, "proxy_token_f1": rng.random(n)}),
    ], ignore_index=True)
    return judgments, scored


def test_the_proxy_cut_is_the_router_drafting_median_the_labels_used():
    judgments, scored = make()
    out = jr.build_report(judgments, scored)
    expected = float(scored[scored.task == "drafting"].proxy_token_f1.median())
    assert out["escalation_under_judge"]["proxy_cut"] == pytest.approx(expected)


def test_d28_depends_on_local_replies_only():
    """The proxy label lives on the adapter's replies. Frontier grades must not move it."""
    a = jr.build_report(*make(frontier_score=5))["d28_proxy_vs_judge"]
    b = jr.build_report(*make(frontier_score=1))["d28_proxy_vs_judge"]
    assert a == b
    assert a["spearman"] > 0.9, "the synthetic proxy was built to track the local grade"


def test_rescued_and_broken_follow_from_the_judge_threshold():
    ids = ["a", "b", "c", "d"]
    local = pd.Series([5.0, 2.0, 5.0, 2.0], index=ids)
    frontier = pd.Series([2.0, 5.0, 5.0, 2.0], index=ids)
    proxy = pd.Series([0.9, 0.1, 0.9, 0.1], index=ids)
    out = jr.escalation_under_judge(local, frontier, proxy, proxy, cut=0.5)
    assert out["judge"]["broken"] == 0.25      # a: local good, frontier bad
    assert out["judge"]["rescued"] == 0.25     # b: local bad, frontier good
    assert out["judge_success_min"] == 4


def test_unparsed_grades_and_unpaired_items_are_left_out():
    judgments, scored = make()
    judgments.loc[judgments.item_id == "local:drafting-r0000", "parsed_ok"] = False
    judgments = judgments[judgments.item_id != "frontier:drafting-r0001"]
    out = jr.build_report(judgments, scored)
    assert out["escalation_under_judge"]["pairs"] == 38


def test_a_partial_grading_run_says_it_is_partial():
    out = jr.build_report(*make(n=40))
    assert out["coverage"]["complete"] is False
    assert "provisional" in out["coverage"]["note"]
