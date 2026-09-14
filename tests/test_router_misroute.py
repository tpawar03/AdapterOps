"""Guards on router-misroute tagging (F29, F37)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import misroute
from adapterops.router.baselines import evaluate_at_budget

SUMMARY = ROOT / "runs" / "router__misroutes.json"


def frame() -> pd.DataFrame:
    # (success, frontier_success): gains +1, -1, 0, 0, +1, 0, -1, 0, 0, 0
    arms = [(0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 1), (1, 0), (1, 1), (0, 0), (1, 1)]
    return pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(10)],
        "task": ["intent", "pii"] * 5,
        "success": [s for s, _ in arms],
        "frontier_success": [f for _, f in arms],
    })


def test_each_decision_gets_the_type_its_gain_implies():
    f = frame()
    # Escalate p1 (breaks a success), p3 (changes nothing) at a 20% budget; p0 and p4 are missed.
    scores = pd.Series([0, 9, 1, 8, 0, 0, 0, 0, 0, 0], dtype=float)
    t = misroute.tag(f, scores, budget=0.2).set_index("pair_id").failure_type
    assert t["p1"] == "harmful_escalation"
    assert t["p3"] == "wasted_escalation"
    assert t["p0"] == "missed_rescue" and t["p4"] == "missed_rescue"
    assert t[["p2", "p5", "p6", "p7", "p8", "p9"]].isna().all()


def test_a_kept_local_pair_that_escalation_would_break_is_not_a_misroute():
    f = frame()
    scores = pd.Series([9, 0, 0, 0, 8, 0, 0, 0, 0, 0], dtype=float)   # escalates both rescues
    t = misroute.tag(f, scores, budget=0.2)
    assert t.failure_type.isna().all()


def test_tags_account_for_quality_exactly_as_the_curve_scores_it():
    f = frame()
    scores = pd.Series([3, 9, 1, 8, 7, 2, 6, 0, 5, 4], dtype=float)
    for budget in (0.1, 0.3, 0.5):
        s = misroute.summarise(misroute.tag(f, scores, budget))
        assert s["quality"] == pytest.approx(s["quality_from_tags"])
        assert s["quality"] == evaluate_at_budget(f, scores, budget)["quality"]


def test_check_refuses_decisions_that_do_not_match_the_published_curve():
    s = misroute.summarise(misroute.tag(frame(), pd.Series([0.0] * 10), 0.2))
    with pytest.raises(ValueError, match="not the decisions the curve measured"):
        misroute.check(s, s["quality"] + 0.01, "router_in_distribution", "confidence")


@pytest.mark.skipif(not SUMMARY.exists(), reason="run `adapterops router-misroutes` first")
def test_the_committed_records_are_router_only_and_passed_their_checks():
    summary = json.loads(SUMMARY.read_text())
    records = pd.read_parquet(ROOT / summary["records"]["file"])
    assert set(records.component) == {"router"}
    assert set(records.failure_type) <= set(misroute.TYPES)
    for by_policy in summary["populations"].values():
        for s in by_policy.values():
            assert s["quality"] == pytest.approx(s["quality_from_tags"])
