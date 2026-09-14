"""Guards on the live request-path run: grading by the curve's rule, agreement, failures as rows."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import request_run as rr

PAIRS = pd.DataFrame([
    {"pair_id": "a", "task": "intent", "text": "card not here", "gold": "card_arrival",
     "recorded_mean_logprob": -0.01, "recorded_local_success": True,
     "recorded_frontier_success": True, "recorded_escalated": False},
    {"pair_id": "b", "task": "urgency", "text": "server down", "gold": "high",
     "recorded_mean_logprob": -0.9, "recorded_local_success": False,
     "recorded_frontier_success": True, "recorded_escalated": True},
    {"pair_id": "c", "task": "pii", "text": "I am Dana Scott", "gold": "GIVENNAME: Dana",
     "recorded_mean_logprob": -0.02, "recorded_local_success": True,
     "recorded_frontier_success": False, "recorded_escalated": False},
    {"pair_id": "d", "task": "drafting", "text": "refund please", "gold": "Sure.",
     "recorded_mean_logprob": -0.5, "recorded_local_success": True,
     "recorded_frontier_success": True, "recorded_escalated": False},
])


def fake_send(body):
    task = body["tasks"][0]
    pair = {
        "a": {"route": "local", "served_by": "local", "score": 0.01,
              "output": {"label": "card_arrival"}},
        "b": {"route": "escalated", "served_by": "frontier", "score": 0.95,
              "output": {"label": "medium"}},
        "c": {"route": "local", "served_by": "local", "score": 0.02,
              "output": {"spans": [{"label": "GIVENNAME", "value": "Dana"}]}},
    }.get(body["ticket_id"])
    if pair is None:
        raise ConnectionError("server went away")
    pair |= {"threshold": 0.4, "cost_usd": 0.001, "latency_ms": 5.0, "local_error": None,
             "frontier_error": None, "judge_score": None}
    return {"pairs": {task: pair}}


def test_served_answers_are_graded_by_the_curves_rule():
    live, _ = rr.drive(PAIRS, fake_send, concurrency=2)
    by_id = live.set_index("pair_id")
    assert by_id.loc["a", "served_success"]
    assert not by_id.loc["b", "served_success"]          # a wrong label served by the frontier
    assert by_id.loc["c", "served_success"]              # PII spans rebuilt and scored strictly
    assert not by_id.loc["d", "served_success"]          # an HTTP failure is a failed request
    assert by_id.loc["d", "http_error"].startswith("ConnectionError")


def test_summary_compares_live_decisions_and_quality_with_the_curve():
    live, _ = rr.drive(PAIRS, fake_send, concurrency=1)
    s = rr.summarise(PAIRS, live, wall_s=2.0, policy="confidence")
    assert s["http_errors"] == 1
    assert s["escalation_rate"] == round(1 / 3, 4)
    agree = s["agreement_with_curve"]
    assert agree["pairs_compared"] == 3 and agree["same_decision"] == 1.0
    graded = s["graded_tasks"]
    assert graded["pairs"] == 3
    assert graded["served_success"] == round(2 / 3, 4)
    # The curve escalated b and the frontier got it right; a and c stayed local and succeeded.
    assert graded["curve_success_at_operating_point"] == 1.0
    assert "judge_mean_local" in s["per_task"]["drafting"]


def test_the_run_writes_no_ticket_text(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(rr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rr, "request_set", lambda: PAIRS)
    assert rr.main(url="", name="t", concurrency=[1], send=fake_send) == 0
    written = (tmp_path / "request_path__t.json").read_text()
    pairs = pd.read_parquet(tmp_path / "request_path__t__pairs.parquet")
    assert "Dana Scott" not in written and "text" not in pairs.columns
