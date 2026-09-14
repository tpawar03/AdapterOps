"""Guards on the live request-path run: grading by the curve's rule, agreement, failures as rows,
duration runs, the direct-to-vLLM mode, and the API's thread limit."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import pipeline as pl
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
    pairs = rr.at_threshold(PAIRS, 0.4)
    live, _ = rr.drive(pairs, fake_send, concurrency=1)
    s = rr.summarise(pairs, live, wall_s=2.0, policy="confidence")
    assert s["http_errors"] == 1
    assert s["escalation_rate"] == round(1 / 3, 4)
    agree = s["agreement_with_curve"]
    assert agree["pairs_compared"] == 3 and agree["same_decision"] == 1.0
    graded = s["graded_tasks"]
    assert graded["pairs"] == 3
    assert graded["served_success"] == round(2 / 3, 4)
    # The curve escalated b and the frontier got it right; a and c stayed local and succeeded.
    assert graded["curve_success_at_operating_point"] == 1.0
    assert graded["recorded_success_at_threshold"] == 1.0
    assert "judge_mean_local" in s["per_task"]["drafting"]
    assert s["frontier_served_rate"] == round(1 / 3, 4) and s["frontier_p95_ms"] == 5.0


def test_frontier_errors_under_load_are_counted_by_kind_and_per_window():
    pairs = rr.at_threshold(PAIRS, 0.4)

    def rate_limited(body):
        out = fake_send(body)
        for pair in out["pairs"].values():
            if pair["route"] == "escalated":
                pair["served_by"] = "local"
                pair["frontier_error"] = "RateLimitError: Error code: 429 - too many requests"
        return out

    live, wall = rr.drive(pairs, rate_limited, concurrency=2, duration=0.2, keep_output=False)
    s = rr.summarise(pairs, live, wall, policy="confidence", duration=0.2, window=0.1)
    assert s["frontier_served_rate"] == 0.0 and s["frontier_errors"] > 0
    (kind, count), = s["frontier_errors_by_kind"].items()
    assert kind.startswith("RateLimitError") and count == s["frontier_errors"]
    assert sum(w["frontier_errors"] for w in s["windows"]) == s["frontier_errors"]


def test_quality_is_compared_with_the_side_that_actually_served():
    # With the frontier off, b escalates but the adapter answers it. The routed figure credits b with
    # GPT-4o-mini's recorded success; the as-served figure holds it to the adapter's recorded failure.
    pairs = rr.at_threshold(PAIRS, 0.4)

    def frontier_off(body):
        out = fake_send(body)
        for pair in out["pairs"].values():
            pair["served_by"] = "local"
        return out

    live, _ = rr.drive(pairs, frontier_off, concurrency=1)
    graded = rr.summarise(pairs, live, wall_s=1.0, policy="confidence")["graded_tasks"]
    assert graded["recorded_success_at_threshold"] == 1.0
    assert graded["recorded_success_as_served"] == round(2 / 3, 4)


def test_agreement_is_against_the_threshold_not_the_budget():
    # The budget escalated b; the live threshold of 1.0 would not, and the live path did not.
    pairs = rr.at_threshold(PAIRS, 1.0)

    def never(body):
        out = fake_send(body)
        for pair in out["pairs"].values():
            pair["route"] = "local"
        return out

    live, _ = rr.drive(pairs, never, concurrency=1)
    s = rr.summarise(pairs, live, wall_s=1.0, policy="confidence")
    assert s["agreement_with_curve"]["same_decision"] == 1.0
    assert s["per_task"]["urgency"]["curve_escalation_rate"] == 1.0
    assert s["per_task"]["urgency"]["threshold_escalation_rate"] == 0.0


def test_a_duration_run_cycles_the_pairs_and_reports_windows():
    pairs = rr.at_threshold(PAIRS, 0.4)
    live, wall = rr.drive(pairs, fake_send, concurrency=3, duration=0.3, keep_output=False)
    assert len(live) > len(pairs)                        # more than one pass
    assert "output" not in live.columns
    assert live.t_start_s.max() <= 0.301                 # nothing starts after the deadline (ms rounding)
    s = rr.summarise(pairs, live, wall, policy="confidence", duration=0.3, window=0.1)
    assert s["passes"] > 1
    assert 1 <= len(s["windows"]) <= 3
    assert all(w["start_s"] < 0.3 for w in s["windows"])
    assert sum(w["requests"] for w in s["windows"]) == len(live)


def test_direct_vllm_records_the_decision_without_acting_on_it(monkeypatch):
    class FakeVLLM:
        def __init__(self, base_url, timeout=30.0):
            pass

        def generate(self, task, text):
            lp = -0.9 if task == "urgency" else -0.01
            return pl.Generation(text={"intent": "card_arrival", "urgency": "high",
                                       "pii": "GIVENNAME: Dana", "drafting": "Sure."}[task],
                                 mean_logprob=lp, latency_s=0.02)

    monkeypatch.setattr(pl, "VLLMBackend", FakeVLLM)
    send = rr.vllm_sender("http://vllm", threshold=0.4)
    urgency = send({"text": "server down", "tasks": ["urgency"], "ticket_id": "b"})["pairs"]["urgency"]
    intent = send({"text": "card not here", "tasks": ["intent"], "ticket_id": "a"})["pairs"]["intent"]
    assert urgency["route"] == "escalated" and urgency["served_by"] == "local"
    assert urgency["cost_usd"] == 0.0 and urgency["output"] == {"label": "high"}
    assert intent["route"] == "local"


def test_the_run_writes_no_ticket_text(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(rr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rr, "request_set", lambda population=rr.POPULATION: PAIRS)
    assert rr.main(url="", name="t", concurrency=[1], send=fake_send) == 0
    written = (tmp_path / "request_path__t.json").read_text()
    pairs = pd.read_parquet(tmp_path / "request_path__t__pairs.parquet")
    assert "Dana Scott" not in written and "text" not in pairs.columns


def test_resummarise_recomputes_from_the_per_request_file_without_sending(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(rr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rr, "request_set", lambda population=rr.POPULATION: PAIRS)
    rr.main(url="", name="old", concurrency=[1, 2], send=fake_send)
    path = tmp_path / "request_path__old.json"
    stale = json.loads(path.read_text())
    for level in stale["levels"]:
        del level["graded_tasks"]["recorded_success_as_served"]   # a run from before the field
    path.write_text(json.dumps(stale))

    def refuse(body):
        raise AssertionError("resummarise must not send")

    monkeypatch.setattr(rr, "http_sender", lambda *a, **k: refuse)
    rr.resummarise("old")
    fresh = json.loads(path.read_text())
    assert [lv["concurrency"] for lv in fresh["levels"]] == [1, 2]
    assert all("recorded_success_as_served" in lv["graded_tasks"] for lv in fresh["levels"])
    assert fresh["levels"][0]["requests"] == len(PAIRS)
    assert "no request was re-sent" in fresh["resummarised"]


def test_the_api_raises_its_thread_limit_past_anyios_default():
    import anyio.to_thread
    from fastapi.testclient import TestClient

    from adapterops.serve.api import create_app

    app = create_app(service=None, max_threads=123)

    @app.get("/threads")
    async def threads() -> dict:                         # on the loop, where the limiter lives
        return {"tokens": anyio.to_thread.current_default_thread_limiter().total_tokens}

    with TestClient(app) as client:
        assert client.get("/threads").json() == {"tokens": 123}
