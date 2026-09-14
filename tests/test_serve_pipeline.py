"""Guards on the online request path (§8): routing, fallback, cost, metrics, tracing, HTTP.

Backends are fakes, so nothing here loads a model or calls an API. What is under test is the
decision logic the live path adds on top of components tested elsewhere.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import pipeline as pl
from adapterops.serve.tracing import JsonlTracer, LangfuseTracer

VOCAB = {"intent": frozenset({"card_arrival", "declined_card_payment"}),
         "urgency": frozenset({"low", "medium", "high"})}
PRICES = pl.Prices(local_usd_per_request=0.00001, frontier_input_per_1m=0.15,
                   frontier_output_per_1m=0.60, source="test")
GOOD = {"intent": "card_arrival", "urgency": "high", "pii": "GIVENNAME: Dana",
        "drafting": "Sorry to hear that — here is what to do."}


class FakeLocal:
    name = "fake-local"

    def __init__(self, outputs=None, logprob=-0.1, raises=()):
        self.outputs = {**GOOD, **(outputs or {})}
        self.logprob = logprob
        self.raises = set(raises)
        self.calls = []

    def generate(self, task, text):
        self.calls.append(task)
        if task in self.raises:
            raise TimeoutError("local timed out")
        lp = self.logprob[task] if isinstance(self.logprob, dict) else self.logprob
        return pl.Generation(text=self.outputs[task], mean_logprob=lp, n_tokens=3)


class FakeFrontier:
    name = "fake-frontier"

    def __init__(self, raises=False):
        self.calls = []
        self.raises = raises

    def generate(self, task, text):
        self.calls.append(task)
        if self.raises:
            raise RuntimeError("rate limited")
        return pl.Generation(text=GOOD[task], prompt_tokens=1000, completion_tokens=100)


class StubRouter:
    name = "router_p_fail"
    before_local = True
    threshold = 0.5

    def __init__(self, p_fail):
        self.p_fail = p_fail

    def decide_before(self, task, text):
        p = self.p_fail[task]
        return pl.Decision(p >= self.threshold, p, self.threshold, self.name)


def service(local=None, policy=None, frontier=None, judge=None, tracer=None):
    return pl.Service(local=local or FakeLocal(), policy=policy or pl.ConfidencePolicy(0.4),
                      frontier=frontier, judge=judge, tracer=tracer, prices=PRICES,
                      vocab=VOCAB, manifest={"version": 7})


def test_confidence_escalates_at_the_threshold_and_keeps_confident_pairs_local():
    local = FakeLocal(logprob={"intent": -0.4, "urgency": -0.39, "pii": -0.05, "drafting": -2.0})
    frontier = FakeFrontier()
    out = service(local, frontier=frontier).handle("my card has not arrived")["pairs"]

    assert out["intent"]["route"] == "escalated"          # score 0.4 >= threshold 0.4
    assert out["urgency"]["route"] == "local"             # 0.39 < 0.4
    assert out["drafting"]["route"] == "escalated"
    assert sorted(frontier.calls) == ["drafting", "intent"]
    assert out["intent"]["served_by"] == "frontier" and out["pii"]["served_by"] == "local"
    assert out["urgency"]["mean_token_probability"] == pytest.approx(0.677, abs=1e-3)


def test_a_local_error_is_a_fallback_and_is_counted_apart_from_escalation():
    svc = service(FakeLocal(raises={"pii"}), frontier=FakeFrontier())
    out = svc.handle("x")["pairs"]
    assert out["pii"]["route"] == "fallback" and out["pii"]["served_by"] == "frontier"
    assert "TimeoutError" in out["pii"]["local_error"]

    m = svc.metrics.snapshot()
    assert m["fallback_rate"] == 0.25 and m["frontier_call_rate"] == 0.0
    assert m["per_task"]["pii"]["fallback_rate"] == 1.0


def test_an_unusable_label_falls_back_but_a_wrong_valid_label_does_not():
    svc = service(FakeLocal(outputs={"intent": "not_a_label", "urgency": "low"}),
                  frontier=FakeFrontier())
    out = svc.handle("x")["pairs"]
    assert out["intent"]["route"] == "fallback"
    assert "not a intent label" in out["intent"]["local_error"]
    assert out["urgency"]["route"] == "local", "a valid but possibly wrong label is not a fallback"


def test_unparseable_pii_lines_fall_back():
    svc = service(FakeLocal(outputs={"pii": "GIVENNAME: Dana\nsomething odd"}),
                  frontier=FakeFrontier())
    assert svc.handle("x")["pairs"]["pii"]["route"] == "fallback"


def test_the_router_decides_before_the_adapter_runs():
    local, frontier = FakeLocal(), FakeFrontier()
    router = StubRouter({"intent": 0.9, "urgency": 0.1, "pii": 0.1, "drafting": 0.1})
    out = service(local, policy=router, frontier=frontier).handle("x")["pairs"]
    assert "intent" not in local.calls, "the adapter ran on a pair the router escalated"
    assert out["intent"]["served_by"] == "frontier"
    assert out["urgency"]["predicted_success"] == pytest.approx(0.9)


def test_without_a_frontier_an_escalation_is_answered_locally_and_says_so():
    router = StubRouter({"intent": 0.9, "urgency": 0.1, "pii": 0.1, "drafting": 0.1})
    out = service(policy=router).handle("x")["pairs"]["intent"]
    assert out["route"] == "escalated" and out["served_by"] == "local"
    assert out["frontier_error"] == "no frontier configured" and out["notes"]


def test_a_pair_nobody_could_answer_is_unanswered_not_silently_local():
    out = service(FakeLocal(raises={"urgency"})).handle("x")["pairs"]["urgency"]
    assert out["served_by"] == "none" and out["output"] is None and not out["valid"]


def test_the_judge_scores_only_a_locally_answered_draft():
    judged = []

    def judge(texts, replies):
        judged.append(replies[0])
        return [4.2]

    local = FakeLocal(logprob={"intent": -0.1, "urgency": -0.1, "pii": -0.1, "drafting": -0.1})
    out = service(local, judge=judge, frontier=FakeFrontier()).handle("x")["pairs"]
    assert out["drafting"]["judge_score"] == 4.2 and out["intent"]["judge_score"] is None

    escalated = FakeLocal(logprob={"intent": -0.1, "urgency": -0.1, "pii": -0.1, "drafting": -5.0})
    out = service(escalated, judge=judge, frontier=FakeFrontier()).handle("x")["pairs"]
    assert out["drafting"]["judge_score"] is None, "the judge scored a GPT-4o-mini reply"


def test_cost_adds_local_compute_and_frontier_tokens_at_list_price():
    local = FakeLocal(logprob={"intent": -1.0, "urgency": -0.1, "pii": -0.1, "drafting": -0.1})
    result = service(local, frontier=FakeFrontier()).handle("x")
    intent = result["pairs"]["intent"]
    # confidence is a cascade: the adapter ran, then GPT-4o-mini answered.
    assert intent["cost_usd"] == pytest.approx(0.00001 + (1000 * 0.15 + 100 * 0.60) / 1e6)
    assert result["cost_usd"] == pytest.approx(sum(p["cost_usd"] for p in result["pairs"].values()))


def test_a_subset_of_tasks_is_routed_and_unknown_tasks_are_refused():
    svc = service()
    assert set(svc.handle("x", tasks=("pii",))["pairs"]) == {"pii"}
    with pytest.raises(ValueError, match="unknown task"):
        svc.handle("x", tasks=("sentiment",))


@pytest.mark.skipif(not pl.CURVE.exists(), reason="no committed operating curve")
def test_policies_use_the_committed_operating_point():
    assert pl.operating_threshold("confidence") == pytest.approx(0.393881)
    assert pl.operating_threshold("router_p_fail") == pytest.approx(0.426448)


def test_a_pinned_checkpoint_whose_weights_moved_refuses_to_load(tmp_path):
    (tmp_path / "judge").mkdir()
    weights = tmp_path / "judge" / "model.safetensors"
    weights.write_bytes(b"weights-v1")
    manifest = {"version": 4, "components": {"judge": {"judge/model.safetensors": {
        "file": "judge/model.safetensors",
        "sha256": "0" * 64}}}}
    with pytest.raises(ValueError, match="refusing to serve"):
        pl.pinned_checkpoint(manifest, "judge", root=tmp_path)
    assert pl.pinned_checkpoint({"components": {"judge": None}}, "judge") is None


def test_jsonl_traces_leave_the_ticket_text_out_by_default(tmp_path):
    path = tmp_path / "traces.jsonl"
    service(tracer=JsonlTracer(path)).handle("Dana Whitfield, dana.w@example.org")
    row = json.loads(path.read_text().splitlines()[0])
    assert row["input"] == {"chars": 34}
    assert "Dana Whitfield" not in json.dumps(row["input"])


def test_langfuse_ingestion_batch_is_one_trace_and_a_span_per_pair():
    import httpx

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(207, json={"successes": [], "errors": []})

    tracer = LangfuseTracer("https://lf.example", "pk", "sk",
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    service(tracer=tracer).handle("secret ticket text")

    assert tracer.sent == 1 and tracer.failures == 0
    request = seen[0]
    assert request.url.path == "/api/public/ingestion"
    assert request.headers["authorization"].startswith("Basic ")
    batch = json.loads(request.content)["batch"]
    assert [e["type"] for e in batch].count("trace-create") == 1
    assert [e["type"] for e in batch].count("span-create") == 4
    assert "secret ticket text" not in request.content.decode()


def test_a_failing_tracer_does_not_fail_the_request():
    import httpx

    def handler(request):
        return httpx.Response(500)

    tracer = LangfuseTracer("https://lf.example", "pk", "sk",
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = service(tracer=tracer).handle("x")
    assert len(result["pairs"]) == 4 and tracer.failures == 1


def test_the_http_api_routes_a_ticket_and_reports_live_metrics():
    from fastapi.testclient import TestClient

    from adapterops.serve.api import create_app

    client = TestClient(create_app(service(frontier=FakeFrontier())))
    r = client.post("/v1/tickets", json={"text": "my card has not arrived", "tasks": ["intent"]})
    assert r.status_code == 200
    assert r.json()["pairs"]["intent"]["output"] == {"label": "card_arrival"}

    assert client.post("/v1/tickets", json={"text": ""}).status_code == 422
    assert client.post("/v1/tickets", json={"text": "x", "tasks": ["sentiment"]}).status_code == 422
    metrics = client.get("/v1/metrics").json()
    assert metrics["tickets"] == 1 and metrics["pairs"] == 1
    assert client.get("/v1/service").json()["manifest_version"] == 7
