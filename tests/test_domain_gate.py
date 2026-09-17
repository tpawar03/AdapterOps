"""Guards on the out-of-domain gate in the request path."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import domain_gate as dg
from adapterops.serve import pipeline as pl

SHOP = ("Hi, my sofa order 5813947 was supposed to arrive last Tuesday and the tracking page still says "
        "processing. Can you tell me when it will ship and whether I can get free delivery for the delay?")


class Gate:
    """Flags one task, as the real gate flags intent, urgency and drafting."""

    def __init__(self, flagged=("intent",)):
        self.checks = dict.fromkeys(flagged)

    def flags(self, task, text):
        return task in self.checks


class Local:
    name = "local"

    def __init__(self):
        self.calls = []

    def generate(self, task, text):
        self.calls.append(task)
        return pl.Generation(text={"intent": "card_arrival", "pii": "", "urgency": "low"}[task], mean_logprob=-0.01)


class Frontier:
    name = "gpt-4o-mini"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def generate(self, task, text):
        self.calls.append(task)
        if self.fail:
            raise TimeoutError("frontier down")
        return pl.Generation(text="card_arrival", prompt_tokens=100, completion_tokens=3)


def service(local, frontier, gate):
    return pl.Service(local=local, policy=pl.NeverEscalate(), frontier=frontier,
                      prices=pl.Prices(0.00001, 0.15, 0.6, "t"),
                      vocab={"intent": frozenset({"card_arrival"}), "urgency": frozenset({"low"})},
                      domain_gate=gate)


def test_a_flagged_pair_goes_to_the_frontier_without_running_locally():
    local, frontier = Local(), Frontier()
    pair = service(local, frontier, Gate()).handle(SHOP, ["intent"])["pairs"]["intent"]
    assert pair["route"] == "out_of_domain" and pair["served_by"] == "frontier"
    assert local.calls == [] and frontier.calls == ["intent"]


def test_unflagged_tasks_run_locally_as_before():
    local, frontier = Local(), Frontier()
    pair = service(local, frontier, Gate()).handle(SHOP, ["pii"])["pairs"]["pii"]
    assert pair["route"] == "local" and local.calls == ["pii"] and frontier.calls == []


def test_without_a_frontier_a_flagged_pair_is_answered_locally_and_says_so():
    local = Local()
    pair = service(local, None, Gate()).handle(SHOP, ["intent"])["pairs"]["intent"]
    assert pair["route"] == "out_of_domain" and pair["served_by"] == "local"
    assert local.calls == ["intent"] and any("no frontier" in n for n in pair["notes"])


def test_a_failing_frontier_falls_back_to_a_local_answer():
    local = Local()
    pair = service(local, Frontier(fail=True), Gate()).handle(SHOP, ["intent"])["pairs"]["intent"]
    assert pair["served_by"] == "local" and pair["output"] == {"label": "card_arrival"}
    assert pair["frontier_error"].startswith("TimeoutError")


def test_metrics_count_gated_pairs_apart_from_confidence_escalations():
    svc = service(Local(), Frontier(), Gate())
    svc.handle(SHOP, ["intent", "pii"])
    snap = svc.metrics.snapshot()
    assert snap["frontier_call_rate"] == 0.0, "the tuned escalation rate must not absorb gated pairs"
    assert snap["out_of_domain_rate"] == 0.5 and snap["frontier_served_rate"] == 0.5
    assert snap["per_task"]["intent"]["out_of_domain_rate"] == 1.0


def test_pii_is_never_a_gated_task():
    assert "pii" not in dg.GATED_TASKS


@pytest.fixture(scope="module")
def pin():
    return json.loads(dg.PIN_FILE.read_text())


def test_the_real_gate_flags_a_shop_ticket_and_passes_banking_and_retail_requests(pin):
    gate = dg.DomainGate(pin)
    assert all(gate.flags(task, SHOP) for task in dg.GATED_TASKS)
    assert not gate.flags("intent", "My card still hasn't arrived, when will it come?")
    assert not gate.flags("drafting", "I want to cancel my order")
    assert not gate.flags("pii", SHOP), "PII is never gated"


def test_a_threshold_that_no_longer_recomputes_is_refused(pin):
    moved = copy.deepcopy(pin)
    moved["tasks"] = {"intent": moved["tasks"]["intent"]}
    moved["tasks"]["intent"]["thresholds"]["similarity_below"] += 0.01
    with pytest.raises(ValueError, match="refusing to serve"):
        dg.DomainGate(moved)


def test_a_changed_training_split_is_refused(pin):
    moved = copy.deepcopy(pin)
    moved["tasks"] = {"intent": moved["tasks"]["intent"]}
    moved["tasks"]["intent"]["train_split"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="has changed"):
        dg.DomainGate(moved)
