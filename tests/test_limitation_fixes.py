"""Guards on three limitation fixes: PII leakage in regression runs, the prompted-baseline floor at
promotion, and the frontier request budget in the request path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import regression as reg
from adapterops.manifest import system as s
from adapterops.serve import pipeline as pl


def test_pii_scoring_reports_leakage_beside_span_f1():
    texts = ["Call Dana Scott today.", "Email dana@example.com now."]
    gold = ["GIVENNAME: Dana\nSURNAME: Scott", "EMAIL: dana@example.com"]
    predictions = ["GIVENNAME: Dana Scott", ""]            # a split name that masks both, then a miss
    out = reg.score("pii", texts, gold, predictions)
    assert out["span_f1_strict"] == 0.0
    assert out["docs_fully_masked"] == 0.5                  # the name is fully masked despite being "wrong"
    assert out["gold_spans_wholly_unmasked"] == round(1 / 3, 4)


def test_the_backfill_refuses_a_run_the_gate_thresholds_pin():
    import pytest

    with pytest.raises(ValueError, match="pinned input"):
        reg.add_redaction(ROOT / "runs" / "regression__v1-baseline-1.json")


def _manifest():
    return {"version": 7, "components": {"adapters": {}}, "eval_splits": {},
            "gate": {"state": "enforcing", "thresholds": {"urgency": 0.0381}}}


def test_promotion_blocks_a_candidate_below_the_prompted_baseline(monkeypatch):
    monkeypatch.setattr(s, "load_current", _manifest)
    monkeypatch.setattr(s, "prompted_floors", lambda root=None: {"urgency": 0.3962})
    inside_noise_but_below_prompting = {"per_task": {"urgency": {
        "drop": 0.03, "random": {"candidate": 0.3900, "baseline": 0.4200, "drop": 0.03}}}}
    reasons = s.blocking_reasons(_manifest(), inside_noise_but_below_prompting)
    assert len(reasons) == 1 and "below the prompted base model's 0.3962" in reasons[0]
    still_ahead = {"per_task": {"urgency": {"drop": 0.02, "random": {"candidate": 0.4000, "drop": 0.02}}}}
    assert s.blocking_reasons(_manifest(), still_ahead) == []


def test_the_prompted_floors_come_from_the_committed_runs():
    floors = s.prompted_floors()
    assert set(floors) == {"intent", "urgency", "pii"}
    assert floors["urgency"] < 0.42 < floors["urgency"] + 0.05


def test_the_budget_refuses_past_its_minute_and_day_windows():
    now = [0.0]
    budget = pl.FrontierBudget(per_minute=2, per_day=3, clock=lambda: now[0])
    assert budget.take() is None and budget.take() is None
    assert "per-minute" in budget.take()
    now[0] = 61.0                                           # the minute window has moved on
    assert budget.take() is None
    assert "daily" in budget.take()
    now[0] = 86_401.0                                       # a new day
    assert budget.take() is None


class Local:
    name = "local"

    def generate(self, task, text):
        return pl.Generation(text="card_arrival", mean_logprob=-2.0, n_tokens=2)


class Frontier:
    name = "frontier"

    def __init__(self):
        self.calls = 0

    def generate(self, task, text):
        self.calls += 1
        return pl.Generation(text="card_arrival", prompt_tokens=100, completion_tokens=5)


def test_an_escalation_over_budget_is_answered_locally_and_counted():
    frontier = Frontier()
    service = pl.Service(
        local=Local(), policy=pl.ConfidencePolicy(0.4), frontier=frontier,
        prices=pl.Prices(0.00001, 0.15, 0.6, "test"), vocab={"intent": frozenset({"card_arrival"}),
                                                              "urgency": frozenset({"low"})},
        frontier_budget=pl.FrontierBudget(per_day=1))
    first = service.handle("my card is late", ["intent"])["pairs"]["intent"]
    second = service.handle("my card is late", ["intent"])["pairs"]["intent"]
    assert first["served_by"] == "frontier" and frontier.calls == 1
    assert second["route"] == "escalated" and second["served_by"] == "local"
    assert second["frontier_error"].startswith("budget:") and frontier.calls == 1
    assert service.metrics.snapshot()["per_task"]["intent"]["frontier_budget_refused"] == 1
