"""Guards on the GPT-4o judge labelling run (F13).

Every test here runs without an API call. The two that matter most protect a spend cap
that would otherwise never fire, and a rubric that would otherwise quietly rebuild the
house-style bias the Phase 2 escalation finding exposed.
"""

import builtins
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import label as jl


def test_the_ledger_prices_gpt4o_not_gpt4o_mini():
    """The frontier arm's Ledger prices gpt-4o-mini. Reused unchanged it would price this
    run about 16x too cheap, and the spend cap would never fire."""
    ledger = jl.JudgeLedger(cap=100.0, request_cap=10)
    ledger.add(1_000_000, 100_000)
    assert ledger.usd == pytest.approx(2.50 + 1.00)

    capped = jl.JudgeLedger(cap=1.0, request_cap=10)
    capped.add(1_000_000, 0)
    assert capped.stopped, "a $2.50 spend did not trip a $1.00 cap"


def test_rubric_declares_the_template_slot_policy_and_has_no_reference():
    assert "Do not reward or penalise" in jl.RUBRIC
    assert "{{Order Number}}" in jl.RUBRIC
    assert "There is no reference answer" in jl.RUBRIC
    messages = jl.build_messages("where is my order", "Please share your {{Order Number}}.")
    assert "Please share your {{Order Number}}." in messages[1]["content"]


def test_the_bitext_reference_never_reaches_the_prompt():
    """Reference-free by design: the proxy already measures overlap with the reference,
    and a reference-guided judge would reward house style one level up."""
    items = jl.plan_items(include_frontier=False).head(60)
    gold = pd.read_parquet(ROOT / "data/router/scored.parquet").set_index("pair_id").gold
    for _, item in items.iterrows():
        prompt = " ".join(m["content"] for m in jl.build_messages(item.instruction, item.reply))
        reference = gold[item.pair_id].strip()
        if reference != item.reply.strip():
            assert reference not in prompt, f"{item.item_id}: reference leaked into prompt"


@pytest.mark.parametrize(("raw", "expected"), [
    ('{"score": 4, "reason": "ok"}', 4),
    ('{"score": "5", "reason": "ok"}', 5),
    ('{"score": 4.0}', 4),
    ('{"score": 4.5}', None),
    ('{"score": 0}', None),
    ('{"score": 6}', None),
    ('{"score": true}', None),
    ('{"reason": "no score"}', None),
    ("[4]", None),
    ("not json", None),
])
def test_parse_accepts_only_an_unambiguous_integer_score(raw, expected):
    assert jl.parse_judgment(raw) == expected


def test_plan_covers_every_router_drafting_pair_and_holds_out_150():
    items = jl.plan_items(include_frontier=False)
    local = items[items.source == "local"]
    scored = pd.read_parquet(ROOT / "data/router/scored.parquet")
    router_ids = set(scored[(scored.task == "drafting")
                            & (scored.purpose == "router")].pair_id)

    assert len(local) == jl.LOCAL_BUDGET
    assert router_ids <= set(local.pair_id), "a router drafting pair would keep its proxy label"
    assert (local.split == "calibration").sum() == jl.CALIBRATION_HOLDOUT
    assert local.item_id.is_unique


def test_plan_is_deterministic():
    pd.testing.assert_frame_equal(jl.plan_items(False), jl.plan_items(False))


def test_frontier_replies_are_evaluation_only():
    items = jl.plan_items(include_frontier=True)
    frontier = items[items.source == "frontier"]
    assert len(frontier) == 750
    assert set(frontier.split) == {"frontier_eval"}, "frontier replies must not train the judge"
    assert items.item_id.is_unique


def test_the_cost_projection_makes_no_api_call(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "openai":
            raise AssertionError("the projection imported openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    estimate = jl.project_cost(jl.plan_items(False).head(20))
    assert estimate["total"]["requests"] == 20
    assert estimate["total"]["usd"] > 0
