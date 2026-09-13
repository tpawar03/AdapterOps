"""Guards on the rules baseline (F26), the golden-set ceiling, judge cost and model cards."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ceiling
from adapterops.judge import cost, m2
from adapterops.manifest import cards
from adapterops.router import rules


def train_frame():
    return pd.DataFrame({
        "task": ["drafting"] * 2 + ["urgency"] * 2,
        "success": [False, True, True, True],
        "frontier_success": [True, True, False, True],
    })


def test_task_order_comes_from_train_gain():
    order = rules.task_order(train_frame())
    assert list(order.index) == ["drafting", "urgency"]
    assert order["drafting"] == pytest.approx(0.5)


def test_task_rule_never_lets_length_cross_a_task_boundary():
    frame = pd.DataFrame({"task": ["urgency", "drafting"], "text": ["x" * 5000, "short"]})
    s = rules.rule_scores(train_frame(), frame)
    assert s.rules_task_length[1] > s.rules_task_length[0]
    assert s.rules_length[0] > s.rules_length[1]


def test_ceiling_plan_covers_every_golden_item_once():
    items = ceiling.plan()
    assert items.pair_id.is_unique
    assert items.groupby("task").size().to_dict() == {
        "drafting": 300, "intent": 770, "pii": 300, "urgency": 300}


def test_ceiling_scores_exact_labels_and_records_casefold_separately():
    items = pd.DataFrame({"pair_id": ["a", "b"], "task": "intent", "split": "random",
                          "text": ["t1", "t2"], "gold": ["card_arrival", "pin_blocked"]})
    cache = {"a": {"prediction": "Card_arrival", "prompt_tokens": 400, "completion_tokens": 4},
             "b": {"prediction": "pin_blocked", "prompt_tokens": 400, "completion_tokens": 16}}
    _, out = ceiling.summarise(items, cache)
    cell = out["per_task"]["intent"]
    assert cell["micro_accuracy"] == 0.5
    assert cell["micro_accuracy_casefold"] == 1.0
    assert cell["at_token_cap"] == 0.5


def test_gpt4o_cost_is_priced_per_job_from_tokens():
    rows = [{"item_id": "local:x", "prompt_tokens": 1_000_000, "completion_tokens": 0, "error": None},
            {"item_id": "m2-adapter:y", "prompt_tokens": 0, "completion_tokens": 100_000,
             "error": None},
            {"item_id": "local:z", "prompt_tokens": 5, "completion_tokens": 5, "error": "Timeout"}]
    out = cost.gpt4o_cost(rows, {"input": 2.5, "output": 10.0})
    assert out["overall"]["grades"] == 2
    assert out["by_job"]["local"]["usd"] == pytest.approx(2.5)
    assert out["by_job"]["m2-adapter"]["usd_per_1k"] == pytest.approx(1000.0)


def test_a_frontier_grading_run_never_overwrites_the_adapter_vs_prompted_record():
    default_out, default_summary = m2.outputs(m2.DEFAULT_SIDES)
    out, summary = m2.outputs(("frontier",))
    assert (default_out, default_summary) == (m2.OUT_FILE, m2.SUMMARY_FILE)
    assert out != m2.OUT_FILE and summary.name == "drafting__m2_gpt4o__frontier.json"


def test_cards_name_the_pinned_revision_and_carry_caveats():
    ctx = cards.load()
    intent = cards.render("intent", ctx)
    assert ctx["pins"]["intent"]["revision"] in intent
    urgency = cards.render("urgency", ctx)
    assert "license: cc-by-nc-4.0" in urgency and "TF-IDF" in urgency


def test_an_unmeasured_ceiling_says_so():
    ctx = {**cards.load(), "ceiling": None, "ceiling_drafting": None}
    assert "not measured" in cards.render("pii", ctx)
    assert "not measured" in cards.render("drafting", ctx)
