"""Guards on the GPT-4o labelling of the out-of-domain sample."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ood_label as ol

VOCAB = {"card_arrival", "lost_or_stolen_card"}


def test_the_intent_prompt_allows_none_and_keeps_the_label_list():
    prompt = ol.intent_prompt("Where is my sofa?", "card_arrival, lost_or_stolen_card")
    assert "reply NONE" in prompt and "card_arrival, lost_or_stolen_card" in prompt


def test_parse_accepts_only_labels_in_the_schema():
    assert ol.parse("intent", "NONE", VOCAB) == "NONE"
    assert ol.parse("intent", "card_arrival\n", VOCAB) == "card_arrival"
    assert ol.parse("intent", "delivery_delay", VOCAB) is None, "an invented label must not become gold"
    assert ol.parse("urgency", "High.", VOCAB) == "high"
    assert ol.parse("urgency", "critical", VOCAB) is None
    assert ol.parse("drafting_grade", '{"score": 4, "reason": "ok"}', VOCAB) == 4


def test_the_projection_counts_every_request_and_states_the_cap():
    items = [{"task": "intent", "max_tokens": 16, "messages": [{"role": "user", "content": "x" * 400}]}] * 3
    p = ol.project(items)
    assert p["requests"] == 3 and p["by_task"] == {"intent": 3} and p["spend_cap_usd"] == ol.SPEND_CAP_USD
    assert p["estimated_usd_upper"] >= 0
