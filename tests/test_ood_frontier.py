"""Guards on the GPT-4o-mini comparison for the out-of-domain gate."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ood_frontier as of
from adapterops.eval import ood_label as ol


def sample():
    return pd.DataFrame({"id": [f"{s}:{i}" for s in ("abcd", "cfpb") for i in range(40)],
                         "source": [s for s in ("abcd", "cfpb") for _ in range(40)],
                         "text": [f"ticket {i}" for i in range(80)]})


def test_mini_is_graded_on_the_same_tickets_as_the_adapter():
    s = sample()
    adapter_graded = [i["id"] for i in ol.requests(s, None) if i["task"] == "drafting_grade"]
    assert of.graded_ids(s) == adapter_graded and len(adapter_graded) == 60


def test_the_projection_prices_each_model_at_its_own_rate():
    item = {"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 0}
    mini = of.project([{**item, "model": of.MODEL}])["estimated_usd_upper"]
    grader = of.project([{**item, "model": of.GRADER}])["estimated_usd_upper"]
    assert grader > mini * 10, "gpt-4o input is ~17x gpt-4o-mini's"


def test_cache_keys_change_with_the_model_and_the_token_cap():
    messages = [{"role": "user", "content": "hi"}]
    assert of._key(of.MODEL, messages, 16) != of._key(of.GRADER, messages, 16)
    assert of._key(of.MODEL, messages, 16) != of._key(of.MODEL, messages, 400)
