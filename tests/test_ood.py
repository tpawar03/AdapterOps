"""Guards on the out-of-domain sample: PII gold from ABCD scenarios, stratification, freezing."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ood


def test_gold_marks_the_full_name_parts_and_scenario_values_in_the_customers_words():
    text = "Hi, my name is Chloe Zhang. Email chloe@example.com, order 5813947. Chloe again."
    scenario = {"personal": {"customer_name": "chloe zhang", "email": "chloe@example.com"},
                "order": {"order_id": "5813947"}}
    gold = ood.abcd_gold(text, scenario)
    got = [(text[g["start"]:g["end"]], g["label"], g["scope"]) for g in gold]
    assert got == [("Chloe", "GIVENNAME", "in"), ("Zhang", "SURNAME", "in"),
                   ("chloe@example.com", "EMAIL", "in"), ("5813947", "ORDER_ID", "out"),
                   ("Chloe", "GIVENNAME", "in")]


def test_a_value_inside_an_earlier_span_is_not_double_counted():
    text = "Username chloezhang81 please."
    gold = ood.abcd_gold(text, {"personal": {"customer_name": "chloe zhang", "username": "chloezhang81"}})
    assert [(text[g["start"]:g["end"]], g["label"]) for g in gold] == [("chloezhang81", "USERNAME")]


def test_cfpb_sample_is_stratified_and_length_capped():
    rows = [{"text": " ".join(["w"] * (50 if i % 5 else 300)), "label": f"p{i % 7}", "product_raw": "x"}
            for i in range(700)]
    sample = ood.cfpb_sample(pd.DataFrame(rows))
    assert len(sample) == ood.PER_SOURCE
    assert sample.stratum.value_counts().max() - sample.stratum.value_counts().min() <= 1
    assert sample.text.str.split().str.len().max() <= ood.MAX_WORDS


def test_a_frozen_sample_refuses_a_different_rebuild(tmp_path):
    path = tmp_path / "s.parquet"
    sample = pd.DataFrame({"id": ["a"], "text": ["t"], "source": ["abcd"]})
    ood.freeze(sample, path)
    with pytest.raises(ValueError, match="frozen"):
        ood.freeze(sample.assign(text=["changed"]), path)


def test_pair_rows_flag_what_the_operating_threshold_would_escalate():
    result = {"manifest_version": 8, "pairs": {
        "intent": {"output": {"label": "card_arrival"}, "score": 0.5, "threshold": 0.39, "valid": True},
        "urgency": {"output": {"label": "high"}, "score": None, "threshold": 0.39, "valid": True}}}
    rows = ood.pair_rows({"id": "abcd:1", "source": "abcd"}, result)
    assert [r["would_escalate"] for r in rows] == [True, False]
    assert json.loads(rows[0]["output"]) == {"label": "card_arrival"}


def test_abcd_sample_keeps_each_flow_label_and_caps_per_flow():
    convos = [{"convo_id": i, "scenario": {"flow": f"flow{i % 3}", "subflow": "s", "personal": {}, "order": {}},
               "original": [["customer", "hello there"]]} for i in range(60)]
    sample = ood.abcd_sample(convos)
    assert sample.stratum.notna().all()
    assert sample.stratum.value_counts().to_dict() == {"flow0": 15, "flow1": 15, "flow2": 15}
