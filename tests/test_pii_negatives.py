"""Guards on the PII false-positive measurement: the screen, the two counts, and the written run."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import pii_negatives as pn


@pytest.mark.parametrize("text", [
    "Why would I have a pending payment?",
    "I want help speaking to a person",
    "Atm took my card",
    "My ATM card is not working. I need help.",
])
def test_plain_requests_pass_the_screen(text):
    assert pn.pii_free(text)


@pytest.mark.parametrize("text", [
    "I paid 30 pounds yesterday",                  # a digit
    "email me at dana@example.com",                # an address
    "cancel order {{Order Number}}",               # a template slot
    "Mr Scott cannot log in",                      # a title
    "my wife lost her card",                       # gendered words
    "I was in London when it happened",            # a capitalised word mid-sentence
])
def test_anything_a_pii_label_could_point_at_is_screened_out(text):
    assert not pn.pii_free(text)


def test_invented_values_and_grounded_spans_are_counted_apart():
    frame = pd.DataFrame({
        "source": ["intent", "intent", "drafting"],
        "text": ["My card still hasn't arrived", "help me with my card", "I want a refund"],
        "output": ["AGE: 3\nSEX: M", "", "GIVENNAME: refund"],
    })
    s = pn.summarise(frame)
    a = s["all"]
    assert a["texts"] == 3 and a["texts_with_any_line"] == 2
    assert a["false_positive_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["texts_with_invented_value"] == 1              # "3" and "M" are not in the text
    assert a["texts_with_grounded_span"] == 1               # "refund" is
    assert a["labels"] == {"AGE": 1, "SEX": 1, "GIVENNAME": 1}
    assert s["per_source"]["drafting"]["false_positive_rate"] == 1.0


def test_a_named_run_on_the_held_out_set_writes_its_own_files(tmp_path, monkeypatch):
    held_out = tmp_path / "val.parquet"
    pd.DataFrame({"text": ["Thank you for your patience.", "We look forward to it."],
                  "source": ["ai4privacy_val"] * 2}).to_parquet(held_out)
    monkeypatch.setattr(pn, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pn, "VAL_SET_FILE", held_out)
    monkeypatch.setattr(pn, "RUN_FILE", tmp_path / "runs" / "pii__false_positives.json")
    monkeypatch.setattr(pn, "OUTPUTS_FILE", tmp_path / "runs" / "pii__false_positives__outputs.parquet")

    assert pn.main(generate=lambda texts: ["", "AGE: 25"], baseline=lambda text: [],
                   set_name="ai4privacy_val", name="negatives-val") == 0
    run = json.loads((tmp_path / "runs" / "pii__false_positives__negatives-val.json").read_text())
    assert run["set"]["name"] == "ai4privacy_val"
    assert run["adapter"]["all"]["texts_with_any_line"] == 1
    assert not (tmp_path / "runs" / "pii__false_positives.json").exists(), "the served run was overwritten"


def test_main_freezes_the_set_and_writes_the_run_without_a_model(tmp_path, monkeypatch):
    golden = tmp_path / "evals" / "golden"
    golden.mkdir(parents=True)
    pd.DataFrame({"text": ["Why is my card declined?", "I paid 30 pounds"]}).to_parquet(
        golden / "intent.parquet")
    pd.DataFrame({"instruction": ["I want a refund", "cancel {{Order Number}}"]}).to_parquet(
        golden / "drafting.parquet")
    monkeypatch.setattr(pn, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pn, "SET_DIR", tmp_path / "evals" / "pii_negatives")
    monkeypatch.setattr(pn, "SET_FILE", tmp_path / "evals" / "pii_negatives" / "set.parquet")
    monkeypatch.setattr(pn, "SET_RECORD", tmp_path / "evals" / "pii_negatives" / "SET.json")
    monkeypatch.setattr(pn, "RUN_FILE", tmp_path / "runs" / "run.json")
    monkeypatch.setattr(pn, "OUTPUTS_FILE", tmp_path / "runs" / "outputs.parquet")

    def fake_generate(texts):
        return ["AGE: 3" if "card" in t else "" for t in texts]

    assert pn.main(generate=fake_generate, baseline=lambda text: []) == 0
    run = json.loads((tmp_path / "runs" / "run.json").read_text())
    assert run["set"]["texts"] == 2                          # the digit and the slot were screened out
    assert run["adapter"]["all"]["texts_with_any_line"] == 1
    assert run["baseline"]["all"]["false_positive_rate"] == 0.0
    assert run["examples"][0]["output"] == "AGE: 3"
    record = json.loads((tmp_path / "evals" / "pii_negatives" / "SET.json").read_text())
    assert record["counts"] == {"intent": 1, "drafting": 1} and len(record["sha256"]) == 64
