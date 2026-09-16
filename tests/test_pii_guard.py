"""Guards on the PII second detector: it adds only where the adapter tagged nothing, and only chosen labels."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import pii_guard as pg

TEXT = "Reach Dana at dana@example.com or 4111 1111 1111 1111 before 12/05/2024."


def test_a_missed_email_is_added_and_the_adapters_spans_are_kept():
    out = pg.augment(TEXT, "GIVENNAME: Dana", ["EMAIL"])
    assert out.splitlines() == ["GIVENNAME: Dana", "EMAIL: dana@example.com"]


def test_nothing_is_added_over_a_span_the_adapter_already_placed():
    answer = "GIVENNAME: Dana\nEMAIL: dana@example.com"
    assert pg.augment(TEXT, answer, ["EMAIL"]) == answer


def test_only_the_chosen_labels_are_added():
    out = pg.augment(TEXT, "", ["CREDITCARDNUMBER"])
    assert out == "CREDITCARDNUMBER: 4111 1111 1111 1111"


def test_the_request_path_adds_a_missed_email_unless_the_guard_is_off():
    from adapterops.serve import pipeline as pl

    class Local:
        name = "local"

        def generate(self, task, text):
            return pl.Generation(text="GIVENNAME: Dana", mean_logprob=-0.01, n_tokens=3)

    def service(labels):
        return pl.Service(local=Local(), policy=pl.NeverEscalate(), prices=pl.Prices(0.00001, 0.15, 0.6, "t"),
                          vocab={"intent": frozenset({"x"}), "urgency": frozenset({"low"})},
                          pii_guard_labels=labels)

    guarded = service(pl.PII_GUARD_LABELS).handle(TEXT, ["pii"])["pairs"]["pii"]
    assert {"label": "EMAIL", "value": "dana@example.com"} in guarded["output"]["spans"]
    assert any("PII guard added" in n for n in guarded["notes"])
    plain = service(()).handle(TEXT, ["pii"])["pairs"]["pii"]
    assert plain["output"]["spans"] == [{"label": "GIVENNAME", "value": "Dana"}]


def test_the_served_labels_are_the_measured_set():
    from adapterops.serve import pipeline as pl

    assert pl.PII_GUARD_LABELS == pg.LABEL_SETS["structured_ids_dates"]
    assert "ZIPCODE" not in pl.PII_GUARD_LABELS


def test_measure_counts_the_leak_the_guard_closes():
    frame = pd.DataFrame({"text": [TEXT], "gold": ["GIVENNAME: Dana\nEMAIL: dana@example.com"],
                          "prediction": ["GIVENNAME: Dana"]})
    before, after = pg.measure(frame, None), pg.measure(frame, ["EMAIL"])
    assert before["gold_spans_wholly_unmasked"] == 1 and after["gold_spans_wholly_unmasked"] == 0
    assert after["spans_added"] == 1 and after["span_f1_strict"] == 1.0
