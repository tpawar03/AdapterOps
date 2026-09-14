"""Guards on the PII negatives split: sentence spans, empty targets, length matching, original rows."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.data import pii_negatives_split as ps


def doc(text, spans):
    mask = [{"label": label, "start": text.index(value), "end": text.index(value) + len(value),
             "value": value} for label, value in spans]
    return {"source_text": text, "privacy_mask": mask}


def test_sentences_carry_only_the_spans_they_wholly_contain():
    text = ("Please email dana@example.com about the invoice. Thank you for your patience today. "
            "Call Dr. Smith tomorrow at noon.")
    frame = pd.DataFrame([doc(text, [("EMAIL", "dana@example.com"), ("SURNAME", "Smith")])])
    rows = ps.sentences(frame).set_index("source_text")
    email = rows.loc["Please email dana@example.com about the invoice."]
    assert email.spans == 1 and email.contained and email.target == "EMAIL: dana@example.com"
    assert "dana@example.com" not in email.blind
    thanks = rows.loc["Thank you for your patience today."]
    assert thanks.spans == 0 and thanks.target == ""
    # "Dr." ends a regex sentence, so "Smith tomorrow at noon." holds the span but "Call Dr." does
    # not; the split sentence that touches it must still contain it whole.
    assert all(r.contained for r in rows.itertuples() if r.spans)


def test_negatives_are_span_free_and_pass_the_strict_screen():
    frame = pd.DataFrame({"source_text": ["Thank you for your patience.", "We met in London last week.",
                                          "Your order ships soon."],
                          "spans": [0, 0, 1], "chars": [28, 27, 21]})
    negatives = ps.negatives_from(frame)
    assert negatives.source_text.tolist() == ["Thank you for your patience."]


def test_length_matching_draws_one_positive_per_negative_from_the_same_bin():
    negatives = pd.DataFrame({"chars": [20] * 10 + [200] * 10})
    positives = pd.DataFrame({"chars": [20] * 50 + [200] * 50 + [120] * 50})
    matched = ps.length_matched(negatives, positives)
    assert len(matched) == 20
    assert sorted(matched.chars.value_counts().to_dict().items()) == [(20, 10), (200, 10)]


def test_the_split_keeps_the_served_adapters_exact_documents(tmp_path, monkeypatch):
    from adapterops.train import qlora

    # The PII sentence and the span-free one are both 27 characters, so length matching can pair
    # them; the numbered sentence makes each document distinct and is neither (a digit, no span).
    texts = [(f"Email from Dana Scott here. Thank you for reading this. Reference {i} is attached.",
              [("GIVENNAME", "Dana"), ("SURNAME", "Scott")]) for i in range(40)]
    train = pd.DataFrame([doc(t, s) | {"target": ""} for t, s in texts])
    source = tmp_path / "data" / "pii" / "split_train.parquet"
    source.parent.mkdir(parents=True)
    train.to_parquet(source)
    train.iloc[:10].to_parquet(tmp_path / "data" / "pii" / "split_val.parquet")
    monkeypatch.setattr(ps, "SOURCE", source)
    monkeypatch.setattr(ps, "VAL_SOURCE", tmp_path / "data" / "pii" / "split_val.parquet")
    monkeypatch.setattr(ps, "DOCUMENTS", 15)
    monkeypatch.setattr(ps, "probe", lambda n, p, seed=ps.SEED: {"skipped": True})
    monkeypatch.setattr(qlora, "REPO_ROOT", tmp_path)       # load_split reads this same file

    rows, facts, val_negatives = ps.build()
    kept = set(rows[rows.kind == "document"].source_text)
    assert kept == set(qlora.load_split("pii", "train", 15).source_text)
    assert (rows[rows.kind == "negative_sentence"].target == "").all()
    assert set(rows[rows.kind == "positive_sentence"].target) == {"GIVENNAME: Dana\nSURNAME: Scott"}
    assert facts["rows"]["document"] == 15 and len(val_negatives) > 0
