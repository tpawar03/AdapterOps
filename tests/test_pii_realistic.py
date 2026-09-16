"""Guards on scoring the PII adapter against outside schemas."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import pii_realistic as pr

WORDS = " ".join(["word"] * 20)


def test_paragraph_offsets_are_relative_and_scope_follows_the_masking_need():
    text = f"Heading\n{WORDS} Mr Galip Yalman on 3 May 1999 in Ankara {WORDS}"
    start = text.index("Galip")
    doc = {"doc_id": "d1", "text": text, "annotations": {"annotator1": {"entity_mentions": [
        {"start_offset": start, "end_offset": start + 12, "entity_type": "PERSON", "identifier_type": "DIRECT"},
        {"start_offset": text.index("Ankara"), "end_offset": text.index("Ankara") + 6, "entity_type": "LOC",
         "identifier_type": "NO_MASK"},
        {"start_offset": text.index("3 May"), "end_offset": text.index("3 May") + 10, "entity_type": "MISC",
         "identifier_type": "QUASI"}]}}}
    [para] = pr.paragraphs(doc)
    gold = json.loads(para["gold"])
    assert para["text"][gold[0]["start"]:gold[0]["end"]] == "Galip Yalman"
    assert [g["scope"] for g in gold] == ["in", "no_mask", "out"]


def test_score_counts_leaks_label_agreement_precision_and_escalation():
    text = "Call Dana Reyes at 555-0100 today."
    gold = json.dumps([{"start": 5, "end": 9, "label": "first_name", "scope": "in"},
                       {"start": 10, "end": 15, "label": "last_name", "scope": "in"},
                       {"start": 19, "end": 27, "label": "phone_number", "scope": "in"}])
    frame = pd.DataFrame({"text": [text], "gold": [gold], "mean_logprob": [-0.5],
                          "prediction": ["GIVENNAME: Dana\nCITY: Reyes\nDATE: today"]})
    r = pr.score(frame, threshold=0.39, mapping=pr.NEMOTRON_LABELS)
    assert r["in_scope_spans"] == 3 and r["in_scope_wholly_unmasked"] == round(1 / 3, 4)
    assert r["docs_fully_masked_in_scope"] == 0.0
    assert r["in_scope_boundary_exact"] == round(2 / 3, 4) and r["label_right_when_boundary_exact"] == 0.5
    assert r["masked_chars_on_personal_data"] == round(9 / 14, 4)
    assert r["texts_with_a_leak_escalated"] == 1


def test_paired_bootstrap_separates_a_real_fall_from_noise():
    spans = [4] * 100
    fell = pr.paired_bootstrap([2] * 100, [0] * 100, spans, resamples=500)
    assert fell["difference"] == -0.5 and fell["ci95"][1] < 0
    same = pr.paired_bootstrap([1, 0] * 50, [0, 1] * 50, spans, resamples=500)
    assert same["difference"] == 0.0 and same["ci95"][0] < 0 < same["ci95"][1]


def test_nemotron_sample_fills_each_cell_and_marks_scope():
    rows = []
    for locale in ("us", "intl"):
        for fmt in ("structured", "unstructured"):
            for i in range(60):
                spans = [{"start": 0, "end": 4, "label": "first_name"}, {"start": 5, "end": 8, "label": "url"}]
                rows.append({"uid": f"{locale}{fmt}{i}", "locale": locale, "document_format": fmt,
                             "text": "Dana abc", "spans": str(spans)})
    rows.append({"uid": "none", "locale": "us", "document_format": "structured", "text": "abc",
                 "spans": str([{"start": 0, "end": 3, "label": "url"}])})
    sample = pr.nemotron_sample(pd.DataFrame(rows))
    assert len(sample) == pr.PER_SOURCE and sample.stratum.value_counts().eq(pr.PER_SOURCE // 4).all()
    assert "none" not in set(sample.id)
    assert [g["scope"] for g in json.loads(sample.gold.iloc[0])] == ["in", "out"]
