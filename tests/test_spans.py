"""Guards on span scoring and the two recovery paths."""

import pandas as pd
import pytest

from adapterops.eval.spans import (
    Span,
    parse_model_output,
    score_spans,
    spans_from_masked,
    spans_from_values,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_strict_requires_exact_boundaries():
    gold = [[Span(0, 5, "NAME")]]
    assert score_spans(gold, [[Span(0, 5, "NAME")]], strict=True)["f1"] == 1.0
    assert score_spans(gold, [[Span(0, 4, "NAME")]], strict=True)["f1"] == 0.0
    assert score_spans(gold, [[Span(0, 4, "NAME")]], strict=False)["f1"] == 1.0


def test_label_must_match_even_when_offsets_do():
    gold = [[Span(0, 5, "NAME")]]
    assert score_spans(gold, [[Span(0, 5, "CITY")]], strict=False)["f1"] == 0.0


def test_hallucinated_values_are_dropped_not_invented():
    spans = spans_from_values("hello world", [("NAME", "nobody")])
    assert spans == []


def test_repeated_value_maps_to_distinct_spans():
    spans = spans_from_values("Ann and Ann", [("NAME", "Ann"), ("NAME", "Ann")])
    assert [(s.start, s.end) for s in spans] == [(0, 3), (8, 11)]


def test_parse_ignores_unparseable_lines():
    out = parse_model_output("GIVENNAME: Ann\nnot a pair\nEMAIL: a@b.c\n")
    assert out == [("GIVENNAME", "Ann"), ("EMAIL", "a@b.c")]


@pytest.mark.skipif(not (ROOT / "data/pii/train.parquet").exists(), reason="mirror absent")
def test_value_format_beats_masked_format_on_real_data():
    """D22's evidence, as a test: masked text cannot recover exact boundaries."""
    df = pd.read_parquet(ROOT / "data/pii/train.parquet").head(200)
    gold, by_value, by_mask = [], [], []
    for src, msk, mask in zip(df.source_text, df.masked_text, df.privacy_mask, strict=True):
        gold.append([Span(s["start"], s["end"], s["label"]) for s in mask])
        by_value.append(spans_from_values(src, [(s["label"], s["value"]) for s in mask]))
        recovered, ok = spans_from_masked(src, msk)
        by_mask.append(recovered if ok else [])
    f_value = score_spans(gold, by_value, strict=True)["f1"]
    f_mask = score_spans(gold, by_mask, strict=True)["f1"]
    assert f_value > 0.99, f_value
    assert f_mask < 0.95, f_mask
