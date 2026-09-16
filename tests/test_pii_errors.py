"""Guards on the PII error taxonomy: one category per span, and the exact count matches the scorer."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import pii_errors as pe
from adapterops.eval.spans import parse_model_output, score_spans, spans_from_values

TEXT = "Call Dr Dana Scott at 555-0199 on May 2nd about the refund."
GOLD = "TITLE: Dr\nGIVENNAME: Dana\nSURNAME: Scott\nTELEPHONENUM: 555-0199\nDATE: May 2nd"
PREDICTION = ("TITLE: Dr\nGIVENNAME: Dana Scott\nSURNAME: Scott\nPHONE: 555-0199\nDATE: 2nd\n"
              "AGE: 44\nCITY: refund")


def test_every_span_gets_one_category_and_exact_matches_the_scorer():
    rows = pd.DataFrame(pe.classify(TEXT, GOLD, PREDICTION))
    kinds = rows.kind.value_counts().to_dict()
    assert kinds == {"exact": 1, "boundary": 2, "label": 1, "repeated_value": 1, "invented": 1,
                     "spurious": 1, "missed": 1}
    # "Dana Scott" claims both names' text, so the later "Scott" has nowhere left to go and SURNAME is missed.
    assert rows[rows.kind == "missed"].gold_label.tolist() == ["SURNAME"]
    assert rows[rows.kind == "label"].iloc[0][["gold_label", "pred_label"]].tolist() == ["TELEPHONENUM", "PHONE"]
    gold = spans_from_values(TEXT, parse_model_output(GOLD))
    pred = spans_from_values(TEXT, parse_model_output(PREDICTION))
    assert score_spans([gold], [pred], strict=True)["tp"] == kinds["exact"]
    # For redaction the split name still masks "Scott", and the phone number is masked under the wrong
    # label; only DATE's "May" is left showing.
    assert pe.coverage(TEXT, gold, pred) == (1, 0)


def test_invented_values_are_counted_though_the_scorer_ignores_them():
    rows = pd.DataFrame(pe.classify("My card is late.", "", "AGE: 25\nGIVENNAME: John"))
    assert rows.kind.tolist() == ["invented", "invented"]
    assert score_spans([[]], [spans_from_values("My card is late.", parse_model_output("AGE: 25"))],
                       strict=True)["fp"] == 0


def test_failed_documents_are_attributed_to_their_causes():
    frame = pd.DataFrame({
        "text": ["Email dana@example.com today.", "Call 555-0199 or visit Leeds soon."],
        "gold": ["EMAIL: dana@example.com", "TELEPHONENUM: 555-0199\nCITY: Leeds"],
        "prediction": ["EMAIL: dana@example.com", "TELEPHONENUM: 555-0199\nCITY: Leeds soon"],
    })
    s = pe.summarise(frame, pe.classify_frame(frame))
    assert s["documents"] == 2 and s["documents_all_spans_right"] == 1
    assert s["failed_documents_by_single_cause"] == {"boundary": 1}
    assert s["boundary_extends_gold"] == 1
    assert s["per_label"]["CITY"]["recall"] == 0.0 and s["per_label"]["EMAIL"]["recall"] == 1.0
