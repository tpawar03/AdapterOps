"""Guards on the out-of-domain report blocks."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ood_report as rep


def outputs(task, rows, invalid=()):
    return pd.DataFrame([{"id": i, "task": task, "output": json.dumps(o), "would_escalate": esc,
                          "judge_score": js, "latency_ms": 10.0, "score": None, "valid": i not in invalid}
                         for i, o, esc, js in rows])


def labels(task, pairs):
    return pd.DataFrame([{"id": i, "task": task, "label": lab} for i, lab in pairs])


def test_intent_separates_no_fit_from_wrong_and_reads_escalation_by_outcome():
    out = outputs("intent", [("a", {"label": "card_arrival"}, False, None),
                             ("b", {"label": "card_arrival"}, True, None),
                             ("c", {"label": "top_up_failed"}, True, None),
                             ("d", {"label": "edit_order"}, False, None)], invalid={"d"})
    lab = labels("intent", [("a", "card_arrival"), ("b", "lost_or_stolen_card"), ("c", "NONE"),
                            ("d", "NONE")])
    r = rep.intent_block(out, lab)
    assert r["no_banking77_label_fits"] == 0.5 and r["invented_label_outside_the_77"] == 0.25
    assert r["accuracy_where_a_label_fits"] == 0.5
    conf, routed = r["escalated_on_confidence"], r["routed_away_confidence_or_fallback"]
    assert conf["when_right"] == 0.0 and conf["when_wrong"] == 1.0
    assert conf["when_no_label_fits"] == 0.5, "the invented label was not escalated on confidence"
    assert routed["when_no_label_fits"] == 1.0, "but the fallback caught it"


def test_cfpb_pii_counts_spans_on_redaction_marks_apart():
    sample = pd.DataFrame({"id": ["x"], "source": ["cfpb"], "text": ["t"], "pii_gold": ["[]"]})
    out = pd.DataFrame([{"id": "x", "task": "pii", "score": None, "output": json.dumps({"spans": [
        {"label": "GIVENNAME", "value": "XXXX"}, {"label": "DATE", "value": "XX/XX/XXXX"},
        {"label": "CITY", "value": "Capital One"}]})}])
    r = rep.pii_block(sample, out, "cfpb")
    assert r["texts_with_any_span"] == 1 and r["spans_on_redaction_marks"] == 2 and r["spans_on_other_text"] == 1


def test_drafting_pairs_the_judge_with_gpt4o_grades_only_where_both_exist():
    sample = pd.DataFrame({"id": list("abcdef"), "source": ["abcd"] * 6})
    out = outputs("drafting", [(i, {"reply": "r"}, False, js)
                               for i, js in zip("abcdef", (4.1, 3.2, 4.5, 2.9, 3.8, 4.0), strict=True)])
    lab = labels("drafting_grade", [("a", "4"), ("b", "3"), ("c", "5"), ("d", "2"), ("e", "4")])
    r = rep.drafting_block(out, lab, "abcd", sample)
    assert r["replies"] == 6 and r["graded_by_gpt4o"] == 5 and r["spearman_judge_vs_gpt4o"] > 0.9
