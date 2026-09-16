"""Guards on the urgency TF-IDF comparison and its frozen rule."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import urgency_tfidf as ut


def test_the_rule_needs_both_conditions():
    gate = 0.0381
    clear = ut.decide({"random": 0.55, "hard": 0.10}, {"random": 0.42, "hard": 0.08}, gate)
    assert clear["serve_tfidf_for_urgency"] is True

    thin = ut.decide({"random": 0.44, "hard": 0.10}, {"random": 0.42, "hard": 0.08}, gate)
    assert thin["beats_adapter_on_golden_by_more_than_the_gate"] is False
    assert thin["serve_tfidf_for_urgency"] is False

    worse_on_hard = ut.decide({"random": 0.55, "hard": 0.01}, {"random": 0.42, "hard": 0.08}, gate)
    assert worse_on_hard["not_worse_on_hard_than_the_gate_allows"] is False
    assert worse_on_hard["serve_tfidf_for_urgency"] is False


def test_the_gate_threshold_comes_from_the_committed_gate():
    assert ut.threshold() == 0.0381
