"""Guards on the hard-cases adjudication screen (F30, D36).

The first test is the design: a frontier miss alone must not quarantine an item, because
hard cases were chosen for difficulty and a second miss is what difficulty predicts.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import adjudicate as adj


def frames(rows):
    """rows: (pair_id, task, gold, adapter_answer, frontier_answer or None)."""
    cand = pd.DataFrame([{"pair_id": p, "task": t, "gold": g, "pred_clean": a,
                          "success": False} for p, t, g, a, _ in rows])
    front = pd.DataFrame([{"pair_id": p, "frontier_pred_clean": f, "frontier_span_f1": 0.5}
                          for p, _, _, _, f in rows])
    return cand, front


def test_a_frontier_miss_alone_does_not_quarantine_but_shared_alternative_does():
    cand, front = frames([
        ("i1", "intent", "card_arrival", "card_linking", "top_up_failed"),   # both wrong, differ
        ("i2", "intent", "card_arrival", "card_linking", "card_linking"),    # both wrong, agree
        ("i3", "intent", "card_arrival", "card_linking", "card_arrival"),    # frontier sides with gold
    ])
    m = adj.adjudicate(cand, front).set_index("pair_id")
    assert m.loc["i1", "frontier_rejects_gold"] and not m.loc["i1", "quarantine"]
    assert m.loc["i2", "independent_agreement_against_gold"] and m.loc["i2", "quarantine"]
    assert not m.loc["i3", "frontier_rejects_gold"] and not m.loc["i3", "quarantine"]


def test_urgency_agreement_is_reported_but_never_quarantines():
    """Three classes: once both models are wrong, coinciding is near a coin flip."""
    cand, front = frames([("u1", "urgency", "high", "medium", "medium")])
    m = adj.adjudicate(cand, front)
    assert bool(m.independent_agreement_against_gold.iloc[0])
    assert not bool(m.quarantine.iloc[0])
    assert m.applicability.iloc[0] == "reported_not_applied"


def test_pii_and_drafting_are_never_quarantined_even_when_answers_match():
    cand, front = frames([
        ("p1", "pii", "EMAIL: a@b.c", "EMAIL: x@y.z", "EMAIL: x@y.z"),
        ("d1", "drafting", "reference reply", "same draft", "same draft"),
    ])
    m = adj.adjudicate(cand, front)
    assert not m.quarantine.any()
    assert not m.independent_agreement_against_gold.any(), "the rule is classification-only"


def test_a_missing_frontier_answer_never_quarantines():
    cand, front = frames([("i1", "intent", "card_arrival", "card_linking", None)])
    m = adj.adjudicate(cand, front)
    assert not m.quarantine.any() and not m.frontier_rejects_gold.any()


def test_chance_rate_is_one_over_the_remaining_wrong_classes():
    assert adj.uniform_chance_given_both_wrong(77) == pytest.approx(1 / 76)
    assert adj.uniform_chance_given_both_wrong(3) == pytest.approx(0.5)
    assert adj.uniform_chance_given_both_wrong(None) is None


def test_per_task_report_compares_observed_agreement_to_chance():
    cand, front = frames([
        ("i1", "intent", "g", "a", "b"),
        ("i2", "intent", "g", "a", "a"),
        ("i3", "intent", "g", "a", "g"),
        ("i4", "intent", "g", "a", "a"),
    ])
    r = adj.per_task(adj.adjudicate(cand, front), classes={"intent": 77})["intent"]
    assert r["prd_literal_rule_rate"] == 0.75
    assert r["independent_agreement_rate"] == 0.5
    assert r["agreement_given_both_wrong"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["uniform_chance_given_both_wrong"] == pytest.approx(1 / 76, abs=1e-4)
    assert r["retained"] + r["quarantined"] == r["candidates"]


def test_freeze_partitions_candidates_and_refuses_to_rescreen(tmp_path, monkeypatch):
    cand, front = frames([
        ("i1", "intent", "g", "a", "a"),
        ("i2", "intent", "g", "a", "b"),
        ("u1", "urgency", "high", "low", "low"),
    ])
    cand.to_parquet(tmp_path / "cand.parquet")
    front.to_parquet(tmp_path / "front.parquet")
    for name, value in {
        "REPO_ROOT": tmp_path, "CANDIDATES": tmp_path / "cand.parquet",
        "FRONTIER": tmp_path / "front.parquet", "HARD_DIR": tmp_path / "evals" / "hard",
        "HARD_FILE": tmp_path / "evals" / "hard" / "hard_cases.parquet",
        "QUARANTINE_FILE": tmp_path / "evals" / "hard" / "quarantine.parquet",
        "MANIFEST": tmp_path / "evals" / "HARD_CASES.json",
    }.items():
        monkeypatch.setattr(adj, name, value)
    monkeypatch.setattr(adj, "n_classes", lambda t: {"intent": 77, "urgency": 3}.get(t))

    assert adj.main() == 0
    kept = pd.read_parquet(adj.HARD_FILE)
    quarantined = pd.read_parquet(adj.QUARANTINE_FILE)
    assert set(kept.pair_id) == {"i2", "u1"} and set(quarantined.pair_id) == {"i1"}
    assert adj.main() == 1, "a frozen hard split was re-screened without --force"
