"""Guards on the regression run (F18) — and the detect → block chain it feeds (M7).

No GPU and no server: an oracle predictor that returns gold, and a broken one that returns a
constant intent label, are enough to prove the chain end to end in code. The last test is
M7's claim with the serving path mocked out: a real regression, measured on the real frozen
splits, reaches the promotion gate and is blocked.
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import regression as reg
from adapterops.manifest import system


@pytest.fixture(scope="module")
def table():
    lookup = {}
    for task in reg.TASKS:
        for which in reg.SPLITS:
            texts, gold = reg.load_split(task, which)
            lookup.update({(task, t): g for t, g in zip(texts, gold, strict=True)})
    return lookup


@pytest.fixture(scope="module")
def oracle_run(table):
    return reg.run(lambda task, texts: [table[(task, t)] for t in texts])


@pytest.fixture(scope="module")
def broken_run(table):
    def predict(task, texts):
        if task == "intent":
            return ["card_arrival"] * len(texts)          # a model that has stopped working
        return [table[(task, t)] for t in texts]
    return reg.run(predict)


def test_every_task_has_both_splits_and_neither_is_empty():
    for task in reg.TASKS:
        for which in reg.SPLITS:
            texts, gold = reg.load_split(task, which)
            assert texts and len(texts) == len(gold), f"{task}/{which}"


def test_an_oracle_scores_perfectly_or_at_the_known_ceiling(oracle_run):
    s = oracle_run["per_split"]
    for which in reg.SPLITS:
        assert s["intent"][which]["micro_accuracy"] == 1.0
        assert s["urgency"][which]["macro_f1"] == 1.0
        # LABEL: value recovery is not lossless — the golden-set ceiling is 0.9942 (D22).
        assert s["pii"][which]["span_f1_strict"] >= 0.95
        assert s["drafting"][which]["judge_score_mean"] is None


def test_no_change_means_zero_drop_and_an_unmeasured_task_stays_none(oracle_run):
    cmp = reg.compare(oracle_run, oracle_run)["per_task"]
    for task in ("intent", "urgency", "pii"):
        assert cmp[task]["random"]["drop"] == 0.0 and cmp[task]["hard"]["drop"] == 0.0
    assert cmp["drafting"]["drop"] is None, "an unmeasured task must not read as no regression"


def test_a_broken_adapter_regresses_its_own_task_on_both_splits_and_nothing_else(
        oracle_run, broken_run):
    cmp = reg.compare(broken_run, oracle_run)["per_task"]
    assert cmp["intent"]["random"]["drop"] > 0.9
    assert cmp["intent"]["hard"]["drop"] > 0.9
    for task in ("urgency", "pii"):
        assert cmp[task]["drop"] == 0.0


def test_random_and_hard_are_reported_apart_and_only_random_gates(oracle_run, broken_run):
    """F32: never blended. F33: the hard split gates only once its threshold is derived."""
    row = reg.compare(broken_run, oracle_run)["per_task"]["intent"]
    assert row["drop"] == row["random"]["drop"]
    assert "hard" in row and row["hard"]["drop"] is not None
    assert "report-only" in row["hard_gate"]


def test_detect_then_block_end_to_end(oracle_run, broken_run, monkeypatch):
    """M7 with serving mocked: a measured regression reaches the gate and is refused.

    The candidate pins exactly what the current manifest pins, so no component or split has
    moved and the only thing the gate can act on is the regression itself.
    """
    candidate = system.build("regression test")
    candidate["gate"] = {"state": "enforcing", "why": "test", "thresholds": {"intent": 0.02}}
    monkeypatch.setattr(system, "load_current", lambda: copy.deepcopy(candidate))

    regressed = reg.compare(broken_run, oracle_run)
    assert any("intent: regression" in r for r in system.blocking_reasons(candidate, regressed))

    unchanged = reg.compare(oracle_run, oracle_run)
    assert not system.blocking_reasons(candidate, unchanged)


def test_predictions_can_be_kept_for_judging_after_the_gpu_is_gone(table):
    """Drafting is judge-scored on CPU after the session. Without the replies kept, scoring it
    on the golden set would mean renting the GPU again just to regenerate them."""
    rows = []
    reg.run(lambda task, texts: [table[(task, t)] for t in texts], tasks=("drafting",),
            collect=rows)
    random_texts, _ = reg.load_split("drafting", "random")
    hard_texts, _ = reg.load_split("drafting", "hard")
    assert len(rows) == len(random_texts) + len(hard_texts)
    assert {r["split"] for r in rows} == {"random", "hard"}
    assert all(r["prediction"] == r["gold"] for r in rows)
