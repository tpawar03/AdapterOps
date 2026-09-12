"""Guards on gate-threshold derivation (F33, D37).

The first test is §11's whole argument in one assertion: two identical baseline runs must not
produce a zero threshold, because a zero threshold blocks every release that moves at all.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import regression as reg
from adapterops.eval import thresholds as th


def run(values):
    return {"per_split": {task: {split: {reg.GATED[task]: v} for split, v in splits.items()}
                          for task, splits in values.items()}}


def test_identical_runs_with_no_training_measurement_give_no_threshold_at_all():
    a = run({"urgency": {"random": 0.55, "hard": 0.40}})
    row = th.derive(a, a, training={})["per_task"]["urgency"]["random"]
    assert row["threshold"] is None
    assert "report-only" in row["note"]


def test_training_variance_sets_the_floor_when_inference_is_quieter():
    a = run({"intent": {"random": 0.931}})
    d = th.derive(a, a, training={("intent", "random"): {"spread": 0.0039, "source": "x"}},
                  multiplier=3.0)
    row = d["per_task"]["intent"]["random"]
    assert row["bound_by"] == "training"
    assert row["threshold"] == pytest.approx(0.0117)
    assert row["provisional"] is False


def test_a_threshold_from_inference_alone_is_provisional_and_not_enforced():
    d = th.derive(run({"pii": {"random": 0.95}}), run({"pii": {"random": 0.94}}), training={})
    row = d["per_task"]["pii"]["random"]
    assert row["bound_by"] == "inference" and row["provisional"] is True
    assert "pii" not in th.gate_from(d)["thresholds"]


def test_a_larger_inference_spread_binds_over_a_measured_training_spread():
    d = th.derive(run({"intent": {"random": 0.93}}), run({"intent": {"random": 0.91}}),
                  training={("intent", "random"): {"spread": 0.0039, "source": "x"}})
    row = d["per_task"]["intent"]["random"]
    assert row["bound_by"] == "inference"
    assert row["threshold"] == pytest.approx(0.06)
    assert row["provisional"] is False


def test_an_unmeasured_task_stays_unmeasured():
    a = run({"drafting": {"random": None}})
    assert "drafting" not in th.derive(a, a, training={})["per_task"]


def test_training_spread_is_read_from_the_committed_run_record_not_copied():
    spreads = th.training_spreads()
    assert spreads[("intent", "random")]["spread"] == pytest.approx(0.0039, abs=1e-4)
    assert ("urgency", "random") not in spreads, "no number may be borrowed from intent"


def test_the_gate_enforces_only_when_a_non_provisional_threshold_exists():
    provisional = th.derive(run({"pii": {"random": 0.95}}), run({"pii": {"random": 0.94}}),
                            training={})
    assert th.gate_from(provisional)["state"] == "report_only"
    real = th.derive(run({"intent": {"random": 0.93}}), run({"intent": {"random": 0.93}}),
                     training={("intent", "random"): {"spread": 0.0039, "source": "x"}})
    gate = th.gate_from(real)
    assert gate["state"] == "enforcing" and set(gate["thresholds"]) == {"intent"}
