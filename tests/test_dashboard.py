"""Guards on the results dashboard (F17, F32).

Rendered against the committed runs, so a test failure here means the dashboard and the evidence
it summarises have come apart.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import dashboard as db

needs_runs = pytest.mark.skipif(not (ROOT / "runs" / "economics.json").exists(),
                                reason="run `adapterops economics` first")


def test_gain_captured_is_zero_at_local_one_at_oracle_negative_below():
    assert db.gain_captured(0.7, 0.7, 0.8) == pytest.approx(0.0)
    assert db.gain_captured(0.8, 0.7, 0.8) == pytest.approx(1.0)
    assert db.gain_captured(0.6, 0.7, 0.8) < 0


def test_gain_is_undefined_when_the_oracle_adds_nothing():
    assert db.gain_captured(0.7, 0.7, 0.7) is None


def test_gate_state_never_reads_a_zero_floor_as_enforced():
    row = {"threshold": None, "provisional": None, "bound_by": None}
    assert "report-only" in db.gate_state(row)
    row = {"threshold": 0.04, "provisional": True, "bound_by": "inference"}
    assert "not enforced" in db.gate_state(row)


@needs_runs
def test_the_two_quality_rows_are_separate_sections():
    text = db.render(db.load())
    random_at = text.index("## 1 · Quality retained — random golden set")
    hard_at = text.index("## 2 · Quality retained — hard cases")
    assert random_at < hard_at
    assert "blend" not in text.lower().replace("never averaged", "")


@needs_runs
def test_m7_numbers_come_from_the_committed_run():
    text = db.render(db.load())
    assert "0.9234" in text
    assert db.FAILURE_HEADING in text


@needs_runs
def test_the_frontier_call_rate_is_labelled_offline():
    assert "Not measured live" in db.render(db.load())
