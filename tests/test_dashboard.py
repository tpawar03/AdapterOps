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
def test_the_frontier_call_rate_shows_live_and_offline_apart():
    rendered = db.render(db.load())
    section = rendered.split("## 3 · Frontier-call rate")[1].split("## 4 ·")[0]
    assert "Measured live, on the curve's own pairs" in section
    assert "The offline operating curve" in section
    assert section.index("Measured live") < section.index("The offline operating curve")


@needs_runs
def test_the_throughput_ceiling_sits_beside_vllm_direct_with_derived_cost():
    text = db.render(db.load())
    section = text[text.index("### Throughput ceiling"):text.index("## 6 · Fallback rate")]
    rows = [line for line in section.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert [row.split(" | ")[0] for row in rows] == ["| 16", "| 32", "| 64", "| 128", "| 256"]
    assert "**Sustained:**" in section and " 0 errors." in section
    assert "Shifted population" in text


@needs_runs
def test_the_pii_score_is_shown_beside_its_false_positive_rate():
    text = db.render(db.load())
    section = text[text.index("## 1 · Quality retained — random"):text.index("## 2 · Quality retained")]
    assert "conditional on the input containing PII" in section
    assert "runs/pii__false_positives.json" in section
    assert "not yet served" in section, "a candidate's numbers must not read as the served adapter's"


@needs_runs
def test_prd_7_misses_are_shown_as_misses():
    text = db.render(db.load())
    section = text[text.index("## 7 · PRD §7 targets"):text.index(db.FAILURE_HEADING)]
    assert section.count("**missed**") == 2          # PII and drafting P95 over 500 ms
    assert "| pii on the A10" in section
    regression = next(line for line in section.splitlines() if "regression run < 25 min" in line)
    assert regression.endswith("| met |") and "a10-v4" in regression


@needs_runs
def test_the_router_retry_is_shown_with_its_verdicts():
    text = db.render(db.load())
    assert "Pre-registered retry (D42)" in text
    assert "loses to confidence" in text
    assert "Rules-based baseline (F26)" in text
    assert "judging 1K drafting replies" in text
    assert "Frontier reference, not a gate" in text
