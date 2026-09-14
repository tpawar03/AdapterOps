"""Guards on the operating-curve chart (F11) and the router latency record (PRD §7)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import latency, plot

CURVE = ROOT / "runs" / "router__operating_curve__judged.json"
LATENCY = ROOT / "runs" / "router__latency.json"


def test_summarise_checks_p95_not_the_mean_against_the_budget():
    ms = [10.0] * 90 + [80.0] * 10          # mean 17 ms, P95 80 ms
    s = latency.summarise(ms, target_ms=50.0)
    assert s["mean_ms"] < 50 < s["p95_ms"]
    assert s["p95_within_target"] is False


def test_ticket_batches_hold_one_pair_per_task():
    frame = pd.DataFrame({"task": ["intent", "pii", "urgency", "drafting"] * 3 + ["intent"],
                          "text": list("abcdefghijklm")})
    batches = latency.ticket_batches(frame)
    assert len(batches) == 3
    assert all(sorted(b.task) == ["drafting", "intent", "pii", "urgency"] for b in batches)


@pytest.mark.skipif(not CURVE.exists(), reason="no committed operating curve")
def test_the_chart_draws_both_populations_and_marks_the_dashboard_operating_point(tmp_path):
    out = tmp_path / "curve.png"
    assert plot.main(CURVE, out) == 0
    assert out.stat().st_size > 10_000
    report = json.loads(CURVE.read_text())
    for name in plot.PANELS:
        op = plot.point(report["populations"][name]["all_tasks"]["curve"], "confidence",
                        plot.OPERATING_BUDGET)
        assert op["escalation_rate"] == pytest.approx(0.2, abs=0.01)


@pytest.mark.skipif(not LATENCY.exists(), reason="run `adapterops router-latency` first")
def test_the_latency_record_is_for_the_pinned_router_and_reproduced_its_scores():
    record = json.loads(LATENCY.read_text())
    system = json.loads((ROOT / "manifests" / "system.json").read_text())
    assert record["router"]["sha256"] == latency.pinned_sha(system)
    assert record["score_check"]["max_abs_diff_vs_committed"] <= latency.SCORE_TOLERANCE
