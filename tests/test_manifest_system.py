"""Guards on promotion blocking (F19) and rollback (F20) — together, M7.

M7 is the project's distinguishing claim, and the only way it means anything is if the
block actually refuses. These run against a temporary repo root so the real
`manifests/system.json` is never touched.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.manifest import system


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo root with one adapter pin and one golden split."""
    (tmp_path / "manifests").mkdir()
    (tmp_path / "evals" / "golden").mkdir(parents=True)
    (tmp_path / "evals" / "golden" / "intent.parquet").write_bytes(b"golden-v1")

    monkeypatch.setattr(system, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(system, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(system, "CURRENT", tmp_path / "manifests" / "system.json")
    monkeypatch.setattr(system, "HISTORY", tmp_path / "manifests" / "history")
    monkeypatch.setattr(system, "ADAPTERS", tmp_path / "manifests" / "adapters.json")

    def pin_adapters(rev: str) -> None:
        (tmp_path / "manifests" / "adapters.json").write_text(json.dumps({"components": {
            "base_model": {"repo": "base", "revision": "b0"},
            "intent": {"repo": "r", "revision": rev, "weight_sha256": rev},
            "urgency": None, "pii": None, "drafting": None,
        }}))

    pin_adapters("aaa1")
    return tmp_path, pin_adapters


def test_the_first_manifest_promotes_because_there_is_nothing_to_regress_from(repo):
    """A regression is a comparison against a previous version, and v1 has none.

    The first version of blocking_reasons blocked v1 for having "moved" every component
    away from nothing — which reads as the gate working until you notice it can never be
    satisfied.
    """
    assert system.promote(note="baseline") == 0
    assert system.load_current()["version"] == 1


def test_moving_an_adapter_without_a_regression_run_is_blocked(repo):
    _, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")                       # a new adapter revision appears

    assert system.promote(note="new adapter") == 1, "an unmeasured model change promoted"
    assert system.load_current()["version"] == 1, "a blocked promotion still wrote"


def test_a_regression_run_unblocks_it_while_the_gate_is_report_only(repo):
    _, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")
    assert system.promote(note="measured", regression={"per_task": {}}) == 0
    assert system.load_current()["version"] == 2


def test_a_moved_eval_split_blocks_even_with_a_regression_run(repo):
    """A score measured against a different bar is not comparable to the previous one, so
    a regression number computed across the move does not license the promotion."""
    tmp, _ = repo
    system.promote(note="baseline")
    (tmp / "evals" / "golden" / "intent.parquet").write_bytes(b"golden-v2-MOVED")

    assert system.promote(note="moved bar", regression={"per_task": {}}) == 1
    reasons = system.blocking_reasons(system.build(), {"per_task": {}})
    assert any("eval splits moved" in r for r in reasons)


def test_force_promotes_but_records_what_it_overrode(repo):
    _, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")
    assert system.promote(note="forced", force=True) == 0
    assert system.load_current()["promoted_despite"], "forcing left no trace"


def test_rollback_restores_the_previous_version_and_keeps_the_one_it_replaced(repo):
    _, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")
    system.promote(note="v2", regression={"per_task": {}})
    assert system.load_current()["version"] == 2

    assert system.rollback() == 0
    current = system.load_current()
    assert current["version"] == 1
    assert current["rolled_back_from"] == 2
    assert (system.HISTORY / "system-0002.json").exists(), "rolled-back version was lost"


def test_the_gate_starts_report_only_and_invents_no_threshold(repo):
    """PRD §11 corrects v2.1 here: a threshold before the variance runs is the
    unfalsifiable gate that raising the golden sets existed to remove (F33)."""
    system.promote(note="baseline")
    gate = system.load_current()["gate"]
    assert gate["state"] == "report_only"
    assert gate["thresholds"] == {}


def test_an_enforcing_gate_blocks_a_drop_past_its_threshold(repo):
    _, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")
    candidate = system.build("regressed")
    candidate["gate"] = {"state": "enforcing", "why": "derived", "thresholds": {"intent": 0.02}}

    reasons = system.blocking_reasons(candidate, {"per_task": {"intent": {"drop": 0.05}}})
    assert any("exceeds threshold" in r for r in reasons)
    ok = system.blocking_reasons(candidate, {"per_task": {"intent": {"drop": 0.01}}})
    assert not any("exceeds threshold" in r for r in ok)


def test_a_derived_threshold_file_becomes_the_gate_a_new_manifest_carries(repo):
    """Until thresholds are derived the gate is report-only; once they are, promotion enforces
    them without anyone copying numbers into the manifest by hand."""
    tmp, _ = repo
    assert system.build()["gate"]["state"] == "report_only"
    (tmp / "evals" / "GATE_THRESHOLDS.json").write_text(json.dumps({
        "gate": {"state": "enforcing", "thresholds": {"intent": 0.0117},
                 "multiplier": 3.0, "why": "D37"}}))
    gate = system.build()["gate"]
    assert gate["state"] == "enforcing" and gate["thresholds"] == {"intent": 0.0117}
    assert gate["derivation"] == "evals/GATE_THRESHOLDS.json"


def test_m7_a_regressed_candidate_is_blocked_then_an_equivalent_one_promotes_and_rolls_back(repo):
    """M7 at the manifest level, on a candidate pin set: detect -> block -> rollback.

    The candidate swaps intent for different weights, as the F21 shuffled adapter will. With a
    derived threshold in force, a measured drop past it is refused; an equivalent candidate is
    promoted; and rollback restores the version before it.
    """
    tmp, _ = repo
    assert system.promote(note="baseline") == 0
    (tmp / "evals" / "GATE_THRESHOLDS.json").write_text(json.dumps({
        "gate": {"state": "enforcing", "thresholds": {"intent": 0.0117},
                 "multiplier": 3.0, "why": "D37"}}))

    candidate = tmp / "manifests" / "candidates" / "intent-shuffled.json"
    candidate.parent.mkdir(parents=True)
    components = json.loads((tmp / "manifests" / "adapters.json").read_text())["components"]
    components["intent"] = {"repo": "r-shuffled", "revision": "shuf1",
                            "weight_sha256": "shuf1", "candidate_of": "r"}
    candidate.write_text(json.dumps({"components": components}))

    regressed = {"per_task": {"intent": {"drop": 0.91}}}
    assert system.promote(note="shuffled", regression=regressed, pins=candidate) == 1
    assert system.load_current()["version"] == 1, "a blocked candidate was promoted"

    equivalent = {"per_task": {"intent": {"drop": 0.0}}}
    assert system.promote(note="equivalent", regression=equivalent, pins=candidate) == 0
    current = system.load_current()
    assert current["version"] == 2 and current["pins"].endswith("intent-shuffled.json")
    assert current["components"]["adapters"]["intent"]["revision"] == "shuf1"

    assert system.rollback() == 0
    assert system.load_current()["version"] == 1


def test_a_component_arriving_from_nothing_needs_no_regression_run(repo):
    """D39: v1 was pinned before the router existed. A component with no previous pin has
    nothing to regress from — the reasoning that exempts v1 itself."""
    tmp, _ = repo
    system.promote(note="baseline")
    components = json.loads((tmp / "manifests" / "adapters.json").read_text())["components"]
    components["urgency"] = {"repo": "u", "revision": "u1", "weight_sha256": "u1"}
    (tmp / "manifests" / "adapters.json").write_text(json.dumps({"components": components}))

    assert system.promote(note="urgency arrives") == 0
    assert system.load_current()["version"] == 2


def test_a_split_refrozen_citing_a_decision_promotes_and_is_recorded(repo):
    tmp, pin_adapters = repo
    system.promote(note="baseline")
    (tmp / "evals" / "golden" / "intent.parquet").write_bytes(b"golden-v2-refrozen")

    assert system.promote(note="re-freeze", refreeze="D39") == 0
    current = system.load_current()
    assert current["refrozen_splits"] == {"decision": "D39",
                                          "splits": ["eval_splits.golden_intent"]}
    assert "promoted_despite" not in current, "a re-freeze was recorded as a forced release"

    pin_adapters("bbb2")                       # the next candidate sees only its own change
    reasons = system.blocking_reasons(system.build(), {"per_task": {}})
    assert not any("eval splits moved" in r for r in reasons)


def test_a_refreeze_is_refused_when_a_model_moves_with_it(repo):
    tmp, pin_adapters = repo
    system.promote(note="baseline")
    (tmp / "evals" / "golden" / "intent.parquet").write_bytes(b"golden-v2-refrozen")
    pin_adapters("bbb2")

    assert system.promote(note="both", refreeze="D39", regression={"per_task": {}}) == 1
    assert system.load_current()["version"] == 1


def test_a_refreeze_with_no_moved_split_is_refused(repo):
    system.promote(note="baseline")
    assert system.promote(note="nothing moved", refreeze="D39") == 1
    assert system.load_current()["version"] == 1


def test_a_promotion_after_a_rollback_never_reuses_a_version_number(repo):
    """Numbered from the current manifest alone, the promotion after a rollback reused the
    rolled-back version's number, and archiving it later would overwrite that version's record."""
    _, pin_adapters = repo
    system.promote(note="v1")
    pin_adapters("bbb2")
    system.promote(note="v2", regression={"per_task": {}})
    system.rollback()                                  # current is v1 again; v2 is archived

    pin_adapters("ccc3")
    assert system.promote(note="after rollback", regression={"per_task": {}}) == 0
    assert system.load_current()["version"] == 3
    assert json.loads((system.HISTORY / "system-0002.json").read_text())["note"] == "v2"


def _router(tmp: Path, weights: bytes) -> None:
    d = tmp / "checkpoints" / "router__judged"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(weights)
    (d / "config.json").write_text("{}")


def test_a_moved_router_needs_its_evaluation_report_not_an_adapter_regression_run(repo):
    """D46: an adapter regression run cannot measure a router, so it cannot license moving one.
    The router's own evidence is its operating curve, and that is what has to be attached."""
    tmp, _ = repo
    _router(tmp, b"router-v1")
    system.promote(note="baseline")
    _router(tmp, b"router-v2")

    assert system.promote(note="router, no report", regression={"per_task": {}}) == 1
    reasons = system.blocking_reasons(system.build(), {"per_task": {}})
    assert any("router" in r and "evaluation report" in r for r in reasons)

    report = tmp / "runs" / "router__curve.json"
    report.parent.mkdir()
    report.write_text('{"populations": {}}')
    assert system.promote(note="router with its curve", evidence={"router": report}) == 0
    current = system.load_current()
    assert current["component_evidence"]["router"]["file"] == "runs/router__curve.json"
    assert "promoted_despite" not in current, "an evidenced move was recorded as a forced release"


def test_router_evidence_does_not_license_an_adapter_move(repo):
    tmp, pin_adapters = repo
    system.promote(note="baseline")
    pin_adapters("bbb2")
    report = tmp / "curve.json"
    report.write_text("{}")
    assert system.promote(note="adapter via router evidence", evidence={"router": report}) == 1


def test_evidence_that_is_not_on_disk_is_refused(repo):
    tmp, _ = repo
    _router(tmp, b"router-v1")
    system.promote(note="baseline")
    _router(tmp, b"router-v2")
    assert system.promote(note="missing report", evidence={"router": tmp / "nope.json"}) == 1
    assert system.load_current()["version"] == 1
