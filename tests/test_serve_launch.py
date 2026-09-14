"""M6 — editing the system manifest changes what is served.

The PRD's measure was a manual test. This is the automated version: the launcher resolves adapter
revisions from `manifests/system.json`, so a promotion or a rollback — the only things that write
it — changes the next launch.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import launch


def write_manifest(path: Path, intent_revision: str) -> None:
    path.write_text(json.dumps({"version": 1, "components": {
        "base_model": {"repo": "Qwen/base", "revision": "b0"},
        "adapters": {"intent": {"repo": "u/intent", "revision": intent_revision},
                     "urgency": None, "pii": None, "drafting": None},
        "router": None, "judge": None}}))


def fake_materialise(pins=None, tasks=None):
    return {task: Path("/adapters") / task / entry["revision"]
            for task, entry in launch.load_components(pins).items()
            if task != "base_model" and entry}


def test_editing_the_system_manifest_changes_what_is_served(tmp_path, monkeypatch, capsys):
    system = tmp_path / "system.json"
    monkeypatch.setattr(launch, "SYSTEM", system)
    monkeypatch.setattr(launch, "materialise", fake_materialise)

    write_manifest(system, "aaa1")
    assert launch.main(["--print-args"]) == 0
    first = capsys.readouterr().out
    assert "intent=/adapters/intent/aaa1" in first and 'BASE_REVISION="b0"' in first

    write_manifest(system, "bbb2")                      # what a promotion or rollback does
    launch.main(["--print-args"])
    second = capsys.readouterr().out
    assert "intent=/adapters/intent/bbb2" in second
    assert "system.json" in second


def test_a_pin_can_name_a_locally_trained_adapter_by_path(tmp_path, monkeypatch):
    import huggingface_hub

    downloaded = []

    def fake_download(**kw):
        downloaded.append(kw["repo_id"])
        return str(tmp_path / kw["repo_id"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(launch, "ADAPTER_DIR", tmp_path / "_adapters")
    pins = tmp_path / "rerun.json"
    pins.write_text(json.dumps({"components": {
        "base_model": {"repo": "Qwen/base", "revision": "b0"},
        "intent": {"repo": "u/intent", "revision": "i1"},
        "urgency": {"path": "checkpoints/urgency-rerun"}}}))
    paths = launch.materialise(pins)
    assert paths["urgency"] == launch.REPO_ROOT / "checkpoints" / "urgency-rerun"
    assert downloaded == ["u/intent"], "a local checkpoint must never be fetched from the Hub"


def test_a_candidate_pin_set_still_overrides_the_manifest(tmp_path, monkeypatch):
    system, candidate = tmp_path / "system.json", tmp_path / "intent-shuffled.json"
    monkeypatch.setattr(launch, "SYSTEM", system)
    write_manifest(system, "aaa1")
    candidate.write_text(json.dumps({"components": {
        "base_model": {"repo": "Qwen/base", "revision": "b0"},
        "intent": {"repo": "u/intent-shuffled", "revision": "shuf1"}}}))
    assert launch.load_components(candidate)["intent"]["revision"] == "shuf1"
    assert launch.load_components()["intent"]["revision"] == "aaa1"
