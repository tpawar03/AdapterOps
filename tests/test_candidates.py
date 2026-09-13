"""Guards on serving a candidate adapter set (M7, M11) without contaminating the real one."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.manifest import registry
from adapterops.serve import launch

BASE = {
    "base_model": {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "b0"},
    "intent": {"repo": "Tanny03/adapterops-intent", "revision": "a1", "weight_sha256": "w1"},
    "pii": {"repo": "Tanny03/adapterops-pii", "revision": "p1", "weight_sha256": "wp"},
}


def test_a_candidate_replaces_one_adapter_and_keeps_every_other_pin():
    resolved = {"repo": "Tanny03/adapterops-intent-shuffled", "revision": "s1",
                "weight_sha256": "ws"}
    out = registry.candidate_components(BASE, "intent", resolved)
    assert out["intent"]["revision"] == "s1"
    assert out["intent"]["candidate_of"] == "Tanny03/adapterops-intent"
    assert out["pii"] == BASE["pii"] and out["base_model"] == BASE["base_model"]
    assert BASE["intent"]["revision"] == "a1", "building a candidate mutated the real pins"


def test_the_base_model_is_not_a_swappable_candidate():
    with pytest.raises(ValueError, match="not a swappable adapter"):
        registry.candidate_components(BASE, "base_model", {"repo": "x", "revision": "y"})


def test_two_pin_sets_never_share_a_download_directory():
    """Both define `intent`. Sharing a directory, the second download overwrites the first and
    vLLM serves whichever weights landed last, under a name that does not say which."""
    real = launch.adapter_dir(None, "intent")
    candidate = launch.adapter_dir(Path("manifests/candidates/intent-m11.json"), "intent")
    assert real != candidate
    assert real.parent.name == "baseline" and candidate.parent.name == "intent-m11"


def test_the_serve_script_passes_a_candidate_pin_set_through():
    script = (ROOT / "scripts" / "phase2_serve.sh").read_text()
    assert '${PINS:+--pins "$PINS"}' in script
    assert "PIN_SET" in script



def _workspace(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(registry, "MANIFEST", tmp_path / "adapters.json")
    monkeypatch.setattr(registry, "CANDIDATES_DIR", tmp_path / "candidates")
    (tmp_path / "adapters.json").write_text(json.dumps({"components": BASE}))
    (tmp_path / "candidates").mkdir()

    def add(name, task, revision, base=BASE):
        comps = registry.candidate_components(
            base, task, {"repo": f"x/{task}-m11", "revision": revision, "weight_sha256": revision})
        (tmp_path / "candidates" / f"{name}.json").write_text(
            json.dumps({"replaces": task, "components": comps}))
    return add


def test_combining_candidates_swaps_each_task_and_keeps_the_rest(tmp_path, monkeypatch):
    import json

    add = _workspace(tmp_path, monkeypatch)
    add("intent-m11", "intent", "m1")
    add("pii-m11", "pii", "m2")
    assert registry.combine_candidates(["intent-m11", "pii-m11"], "all-m11") == 0
    out = json.loads((tmp_path / "candidates" / "all-m11.json").read_text())
    assert out["components"]["intent"]["revision"] == "m1"
    assert out["components"]["pii"]["revision"] == "m2"
    assert out["components"]["base_model"] == BASE["base_model"]
    assert sorted(out["replaces"]) == ["intent", "pii"]


def test_two_candidates_replacing_the_same_task_cannot_combine(tmp_path, monkeypatch):
    add = _workspace(tmp_path, monkeypatch)
    add("a", "intent", "m1")
    add("b", "intent", "m2")
    with pytest.raises(ValueError, match="more than one candidate"):
        registry.combine_candidates(["a", "b"], "clash")


def test_a_candidate_built_against_different_pins_cannot_combine(tmp_path, monkeypatch):
    """Merging it would silently serve a base the individual pin never recorded."""
    import copy

    add = _workspace(tmp_path, monkeypatch)
    drifted = copy.deepcopy(BASE)
    drifted["pii"]["revision"] = "moved"
    add("stale", "intent", "m1", base=drifted)
    with pytest.raises(ValueError, match="different pins"):
        registry.combine_candidates(["stale"], "bad")
