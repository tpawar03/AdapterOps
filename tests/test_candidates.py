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
