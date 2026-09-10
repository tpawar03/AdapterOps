"""Guards on the two properties that are expensive to get wrong."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals" / "SPLITS.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="splits not frozen yet")


def manifest():
    return json.loads(MANIFEST.read_text())


def test_intent_golden_is_exactly_ten_per_class():
    """PRD §11: 770 = 10 per class over 77 classes. Anything else invalidates D10."""
    entry = next(t for t in manifest()["tasks"] if t["task"] == "intent")
    assert entry["golden_per_class"] == {"min": 10, "max": 10, "classes": 77}
    assert entry["splits"]["golden"]["rows"] == 770


def test_golden_never_overlaps_train_or_val():
    """A golden row appearing in training makes every later comparison meaningless."""
    for entry in manifest()["tasks"]:
        text_col = "text" if entry["task"] == "intent" else "instruction"
        frames = {n: pd.read_parquet(ROOT / s["file"]) for n, s in entry["splits"].items()}
        golden = set(frames["golden"][text_col])
        for other in ("train", "val"):
            overlap = golden & set(frames[other][text_col])
            assert not overlap, f"{entry['task']}: {len(overlap)} golden rows leaked into {other}"


def test_frozen_splits_refuse_to_be_overwritten():
    """`splits` must not silently move the bar."""
    r = subprocess.run(["uv", "run", "adapterops", "splits"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 1
    assert "frozen" in r.stdout
