"""Guards on the hard split mined from failures of systems the gate never compares."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import shared_hard as sh


def test_an_item_is_hard_only_when_every_source_fails_it():
    assert sh.mine({"a": [False, False, True, True], "b": [False, True, False, True]}) == [0]


def test_no_compared_model_is_a_source():
    sources = {s for v in sh.SOURCES.values() for s in v}
    assert not sources & set(sh.COMPARED)


def test_successes_follow_each_tasks_rule():
    frame = pd.DataFrame({"text": ["Dana wrote", "x"], "gold": ["GIVENNAME: Dana", "high"],
                          "prediction": ["GIVENNAME: Dana", "high\nextra"]})
    assert sh.successes("pii", frame.iloc[:1]) == [True]
    assert sh.successes("urgency", frame.iloc[1:]) == [True]
    assert sh.successes("drafting", frame, [4.0, 3.9]) == [True, False]


def test_a_frozen_split_is_reused_and_a_different_rebuild_refused(tmp_path):
    path = tmp_path / "split.parquet"
    split = pd.DataFrame({"task": ["intent"], "text": ["t"], "gold": ["g"], "source_row": [3], "sources": ["s"]})
    sh.freeze(split, path)
    assert sh.freeze(split, path).equals(split)
    with pytest.raises(ValueError, match="frozen"):
        sh.freeze(split.assign(source_row=[4]), path)
