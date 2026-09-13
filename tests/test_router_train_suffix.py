"""D38 retrains the router on judge-labelled splits written beside the frozen ones."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import train as rt


def test_the_data_suffix_selects_which_splits_are_read(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="router_train__judged"):
        rt.train(rt.RouterConfig(data_suffix="__judged"))


def test_the_default_still_reads_the_frozen_splits():
    assert rt.RouterConfig().data_suffix == ""
