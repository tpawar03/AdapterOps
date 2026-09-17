"""Guards on the mixed-generator judge's frozen splits and its comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import mixed


def frame():
    rows = []
    for i in range(40):  # router-study requests: adapter and mini replies, a few calibration items
        for gen in ("adapter", "gpt4o_mini"):
            rows.append({"item_id": f"r{i}{gen}", "generator": gen, "origin": "router_study",
                         "instruction": f"router request {i}", "reply": "r", "score": 4.0,
                         "calibration": gen == "adapter" and i < 5, "domain": "in"})
    for i in range(30):  # M2 golden: three generators per request
        for gen in ("adapter", "prompted_base", "gpt4o_mini"):
            rows.append({"item_id": f"m{i}{gen}", "generator": gen, "origin": "m2_golden",
                         "instruction": f"golden request {i}", "reply": "r", "score": 3.0,
                         "calibration": False, "domain": "in"})
    rows.append({"item_id": "o1", "generator": "adapter", "origin": "abcd", "instruction": "router request 7",
                 "reply": "r", "score": 2.0, "calibration": False, "domain": "ood"})
    return pd.DataFrame(rows)


def test_holdouts_are_disjoint_from_training_by_request():
    split = mixed.assign_splits(frame())
    mixed.check_disjoint(split)
    assert set(split.split) >= {"train", "validation", "h1_adapter_calibration", "h2_across_generators", "h3_out_of_domain"}


def test_a_request_held_out_anywhere_leaves_training_entirely():
    split = mixed.assign_splits(frame())
    # "router request 7" is an out-of-domain holdout, so its router-study replies cannot train.
    rows = split[split.instruction == "router request 7"]
    assert not rows.split.isin(["train", "validation"]).any()
    assert (rows.split == "dropped_shares_a_held_out_request").sum() == 2


def test_h2_holds_out_whole_requests_with_every_generator():
    split = mixed.assign_splits(frame())
    h2 = split[split.split == "h2_across_generators"]
    assert (h2.groupby("group").generator.nunique() == 3).all()
    assert h2.group.nunique() == round(30 * mixed.H2_FRACTION)


def test_check_disjoint_catches_a_leak():
    split = mixed.assign_splits(frame())
    leaked = split.copy()
    leaked.loc[leaked.split == "h2_across_generators", "group"] = leaked[leaked.split == "train"].group.iloc[0]
    with pytest.raises(RuntimeError, match="shares requests"):
        mixed.check_disjoint(leaked)


def test_paired_bootstrap_sees_a_better_ranker():
    rng = np.random.default_rng(0)
    gold = rng.integers(1, 6, 200).astype(float)
    served = rng.normal(size=200)
    better = gold + rng.normal(scale=0.5, size=200)
    r = mixed.paired_spearman_bootstrap(gold, served, better, resamples=300)
    assert r["mixed"] > r["served"] and r["ci95"][0] > 0


def test_version_two_refuses_training_replies_on_a_held_out_request(tmp_path, monkeypatch):
    from adapterops.judge import ood_train

    v1 = mixed.assign_splits(frame())
    held_instruction = v1[v1.split == "h2_across_generators"].instruction.iloc[0]
    graded = tmp_path / "graded.parquet"
    pd.DataFrame([{"item_id": "oodtrain:adapter:x", "generator": "adapter", "origin": "abcdtrain",
                   "instruction": held_instruction, "reply": "r", "score": 3.0, "calibration": False,
                   "domain": "ood_train"}]).to_parquet(graded)
    monkeypatch.setattr(mixed, "frames_frame", lambda version=1: v1)
    monkeypatch.setattr(ood_train, "GRADED", graded)
    with pytest.raises(ValueError, match="also held-out requests"):
        mixed.build_v2()


def test_version_two_paths_never_touch_version_one():
    one, two = mixed.paths(1), mixed.paths(2)
    assert all(one[k] != two[k] for k in ("data", "record", "out", "run", "tag"))
