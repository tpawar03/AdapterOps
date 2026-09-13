"""Guards on router v2's evaluation (D42).

The first test is the one the bootstrap rests on: its fast quality function must agree with the
operating curve's, ties included, or every interval it produces describes a different policy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import cascade as cc
from adapterops.router.baselines import evaluate_at_budget
from adapterops.router.train import positive


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pair_id": [f"p{i:03d}" for i in rng.permutation(n)],
        "task": rng.choice(list(cc.TASKS), n),
        "success": rng.random(n) < 0.7,
        "frontier_success": rng.random(n) < 0.5,
        "mean_logprob": -rng.random(n),
        "min_logprob": -2 * rng.random(n),
        "n_tokens": rng.integers(1, 200, n),
        "finish_reason": rng.choice(["stop", "length"], n),
    })


@pytest.mark.parametrize("budget", [0.0, 0.05, 0.2, 0.5, 1.0])
def test_fast_quality_matches_the_operating_curve_including_ties(budget):
    f = frame()
    scores = pd.Series(np.round(np.random.default_rng(1).random(len(f)), 1))  # many ties
    expected = evaluate_at_budget(f, scores, budget)["quality"]
    got = cc.quality_at_budget(f.success.to_numpy(float), f.frontier_success.to_numpy(float),
                               scores.to_numpy(), f.pair_id.rank(method="dense").to_numpy(),
                               budget)
    assert got == pytest.approx(expected, abs=1e-4)


def test_identical_policies_give_a_zero_interval():
    f = frame()
    s = -f.mean_logprob.to_numpy()
    r = cc.paired_bootstrap(f, s, s, 0.2, resamples=50, seed=0)
    assert r["difference"] == 0 and r["ci95"] == [0.0, 0.0]


def test_per_task_confidence_uses_training_statistics_only():
    train, other = frame(seed=2), frame(seed=3)
    before = cc.confidence_per_task(train, other)
    shifted = other.assign(mean_logprob=other.mean_logprob - 5)
    after = cc.confidence_per_task(train, shifted)
    # were eval statistics used, a uniform shift would cancel out
    assert (after > before).all()


def test_design_gives_each_task_its_own_slopes():
    f = frame()
    x = cc.design(f)
    intent = f.task == "intent"
    assert (x.loc[~intent, "intent__mean_logprob"] == 0).all()
    assert (x.loc[intent, "intent__mean_logprob"] == f.loc[intent, "mean_logprob"]).all()


def test_design_refuses_missing_signals():
    f = frame()
    f.loc[0, "min_logprob"] = np.nan
    with pytest.raises(ValueError, match="missing generation signals"):
        cc.design(f)


def test_a_win_needs_the_interval_above_zero_and_shift_to_agree():
    assert cc.decide({"ci95": [0.01, 0.05]}, {"difference": 0.01}) == "beats confidence"
    assert cc.decide({"ci95": [0.01, 0.05]}, {"difference": -0.01}) == "inconclusive"
    assert cc.decide({"ci95": [-0.01, 0.05]}, {"difference": 0.03}) == "inconclusive"
    assert cc.decide({"ci95": [-0.05, -0.01]}, {"difference": 0.03}) == "loses to confidence"


def test_seeds_that_disagree_are_inconclusive():
    assert cc.combine(["beats confidence"] * 3) == "beats confidence"
    assert cc.combine(["beats confidence", "inconclusive", "beats confidence"]) == "inconclusive"


def test_rescue_needs_the_frontier_to_succeed_where_local_failed():
    joined = pd.DataFrame({"pair_id": ["a", "b", "c"], "success": [False, False, True],
                           "frontier_success": [True, False, True]})
    split = joined[["pair_id", "success"]]
    out = cc.with_rescue(split, joined)
    assert out.rescue.tolist() == [True, False, False]


def test_disagreeing_success_labels_are_refused():
    joined = pd.DataFrame({"pair_id": ["a"], "success": [True], "frontier_success": [True]})
    with pytest.raises(ValueError, match="different rules"):
        cc.with_rescue(pd.DataFrame({"pair_id": ["a"], "success": [False]}), joined)


def test_router_targets():
    f = pd.DataFrame({"success": [True, False], "rescue": [False, True]})
    assert positive(f, "failure").tolist() == [0, 1]
    assert positive(f, "rescue").tolist() == [0, 1]
    with pytest.raises(KeyError, match="__rescue"):
        positive(f.drop(columns="rescue"), "rescue")
