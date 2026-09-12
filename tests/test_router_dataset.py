"""Guards on the F31 hard-cases exclusion (D7, D29).

Synthetic scored data, because the real thing needs a GPU — and because synthetic data can
be given a known failure rate, which is what makes the two counter-example tests possible.
Those two are the point of this file: both failed designs produced a dataset that looked
entirely normal, and the only symptom of either was a number that was too good.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import dataset

TASKS = ("intent", "urgency", "pii", "drafting")
FAILURE_RATE = 0.22


def synthetic(router_per_task: int = 750, mining_per_task: int = 700,
              seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, sides = [], []
    for task in TASKS:
        for purpose, n in (("router", router_per_task), ("mining", mining_per_task)):
            for i in range(n):
                pair_id = f"{task}-{purpose[0]}{i:04d}"
                success = rng.random() > FAILURE_RATE
                rows.append({
                    "pair_id": pair_id, "task": task, "purpose": purpose,
                    "success": success, "span_precision": 0.8, "span_recall": 0.9,
                    "span_f1": 1.0 if success else 0.85, "span_f1_relaxed": 0.9,
                    "pred_spans": 4, "mean_logprob": -0.5,
                })
                if purpose == "router":
                    sides.append({"pair_id": pair_id, "task": task,
                                  "side": "shift" if i % 3 == 0 else "in_distribution"})
    return pd.DataFrame(rows), pd.DataFrame(sides)


def test_no_split_shares_a_pair_with_any_other():
    """BUILD-PLAN Phase 2 calls this the most important line in the phase, so it is
    asserted rather than reasoned about."""
    splits, _ = dataset.build(*synthetic())
    assert set(dataset.verify(splits).values()) == {0}


def test_hard_candidates_come_from_the_mining_slice_alone():
    splits, _ = dataset.build(*synthetic())
    hard = splits["hard_candidates"]
    assert len(hard), "nothing mined — the exclusion would be vacuous"
    assert (hard.purpose == "mining").all()
    assert not hard.success.any(), "a success is not a hard case"
    for other in ("train", "eval", "shift_eval"):
        assert not set(hard.pair_id) & set(splits[other].pair_id)


def test_the_router_still_has_failures_to_learn_from():
    """Attempt 2's failure, and the reason the mining slice exists (D29).

    Reserving hard cases out of the training portion took every failure with it and left
    the router training at a 1.00 success rate — a dataset with one class in it, which
    trains without error and predicts a constant.
    """
    splits, _ = dataset.build(*synthetic())
    train = splits["train"]
    assert not train.success.all(), "router training has no failures — nothing to learn"
    assert train.success.mean() == pytest.approx(1 - FAILURE_RATE, abs=0.04)
    for task in TASKS:
        assert not train.query("task == @task").success.all(), task


def test_the_eval_set_keeps_its_failures_too():
    """Attempt 1's failure: mining from everything and holding it out of both router
    splits stripped the eval set of nearly all the failures the curve exists to escalate.
    """
    splits, _ = dataset.build(*synthetic())
    for name in ("eval", "shift_eval"):
        held = splits[name]
        assert (1 - held.success.mean()) == pytest.approx(FAILURE_RATE, abs=0.04), name


def test_train_and_eval_carry_the_same_base_rate():
    """A policy comparison read against a different base rate is not the same problem."""
    splits, _ = dataset.build(*synthetic())
    for task in TASKS:
        train = splits["train"].query("task == @task").success.mean()
        held = splits["eval"].query("task == @task").success.mean()
        assert abs(train - held) < 0.06, task


def test_hard_candidates_are_spread_across_failure_buckets():
    """An uncapped mine fills the split with whichever failure mode is most common and
    calls the result a hard-cases set."""
    _, stats = dataset.build(*synthetic())
    pii = stats["hard_candidates"]["pii"]
    assert pii["reserved"] <= pii["cap"]
    assert sum(pii["by_bucket"].values()) == pii["reserved"]


def test_an_unbound_cap_is_recorded_as_a_measurement_not_a_shortfall():
    """150 hard cases per task needs ~150 failures per task. An adapter scoring 0.93 does
    not produce them from 700 pairs, and the honest output is the number plus the reason.
    """
    _, stats = dataset.build(*synthetic(mining_per_task=200))
    for task, report in stats["hard_candidates"].items():
        assert report["reserved"] <= min(report["cap"], report["failures_in_mining_slice"])
        assert report["cap_bound"] is False, task
        assert report["failure_rate"] == pytest.approx(FAILURE_RATE, abs=0.06)
