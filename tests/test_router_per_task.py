"""Guards on per-task thresholds: exact allocation, thresholds that follow the ranking, and the applied rule."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import per_task as pt


def test_allocation_is_exact_where_a_greedy_split_would_miss():
    # Task a pays nothing for its first pair and 5 for the second; a greedy step-by-step split
    # spends the budget on b's steady +1s. The exact search sees that a's pair is worth taking.
    gains = {"a": np.array([0, 0, 5]), "b": np.array([0, 1, 2, 3])}
    assert pt.allocate(gains, 2) == {"a": 2, "b": 0}
    assert pt.allocate(gains, 3) == {"a": 2, "b": 1}


def test_losing_tasks_get_no_escalations():
    gains = {"drafting": np.array([0, 1, 2]), "urgency": np.array([0, -1, -2])}
    assert pt.allocate(gains, 2) == {"drafting": 2, "urgency": 0}


def frame():
    return pd.DataFrame({
        "pair_id": ["d1", "d2", "d3", "u1", "u2"],
        "task": ["drafting"] * 3 + ["urgency"] * 2,
        "mean_logprob": [-0.9, -0.5, -0.1, -0.8, -0.2],
        "success": [False, False, True, True, True],
        "frontier_success": [True, True, True, False, False],
    })


def test_thresholds_follow_the_ranking_and_the_rule_applies_them():
    f = frame()
    thresholds = pt.thresholds_from(f, {"drafting": 2, "urgency": 0})
    assert thresholds["drafting"] == 0.5 and math.isinf(thresholds["urgency"])
    applied = pt.apply(f, thresholds)
    assert applied["escalation__drafting"] == round(2 / 3, 4) and applied["escalation__urgency"] == 0.0
    assert applied["quality"] == 1.0                         # both rescues taken, no harmful escalation
