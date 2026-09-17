"""Guards on the input-side out-of-domain check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import ood_detect as od

TRAIN = [f"my card payment {i} was declined at the shop" for i in range(20)] + \
        [f"how do i top up my account with {i} euros" for i in range(20)]


def test_similar_text_scores_higher_than_unrelated_text():
    sim = od.Similarity(TRAIN, k=3)
    near, far = sim.score(["my card payment was declined", "the sofa delivery is three weeks late"])
    assert near > far


def test_thresholds_come_from_the_in_domain_quantiles():
    limits = od.thresholds(np.linspace(0, 1, 101), np.arange(1, 102, dtype=float))
    assert abs(limits["similarity_below"] - 0.05) < 1e-9 and abs(limits["length_above"] - 96.0) < 1e-9


def test_usefulness_compares_flag_rates_on_wrong_and_right_answers():
    flagged = pd.Series([True, True, False, False], index=list("abcd"))
    wrong = pd.Series([True, False, True, False], index=list("abcd"))
    r = od.usefulness(flagged, wrong)
    assert r == {"n": 4, "wrong": 2, "flagged_when_wrong": 0.5, "flagged_when_right": 0.5}


def test_usefulness_ignores_tickets_without_a_verdict():
    r = od.usefulness(pd.Series([True, False], index=["a", "b"]), pd.Series([True], index=["a"]))
    assert r["n"] == 1 and r["flagged_when_wrong"] == 1.0
