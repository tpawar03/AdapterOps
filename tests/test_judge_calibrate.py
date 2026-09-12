"""Guards on judge calibration (F15) and the two comparisons built on it.

Two tests document a limit rather than a feature: correlation cannot see a consistent
offset, and raw agreement on a 50/50 label is half chance. Both are why the module reports
more than the headline number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import calibrate as cal


def test_a_perfect_student_scores_one_everywhere():
    t = [1, 2, 3, 4, 5, 3, 2, 4]
    out = cal.agreement(t, t)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pearson"] == pytest.approx(1.0)
    assert out["exact_agreement"] == 1.0
    assert out["within_1_agreement"] == 1.0
    assert out["mean_absolute_error"] == 0.0


def test_correlation_is_blind_to_a_consistent_offset_so_bias_is_reported():
    """A student running a full point generous ranks perfectly — 1.0 on both correlations —
    and agrees exactly on nothing. Only the bias and exact-agreement figures show it."""
    teacher = [1, 2, 3, 4, 2, 3, 1, 4]
    student = [x + 1 for x in teacher]
    out = cal.agreement(teacher, student)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pearson"] == pytest.approx(1.0)
    assert out["exact_agreement"] == 0.0
    assert out["within_1_agreement"] == 1.0
    assert out["mean_bias_student_minus_teacher"] == pytest.approx(1.0)


def test_a_continuous_student_is_rounded_for_agreement_only():
    out = cal.agreement([3, 4, 2], [3.4, 3.6, 2.2])
    assert out["exact_agreement"] == 1.0
    assert out["mean_absolute_error"] == pytest.approx((0.4 + 0.4 + 0.2) / 3, abs=1e-4)


def test_a_constant_score_reports_undefined_correlation_rather_than_crashing():
    out = cal.agreement([3, 3, 3, 3], [2, 3, 4, 5])
    assert out["spearman"] is None and out["pearson"] is None
    assert "undefined" in out["note"]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="teacher scores but"):
        cal.agreement([1, 2, 3], [1, 2])


def test_a_proxy_that_orders_drafts_like_the_judge_agrees_fully():
    judge = [1, 2, 3, 4, 5, 5, 4, 2]
    proxy = [0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.25]
    out = cal.proxy_vs_judge(proxy, judge, proxy_cut=0.5, judge_success_min=4)
    assert out["spearman"] > 0.9
    assert out["binary_agreement"] == 1.0
    assert out["cohen_kappa"] == pytest.approx(1.0)


def test_raw_agreement_on_a_balanced_label_is_half_chance_so_kappa_is_reported():
    """Two unrelated 50/50 labels agree about half the time. Reporting that 0.5 as
    'the proxy agrees with the judge on half the drafts' would sound like a finding."""
    rng = np.random.default_rng(0)
    proxy = rng.random(4000)
    judge = rng.integers(1, 6, 4000)
    out = cal.proxy_vs_judge(proxy, judge, proxy_cut=np.median(proxy), judge_success_min=4)
    assert 0.40 < out["binary_agreement"] < 0.60
    assert abs(out["cohen_kappa"]) < 0.05


def test_the_proxy_rule_is_reproduced_including_its_zero_precondition():
    out = cal.proxy_vs_judge([0.0, 0.0, 0.9, 0.9], [1, 1, 5, 5], proxy_cut=0.0)
    assert out["proxy_success_rate"] == 0.5, "zero token overlap must never count as success"


def test_same_family_check_reports_the_gap_and_its_caveat():
    ids = [f"drafting-r{i:04d}" for i in range(40)]
    local = pd.Series([3] * 40, index=ids, dtype=float)
    frontier = pd.Series([4] * 30 + [3] * 10, index=ids, dtype=float)
    out = cal.same_family_check(local, frontier)
    assert out["mean_difference_frontier_minus_local"] == pytest.approx(0.75)
    assert out["frontier_higher"] == pytest.approx(0.75)
    assert out["tied"] == pytest.approx(0.25)
    assert "cannot be separated" in out["caveat"]


def test_same_family_check_declines_with_too_few_pairs():
    ids = ["a", "b", "c"]
    out = cal.same_family_check(pd.Series([3.0] * 3, index=ids), pd.Series([4.0] * 3, index=ids))
    assert "too few" in out["note"]
