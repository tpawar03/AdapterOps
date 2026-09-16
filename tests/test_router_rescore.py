"""Guards on re-scoring routing pairs with a replaced adapter: the score's token span, and the summary."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import rescore as rs


def test_the_mean_stops_at_the_first_end_of_text_and_ignores_padding():
    torch = pytest.importorskip("torch")
    eos, pad = 151645, 151643
    generated = torch.tensor([[11, 12, eos, pad, pad],     # finished early, then padded
                              [21, 22, 23, 24, 25]])       # hit the cap, no end-of-text
    scores = torch.tensor([[-0.1, -0.3, -0.2, -9.0, -9.0],
                           [-1.0, -1.0, -1.0, -1.0, -2.0]])
    (first, n1), (second, n2) = rs.mean_logprobs(scores, generated, [eos, pad])
    assert n1 == 3 and first == pytest.approx(-0.2)          # the -9.0 padding never enters the mean
    assert n2 == 5 and second == pytest.approx(-1.2)


def test_summary_counts_escalations_agreement_and_the_population_effect():
    population = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3", "i1", "i2"],
        "recorded_mean_logprob": [-0.01, -0.02, -0.03, -0.9, -0.1],
    })
    task_rows = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3"],
        "recorded_mean_logprob": [-0.01, -0.02, -0.03],
        "old_mean_logprob": [-0.011, -0.021, -0.031],
        "new_mean_logprob": [-0.5, -0.36, -0.02],            # one past 0.4, one just under it
        "recorded_local_success": [True, False, True],
        "recorded_frontier_success": [True, True, False],
        "old_success": pd.Series([True, False, True]),
        "new_success": pd.Series([False, True, True]),
    })
    s = rs.summarise(population, task_rows, threshold=0.4, budget_escalations=1)
    assert s["recorded_vllm_old_adapter"]["escalated_at_threshold"] == 0
    assert s["laptop_new_adapter"]["escalated_at_threshold"] == 1
    assert s["laptop_new_adapter"]["within_near_below_threshold"] == 1
    assert s["laptop_old_vs_recorded"]["same_decision"] == 1.0
    assert s["new_vs_old_same_decision"] == pytest.approx(2 / 3, abs=1e-4)
    # p1 escalates under the new scores and GPT-4o-mini got it right; p2 and p3 stay local.
    assert s["laptop_new_adapter"]["quality_at_threshold"] == pytest.approx(1.0)
    assert s["population_escalations"] == {"budget_20pct": 1, "recorded": 1, "with_new_adapter_scores": 2}
