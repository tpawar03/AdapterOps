"""Guards on the router's computed label and on the policy comparison it feeds.

Both modules run on a laptop and neither needs a GPU, which is the point of splitting
them off from generation: the success rule and the operating curve can be wrong and
corrected for free, while re-running the adapters cannot.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import baselines, scoring  # noqa: E402


# --- the success rule -------------------------------------------------------------

def test_classification_success_is_exact_match_on_the_first_line():
    """Adapters emit a label and sometimes keep talking; only the label is scored."""
    assert scoring.score_pair("intent", "t", "card_arrival", "card_arrival\nblah")["exact"] == 1.0
    assert scoring.score_pair("urgency", "t", "high", " high ")["exact"] == 1.0
    assert scoring.score_pair("urgency", "t", "high", "medium")["exact"] == 0.0


def test_pii_success_needs_every_span_and_invents_none():
    text = "Call Ana at ana@x.com"
    gold = "GIVENNAME: Ana\nEMAIL: ana@x.com"

    perfect = scoring.score_pair("pii", text, gold, gold)
    assert perfect["span_f1"] == 1.0

    missed = scoring.score_pair("pii", text, gold, "EMAIL: ana@x.com")
    assert missed["span_recall"] < 1.0

    invented = scoring.score_pair("pii", text, gold, gold + "\nCITY: Call")
    assert invented["span_recall"] == 1.0
    assert invented["span_precision"] < 1.0


def test_pii_label_rejects_a_document_that_is_merely_mostly_right():
    """One span missed is one document leaked — the rule is F1 = 1.0, not a threshold."""
    scored = pd.DataFrame([
        {"pair_id": "pii-0", "task": "pii", "span_f1": 1.0},
        {"pair_id": "pii-1", "task": "pii", "span_f1": 0.9333},
    ])
    assert scoring.label(scored).success.tolist() == [True, False]


def test_token_f1_is_bounded_and_order_free():
    assert scoring.token_f1("thanks for your patience", "thanks for your patience") == 1.0
    assert scoring.token_f1("zzz", "thanks for your patience") == 0.0
    a = scoring.token_f1("patience your for thanks", "thanks for your patience")
    assert a == 1.0, "a bag-of-tokens proxy must not reward word order it cannot see"


def test_drafting_label_is_marked_as_a_proxy_and_names_its_arbitrary_cut():
    """The weakest label in the project. It is allowed to be weak; it is not allowed to
    be silent about it — Phase 3 replaces it and measures the disagreement."""
    scored = pd.DataFrame({
        "pair_id": [f"d-{i}" for i in range(4)],
        "task": "drafting",
        "proxy_token_f1": [0.1, 0.3, 0.5, 0.7],
    })
    out = scoring.label(scored)
    assert out.label_is_proxy.all()
    assert out.success.tolist() == [False, False, True, True]
    assert "PROXY" in out.label_rule.iloc[0] and "arbitrary" in out.label_rule.iloc[0]


def test_only_drafting_carries_a_proxy_label():
    scored = pd.DataFrame({
        "pair_id": ["i-0", "d-0"], "task": ["intent", "drafting"],
        "exact": [1.0, float("nan")], "proxy_token_f1": [float("nan"), 0.5],
    })
    assert scoring.label(scored).label_is_proxy.tolist() == [False, True]


# --- the operating curve ----------------------------------------------------------

def synthetic(n: int = 100, failures: int = 30) -> pd.DataFrame:
    """Adapter fails on `failures` pairs, and is least confident on exactly those —
    a confidence signal that is perfect, so a broken curve cannot hide behind noise."""
    success = [False] * failures + [True] * (n - failures)
    return pd.DataFrame({
        "pair_id": [f"p-{i:03d}" for i in range(n)],
        "task": ["intent"] * (n // 2) + ["pii"] * (n - n // 2),
        "success": success,
        "mean_logprob": [-5.0] * failures + [-0.1] * (n - failures),
        "frontier_success": [True] * n,
    })


def test_budget_zero_and_one_are_the_two_F8_baselines():
    df = synthetic()
    curve = baselines.operating_curve(df, "random")
    assert curve.loc[curve.budget == 0.0, "quality"].iloc[0] == pytest.approx(0.70)
    assert curve.loc[curve.budget == 1.0, "quality"].iloc[0] == pytest.approx(1.00)


def test_oracle_bounds_every_other_policy_at_every_budget():
    df = synthetic()
    oracle = baselines.operating_curve(df, "oracle").set_index("budget").quality
    for policy in ("random", "confidence"):
        other = baselines.operating_curve(df, policy).set_index("budget").quality
        assert (other <= oracle + 1e-9).all(), f"{policy} beat the oracle — the curve is wrong"


def test_a_perfect_confidence_signal_matches_the_oracle():
    df = synthetic()
    conf = baselines.operating_curve(df, "confidence").set_index("budget").quality
    oracle = baselines.operating_curve(df, "oracle").set_index("budget").quality
    assert conf.loc[0.30] == pytest.approx(oracle.loc[0.30])
    assert conf.loc[0.30] == pytest.approx(1.0)


def test_random_gains_roughly_its_budget_and_no_more():
    """The no-information floor: escalating b% of a 30%-failure pool recovers ~0.3b."""
    df = synthetic(n=1000, failures=300)
    curve = baselines.operating_curve(df, "random").set_index("budget").quality
    assert curve.loc[0.20] == pytest.approx(0.70 + 0.30 * 0.20, abs=0.03)


def test_escalation_quality_must_come_from_somewhere():
    df = synthetic().drop(columns=["frontier_success"])
    with pytest.raises(ValueError, match="flatters every escalating policy"):
        baselines.operating_curve(df, "confidence")
    assumed = baselines.operating_curve(df, "confidence", assume_frontier_success=1.0)
    assert (assumed.assumed_frontier_success == 1.0).all(), "the assumption must be recorded"


def test_curve_is_identical_across_runs_even_when_scores_tie():
    """oracle scores are 0 or 1 with massive ties; without the pair_id tie-break the
    curve would move whenever the pool was re-ordered."""
    df = synthetic()
    first = baselines.operating_curve(df, "oracle")
    shuffled = df.sample(frac=1.0, random_state=7)
    second = baselines.operating_curve(shuffled, "oracle")
    pd.testing.assert_frame_equal(first, second)


def test_headroom_captured_reports_share_of_the_available_gain():
    df = synthetic()
    conf = baselines.operating_curve(df, "confidence")
    oracle = baselines.operating_curve(df, "oracle")
    assert baselines.headroom_captured(conf, oracle, 0.30) == 1.0
    rand = baselines.operating_curve(df, "random")
    assert 0.0 <= baselines.headroom_captured(rand, oracle, 0.30) < 1.0


def test_compare_puts_every_policy_on_one_frame_with_per_task_columns():
    out = baselines.compare(synthetic())
    assert set(out.policy) == {"random", "confidence", "oracle"}
    assert {"quality__intent", "quality__pii",
            "escalation__intent", "escalation__pii"} <= set(out.columns)


# --- the CPU half of the generation run -------------------------------------------

def pool_sample(per_task: int = 4) -> pd.DataFrame:
    pool = pd.read_parquet(ROOT / "data/router/pool.parquet")
    return pool.groupby("task", group_keys=False).head(per_task).reset_index(drop=True)


def test_a_perfect_adapter_scores_one_on_every_task():
    """Exercises the real gold of all four tasks through the real scorers, no GPU.

    Worth its runtime because PII's gold is span offsets recovered from a `LABEL: value`
    string — if that round-trip were lossy, every PII success label would be wrong and the
    only symptom would be a suspiciously low success rate on the GPU run.
    """
    from adapterops.router.generate import score_frame

    frame = pool_sample().assign(prediction=lambda d: d.gold, error=None)
    scored = scoring.label(score_frame(frame))
    assert scored.success.all(), scored.loc[~scored.success, ["task", "gold"]]


def test_a_failed_request_scores_as_a_failure_rather_than_vanishing():
    """A pair local inference could not answer is exactly a pair that should have been
    escalated. Dropping it would also hide the fallback rate PRD §11 monitors."""
    from adapterops.router.generate import score_frame

    frame = pool_sample(2).assign(prediction="", error="ReadTimeout")
    scored = scoring.label(score_frame(frame))
    assert not scored.success.any()
    assert len(scored) == len(frame)


def test_requests_are_interleaved_across_adapters_not_batched_by_task():
    """Task-by-task order would keep one adapter resident and measure a single-adapter
    server — then report the number as concurrent multi-LoRA latency."""
    pool = pool_sample(3)
    order = pool.sort_values("pair_id", key=lambda s: s.str.slice(-4)).task.tolist()
    assert len(set(order[:4])) == 4, f"first four requests hit {set(order[:4])}"


def test_router_weights_load_in_fp32_because_fp16_adamw_returns_nan():
    """deberta-v3-small ships fp16 weights and transformers 5 keeps a checkpoint's dtype.

    Training that with AdamW diverges on the first step — finite gradients, but
    `exp_avg_sq = (1-b2)*g**2` underflows fp16 and the division back out overflows it, so
    most parameters come back non-finite and the run completes reporting `nan`. This
    asserts the upstream default rather than the workaround: if a future release upcasts
    again, the guard in train.py becomes unnecessary and this test says so.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    from adapterops.router.train import RouterConfig

    shipped = AutoModelForSequenceClassification.from_pretrained(
        RouterConfig.base_model, num_labels=2)
    dtypes = {p.dtype for p in shipped.parameters()}
    assert dtypes == {torch.float16}, (
        f"deberta-v3-small no longer loads as fp16 ({dtypes}) — re-check whether the "
        f"explicit dtype=float32 in router/train.py is still needed")
