"""Guards on drafting's M2 grading: paired sides, and the Phase 3 grades left alone."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import label, m2


def test_both_sides_are_paired_on_the_same_golden_rows_under_their_own_ids():
    items = m2.plan_items()
    assert items.groupby("source").size().to_dict() == {"adapter": 300, "prompted": 300}
    assert items.item_id.is_unique
    assert not items.item_id.str.startswith(("local:", "frontier:")).any()
    wide = items.pivot(index="pair", columns="source", values="instruction")
    assert (wide.adapter == wide.prompted).all()


def test_a_grading_run_never_rewrites_the_phase_3_judgments_file(tmp_path, monkeypatch):
    """judge-label rewrites judgments.parquet from its own plan; this must not go near it."""
    monkeypatch.setattr(label, "JUDGE_DIR", tmp_path)
    monkeypatch.setattr(label, "CACHE_FILE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(label, "LOCK_FILE", tmp_path / "judge.lock")
    monkeypatch.setattr(label, "OUT_FILE", tmp_path / "judgments.parquet")
    monkeypatch.setattr(m2, "OUT_FILE", tmp_path / "m2_golden.parquet")
    monkeypatch.setattr(m2, "SUMMARY_FILE", tmp_path / "summary.json")
    monkeypatch.setenv("OPENAI_API_KEY", "unused")

    def fake_batch(items, ledger, workers):
        rows = [{"item_id": r.item_id, "raw": "{}", "score": 3 if r.source == "prompted" else 4,
                 "prompt_tokens": 300, "completion_tokens": 20, "error": None}
                for r in items.itertuples()]
        label.CACHE_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    monkeypatch.setattr(label, "call_batch", fake_batch)
    assert m2.main(judge=lambda i, r: [4.0] * len(r), count_tokens=len) == 0
    assert not label.OUT_FILE.exists(), "the Phase 3 judgments file was written"
    summary = json.loads(m2.SUMMARY_FILE.read_text())
    assert summary["paired"] == {"n": 300, "prompted_higher": 0.0, "adapter_higher": 1.0,
                                 "tied": 0.0, "mean_difference": -1.0}


def test_cut_off_and_complete_replies_are_reported_apart():
    graded = pd.DataFrame({
        "source": ["prompted"] * 4, "pair": range(4), "instruction": ["q"] * 4,
        "reply": ["long", "long", "s", "s"], "score": [2, 3, 5, 5],
        "prompt_tokens": [1] * 4, "completion_tokens": [1] * 4,
    })
    cell = m2.summarise(graded, count_tokens=lambda r: 999 if r == "long" else 5)["sides"]["prompted"]
    assert cell["at_cap_share"] == 0.5
    assert cell["gpt4o_mean_at_cap"] == 2.5 and cell["gpt4o_mean_complete"] == 5.0
