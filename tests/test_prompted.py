"""Guards on the prompted baselines (M2) — the comparison that decides if fine-tuning earned it.

A baseline that saw golden rows in its demonstrations, or ran with a different decoding
budget, would make every adapter delta meaningless in the direction of looking good.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import prompted as pr
from adapterops.eval import regression as reg
from adapterops.train.qlora import COLUMNS


@pytest.mark.parametrize(("task", "recipe"), [
    ("intent", "fewshot"), ("intent", "per-class"), ("urgency", "fewshot"),
    ("pii", "fewshot"), ("drafting", "fewshot"),
])
def test_demonstrations_are_never_golden_rows(task, recipe):
    text_col, _ = COLUMNS[task]
    golden = set(pd.read_parquet(ROOT / "evals" / "golden" / f"{task}.parquet")[text_col])
    assert not set(pr.demonstrations(task, recipe).text) & golden


def test_intent_per_class_covers_every_label_once_and_fewshot_is_the_recorded_ten():
    """The recorded baseline covered 10 of 77 classes and said so. per-class fixes that."""
    per_class = pr.demonstrations("intent", "per-class")
    assert len(per_class) == 77 and per_class.gold.is_unique
    assert len(pr.demonstrations("intent", "fewshot")) == 10


def test_urgency_demonstrations_are_balanced_across_its_three_classes():
    demos = pr.demonstrations("urgency", "fewshot")
    assert demos.gold.value_counts().to_dict() == {"high": 4, "low": 4, "medium": 4}


def test_headers_are_deterministic_and_intent_lists_every_label():
    assert pr.header("urgency") == pr.header("urgency")
    head = pr.header("intent", "per-class")
    labels = pd.read_parquet(ROOT / "data/intent/split_train.parquet").label_text.unique()
    assert all(label in head for label in labels)


def test_the_context_check_catches_a_prompt_the_server_would_reject():
    """One demonstration per intent class does not fit the 1,536-token Phase 2 server."""
    assert not pr.budget("intent", "per-class", max_model_len=1536)["fits"]
    assert pr.budget("intent", "per-class", max_model_len=8192)["fits"]


def test_a_perfect_generator_scores_perfectly_through_the_shared_scorers():
    texts, gold = reg.load_split("urgency", "random")
    by_query = {pr.query("urgency", t): g for t, g in zip(texts, gold, strict=True)}
    head = pr.header("urgency")

    def generate(prompts):
        return [by_query[p[len(head):]] for p in prompts]

    result = pr.run("urgency", generate)
    assert result["metrics"]["macro_f1"] == 1.0
    assert result["metrics"]["exact_label_rate"] == 1.0
    assert result["system"] == "prompted-fewshot"


def test_a_recorded_baseline_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pr, "REPO_ROOT", ROOT)
    (tmp_path / "urgency__prompted-fewshot.json").write_text("{}")
    monkeypatch.setattr(pr, "budget", lambda *a, **k: {
        "header_tokens": 1, "longest_query_tokens": 1, "max_new_tokens": 6,
        "needed": 8, "max_model_len": 1536, "fits": True})
    monkeypatch.setattr(pr, "RUNS_DIR", tmp_path)
    assert pr.main("urgency", base_url="http://unused") == 1


def test_saved_replies_come_back_in_the_regression_layout_judge_score_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(pr, "budget", lambda *a, **k: {
        "header_tokens": 1, "longest_query_tokens": 1, "max_new_tokens": 448,
        "needed": 450, "max_model_len": 4096, "fits": True})
    monkeypatch.setattr(pr, "http_generate",
                        lambda base_url, task: lambda prompts: [f"reply {i}" for i in
                                                                range(len(prompts))])
    assert pr.main("drafting", base_url="http://unused", save_predictions=True) == 0

    result = __import__("json").loads((tmp_path / "drafting__prompted-fewshot.json").read_text())
    rows = pd.read_parquet(ROOT / result["predictions_file"])
    texts, _ = reg.load_split("drafting", "random")
    assert list(rows.columns) == ["task", "split", "text", "gold", "prediction"]
    assert len(rows) == len(texts) and set(rows.task) == {"drafting"}
    assert set(rows.split) == {"random"} and rows.prediction.iloc[0] == "reply 0"
