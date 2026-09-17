"""Guards on scoring saved drafting replies with the distilled judge after the GPU session."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import score as js

JUDGE_PRESENT = (ROOT / "checkpoints" / "judge" / "model.safetensors").exists()


def run_doc():
    return {"per_split": {
        "intent": {"random": {"micro_accuracy": 0.93}, "hard": {"micro_accuracy": 0.5}},
        "drafting": {"random": {"judge_score_mean": None, "note": "no judge"},
                     "hard": {"judge_score_mean": None, "note": "no judge"}},
    }, "predictions_file": "runs/x.parquet"}


def test_filling_a_run_sets_drafting_only_and_clears_the_placeholder_note():
    filled = js.fill_regression_run(run_doc(), {"random": {"judge_score_mean": 4.1, "n": 3},
                                                "hard": {"judge_score_mean": 3.6, "n": 2}})
    assert filled["per_split"]["drafting"]["random"]["judge_score_mean"] == 4.1
    assert "note" not in filled["per_split"]["drafting"]["hard"]
    assert filled["per_split"]["intent"] == run_doc()["per_split"]["intent"]


def test_only_drafting_rows_are_scored_and_per_split():
    preds = pd.DataFrame([
        {"task": "drafting", "split": "random", "text": "q", "prediction": "abc"},
        {"task": "drafting", "split": "random", "text": "q", "prediction": "abcde"},
        {"task": "drafting", "split": "hard", "text": "q", "prediction": "ab"},
        {"task": "intent", "split": "random", "text": "q", "prediction": "zzzzzzzzzz"},
    ])
    out = js.score_predictions(preds, lambda texts, replies: [float(len(r)) for r in replies])
    assert out == {"hard": {"n": 1, "judge_score_mean": 2.0},
                   "random": {"n": 2, "judge_score_mean": 4.0}}


def test_a_run_without_saved_predictions_is_refused(tmp_path):
    path = tmp_path / "run.json"
    doc = run_doc()
    doc.pop("predictions_file")
    path.write_text(json.dumps(doc))
    assert js.main(str(path)) == 2


@pytest.mark.skipif(not JUDGE_PRESENT, reason="no trained judge checkpoint on this machine")
def test_the_scoring_path_reproduces_the_judges_own_calibration_predictions():
    """Same text through a different code path must give the same score, or this is not the judge."""
    cal = pd.read_parquet(ROOT / "data" / "judge" / "calibration_scored.parquet").head(8)
    scores = js.load_judge()(cal.instruction.tolist(), cal.reply.tolist())
    assert np.allclose(scores, cal.judge_score.to_numpy(), atol=0.02), (
        list(zip(scores, cal.judge_score.to_numpy(), strict=True)))


def test_a_prompted_baselines_replies_are_scored_into_its_metrics(tmp_path, monkeypatch):
    """M2 for drafting: the prompted run has one golden split under `metrics`."""
    preds = tmp_path / "p.parquet"
    pd.DataFrame([{"task": "drafting", "split": "random", "text": "q", "gold": "g",
                   "prediction": "abc"},
                  {"task": "drafting", "split": "random", "text": "q", "gold": "g",
                   "prediction": "abcde"}]).to_parquet(preds)
    path = tmp_path / "drafting__prompted-fewshot.json"
    path.write_text(json.dumps({"system": "prompted-fewshot", "predictions_file": str(preds),
                                "metrics": {"n": 2, "judge_score_mean": None,
                                            "note": "no judge available"}}))
    monkeypatch.setattr(js, "load_judge",
                        lambda path=None: lambda texts, replies: [float(len(r)) for r in replies])

    assert js.main(str(path)) == 0
    metrics = json.loads(path.read_text())["metrics"]
    assert metrics["judge_score_mean"] == 4.0 and "note" not in metrics
    assert js.main(str(path)) == 1, "an already-scored baseline was silently rescored"
