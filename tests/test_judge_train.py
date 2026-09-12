"""Guards on judge training (F14) that need no model: what it is allowed to learn from.

The calibration holdout is the M5 number. If it leaks into training — or only into
checkpoint selection — the correlation it reports is one the model was tuned to produce.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.judge import train as jt


def labelled(n_train: int = 60, n_cal: int = 20, n_frontier: int = 15,
             n_unparsed: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_train):
        rows.append({"item_id": f"local:t{i}", "source": "local", "split": "train",
                     "parsed_ok": True, "score": 1 + i % 5,
                     "instruction": f"q{i}", "reply": f"r{i}"})
    for i in range(n_cal):
        rows.append({"item_id": f"local:c{i}", "source": "local", "split": "calibration",
                     "parsed_ok": True, "score": 1 + i % 5,
                     "instruction": f"cq{i}", "reply": f"cr{i}"})
    for i in range(n_frontier):
        rows.append({"item_id": f"frontier:f{i}", "source": "frontier",
                     "split": "frontier_eval", "parsed_ok": True, "score": 5,
                     "instruction": f"fq{i}", "reply": f"fr{i}"})
    for i in range(n_unparsed):
        rows.append({"item_id": f"local:u{i}", "source": "local", "split": "train",
                     "parsed_ok": False, "score": None,
                     "instruction": f"uq{i}", "reply": f"ur{i}"})
    return pd.DataFrame(rows)


def test_calibration_items_never_reach_fit_or_checkpoint_selection():
    fit, val, cal = jt.split_for_training(labelled())
    assert set(cal.item_id) == {f"local:c{i}" for i in range(20)}
    assert not set(cal.item_id) & set(fit.item_id), "calibration leaked into training"
    assert not set(cal.item_id) & set(val.item_id), "calibration used for checkpoint selection"
    assert not set(fit.item_id) & set(val.item_id)


def test_frontier_replies_and_unparsed_grades_are_excluded():
    fit, val, cal = jt.split_for_training(labelled())
    used = set(fit.item_id) | set(val.item_id) | set(cal.item_id)
    assert not any(i.startswith("frontier:") for i in used)
    assert not any(i.startswith("local:u") for i in used)
    assert len(fit) + len(val) == 60


def test_the_judge_sees_what_the_teacher_saw_and_no_reference():
    frame = pd.DataFrame({"instruction": ["where is it"], "reply": ["Share your {{Order Number}}."]})
    assert jt.to_text(frame) == ["REQUEST:\nwhere is it\n\nREPLY:\nShare your {{Order Number}}."]


def test_training_refuses_a_half_finished_grading_run(tmp_path, monkeypatch):
    """A partial run would otherwise produce a judge and an M5 number that look complete."""
    path = tmp_path / "judgments.parquet"
    labelled(n_train=40).to_parquet(path)
    monkeypatch.setattr(jt, "LABELS_FILE", path)
    with pytest.raises(ValueError, match="Finish the grading run"):
        jt.train(jt.JudgeConfig(min_fit=200))


def test_target_scaling_centres_grades_and_round_trips_exactly():
    """A head initialised near zero regressing raw 1-5 grades spends its first steps learning
    the mean — on the first smoke run, that was all it learned (Spearman -0.45)."""
    import numpy as np

    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert jt.to_target(scores).tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert jt.from_target(jt.to_target(scores)).tolist() == scores.tolist()
