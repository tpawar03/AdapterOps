"""Guards on the training-variance measurement that lets provisional gates enforce (F33, D37)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import regression as reg
from adapterops.eval import training_variance as tv


def run(values):
    return {"per_split": {task: {split: {reg.GATED[task]: v} for split, v in splits.items()}
                          for task, splits in values.items()}}


def test_spread_is_the_absolute_difference_of_the_gated_metric_per_split():
    original = run({"urgency": {"random": 0.4269, "hard": 0.0068},
                    "pii": {"random": 0.9455, "hard": 0.8492}})
    rerun = run({"urgency": {"random": 0.4412, "hard": 0.0068},
                 "pii": {"random": 0.9431, "hard": 0.8533}})
    out = tv.spreads([original, rerun], ["urgency", "pii"])
    assert out["urgency"]["random"]["spread"] == pytest.approx(0.0143)
    assert out["urgency"]["hard"]["spread"] == 0.0
    assert out["pii"]["hard"]["spread"] == pytest.approx(0.0041)
    assert out["pii"]["random"]["metric"] == "span_f1_strict"


def test_an_unscored_drafting_run_is_refused_not_skipped():
    original = run({"drafting": {"random": 4.27, "hard": 3.72}})
    rerun = run({"drafting": {"random": None, "hard": None}})
    with pytest.raises(ValueError, match="judge-score"):
        tv.spreads([original, rerun], ["drafting"])


def test_main_pins_its_inputs_and_the_rerun_training_records(tmp_path, monkeypatch):
    monkeypatch.setattr(tv, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tv, "OUT", tmp_path / "runs" / "training_variance.json")
    (tmp_path / "runs").mkdir()
    a, b = tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"
    a.write_text(json.dumps(run({"urgency": {"random": 0.42, "hard": 0.0}})))
    b.write_text(json.dumps(run({"urgency": {"random": 0.44, "hard": 0.0}})))
    (tmp_path / "runs" / "urgency-rerun__train.json").write_text("{}")

    assert tv.main([str(a), str(b)], ["urgency"]) == 0
    written = json.loads(tv.OUT.read_text())
    assert [i["file"] for i in written["inputs"]] == ["runs/a.json", "runs/b.json"]
    assert written["rerun_training_records"]["urgency"][0]["file"] == "runs/urgency-rerun__train.json"
    assert written["per_task"]["urgency"]["random"]["spread"] == pytest.approx(0.02)
    assert tv.main([str(a), str(b)], ["urgency"]) == 1, "a measured variance was silently replaced"


def test_three_runs_give_a_range_and_a_standard_deviation():
    runs = [run({"pii": {"random": v, "hard": 0.85}}) for v in (0.9421, 0.9455, 0.9402)]
    row = tv.spreads(runs, ["pii"])["pii"]["random"]
    assert row["runs"] == 3 and row["values"] == [0.9421, 0.9455, 0.9402]
    assert row["spread"] == pytest.approx(0.0053) and row["stdev"] == pytest.approx(0.002683, abs=1e-5)
    assert "original" not in row, "the two-run keys should only appear for two runs"


def test_fewer_than_two_runs_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        tv.main(["one.json"], ["pii"])
