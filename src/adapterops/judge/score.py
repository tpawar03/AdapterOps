"""Score saved replies with the distilled judge (F14) — drafting's gated metric, after the GPU is gone.

Drafting is the one task whose gated metric is a model rather than a comparison with gold, and that
model — the distilled judge — lives on the laptop, not on the rented box. So the regression run keeps
its replies (`regress --save-predictions`), and this scores them afterwards on CPU and writes the
result back into the run, where `compare()` and `derive-thresholds` expect to find it.

**Order matters.** `derive-thresholds` pins each input run by sha256. Score the baseline runs *before*
deriving thresholds, or the derivation pins runs whose drafting values are still empty.

**A candidate run's comparison is recomputed** when a baseline is given, because the comparison
written at regression time had no drafting value on either side.

**The scoring path is checked against training.** Its test loads the saved judge and reproduces the
calibration predictions `Trainer.predict` made when the judge was trained — the same text through a
different code path must give the same score, or this is scoring with something that is not the judge.

    uv run adapterops judge-score --run runs/regression__v1-baseline-1.json
    uv run adapterops judge-score --run runs/regression__all-m11.json \\
        --baseline runs/regression__v1-baseline-1.json
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.judge.train import OUT_DIR, from_target, to_text

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE = Callable[[Sequence[str], Sequence[str]], list[float]]


def load_judge(path: Path | None = None, batch_size: int = 16) -> JUDGE:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = path or OUT_DIR
    if not (Path(path) / "model.safetensors").exists():
        msg = f"no trained judge at {path} — run `adapterops judge-train` first"
        raise FileNotFoundError(msg)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()

    def judge(instructions: Sequence[str], replies: Sequence[str]) -> list[float]:
        texts = to_text(pd.DataFrame({"instruction": list(instructions), "reply": list(replies)}))
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=512)
                raw = model(**enc).logits.reshape(-1).numpy()
                scores.extend(float(x) for x in from_target(raw))
        return scores

    return judge


def pinned_judge() -> Path:
    """The judge the current manifest pins, so the gate scores drafting with the judge that is served."""
    from adapterops.manifest.system import load_current
    from adapterops.serve.pipeline import pinned_checkpoint

    return pinned_checkpoint(load_current(), "judge") or OUT_DIR


def score_predictions(predictions: pd.DataFrame, judge: JUDGE) -> dict:
    """Mean judge score per split, over drafting rows only."""
    drafting = predictions[predictions.task == "drafting"]
    out = {}
    for split, rows in drafting.groupby("split"):
        scores = judge(rows.text.tolist(), rows.prediction.tolist())
        out[split] = {"n": len(rows), "judge_score_mean": round(float(np.mean(scores)), 4)}
    return out


def fill_regression_run(run: dict, drafting: dict, judge: str = "checkpoints/judge") -> dict:
    out = copy.deepcopy(run)
    for split, result in drafting.items():
        cell = out["per_split"]["drafting"][split]
        cell["judge_score_mean"] = result["judge_score_mean"]
        cell["judge"] = f"distilled judge, {judge} (F14)"
        cell.pop("note", None)
    return out


def fill_prompted_run(run: dict, drafting: dict, judge: str = "checkpoints/judge") -> dict:
    """A prompted baseline (M2) holds one golden split under `metrics`, not `per_split`."""
    out = copy.deepcopy(run)
    out["metrics"]["judge_score_mean"] = drafting["random"]["judge_score_mean"]
    out["metrics"]["judge"] = f"distilled judge, {judge} (F14)"
    out["metrics"].pop("note", None)
    return out


def main(run_path: str, baseline: str | None = None, force: bool = False, out: str | None = None) -> int:
    """`out` writes the scored run to a new file, leaving a sha-pinned run as it was."""
    from adapterops.eval.regression import compare

    path = Path(run_path)
    run = json.loads(path.read_text())
    if "predictions_file" not in run:
        print(f"  {run_path} kept no predictions — re-run regress with --save-predictions")
        return 2
    judge_dir = pinned_judge()
    judge_name = str(judge_dir.relative_to(REPO_ROOT)) if judge_dir.is_relative_to(REPO_ROOT) else str(judge_dir)
    dest = Path(out) if out else path
    if "per_split" not in run:
        return _main_prompted(path, run, force or bool(out), dest, judge_dir, judge_name)
    already = [s for s, cell in run["per_split"]["drafting"].items()
               if cell.get("judge_score_mean") is not None]
    if already and not (force or out):
        print(f"  drafting is already judge-scored on {already} — --force to rescore")
        return 1

    drafting = score_predictions(pd.read_parquet(REPO_ROOT / run["predictions_file"]),
                                 load_judge(judge_dir))
    filled = fill_regression_run(run, drafting, judge_name)
    if baseline:
        filled["comparison"] = compare(filled, json.loads(Path(baseline).read_text()))
    dest.write_text(json.dumps(filled, indent=2) + "\n", encoding="utf-8")
    for split, result in drafting.items():
        print(f"  drafting {split:6s} n {result['n']:>3} · judge score {result['judge_score_mean']}")
    if baseline:
        print(f"  drafting drop vs baseline: {filled['comparison']['per_task']['drafting']['drop']}")
    return 0


def _main_prompted(path: Path, run: dict, force: bool, dest: Path, judge_dir: Path, judge_name: str) -> int:
    if run["metrics"].get("judge_score_mean") is not None and not force:
        print(f"  {path} is already judge-scored — --force to rescore")
        return 1
    drafting = score_predictions(pd.read_parquet(REPO_ROOT / run["predictions_file"]),
                                 load_judge(judge_dir))
    dest.write_text(json.dumps(fill_prompted_run(run, drafting, judge_name), indent=2) + "\n",
                    encoding="utf-8")
    print(f"  {run['system']} drafting n {drafting['random']['n']} · "
          f"judge score {drafting['random']['judge_score_mean']}")
    return 0
