"""On-demand regression run (F18) — the *detect* in detect → block → rollback (M7).

GitHub Actions' free runners are CPU-only (D4), so this runs during a rented GPU session
rather than nightly. It scores a candidate system on **both** eval splits — the frozen random
golden sets and the adjudicated hard split — and compares each with a baseline run of the
last known-good manifest.

**Two rows, never blended (F32).** Each task reports its random-split and hard-split results
separately. Only the random-split drop is handed to the promotion gate as `drop`; the hard
split is report-only until F33 derives its threshold from the run-to-run variance of two
baseline runs. A blended number can show a model improving on average while regressing on
exactly the cases that matter — the failure the two splits exist to expose.

**Every gated metric is higher-is-better,** so a drop is `baseline − candidate`, and positive
means regression. Drafting's gated metric is the distilled judge's mean score. Without a judge
that value is recorded as unavailable and its drop is `None`, which the gate skips rather than
reading as zero — an unmeasured task is not a task that did not regress.

    uv run adapterops regress --base-url http://localhost:8000 --name v2 \\
        --baseline runs/regression__v1.json
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from adapterops.eval.spans import parse_model_output, score_spans, spans_from_values
from adapterops.train.qlora import COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
HARD_FILE = REPO_ROOT / "evals" / "hard" / "hard_cases.parquet"

TASKS = ("intent", "urgency", "pii", "drafting")
SPLITS = ("random", "hard")
GATED = {
    "intent": "micro_accuracy",       # 77 classes — macro-F1 is indicative only (D10)
    "urgency": "macro_f1",
    "pii": "span_f1_strict",
    "drafting": "judge_score_mean",
}

Predictor = Callable[[str, Sequence[str]], list[str]]
Judge = Callable[[Sequence[str], Sequence[str]], Sequence[float]]


def load_split(task: str, which: str) -> tuple[list[str], list[str]]:
    """(inputs, gold) for one task on one split, in the column convention training used."""
    if which == "random":
        frame = pd.read_parquet(GOLDEN_DIR / f"{task}.parquet")
        text_col, gold_col = COLUMNS[task]
        return frame[text_col].astype(str).tolist(), frame[gold_col].astype(str).tolist()
    if which == "hard":
        frame = pd.read_parquet(HARD_FILE)
        frame = frame[frame.task == task]
        return frame.text.astype(str).tolist(), frame.gold.astype(str).tolist()
    msg = f"unknown split {which!r}"
    raise ValueError(msg)


def score(task: str, texts: Sequence[str], gold: Sequence[str], predictions: Sequence[str],
          judge: Judge | None = None) -> dict:
    n = len(gold)
    if task in ("intent", "urgency"):
        pred = [p.strip().split("\n")[0].strip() for p in predictions]
        return {"n": n,
                "micro_accuracy": round(float(accuracy_score(gold, pred)), 4),
                "macro_f1": round(float(f1_score(gold, pred, average="macro",
                                                 zero_division=0)), 4)}
    if task == "pii":
        gold_spans = [spans_from_values(t, parse_model_output(g)) for t, g in zip(texts, gold)]
        pred_spans = [spans_from_values(t, parse_model_output(p))
                      for t, p in zip(texts, predictions)]
        return {"n": n,
                "span_f1_strict": round(float(score_spans(gold_spans, pred_spans,
                                                          strict=True)["f1"]), 4)}
    if task == "drafting":
        if judge is None:
            return {"n": n, "judge_score_mean": None,
                    "note": "no judge available — drafting is not gated on this run"}
        scores = [float(s) for s in judge(texts, predictions)]
        return {"n": n, "judge_score_mean": round(sum(scores) / len(scores), 4)}
    msg = f"unknown task {task!r}"
    raise ValueError(msg)


def run(predictor: Predictor, judge: Judge | None = None,
        tasks: Sequence[str] = TASKS, collect: list[dict] | None = None) -> dict:
    """Score every task on both splits. Pass `collect` to keep each prediction as a row.

    Drafting is the reason to keep them: its gated metric is the distilled judge's score, and
    the judge runs on CPU after the GPU session ends. Without the replies, scoring drafting on
    the golden set would mean renting the GPU again to regenerate them.
    """
    per_split: dict = {}
    for task in tasks:
        per_split[task] = {}
        for which in SPLITS:
            texts, gold = load_split(task, which)
            if not texts:
                per_split[task][which] = {"n": 0, GATED[task]: None, "note": "empty split"}
                continue
            predictions = predictor(task, texts)
            if len(predictions) != len(texts):
                msg = (f"{task}/{which}: predictor returned {len(predictions)} predictions "
                       f"for {len(texts)} inputs")
                raise ValueError(msg)
            per_split[task][which] = score(task, texts, gold, predictions, judge)
            if collect is not None:
                collect.extend({"task": task, "split": which, "text": t, "gold": g,
                                "prediction": p}
                               for t, g, p in zip(texts, gold, predictions, strict=True))
    return {"gated_metric": GATED, "per_split": per_split}


def compare(candidate: dict, baseline: dict) -> dict:
    """Per-task drops, random and hard kept apart. `drop` is the random split's — it gates."""
    per_task = {}
    for task, metric in GATED.items():
        rows = {}
        for which in SPLITS:
            c = candidate["per_split"].get(task, {}).get(which, {}).get(metric)
            b = baseline["per_split"].get(task, {}).get(which, {}).get(metric)
            rows[which] = {"candidate": c, "baseline": b,
                           "drop": None if c is None or b is None else round(b - c, 4)}
        per_task[task] = {
            "gated_metric": metric,
            "drop": rows["random"]["drop"],
            "random": rows["random"],
            "hard": rows["hard"],
            "hard_gate": "report-only until F33 derives a threshold from two baseline runs",
        }
    return {"per_task": per_task}


def http_predictor(base_url: str, concurrency: int = 16) -> Predictor:
    """The served adapters, addressed by task name as scripts/phase2_serve.sh registers them."""
    from adapterops.router.generate import MAX_TOKENS, complete
    from adapterops.train.qlora import PROMPTS

    def predict(task: str, texts: Sequence[str]) -> list[str]:
        def one(text: str) -> str:
            return complete(base_url, task, PROMPTS[task].format(text=text),
                            MAX_TOKENS[task])["prediction"]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, texts))

    return predict


def main(base_url: str, name: str, baseline: str | None = None,
         save_predictions: bool = False) -> int:
    rows: list[dict] | None = [] if save_predictions else None
    result = {"name": name, **run(http_predictor(base_url), collect=rows)}
    if baseline:
        result["comparison"] = compare(result, json.loads(Path(baseline).read_text()))
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"regression__{name}.json"
    if rows is not None:
        predictions = RUNS_DIR / f"regression__{name}__predictions.parquet"
        pd.DataFrame(rows).to_parquet(predictions)
        result["predictions_file"] = str(predictions.relative_to(REPO_ROOT))
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for task, splits in result["per_split"].items():
        metric = GATED[task]
        cells = "  ".join(f"{w} {splits[w].get(metric)}" for w in SPLITS)
        drop = result.get("comparison", {}).get("per_task", {}).get(task, {}).get("drop")
        print(f"  {task:9s} {metric:18s} {cells}" + (f"  · drop {drop}" if baseline else ""))
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")
    return 0
