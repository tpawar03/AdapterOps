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
from dataclasses import replace
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

LABEL_GROUPS = {
    **dict.fromkeys(("GENDER", "SEX"), "SEX_OR_GENDER"),
    **dict.fromkeys(("IDCARDNUM", "DRIVERLICENSENUM", "PASSPORTNUM", "SOCIALNUM", "TAXNUM"), "ID_NUMBER"),
    **dict.fromkeys(("GIVENNAME", "SURNAME"), "NAME"),
}
"""PII labels the dataset cannot tell apart from the text: "Male" is GENDER or SEX with no cue in 908 of 1,727
training spans; without a cue word an ID's format gives its label about half the time (`runs/pii__relabel.json`);
invented three-word names are split given-given-surname 2,372 times and given-surname-surname 1,755 times.
`span_f1_grouped` scores each group as one label, with a name's adjacent parts as one span; the gate stays on strict."""

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


def redaction(texts: Sequence[str], gold_spans: Sequence, pred_spans: Sequence) -> dict:
    """What a redaction step would leak, beside the strict span F1 that counts a harmless relabel the
    same as a leak: documents with all personal text masked by some span of any label, and gold spans
    left partly or wholly unmasked."""
    from adapterops.eval.pii_errors import coverage

    leaks = [coverage(t, g, p) for t, g, p in zip(texts, gold_spans, pred_spans, strict=True)]
    spans = sum(len(g) for g in gold_spans)
    return {
        "docs_fully_masked": round(sum(partly == 0 for partly, _ in leaks) / len(leaks), 4) if leaks else None,
        "gold_spans_partly_unmasked": round(sum(p for p, _ in leaks) / spans, 4) if spans else None,
        "gold_spans_wholly_unmasked": round(sum(w for _, w in leaks) / spans, 4) if spans else None,
    }


def grouped_f1(texts: Sequence[str], gold_spans: Sequence, pred_spans: Sequence) -> float:
    """Strict span F1 with the labels in each of `LABEL_GROUPS` counted as one, and NAME spans separated only by
    whitespace joined into one."""
    def merge(text, doc):
        out = []
        for s in sorted((replace(s, label=LABEL_GROUPS.get(s.label, s.label)) for s in doc), key=lambda s: s.start):
            if out and s.label == "NAME" == out[-1].label and not text[out[-1].end:s.start].strip():
                out[-1] = replace(out[-1], end=s.end)
            else:
                out.append(s)
        return out

    gold = [merge(t, d) for t, d in zip(texts, gold_spans, strict=True)]
    pred = [merge(t, d) for t, d in zip(texts, pred_spans, strict=True)]
    return round(float(score_spans(gold, pred, strict=True)["f1"]), 4)


def add_redaction(run_path: Path) -> dict:
    """Backfill the redaction numbers into a committed run from its saved predictions.

    Refuses a run the gate thresholds pin by sha256: rewriting it would break the recorded provenance of
    every threshold derived from it."""
    thresholds = REPO_ROOT / "evals" / "GATE_THRESHOLDS.json"
    if thresholds.exists():
        pinned = {Path(i["file"]).name for i in json.loads(thresholds.read_text()).get("inputs", [])}
        if run_path.name in pinned:
            msg = f"{run_path.name} is a pinned input to {thresholds.name}; rewriting it would break its sha256"
            raise ValueError(msg)
    run = json.loads(run_path.read_text())
    predictions = pd.read_parquet(REPO_ROOT / run["predictions_file"])
    for which in SPLITS:
        frame = predictions[(predictions.task == "pii") & (predictions.split == which)]
        if frame.empty:
            continue
        gold = [spans_from_values(t, parse_model_output(g)) for t, g in zip(frame.text, frame.gold)]
        pred = [spans_from_values(t, parse_model_output(p)) for t, p in zip(frame.text, frame.prediction)]
        run["per_split"]["pii"][which].update(redaction(list(frame.text), gold, pred),
                                              span_f1_grouped=grouped_f1(list(frame.text), gold, pred))
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return run["per_split"]["pii"]


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
                                                          strict=True)["f1"]), 4),
                "span_f1_grouped": grouped_f1(texts, gold_spans, pred_spans),
                **redaction(texts, gold_spans, pred_spans)}
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
    import threading

    from adapterops.router.generate import MAX_TOKENS, complete
    from adapterops.train.qlora import PROMPTS

    stats = {"requests": 0, "errors": 0}
    lock = threading.Lock()

    def predict(task: str, texts: Sequence[str]) -> list[str]:
        def one(text: str) -> str:
            # A failed request scores as a wrong answer rather than aborting the run, and is
            # counted: F17's fallback rate is the share of requests local serving could not answer.
            try:
                prediction, failed = complete(base_url, task, PROMPTS[task].format(text=text),
                                              MAX_TOKENS[task])["prediction"], False
            except Exception:                            # noqa: BLE001 - counted below
                prediction, failed = "", True
            with lock:
                stats["requests"] += 1
                stats["errors"] += failed
            return prediction

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, texts))

    predict.stats = stats
    return predict


def fallback_summary(stats: dict) -> dict:
    n = stats["requests"]
    return {"requests": n, "errors": stats["errors"],
            "fallback_rate": round(stats["errors"] / n, 4) if n else None,
            "definition": "share of requests local serving failed to answer (PRD §11)"}


def main(base_url: str, name: str, baseline: str | None = None,
         save_predictions: bool = False, pins: str | None = None) -> int:
    import time

    from adapterops.serve.classical import with_classical_predictor
    from adapterops.serve.launch import load_components

    rows: list[dict] | None = [] if save_predictions else None
    # Tasks pinned to a classical model are answered here, not by vLLM, which never loads them.
    predictor = with_classical_predictor(http_predictor(base_url),
                                         load_components(Path(pins) if pins else None))
    started = time.perf_counter()
    result = {"name": name, **run(predictor, collect=rows)}
    # PRD §7 budgets a regression run at < 25 min; the Phase 4 runs recorded no timing, so the
    # target could not be checked. Generation and judge scoring, not the file writes.
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    result["fallback"] = fallback_summary(predictor.stats)
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
