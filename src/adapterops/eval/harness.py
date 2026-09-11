"""Golden-set evaluation harness (F6).

Takes a **split path as an argument** from its first version, deliberately: the Phase 4
hard-cases split (F31) then becomes a second invocation rather than a refactor.

A *system under test* is any callable `list[str] -> list[str]`. Three arrive later —
the tuned adapter, the prompted baseline, the frontier ceiling. `EchoPredictor` and
`MajorityPredictor` exist now so the plumbing is exercised before any model exists;
`MajorityPredictor` is also the floor any real system must clear.

Run:  uv run adapterops eval --split evals/golden/intent.parquet --task intent
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"

Predictor = Callable[[Sequence[str]], Sequence[str]]

TASK_COLUMNS = {
    "intent": ("text", "label"),
    "urgency": ("text", "label"),
    "pii": ("source_text", "privacy_mask"),   # span offsets, not a document label (PRD changelog 24)
    "drafting": ("instruction", "response"),
}

GATED_METRIC = {
    "intent": "micro_accuracy",   # 77 classes — macro-F1 is indicative only (PRD §11)
    "urgency": "macro_f1",
    "pii": "span_f1",
    "drafting": "judge_score",
}


@dataclass
class Result:
    task: str
    split: str
    split_sha256: str
    system: str
    n: int
    gated_metric: str
    metrics: dict
    seconds: float
    host: str


def majority_predictor(train_labels: Sequence[str]) -> Predictor:
    """The floor. A system that cannot beat this has learned nothing."""
    top = pd.Series(train_labels).value_counts().idxmax()
    return lambda texts: [top] * len(texts)


def score(task: str, gold: Sequence[str], pred: Sequence[str]) -> dict:
    if task == "drafting":
        msg = "drafting is judge-scored (Phase 3); no label metric applies"
        raise NotImplementedError(msg)
    if task == "pii":
        msg = ("pii is span detection: use adapterops.eval.spans.score_spans on "
               "Span lists, not score() on label strings (PRD changelog 24-25)")
        raise NotImplementedError(msg)
    return {
        "micro_accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
    }


def evaluate(split_path: Path, task: str, system_name: str, predictor: Predictor) -> Result:
    import hashlib

    if task not in TASK_COLUMNS:
        msg = f"unknown task {task!r}; expected one of {sorted(TASK_COLUMNS)}"
        raise ValueError(msg)
    text_col, label_col = TASK_COLUMNS[task]
    df = pd.read_parquet(split_path)
    for col in (text_col, label_col):
        if col not in df.columns:
            msg = f"{split_path} has no {col!r} column (found {list(df.columns)})"
            raise ValueError(msg)

    started = time.perf_counter()
    pred = predictor(df[text_col].tolist())
    elapsed = time.perf_counter() - started
    if len(pred) != len(df):
        msg = f"predictor returned {len(pred)} predictions for {len(df)} rows"
        raise ValueError(msg)

    return Result(
        task=task,
        split=str(split_path.relative_to(REPO_ROOT)),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        system=system_name,
        n=len(df),
        gated_metric=GATED_METRIC[task],
        metrics=score(task, df[label_col].astype(str).tolist(), [str(p) for p in pred]),
        seconds=round(elapsed, 4),
        host=platform.platform(),
    )


def record(result: Result) -> Path:
    """Append the result to runs/ — the dashboard reads real historical runs (§11)."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{result.task}__{result.system}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return path
