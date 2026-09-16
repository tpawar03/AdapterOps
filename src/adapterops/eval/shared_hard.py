"""A hard split mined from the failures of systems the gate never compares (TODO.md §1).

`evals/hard/hard_cases.parquet` holds items the first adapter got wrong, so that adapter scores 0 on intent by
construction and any different model scores higher (D32). This mines the golden items every *source* fails, where
no source is a model M11 compares: GPT-4o-mini for every task, and for PII also the retrained adapter served since
manifest v6, whose weights differ from v1's. The served intent, urgency and drafting adapters are v1's weights, so
they cannot be sources. Drafting fails where the distilled judge grades a reply below 4, the rule the first hard
split used with the GPT-4o teacher.

The split is written once and then only read: a rebuild that differs from it is refused. `main` then re-runs M11's
sensitivity check — v1 against its under-trained checkpoints — on the random split, the first hard split and this one.

    uv run adapterops hard-shared
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from adapterops.eval import regression as reg
from adapterops.eval.spans import parse_model_output, spans_from_values

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "runs"
SPLIT_FILE = REPO_ROOT / "evals" / "hard" / "shared_failures.parquet"
RUN_FILE = RUNS / "hard__shared.json"
FRONTIER = RUNS / "frontier__golden__predictions.parquet"
SOURCES = {
    "intent": ("gpt-4o-mini",),
    "urgency": ("gpt-4o-mini",),
    "pii": ("gpt-4o-mini", "a10-v5-pii-negatives"),
    "drafting": ("gpt-4o-mini",),
}
COMPARED = ("v1-baseline-1", "v1-baseline-2", "all-m11")
DRAFT_PASS = 4.0


def golden_predictions(system: str) -> pd.DataFrame:
    """task, text, gold and prediction for one system on the golden (random) split."""
    if system == "gpt-4o-mini":
        frame = pd.read_parquet(FRONTIER)
    else:
        frame = pd.read_parquet(RUNS / f"regression__{system}__predictions.parquet")
        frame = frame[frame.split == "random"]
    return frame[["task", "text", "gold", "prediction"]].reset_index(drop=True)


def successes(task: str, frame: pd.DataFrame, judge_scores: Sequence[float] | None = None) -> list[bool]:
    if task in ("intent", "urgency"):
        return [p.strip().split("\n")[0].strip() == g for g, p in zip(frame.gold, frame.prediction, strict=True)]
    if task == "pii":
        return [set(spans_from_values(t, parse_model_output(g))) == set(spans_from_values(t, parse_model_output(p)))
                for t, g, p in zip(frame.text, frame.gold, frame.prediction, strict=True)]
    return [s >= DRAFT_PASS for s in judge_scores]


def mine(source_ok: dict[str, Sequence[bool]]) -> list[int]:
    """Row positions every source fails."""
    oks = list(source_ok.values())
    return [i for i in range(len(oks[0])) if not any(ok[i] for ok in oks)]


def freeze(split: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write the split the first time; afterwards return the frozen one, refusing a rebuild that differs."""
    if path.exists():
        frozen = pd.read_parquet(path)
        key = ["task", "source_row", "text", "gold"]
        if not frozen[key].reset_index(drop=True).equals(split[key].reset_index(drop=True)):
            msg = f"{path.name} is frozen and a rebuild differs from it — delete it deliberately to re-mine"
            raise ValueError(msg)
        return frozen
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(path, index=False)
    return split


def metric(task: str, frame: pd.DataFrame, rows: Sequence[int], judge_scores: Sequence[float] | None) -> float:
    if task == "drafting":
        return round(sum(judge_scores[i] for i in rows) / len(rows), 4)
    sub = frame.iloc[list(rows)]
    return reg.score(task, list(sub.text), list(sub.gold), list(sub.prediction))[reg.GATED[task]]


def main() -> int:
    from adapterops.judge.score import load_judge

    judge = load_judge()
    systems = sorted({s for v in SOURCES.values() for s in v} | set(COMPARED))
    frames = {s: golden_predictions(s) for s in systems}
    per: dict[tuple[str, str], tuple[pd.DataFrame, list[bool], list[float] | None]] = {}
    for task in SOURCES:
        reference = None
        for s in systems:
            f = frames[s][frames[s].task == task].reset_index(drop=True)
            if reference is None:
                reference = f
            elif not (f.text.equals(reference.text) and f.gold.equals(reference.gold)):
                msg = f"{s}/{task}: golden items are not in the same order as {systems[0]}'s"
                raise ValueError(msg)
            scores = [float(x) for x in judge(list(f.text), list(f.prediction))] if task == "drafting" else None
            per[(s, task)] = (f, successes(task, f, scores), scores)

    recorded = json.loads((RUNS / "regression__v1-baseline-1.json").read_text())["per_split"]
    v1_frame, _, v1_scores = per[("v1-baseline-1", "drafting")]
    rescored = metric("drafting", v1_frame, range(len(v1_frame)), v1_scores)
    if abs(rescored - recorded["drafting"]["random"]["judge_score_mean"]) > 0.005:
        msg = f"judge gives v1 drafting {rescored}, the run recorded {recorded['drafting']['random']['judge_score_mean']}"
        raise ValueError(msg)

    parts = []
    for task, sources in SOURCES.items():
        rows = mine({s: per[(s, task)][1] for s in sources})
        f = per[(sources[0], task)][0]
        parts.append(f.iloc[rows][["task", "text", "gold"]].assign(source_row=rows, sources=" + ".join(sources)))
    split = freeze(pd.concat(parts, ignore_index=True), SPLIT_FILE)

    thresholds = json.loads((REPO_ROOT / "evals" / "GATE_THRESHOLDS.json").read_text())["derivation"]["per_task"]
    runs = {s: json.loads((RUNS / f"regression__{s}.json").read_text())["per_split"] for s in COMPARED}
    check = {}
    for task in SOURCES:
        f = per[("v1-baseline-1", task)][0]
        rows = {"random": range(len(f)), "shared_hard": list(split[split.task == task].source_row)}
        out = {}
        for name, r in rows.items():
            vals = {s: metric(task, per[(s, task)][0], r, per[(s, task)][2]) for s in COMPARED}
            out[name] = {"n": len(r), **vals, "m11_drop": round(vals["v1-baseline-1"] - vals["all-m11"], 4),
                         "baseline_spread": round(abs(vals["v1-baseline-1"] - vals["v1-baseline-2"]), 4)}
        old = {s: runs[s][task]["hard"][reg.GATED[task]] for s in COMPARED}
        out["first_hard_split"] = {**old, "m11_drop": round(old["v1-baseline-1"] - old["all-m11"], 4)}
        threshold = thresholds.get(task, {}).get("random", {}).get("threshold")
        out["random"]["gate_threshold"] = threshold
        out["random"]["flags_m11"] = threshold is not None and out["random"]["m11_drop"] > threshold
        out["shared_hard"]["flags_m11_at_3x_baseline_spread"] = (
            out["shared_hard"]["m11_drop"] > 3 * out["shared_hard"]["baseline_spread"])
        check[task] = out

    result = {
        "sources": SOURCES, "compared_never_sources": COMPARED, "drafting_pass": DRAFT_PASS,
        "split": {"file": str(SPLIT_FILE.relative_to(REPO_ROOT)),
                  "sha256": hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
                  "rows_per_task": split.task.value_counts().to_dict()},
        "m11_sensitivity": check,
    }
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for task, out in check.items():
        print(f"  {task:8s} random drop {out['random']['m11_drop']:+.4f} (gate {out['random']['gate_threshold']}) · "
              f"first hard {out['first_hard_split']['m11_drop']:+.4f} · shared hard n={out['shared_hard']['n']} "
              f"v1 {out['shared_hard']['v1-baseline-1']} m11 {out['shared_hard']['all-m11']} "
              f"drop {out['shared_hard']['m11_drop']:+.4f} (spread {out['shared_hard']['baseline_spread']})")
    print(f"  wrote {SPLIT_FILE.relative_to(REPO_ROOT)} and {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
