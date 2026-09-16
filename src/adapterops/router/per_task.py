"""Per-task confidence thresholds, chosen on the router's training pairs and scored on held-out ones.

The request path escalates on one pooled threshold (0.3939, the confidence curve's 20% operating point), but
escalation pays on drafting and costs quality on the other tasks. This splits the same 20% budget across tasks:
on the training pairs, an exact search picks how many pairs each task escalates to maximise served quality,
and each task's count becomes a threshold on `-mean_logprob` — a fixed rule the live path can apply. The
thresholds are then scored on the in-distribution and shifted eval pairs, beside the pooled threshold at the
same escalation rate and the oracle, so a win cannot come from tuning on the pairs it is measured on.

    uv run adapterops router-per-task
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.router.baselines import _ordering, escalation_scores, evaluate_at_budget

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
RUN_FILE = REPO_ROOT / "runs" / "router__per_task.json"
BUDGET = 0.20


def training_frame() -> pd.DataFrame:
    from adapterops.router.report import frontier_grades, join_judged

    local = pd.read_parquet(DATA_DIR / "router_train__judged.parquet")
    return join_judged(local, pd.read_parquet(DATA_DIR / "frontier.parquet"), frontier_grades())


def cumulative_gains(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For one task, pairs ranked by confidence score (least confident first): the served-quality gain of
    escalating the top k, for every k, and the scores in that order."""
    scores = escalation_scores(frame, "confidence")
    order = _ordering(scores, frame.pair_id)
    gain = (frame.frontier_success.astype(int) - frame.success.astype(int)).to_numpy()[order]
    return np.concatenate([[0], np.cumsum(gain)]), scores.to_numpy()[order]


def allocate(gains: dict[str, np.ndarray], total: int) -> dict[str, int]:
    """The per-task counts, summing to at most `total`, that maximise the summed gain — exact, because the
    gain curves are not concave and a greedy split can miss the best one."""
    tasks = list(gains)
    best = {0: (0, {})}
    for task in tasks:
        curve, nxt = gains[task], {}
        for used, (value, picks) in best.items():
            for k in range(min(len(curve) - 1, total - used) + 1):
                candidate = (value + int(curve[k]), {**picks, task: k})
                if used + k not in nxt or candidate[0] > nxt[used + k][0]:
                    nxt[used + k] = candidate
        best = nxt
    return max(best.values(), key=lambda v: v[0])[1]


def thresholds_from(frame: pd.DataFrame, counts: dict[str, int]) -> dict[str, float]:
    """Each task's threshold: the confidence score of the last pair its count escalates (inf for none)."""
    out = {}
    for task, rows in frame.groupby("task"):
        k = counts.get(task, 0)
        _, ordered = cumulative_gains(rows)
        out[task] = float(ordered[k - 1]) if k else math.inf
    return out


def apply(frame: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    """Escalate each pair whose score reaches its task's threshold; quality and rates as the curve reports."""
    scores = escalation_scores(frame, "confidence").to_numpy()
    limit = frame.task.map(thresholds).to_numpy()
    escalate = scores >= limit
    realised = np.where(escalate, frame.frontier_success.astype(float), frame.success.astype(float))
    row = {"escalation_rate": round(float(escalate.mean()), 4), "quality": round(float(realised.mean()), 4)}
    for task in sorted(frame.task.unique()):
        pos = (frame.task == task).to_numpy()
        row[f"escalation__{task}"] = round(float(escalate[pos].mean()), 4)
        row[f"quality__{task}"] = round(float(realised[pos].mean()), 4)
    return row


def escalations(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return escalation_scores(frame, "confidence").to_numpy() >= frame.task.map(thresholds).to_numpy()


def pooled_escalations(frame: pd.DataFrame, rate: float) -> np.ndarray:
    """The pooled rule's decisions at `rate`, ranked exactly as `evaluate_at_budget` ranks them."""
    scores = escalation_scores(frame, "confidence")
    out = np.zeros(len(frame), dtype=bool)
    k = round(rate * len(frame))
    if k:
        out[_ordering(scores, frame.pair_id)[:k]] = True
    return out


def paired_bootstrap(frame: pd.DataFrame, first: np.ndarray, second: np.ndarray,
                     draws: int = 2000, seed: int = 20260909) -> dict:
    """Quality of `first` minus `second` on the same pairs, with a 95% interval over resampled pairs."""
    local = frame.success.astype(float).to_numpy()
    frontier = frame.frontier_success.astype(float).to_numpy()
    diff = np.where(first, frontier, local) - np.where(second, frontier, local)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
    return {"difference": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "pairs_that_differ": int((first != second).sum())}


def _captured(quality: float, local: float, oracle: float) -> float | None:
    return None if oracle == local else round((quality - local) / (oracle - local), 4)


def main() -> int:
    from adapterops.router.misroute import frames

    train = training_frame()
    gains = {task: cumulative_gains(rows)[0] for task, rows in train.groupby("task")}
    counts = allocate(gains, round(BUDGET * len(train)))
    thresholds = thresholds_from(train, counts)
    result = {
        "budget": BUDGET,
        "training_pairs": len(train),
        "training_escalations_per_task": counts,
        "thresholds": {t: (None if math.isinf(v) else round(v, 6)) for t, v in thresholds.items()},
        "populations": {},
    }
    for population, frame in frames().items():
        local = float(frame.success.astype(float).mean())
        oracle = evaluate_at_budget(frame, escalation_scores(frame, "oracle"), BUDGET)["quality"]
        per_task = apply(frame, thresholds)
        pooled = evaluate_at_budget(frame, escalation_scores(frame, "confidence"), BUDGET)
        matched = evaluate_at_budget(frame, escalation_scores(frame, "confidence"), per_task["escalation_rate"])

        mine = escalations(frame, thresholds)
        result["populations"][population] = {
            "pairs": len(frame),
            "never_escalate_quality": round(local, 4),
            "oracle_quality_at_20pct": oracle,
            "per_task_thresholds": {**per_task,
                                    "gain_captured_vs_20pct_oracle": _captured(per_task["quality"], local, oracle)},
            "pooled_threshold_at_20pct": {k: pooled[k] for k in ("escalation_rate", "quality")}
            | {"gain_captured": _captured(pooled["quality"], local, oracle)},
            "pooled_at_matched_rate": {k: matched[k] for k in ("escalation_rate", "quality")},
            "per_task_minus_pooled_20pct": paired_bootstrap(frame, mine, pooled_escalations(frame, BUDGET)),
            "per_task_minus_pooled_matched": paired_bootstrap(
                frame, mine, pooled_escalations(frame, per_task["escalation_rate"])),
        }
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  training allocation {counts} · thresholds {result['thresholds']}")
    for population, r in result["populations"].items():
        print(f"  {population}: per-task {r['per_task_thresholds']['quality']} at "
              f"{r['per_task_thresholds']['escalation_rate']} · pooled at 20% {r['pooled_threshold_at_20pct']['quality']}"
              f" · pooled at matched rate {r['pooled_at_matched_rate']['quality']} · never {r['never_escalate_quality']}")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
