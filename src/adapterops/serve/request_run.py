"""Put the recorded routing pairs through the live request path (PRD §8, §11; F17, F25).

The operating curve was computed offline: every pair was generated once, scored, and each policy's
decision was replayed on the recorded log-probabilities. This sends the same 392 held-out
(ticket, task) pairs — the router's in-distribution eval split, every decision the published curve
counted — to a running `adapterops serve-api` over HTTP, and records what the live path did:

- **how often it escalated and fell back**, overall and per task, against the curve's 20% budget;
- **whether it made the curve's decision** on each pair (confidence policy), and how far the live
  confidence score sits from the recorded vLLM one;
- **what the served answers scored** against gold, on the three exactly graded tasks, beside the
  quality the curve recorded for the same pairs at the same operating point;
- **latency, throughput and cost** at each client concurrency, measured at the client.

**Drafting is not graded here.** The curve's drafting success is a GPT-4o grade of 4 or more on both
sides; re-grading live replies would be a separate paid run. Drafting is routed, timed and costed,
and a locally answered draft carries the distilled judge's score.

**The curve's decisions are a threshold on a ranking, the live path's are a fixed threshold.** At
the 20% budget they coincide on the recorded scores, so a live disagreement means the live score
moved — the transformers backend's precision, or a vLLM version change — not a different policy.

Ticket text is not written to the output; pairs are identified by `pair_id`.

    uv run adapterops request-path-run --url http://127.0.0.1:8080 --name laptop --concurrency 1
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.router.scoring import score_pair

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
POPULATION = "router_in_distribution"
GRADED = ("intent", "urgency", "pii")
Send = Callable[[dict], dict]


def request_set() -> pd.DataFrame:
    """The curve's in-distribution pairs, each with the confidence decision the curve made."""
    from adapterops.router.baselines import escalation_scores
    from adapterops.router.misroute import frames, tag

    frame = frames()[POPULATION]
    tagged = tag(frame, escalation_scores(frame, "confidence"))
    keep = ["pair_id", "task", "text", "gold", "mean_logprob", "success", "frontier_success",
            "escalated"]
    return tagged[keep].rename(columns={"mean_logprob": "recorded_mean_logprob",
                                        "success": "recorded_local_success",
                                        "frontier_success": "recorded_frontier_success",
                                        "escalated": "recorded_escalated"}).reset_index(drop=True)


def http_sender(url: str, timeout: float = 300.0) -> Send:
    import requests

    session = requests.Session()

    def send(body: dict) -> dict:
        r = session.post(f"{url}/v1/tickets", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return send


def served_success(task: str, text: str, gold: str, pair: dict) -> bool | None:
    """Whether the answer the path served is right, by the curve's own rule. None for drafting."""
    if task not in GRADED:
        return None
    output = pair.get("output")
    if not output:
        return False
    if task == "pii":
        prediction = "\n".join(f"{s['label']}: {s['value']}" for s in output["spans"])
        return score_pair(task, text, gold, prediction)["span_f1"] >= 1.0
    return score_pair(task, text, gold, output["label"])["exact"] == 1.0


def drive(pairs: pd.DataFrame, send: Send, concurrency: int) -> tuple[pd.DataFrame, float]:
    """Every pair as a one-task ticket, `concurrency` at a time. Request failures are rows, not
    aborts: an HTTP error is the service failing a request, which is what this is measuring."""

    def one(row) -> dict:
        body = {"text": row.text, "tasks": [row.task], "ticket_id": row.pair_id}
        started = time.perf_counter()
        try:
            response, error = send(body), None
        except Exception as exc:                          # noqa: BLE001 - recorded per pair
            response, error = None, f"{type(exc).__name__}: {exc}"
        client_ms = (time.perf_counter() - started) * 1000
        pair = (response or {}).get("pairs", {}).get(row.task, {})
        return {
            "pair_id": row.pair_id, "task": row.task, "concurrency": concurrency,
            "http_error": error, "client_ms": round(client_ms, 1),
            "route": pair.get("route"), "served_by": pair.get("served_by"),
            "score": pair.get("score"), "threshold": pair.get("threshold"),
            "served_success": (served_success(row.task, row.text, row.gold, pair)
                               if response else False),
            "judge_score": pair.get("judge_score"), "cost_usd": pair.get("cost_usd", 0.0),
            "server_ms": pair.get("latency_ms"), "local_error": pair.get("local_error"),
            "frontier_error": pair.get("frontier_error"),
            "output": json.dumps(pair.get("output")) if pair.get("output") is not None else None,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, pairs.itertuples(index=False)))
    return pd.DataFrame(rows), time.perf_counter() - started


def _p(values: Sequence[float], q: float) -> float | None:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return round(float(np.percentile(values, q)), 1) if values else None


def _rate(mask: pd.Series) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def summarise(pairs: pd.DataFrame, live: pd.DataFrame, wall_s: float, policy: str) -> dict:
    """One concurrency level: rates, agreement with the curve, graded quality, latency, cost."""
    joined = pairs.drop(columns=["text", "gold"]).merge(live, on=["pair_id", "task"])
    answered = joined[joined.http_error.isna()]
    out: dict = {
        "concurrency": int(live.concurrency.iloc[0]),
        "pairs": len(joined),
        "http_errors": int(joined.http_error.notna().sum()),
        "wall_seconds": round(wall_s, 1),
        "throughput_pairs_per_s": round(len(joined) / wall_s, 2) if wall_s else None,
        "escalation_rate": _rate(answered.route == "escalated"),
        "fallback_rate": _rate(answered.route == "fallback"),
        "unanswered": int((answered.served_by == "none").sum()),
        "frontier_errors": int(answered.frontier_error.notna().sum()),
        "client_p50_ms": _p(joined.client_ms, 50),
        "client_p95_ms": _p(joined.client_ms, 95),
        "cost_usd": round(float(joined.cost_usd.sum()), 6),
        "cost_per_1k_pairs_usd": (round(float(joined.cost_usd.sum()) / len(joined) * 1000, 4)
                                  if len(joined) else None),
        "per_task": {},
    }

    graded = joined[joined.task.isin(GRADED)]
    recorded_quality = np.where(graded.recorded_escalated, graded.recorded_frontier_success,
                                graded.recorded_local_success)
    out["graded_tasks"] = {
        "tasks": list(GRADED),
        "pairs": len(graded),
        "served_success": _rate(graded.served_success.astype(bool)),
        "curve_success_at_operating_point": _rate(pd.Series(recorded_quality, dtype=bool)),
        "curve_local_only_success": _rate(graded.recorded_local_success.astype(bool)),
    }

    if policy == "confidence":
        scored = answered[answered.score.notna()]
        live_escalated = scored.route == "escalated"
        out["agreement_with_curve"] = {
            "pairs_compared": len(scored),
            "same_decision": _rate(live_escalated == scored.recorded_escalated.astype(bool)),
            "escalated_live_not_on_curve": int((live_escalated & ~scored.recorded_escalated).sum()),
            "escalated_on_curve_not_live": int((~live_escalated & scored.recorded_escalated).sum()),
            "score_spearman_vs_recorded": (
                round(float(scored.score.rank().corr((-scored.recorded_mean_logprob).rank())), 4)
                if len(scored) > 2 else None),
            "score_median_abs_diff": (
                round(float((scored.score + scored.recorded_mean_logprob).abs().median()), 4)
                if len(scored) else None),
            "note": ("fallbacks carry no score and are left out; the recorded score is vLLM's "
                     "-mean_logprob from the generation run"),
        }

    for task, t in joined.groupby("task", sort=True):
        ok = t[t.http_error.isna()]
        row = {"pairs": len(t),
               "escalation_rate": _rate(ok.route == "escalated"),
               "curve_escalation_rate": _rate(t.recorded_escalated.astype(bool)),
               "fallback_rate": _rate(ok.route == "fallback"),
               "client_p95_ms": _p(t.client_ms, 95),
               "server_p95_ms": _p(t.server_ms, 95)}
        if task in GRADED:
            row["served_success"] = _rate(t.served_success.astype(bool))
            row["curve_local_only_success"] = _rate(t.recorded_local_success.astype(bool))
        else:
            judged = ok.judge_score.dropna()
            row["judge_mean_local"] = round(float(judged.mean()), 3) if len(judged) else None
        out["per_task"][task] = row
    return out


def main(url: str, name: str, concurrency: Sequence[int] = (1,), limit: int | None = None,
         send: Send | None = None) -> int:
    pairs = request_set()
    if limit:
        pairs = pairs.groupby("task", group_keys=False).head(limit).reset_index(drop=True)
    send = send or http_sender(url)

    import requests

    service = metrics_before = None
    if url:
        service = requests.get(f"{url}/v1/service", timeout=30).json()
        metrics_before = requests.get(f"{url}/v1/metrics", timeout=30).json()
    policy = (service or {}).get("policy", "confidence")

    levels, frames = [], []
    for c in concurrency:
        live, wall = drive(pairs, send, c)
        frames.append(live)
        level = summarise(pairs, live, wall, policy)
        levels.append(level)
        g = level["graded_tasks"]
        print(f"  c={c:<3} {level['pairs']} pairs in {level['wall_seconds']}s · "
              f"escalated {level['escalation_rate']} · fallback {level['fallback_rate']} · "
              f"p95 {level['client_p95_ms']} ms · served {g['served_success']} vs curve "
              f"{g['curve_success_at_operating_point']} · ${level['cost_usd']}")

    result = {
        "name": name,
        "population": POPULATION,
        "pairs": len(pairs),
        "service": service,
        "levels": levels,
        "server_metrics": (requests.get(f"{url}/v1/metrics", timeout=30).json() if url else None),
        "server_metrics_before": metrics_before,
        "notes": ("Drafting is routed and costed but not graded (the curve's drafting grade is "
                  "GPT-4o's). Client latency includes HTTP; server latency is the pair's own. "
                  "Throughput is one-task tickets per second at that client concurrency."),
    }
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"request_path__{name}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_parquet(RUNS_DIR / f"request_path__{name}__pairs.parquet")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")
    return 0
