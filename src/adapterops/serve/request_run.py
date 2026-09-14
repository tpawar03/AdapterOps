"""Put the recorded routing pairs through the live request path (PRD §7, §8, §11; F17, F25).

The operating curve was computed offline: every pair was generated once, scored, and each policy's
decision was replayed on the recorded log-probabilities. This sends the same held-out (ticket, task)
pairs — every decision a published curve counted — to a running `adapterops serve-api` over HTTP,
and records what the live path did:

- **how often it escalated and fell back**, overall and per task, against the curve's 20% budget;
- **whether it made the recorded decision** on each pair (confidence policy): the live threshold
  applied to the recorded vLLM score, and how far the live score sits from that recorded one;
- **what the served answers scored** against gold, on the three exactly graded tasks, beside the
  quality the same pairs record at the same threshold;
- **latency, throughput and cost** at each client concurrency, measured at the client.

Three ways to run it:

- **one pass per concurrency** (default) — every pair once;
- **for a duration** (`--duration`) — workers cycle through the pairs until the time is up, and the
  run is also summarised per window, so drift over a sustained load shows as a row that moves;
- **straight at vLLM** (`--direct-vllm`) — no API, no routing, no frontier: the same pairs against
  the served adapters, so a saturation sweep can separate vLLM's ceiling from the request path's.
  Each pair still records the decision the live threshold would make, without acting on it.

**Two populations.** `router_in_distribution` (392 pairs) and `router_shift` (1,047) are the two
the curve was measured on. The live threshold is the in-distribution curve's; on the shifted
population the curve's own 20% budget draws a slightly different line (202 vs 209 escalations on the
recorded scores), so agreement is measured against the threshold and the budget is reported beside it.

**Drafting is not graded here.** The curve's drafting success is a GPT-4o grade of 4 or more on both
sides; re-grading live replies would be a separate paid run. Drafting is routed, timed and costed,
and a locally answered draft carries the distilled judge's score when the server has the judge.

Ticket text is not written to the output; pairs are identified by `pair_id`. A duration run also
drops the answers, which would otherwise run to tens of megabytes.

    uv run adapterops request-path-run --url http://127.0.0.1:8080 --name laptop --concurrency 1
    uv run adapterops request-path-run --name sustained --concurrency 32 --duration 900
    uv run adapterops request-path-run --direct-vllm http://localhost:8000 --name sweep \\
        --concurrency 16 32 64 128 256 --duration 60
"""

from __future__ import annotations

import json
import threading
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
POPULATIONS = ("router_in_distribution", "router_shift")
GRADED = ("intent", "urgency", "pii")
DIRECT = "direct_vllm"
Send = Callable[[dict], dict]


def request_set(population: str = POPULATION) -> pd.DataFrame:
    """A curve population's pairs, each with the confidence decision the curve made at its budget."""
    from adapterops.router.baselines import escalation_scores
    from adapterops.router.misroute import frames, tag

    if population not in POPULATIONS:
        msg = f"unknown population {population!r}; expected one of {POPULATIONS}"
        raise ValueError(msg)
    frame = frames()[population]
    tagged = tag(frame, escalation_scores(frame, "confidence"))
    keep = ["pair_id", "task", "text", "gold", "mean_logprob", "success", "frontier_success",
            "escalated"]
    return tagged[keep].rename(columns={"mean_logprob": "recorded_mean_logprob",
                                        "success": "recorded_local_success",
                                        "frontier_success": "recorded_frontier_success",
                                        "escalated": "recorded_escalated"}).reset_index(drop=True)


def at_threshold(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The decision the live threshold makes on each pair's recorded vLLM score."""
    return pairs.assign(recorded_escalated_at_threshold=-pairs.recorded_mean_logprob >= threshold)


def http_sender(url: str, timeout: float = 300.0, pool: int = 64) -> Send:
    """One session with a connection pool as large as the client's concurrency. The default pool
    holds 10, so a sweep past that would spend its time opening connections."""
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=pool)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def send(body: dict) -> dict:
        r = session.post(f"{url}/v1/tickets", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return send


def vllm_sender(base_url: str, threshold: float, timeout: float = 300.0) -> Send:
    """The served adapters with nothing in front: no API, routing or frontier. A pair past the
    threshold is marked escalated — the decision the request path would make — but answered locally,
    and output the task cannot use is not detected as a fallback."""
    from adapterops.serve.pipeline import VLLMBackend, parse_output

    backend = VLLMBackend(base_url, timeout=timeout)

    def send(body: dict) -> dict:
        task = body["tasks"][0]
        g = backend.generate(task, body["text"])
        score = None if g.mean_logprob is None else -g.mean_logprob
        escalate = score is not None and score >= threshold
        return {"pairs": {task: {
            "route": "escalated" if escalate else "local", "served_by": "local",
            "score": score, "threshold": threshold, "output": parse_output(task, g.text),
            "cost_usd": 0.0, "latency_ms": round(g.latency_s * 1000, 1), "local_error": None,
            "frontier_error": None, "judge_score": None}}}

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


def drive(pairs: pd.DataFrame, send: Send, concurrency: int, duration: float | None = None,
          keep_output: bool = True) -> tuple[pd.DataFrame, float]:
    """Every pair as a one-task ticket, `concurrency` at a time — once, or cycling until `duration`
    seconds have passed. Request failures are rows, not aborts: an HTTP error is the service failing
    a request, which is what this is measuring. Requests started before the deadline all finish."""
    rows = list(pairs.itertuples(index=False))
    started = time.perf_counter()

    def one(row) -> dict:
        body = {"text": row.text, "tasks": [row.task], "ticket_id": row.pair_id}
        t0 = time.perf_counter()
        try:
            response, error = send(body), None
        except Exception as exc:                          # noqa: BLE001 - recorded per pair
            response, error = None, f"{type(exc).__name__}: {exc}"
        client_ms = (time.perf_counter() - t0) * 1000
        pair = (response or {}).get("pairs", {}).get(row.task, {})
        record = {
            "pair_id": row.pair_id, "task": row.task, "concurrency": concurrency,
            "t_start_s": round(t0 - started, 3),
            "http_error": error, "client_ms": round(client_ms, 1),
            "route": pair.get("route"), "served_by": pair.get("served_by"),
            "score": pair.get("score"), "threshold": pair.get("threshold"),
            "served_success": (served_success(row.task, row.text, row.gold, pair)
                               if response else False),
            "judge_score": pair.get("judge_score"), "cost_usd": pair.get("cost_usd", 0.0),
            "server_ms": pair.get("latency_ms"), "local_error": pair.get("local_error"),
            "frontier_error": pair.get("frontier_error"),
        }
        if keep_output:
            record["output"] = (json.dumps(pair.get("output"))
                                if pair.get("output") is not None else None)
        return record

    if duration is None:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            out = list(pool.map(one, rows))
    else:
        deadline = started + duration
        lock = threading.Lock()
        position = [0]
        out = []

        def worker() -> None:
            mine = []
            while time.perf_counter() < deadline:
                with lock:
                    row = rows[position[0] % len(rows)]
                    position[0] += 1
                mine.append(one(row))
            with lock:
                out.extend(mine)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return pd.DataFrame(out), time.perf_counter() - started


def _p(values: Sequence[float], q: float) -> float | None:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return round(float(np.percentile(values, q)), 1) if values else None


def _rate(mask: pd.Series) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def windows(live: pd.DataFrame, width: float, duration: float) -> list[dict]:
    """Rates per window of request start time, so drift across a sustained run is visible. A start
    recorded at the deadline itself — it passed the check a moment before, and starts are rounded to
    the millisecond — belongs to the last window, so every request is counted once."""
    rows = []
    starts = (live.t_start_s.clip(lower=0, upper=duration - 1e-9) // width).astype(int)
    for w, g in live.groupby(starts, sort=True):
        ok = g[g.http_error.isna()]
        span = min(width, duration - w * width)
        rows.append({"start_s": round(w * width, 1), "requests": len(g),
                     "throughput_per_s": round(len(g) / span, 2),
                     "escalation_rate": _rate(ok.route == "escalated"),
                     "fallback_rate": _rate(ok.route == "fallback"),
                     "http_errors": int(g.http_error.notna().sum()),
                     "client_p50_ms": _p(g.client_ms, 50), "client_p95_ms": _p(g.client_ms, 95)})
    return rows


def summarise(pairs: pd.DataFrame, live: pd.DataFrame, wall_s: float, policy: str,
              duration: float | None = None, window: float = 60.0) -> dict:
    """One concurrency level: rates, agreement with the recorded decisions, graded quality, latency,
    cost — and, for a duration run, the same rates per window."""
    if "recorded_escalated_at_threshold" not in pairs:
        pairs = pairs.assign(recorded_escalated_at_threshold=pairs.recorded_escalated)
    joined = pairs.drop(columns=["text", "gold"]).merge(live, on=["pair_id", "task"])
    answered = joined[joined.http_error.isna()]
    out: dict = {
        "concurrency": int(live.concurrency.iloc[0]),
        "requests": len(joined),
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
    if duration is not None:
        out["duration_s"] = duration
        out["passes"] = round(len(joined) / len(pairs), 2)
        out["window_s"] = window
        out["windows"] = windows(live, window, duration)

    graded = joined[joined.task.isin(GRADED)]
    recorded_quality = np.where(graded.recorded_escalated, graded.recorded_frontier_success,
                                graded.recorded_local_success)
    threshold_quality = np.where(graded.recorded_escalated_at_threshold,
                                 graded.recorded_frontier_success, graded.recorded_local_success)
    out["graded_tasks"] = {
        "tasks": list(GRADED),
        "pairs": len(graded),
        "served_success": _rate(graded.served_success.astype(bool)),
        "curve_success_at_operating_point": _rate(pd.Series(recorded_quality, dtype=bool)),
        "recorded_success_at_threshold": _rate(pd.Series(threshold_quality, dtype=bool)),
        "curve_local_only_success": _rate(graded.recorded_local_success.astype(bool)),
    }

    if policy in ("confidence", DIRECT):
        scored = answered[answered.score.notna()]
        live_escalated = scored.route == "escalated"
        recorded = scored.recorded_escalated_at_threshold.astype(bool)
        out["agreement_with_curve"] = {
            "pairs_compared": len(scored),
            "same_decision": _rate(live_escalated == recorded),
            "escalated_live_not_on_curve": int((live_escalated & ~recorded).sum()),
            "escalated_on_curve_not_live": int((~live_escalated & recorded).sum()),
            "score_spearman_vs_recorded": (
                round(float(scored.score.rank().corr((-scored.recorded_mean_logprob).rank())), 4)
                if len(scored) > 2 else None),
            "score_median_abs_diff": (
                round(float((scored.score + scored.recorded_mean_logprob).abs().median()), 4)
                if len(scored) else None),
            "note": ("the recorded decision is the live threshold applied to vLLM's recorded "
                     "-mean_logprob; fallbacks carry no score and are left out"),
        }

    for task, t in joined.groupby("task", sort=True):
        ok = t[t.http_error.isna()]
        row = {"pairs": len(t),
               "escalation_rate": _rate(ok.route == "escalated"),
               "curve_escalation_rate": _rate(t.recorded_escalated.astype(bool)),
               "threshold_escalation_rate": _rate(t.recorded_escalated_at_threshold.astype(bool)),
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
         send: Send | None = None, population: str = POPULATION, duration: float | None = None,
         window: float = 60.0, direct_vllm: str | None = None) -> int:
    import requests

    from adapterops.serve.pipeline import operating_threshold

    pairs = request_set(population)
    if limit:
        pairs = pairs.groupby("task", group_keys=False).head(limit).reset_index(drop=True)

    service = metrics_before = None
    if direct_vllm:
        service = {"policy": DIRECT, "local_backend": "vllm", "base_url": direct_vllm,
                   "threshold": operating_threshold("confidence"), "frontier": None,
                   "note": "no API, routing or frontier; decisions recorded, not acted on"}
    elif url:
        service = requests.get(f"{url}/v1/service", timeout=30).json()
        metrics_before = requests.get(f"{url}/v1/metrics", timeout=30).json()
    policy = (service or {}).get("policy", "confidence")
    threshold = (service or {}).get("threshold") or operating_threshold("confidence")
    pairs = at_threshold(pairs, threshold)
    if send is None:
        send = (vllm_sender(direct_vllm, threshold) if direct_vllm
                else http_sender(url, pool=max(concurrency)))

    levels, frames = [], []
    for c in concurrency:
        live, wall = drive(pairs, send, c, duration=duration, keep_output=duration is None)
        frames.append(live)
        level = summarise(pairs, live, wall, policy, duration=duration, window=window)
        levels.append(level)
        g = level["graded_tasks"]
        spread = ""
        if duration is not None and level["windows"]:
            tp = [w["throughput_per_s"] for w in level["windows"]]
            spread = f" · windows {min(tp)}–{max(tp)}/s"
        print(f"  c={c:<3} {level['requests']} requests in {level['wall_seconds']}s "
              f"({level['throughput_pairs_per_s']}/s{spread}) · escalated {level['escalation_rate']} · "
              f"fallback {level['fallback_rate']} · errors {level['http_errors']} · "
              f"p95 {level['client_p95_ms']} ms · served {g['served_success']} vs recorded "
              f"{g['recorded_success_at_threshold']} · ${level['cost_usd']}")

    result = {
        "name": name,
        "population": population,
        "pairs": len(pairs),
        "mode": DIRECT if direct_vllm else "request_path",
        "duration_s": duration,
        "service": service,
        "levels": levels,
        "server_metrics": (requests.get(f"{url}/v1/metrics", timeout=30).json()
                           if url and not direct_vllm else None),
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
