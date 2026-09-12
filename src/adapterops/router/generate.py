"""Run the router pool through the four served adapters (F7) — the paid half of Phase 2.

One GPU session, three deliverables, because the marginal cost of the second and third is
zero once the box is up:

1. **M1** — all four adapters serving concurrently, with evidence that they are four
   adapters and not one name resolved four times.
2. **The per-adapter latency benchmark** Phase 1 still owes, measured under the same
   interleaved load rather than task by task.
3. **The scored pool** the router trains on, with per-pair log-probabilities so the
   confidence baseline (F9) costs nothing extra.

**The distinctness probe runs first and can abort the session.** Gate 0.5 found that the
one failure vLLM does not report is silently resolving several adapter names to the same
weights: throughput, latency and error rate all look healthy and every downstream number
is meaningless. There the discriminator was two adapters of different quality disagreeing.
Here it is sharper — the four adapters were tuned on four different output formats, so the
*same* prompt sent to all four names must produce four different answers. If any two agree
on every probe prompt, the run stops before it spends anything.

**Requests are interleaved across tasks, not batched by task.** Running 750 intent pairs
then 750 PII pairs would keep one adapter resident at a time and prove nothing about
concurrency — and the latency it measured would be the latency of a single-adapter server.

Usage, on the box with `scripts/phase2_serve.sh` running in another shell:
    uv run python -m adapterops.router.generate --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from adapterops.router.scoring import label, score_pair, success_rates
from adapterops.train.qlora import PROMPTS

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_FILE = REPO_ROOT / "data" / "router" / "pool.parquet"
SCORED_FILE = REPO_ROOT / "data" / "router" / "scored.parquet"
SERVING_RUN = REPO_ROOT / "runs" / "m1_serving.json"

MAX_TOKENS = {"intent": 12, "urgency": 6, "pii": 192, "drafting": 400}
"""Sized from the pool's own gold lengths: PII targets reach 742 characters and drafting
references 2,147, so a shared cap would truncate drafting mid-reply and score the
truncation as a quality failure."""

PROBE_PROMPT = "My card payment was declined at the supermarket this morning."


def complete(base_url: str, adapter: str, prompt: str, max_tokens: int,
             timeout: float = 180.0) -> dict:
    """One completion, with the log-probabilities the confidence baseline needs."""
    import requests

    started = time.perf_counter()
    r = requests.post(
        f"{base_url}/v1/completions",
        json={"model": adapter, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0.0, "logprobs": 1},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    r.raise_for_status()
    choice = r.json()["choices"][0]
    logprobs = [lp for lp in (choice.get("logprobs") or {}).get("token_logprobs") or []
                if lp is not None]
    return {
        "prediction": choice["text"],
        "latency_s": elapsed,
        "n_tokens": len(logprobs),
        "mean_logprob": statistics.fmean(logprobs) if logprobs else None,
        "min_logprob": min(logprobs) if logprobs else None,
        "finish_reason": choice.get("finish_reason"),
    }


def distinctness_probe(base_url: str, adapters: list[str], n: int = 8) -> dict:
    """Four names, one prompt. Four different answers, or the session is not worth paying for.

    Deliberately not an equality check on a single prompt: two adapters could agree once by
    chance. A pair is only reported as suspect if it agrees on *every* probe prompt.
    """
    pool = pd.read_parquet(POOL_FILE)
    texts = [PROBE_PROMPT, *pool[pool.task == "intent"].text.head(n - 1).tolist()]
    answers = {
        a: [complete(base_url, a, PROMPTS["intent"].format(text=t), 16)["prediction"].strip()
            for t in texts]
        for a in adapters
    }
    identical = [
        (a, b) for i, a in enumerate(adapters) for b in adapters[i + 1:]
        if answers[a] == answers[b]
    ]
    return {
        "probe_prompts": len(texts),
        "identical_pairs": [list(p) for p in identical],
        "distinct": not identical,
        "sample": {a: v[0] for a, v in answers.items()},
    }


def run(base_url: str, concurrency: int, limit: int | None = None) -> pd.DataFrame:
    pool = pd.read_parquet(POOL_FILE)
    if limit:
        pool = pool.groupby("task", group_keys=False).head(limit)
    # Interleave: task-by-task order would keep one adapter resident and measure a
    # single-adapter server while calling the result concurrent multi-LoRA.
    pool = pool.sort_values("pair_id", key=lambda s: s.str.slice(-4)).reset_index(drop=True)

    def one(row: pd.Series) -> dict:
        prompt = PROMPTS[row.task].format(text=row.text)
        try:
            out = complete(base_url, row.task, prompt, MAX_TOKENS[row.task])
            out["error"] = None
        except Exception as exc:                       # noqa: BLE001 - a fallback, counted
            out = {"prediction": "", "latency_s": None, "n_tokens": 0, "mean_logprob": None,
                   "min_logprob": None, "finish_reason": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        return {"pair_id": row.pair_id, **out}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool_exec:
        results = list(pool_exec.map(one, (r for _, r in pool.iterrows())))
    wall = time.perf_counter() - started

    frame = pool.merge(pd.DataFrame(results), on="pair_id")
    frame.attrs["wall_seconds"] = wall
    return frame


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-task score components. Pure CPU — re-runnable without the GPU.

    A failed request scores as a failure rather than being dropped: a pair the local path
    could not answer is exactly a pair that should have been escalated, and dropping it
    would hide the fallback rate PRD §11 asks to monitor.
    """
    frame = frame.reset_index(drop=True)
    components = pd.DataFrame([
        score_pair(r.task, r.text, r.gold, r.prediction) for _, r in frame.iterrows()
    ])
    return pd.concat([frame, components], axis=1)


def summarise(frame: pd.DataFrame, probe: dict, concurrency: int) -> dict:
    wall = frame.attrs["wall_seconds"]
    ok = frame[frame.error.isna()]
    per_adapter = {}
    for task, g in ok.groupby("task"):
        lat = sorted(g.latency_s)
        per_adapter[task] = {
            "requests": len(g),
            "p50_ms": round(statistics.median(lat) * 1000, 1),
            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))] * 1000, 1),
            "max_new_tokens": MAX_TOKENS[task],
            "mean_generated_tokens": round(float(g.n_tokens.mean()), 1),
            "truncated": int((g.finish_reason == "length").sum()),
        }
    return {
        "milestone": "M1 — four adapters served concurrently on one base",
        "host": platform.platform(),
        "concurrency": concurrency,
        "requests": len(frame),
        "errors": int(frame.error.notna().sum()),
        "fallback_rate": round(float(frame.error.notna().mean()), 4),
        "wall_seconds": round(wall, 1),
        "throughput_rps": round(len(frame) / wall, 1),
        "adapters_distinct": probe["distinct"],
        "distinctness_probe": probe,
        "per_adapter": per_adapter,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="pairs per task — smoke-test the path before the full 3,000")
    ap.add_argument("--skip-probe", action="store_true",
                    help="not recommended; the probe is what makes the rest meaningful")
    args = ap.parse_args()

    adapters = ["intent", "urgency", "pii", "drafting"]
    probe = {"distinct": None, "skipped": True}
    if not args.skip_probe:
        probe = distinctness_probe(args.base_url, adapters)
        for a, s in probe["sample"].items():
            print(f"  probe {a:9s} -> {s[:60]!r}")
        if not probe["distinct"]:
            print(f"\n  ABORT: these adapter names returned identical answers on every "
                  f"probe prompt: {probe['identical_pairs']}")
            print("  vLLM is serving one set of weights under several names. Every "
                  "number this run would produce is meaningless.")
            return 1
        print("  adapters are distinct\n")

    frame = run(args.base_url, args.concurrency, args.limit)

    scored = label(score_frame(frame))

    SCORED_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_FILE)

    summary = summarise(frame, probe, args.concurrency)
    SERVING_RUN.parent.mkdir(exist_ok=True)
    SERVING_RUN.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(success_rates(scored).to_string())
    print(f"\n  {summary['requests']:,} requests · {summary['errors']} errors · "
          f"{summary['throughput_rps']} req/s · wall {summary['wall_seconds']}s")
    for task, m in summary["per_adapter"].items():
        print(f"    {task:9s} P50 {m['p50_ms']:>7.1f} ms · P95 {m['p95_ms']:>7.1f} ms · "
              f"{m['truncated']} truncated")
    print(f"\n  wrote {SCORED_FILE.relative_to(REPO_ROOT)} and "
          f"{SERVING_RUN.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
