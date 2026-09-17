"""Where PII's and drafting's P95 goes: time to first token against decode time (PRD §7, TODO.md §2).

§7 asks for 500 ms P95 per adapter. Intent and urgency meet it; PII (2,259 ms) and drafting (3,000 ms)
miss it, and both generate far more tokens (80 and 124 on average). A full reply of that length cannot
decode in 500 ms on an A10, so the question is which target is meaningful: if the first token arrives
inside 500 ms, streaming meets §7 for drafting, and PII — whose spans are used only once complete — needs
its own target. This measures the parts instead of guessing them.

Streams the 392 in-distribution pairs straight to vLLM, the serving prompt and token caps unchanged, at
each concurrency, and records per task: time to first token, total time, time per output token and
tokens generated, as p50 / p95. Nothing is routed or graded.

    uv run adapterops latency-profile --base-url http://localhost:8000 --name a10-v11 --concurrency 1 16
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_MS = 500.0


def stream_one(base_url: str, task: str, text: str, timeout: float = 300.0) -> dict:
    import requests

    from adapterops.router.generate import MAX_TOKENS
    from adapterops.train.qlora import PROMPTS

    started = time.perf_counter()
    first = None
    tokens = 0
    with requests.post(f"{base_url}/v1/completions", stream=True, timeout=timeout, json={
            "model": task, "prompt": PROMPTS[task].format(text=text), "max_tokens": MAX_TOKENS[task],
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if first is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                first = time.perf_counter()
    total = time.perf_counter() - started
    ttft = (first or time.perf_counter()) - started
    return {"task": task, "ttft_ms": ttft * 1000, "total_ms": total * 1000, "tokens": tokens,
            "tpot_ms": (total - ttft) * 1000 / (tokens - 1) if tokens > 1 else None}


def summarise(rows: pd.DataFrame) -> dict:
    def pct(s: pd.Series) -> dict:
        s = s.dropna()
        return {"p50": round(float(np.percentile(s, 50)), 1), "p95": round(float(np.percentile(s, 95)), 1)}

    return {task: {"requests": len(g), "ttft_ms": pct(g.ttft_ms), "total_ms": pct(g.total_ms),
                   "tpot_ms": pct(g.tpot_ms), "tokens": pct(g.tokens.astype(float)),
                   "ttft_within_target": round(float((g.ttft_ms <= TARGET_MS).mean()), 4),
                   "total_within_target": round(float((g.total_ms <= TARGET_MS).mean()), 4)}
            for task, g in rows.groupby("task")}


def main(base_url: str, name: str, concurrency: Sequence[int] = (1, 16), limit: int | None = None) -> int:
    import requests

    from adapterops.serve.request_run import request_set

    served = {m["id"] for m in requests.get(f"{base_url}/v1/models", timeout=30).json()["data"]}
    pairs = request_set()
    pairs = pairs[pairs.task.isin(served)]   # urgency is served by TF-IDF, not vLLM
    if limit:
        pairs = pairs.groupby("task", group_keys=False).head(limit)
    levels = []
    for c in concurrency:
        with ThreadPoolExecutor(c) as pool:
            rows = pd.DataFrame(pool.map(lambda r: stream_one(base_url, r.task, r.text),
                                         pairs.itertuples()))
        per_task = summarise(rows)
        levels.append({"concurrency": c, "per_task": per_task})
        for task, t in per_task.items():
            print(f"  c{c:<3} {task:9s} ttft p95 {t['ttft_ms']['p95']:>7} ms · total p95 {t['total_ms']['p95']:>7} ms"
                  f" · tpot p50 {t['tpot_ms']['p50']} ms · tokens p50 {t['tokens']['p50']}")
    out = REPO_ROOT / "runs" / f"latency__{name}.json"
    out.write_text(json.dumps({
        "question": "For each adapter, how much of P95 is time to first token and how much is decoding?",
        "target_ms": TARGET_MS, "base_url": base_url, "pairs": len(pairs),
        "population": "router_in_distribution", "levels": levels,
        "notes": "Streamed straight to vLLM with the serving prompts and token caps; no routing. "
                 "Client-side timing, so it includes HTTP.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return 0
