"""Gate 0.5 probe — does vLLM actually serve two adapters concurrently?

Run against `scripts/gate05_serve.sh`. Answers four questions, in order of how badly a
wrong answer would mislead the project:

1. **Are the adapters distinct?** If vLLM silently resolved both names to one adapter,
   every other measurement would look fine and the gate would read as passed. The trained
   and under-trained adapters differ in quality, so disagreement between them is the
   evidence that routing works. Identical outputs on every prompt is a FAIL, not a pass.
2. **Do both serve under concurrent load?** Requests are interleaved across adapters from
   many threads at once, not run one adapter after the other.
3. **What does it cost in latency?** P50/P95 measured under that concurrency — the
   numbers PRD §7 wants, and the reason they cannot come from a free Colab T4.
4. **Is quality preserved through the server?** The trained adapter should score near its
   recorded 0.9312 offline number; a large drop means the serving path changed behaviour.

Usage:  uv run python -m adapterops.serve.gate05_probe --n 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = (
    "Classify the customer's banking request into one intent label.\n"
    "Request: {text}\n"
    "Intent:"
)
TRAINED = "intent"
UNDERTRAINED = "intent-undertrained"


def complete(base_url: str, adapter: str, text: str) -> tuple[str, float]:
    started = time.perf_counter()
    r = requests.post(
        f"{base_url}/v1/completions",
        json={"model": adapter, "prompt": PROMPT.format(text=text),
              "max_tokens": 12, "temperature": 0.0},
        timeout=120,
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - started
    return r.json()["choices"][0]["text"].strip().split("\n")[0].strip(), elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200, help="golden rows to send per adapter")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="runs/gate05.json")
    args = ap.parse_args()

    served = {m["id"] for m in requests.get(f"{args.base_url}/v1/models", timeout=30).json()["data"]}
    print(f"  server reports models: {sorted(served)}")
    for name in (TRAINED, UNDERTRAINED):
        if name not in served:
            print(f"  FAIL: adapter {name!r} not served")
            return 1

    golden = pd.read_parquet(REPO_ROOT / "evals/golden/intent.parquet").head(args.n)
    jobs = [(a, t) for t in golden.text for a in (TRAINED, UNDERTRAINED)]

    print(f"  sending {len(jobs)} interleaved requests at concurrency {args.concurrency}")
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda j: (j[0], *complete(args.base_url, j[0], j[1])), jobs))
    wall = time.perf_counter() - wall

    by = {TRAINED: [], UNDERTRAINED: []}
    lat = {TRAINED: [], UNDERTRAINED: []}
    for adapter, text, seconds in results:
        by[adapter].append(text)
        lat[adapter].append(seconds)

    gold = golden.label_text.tolist()
    acc = {a: sum(p == g for p, g in zip(by[a], gold, strict=True)) / len(gold) for a in by}
    disagree = sum(a != b for a, b in zip(by[TRAINED], by[UNDERTRAINED], strict=True))

    def pct(xs: list[float], q: float) -> float:
        return round(statistics.quantiles(xs, n=100)[int(q) - 1] * 1000, 1)

    report = {
        "gate": "0.5 — vLLM concurrent multi-LoRA",
        "concurrency": args.concurrency,
        "requests": len(jobs),
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(len(jobs) / max(wall, 1e-6), 2),
        "adapters_distinct": {
            "disagreements": disagree,
            "of": len(gold),
            "rate": round(disagree / len(gold), 4),
            "verdict": "PASS" if disagree > 0 else "FAIL — both names resolved to one adapter",
        },
        "accuracy": {a: round(v, 4) for a, v in acc.items()},
        "latency_ms": {
            a: {"p50": pct(lat[a], 50), "p95": pct(lat[a], 95),
                "mean": round(statistics.mean(lat[a]) * 1000, 1)}
            for a in lat
        },
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n  wrote {args.out}")

    if disagree == 0:
        print("  GATE FAIL: the two adapters produced identical output on every prompt.")
        return 1
    print(f"  trained adapter accuracy {acc[TRAINED]:.4f} "
          f"(recorded offline: 0.9312) · under-trained {acc[UNDERTRAINED]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
