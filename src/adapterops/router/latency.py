"""Router decision latency on CPU — PRD §7's "< 50 ms" budget, measured rather than estimated.

§7 set the budget as an estimate ("44M-param classifier, CPU-viable") and nothing measured it. This
times the router the manifest pins, on the held-out eval pairs it was scored on, on this machine's
CPU.

**A decision is tokenise + forward + softmax** for one (ticket, task) pair — everything between a
pair arriving and an escalate/stay answer. Model loading is timed separately and excluded.

**Two batch shapes, two thread counts.** Batch 1 is one pair; batch 4 is one ticket's four pairs
decided together, which is how §8's request path would call it. Default threads is what an idle
laptop gives; one thread is the honest figure for a small shared server. The budget is checked
against P95 per pair for batch 1, and per ticket for batch 4.

**The timed path is checked against the recorded one.** The pinned checkpoint's sha256 must match
`manifests/system.json`, and its batch-1 probabilities must reproduce the committed `router_p_fail`
column of the run that trained it — so the number is for the router that produced the published
scores, not a stand-in. Whichever router the manifest pins is the one timed (D46).

    uv run adapterops router-latency
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.router.train import to_text

REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM = REPO_ROOT / "manifests" / "system.json"
OUT_FILE = REPO_ROOT / "runs" / "router__latency.json"
SCORED = {
    "checkpoints/router": ("data/router/router_eval_scored.parquet", "runs/router__train.json"),
    "checkpoints/router__judged": ("data/router/router_eval_scored__judged.parquet",
                                   "runs/router__train__judged.json"),
}
"""Each router checkpoint's committed eval predictions, and the run that trained it."""
TARGET_MS = 50.0
WARMUP = 16
SCORE_TOLERANCE = 1e-3


def summarise(ms: list[float], target_ms: float = TARGET_MS) -> dict:
    """Percentiles of one timing series and whether its P95 is inside the budget."""
    a = np.asarray(ms, dtype=float)
    p95 = float(np.percentile(a, 95))
    return {"n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(p95, 2),
            "max_ms": round(float(a.max()), 2), "mean_ms": round(float(a.mean()), 2),
            "target_ms": target_ms, "p95_within_target": p95 < target_ms}


def ticket_batches(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Groups of four pairs, one per task — the shape of one ticket's decisions."""
    by_task = [g.reset_index(drop=True) for _, g in frame.groupby("task")]
    n = min(len(g) for g in by_task)
    return [pd.concat([g.iloc[[i]] for g in by_task], ignore_index=True) for i in range(n)]


def pinned_weights(system: dict) -> dict:
    pins = system["components"]["router"]
    return next(v for k, v in pins.items() if k.endswith("model.safetensors"))


def pinned_sha(system: dict) -> str:
    return pinned_weights(system)["sha256"]


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    system = json.loads(SYSTEM.read_text())
    weights = pinned_weights(system)
    checkpoint = (REPO_ROOT / weights["file"]).parent
    rel = str(checkpoint.relative_to(REPO_ROOT))
    if rel not in SCORED:
        msg = f"no committed eval predictions are registered for {rel}"
        raise KeyError(msg)
    eval_frame, run_file = (REPO_ROOT / p for p in SCORED[rel])

    sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if sha != weights["sha256"]:
        msg = f"{rel} is not the router manifest v{system['version']} pins ({sha[:12]}…)"
        raise ValueError(msg)
    max_length = json.loads(run_file.read_text())["config"]["max_length"]

    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float32)
    model.eval()
    load_seconds = time.perf_counter() - started

    frame = pd.read_parquet(eval_frame)

    def decide(rows: pd.DataFrame) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        enc = tok(to_text(rows), truncation=True, max_length=max_length, padding=True,
                  return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()
        return p, (time.perf_counter() - t0) * 1000

    default_threads = torch.get_num_threads()
    runs: dict = {}
    reproduced = None
    for threads in (default_threads, 1):
        torch.set_num_threads(threads)
        for i in range(WARMUP):
            decide(frame.iloc[[i]])
        p_fail, per_pair = [], []
        for i in range(len(frame)):
            p, ms = decide(frame.iloc[[i]])
            p_fail.append(float(p[0]))
            per_pair.append(ms)
        if reproduced is None:
            gap = float(np.max(np.abs(np.asarray(p_fail) - frame.router_p_fail.to_numpy())))
            if gap > SCORE_TOLERANCE:
                msg = (f"timed router differs from the committed router_p_fail by {gap:.4g} — "
                       f"not the scoring path the published curve used")
                raise ValueError(msg)
            reproduced = gap
        per_ticket = [decide(b)[1] for b in ticket_batches(frame)]
        runs[f"threads_{threads}"] = {"torch_threads": threads,
                                      "batch_1_per_pair": summarise(per_pair),
                                      "batch_4_per_ticket": summarise(per_ticket)}

    out = {
        "prd": "§7 router decision latency < 50 ms",
        "router": {"checkpoint": rel, "sha256": sha,
                   "pinned_in_manifest_version": system["version"],
                   "trained_by": str(run_file.relative_to(REPO_ROOT)), "max_length": max_length},
        "pairs": len(frame),
        "pairs_source": str(eval_frame.relative_to(REPO_ROOT)),
        "decision": "tokenise + forward + softmax; model load excluded",
        "load_seconds": round(load_seconds, 2),
        "score_check": {"max_abs_diff_vs_committed": reproduced, "tolerance": SCORE_TOLERANCE},
        "device": "cpu",
        "host": f"{platform.system()} {platform.machine()} ({platform.processor() or 'unknown'})",
        "runs": runs,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    for name, r in runs.items():
        one, four = r["batch_1_per_pair"], r["batch_4_per_ticket"]
        print(f"  {name}: per pair P50 {one['p50_ms']} ms · P95 {one['p95_ms']} ms · "
              f"per ticket (4 pairs) P95 {four['p95_ms']} ms")
    print(f"  scores reproduce the committed run (max diff {reproduced:.2g})")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
