"""Cost per 1K requests and weight footprint (PRD §11 dashboard row 4; M2's cost/1K) — D41.

**Derived, not measured.** Nothing here makes a request. Every input is already committed:
frontier token counts from the F8 run's cache, local throughput from M1, adapter and base weight
sizes from the Hub at their pinned revisions. The GPU hourly rate is the figure every GPU session
was budgeted at (`PHASE-2-RUN.md`, `PHASE-4-RUN.md`), not an invoice line.

**The headline is a break-even rate, not a savings ratio.** Dividing the two per-1K costs gives a
tidy multiple, but it silently assumes the GPU is busy every second it is rented. A rented GPU
bills whether or not requests arrive, so local serving is cheaper only while sustained load stays
above the rate at which the hour's rent buys the same number of frontier calls. That rate is the
claim that survives; the multiple is recorded beside it with the assumption spelled out.

What is not claimed, because it was never measured: GPU memory at serving time, the A10's maximum
throughput (M1 was a 246-second burst at concurrency 16, not a saturation test), and frontier
latency (the F8 run recorded tokens, not timings).

    uv run adapterops economics
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_FILE = REPO_ROOT / "data" / "router" / "frontier_cache.jsonl"
SERVING_RUN = REPO_ROOT / "runs" / "m1_serving.json"
PINS_FILE = REPO_ROOT / "manifests" / "adapters.json"
OUT_FILE = REPO_ROOT / "runs" / "economics.json"

GPU_USD_PER_HOUR = 0.75
"""A10-class rate the Phase 2 and Phase 4 sessions were budgeted at. Consistent with Gate 0.5's
~$0.44 for ~35 minutes; still a budgeting figure, so it is recorded as an assumption."""

BASE_WEIGHT_FILE = "model.safetensors"


def frontier_usd(prompt_tokens: int, completion_tokens: int,
                 price_per_1m: Mapping[str, float]) -> float:
    return (prompt_tokens * price_per_1m["input"]
            + completion_tokens * price_per_1m["output"]) / 1_000_000


def local_usd_per_1k(gpu_usd_per_hour: float, throughput_rps: float) -> float:
    """What 1,000 requests cost on a GPU kept busy at `throughput_rps` — a lower bound in practice,
    since idle rented time adds cost and no requests."""
    return gpu_usd_per_hour / 3600 / throughput_rps * 1000


def break_even_rps(gpu_usd_per_hour: float, frontier_usd_per_1k: float) -> float:
    """Sustained request rate above which renting the GPU costs less than calling the frontier."""
    return gpu_usd_per_hour / 3600 / (frontier_usd_per_1k / 1000)


def frontier_costs(rows: Iterable[Mapping], price_per_1m: Mapping[str, float]) -> dict:
    """Per-task and pooled cost from cached frontier calls. The last row per pair wins, so a pair
    called twice (the concurrent-run incident in COST-LOG) counts once — this prices the workload,
    not the bill."""
    latest = {r["pair_id"]: r for r in rows if r.get("error") is None}
    by_task: dict[str, dict] = {}
    for pair_id, r in latest.items():
        t = by_task.setdefault(pair_id.split("-")[0],
                               {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        t["requests"] += 1
        t["prompt_tokens"] += int(r["prompt_tokens"])
        t["completion_tokens"] += int(r["completion_tokens"])

    def priced(t: dict) -> dict:
        usd = frontier_usd(t["prompt_tokens"], t["completion_tokens"], price_per_1m)
        return {**t, "usd": round(usd, 4), "usd_per_1k": round(usd / t["requests"] * 1000, 4)}

    pooled = {k: sum(t[k] for t in by_task.values())
              for k in ("requests", "prompt_tokens", "completion_tokens")}
    return {"per_task": {k: priced(v) for k, v in sorted(by_task.items())},
            "pooled": priced(pooled) if pooled["requests"] else None}


def footprint(base_bytes: int, adapter_bytes: Mapping[str, int]) -> dict:
    """Weights on disk: one base plus adapters, against one full fine-tuned copy per task at the
    base's own precision. Disk bytes, not GPU memory — serving memory was never measured."""
    shared = base_bytes + sum(adapter_bytes.values())
    separate = base_bytes * len(adapter_bytes)
    return {
        "base_bytes": base_bytes,
        "adapter_bytes": dict(adapter_bytes),
        "one_base_plus_adapters_bytes": shared,
        "full_copy_per_task_bytes": separate,
        "ratio": round(separate / shared, 2),
        "note": "safetensors file sizes at the pinned revisions; not GPU memory",
    }


def derive(frontier_rows: Iterable[Mapping], serving: Mapping, price_per_1m: Mapping[str, float],
           base_bytes: int, adapter_bytes: Mapping[str, int],
           gpu_usd_per_hour: float = GPU_USD_PER_HOUR) -> dict:
    frontier = frontier_costs(frontier_rows, price_per_1m)
    pooled = frontier["pooled"]
    rps = float(serving["throughput_rps"])
    local = local_usd_per_1k(gpu_usd_per_hour, rps)
    return {
        "purpose": "Derived cost per 1K requests and weight footprint (D41). No request was made.",
        "assumptions": {
            "gpu_usd_per_hour": gpu_usd_per_hour,
            "gpu_rate_source": "the A10 rate the GPU sessions were budgeted at — not an invoice",
            "local_throughput_rps": rps,
            "local_throughput_source": "runs/m1_serving.json — a 246 s burst at concurrency 16 "
                                       "over the same 5,800 pairs, not a saturation test",
            "frontier_price_per_1m": dict(price_per_1m),
            "frontier_source": "data/router/frontier_cache.jsonl — token counts of the F8 run, "
                               "one row per pair, priced at list",
        },
        "frontier": frontier,
        "local": {
            "usd_per_1k_at_m1_throughput": round(local, 4),
            "per_task": None,
            "per_task_note": "M1 measured pooled throughput only; per-task GPU cost is not "
                             "separable from it",
        },
        "same_workload": {
            "pairs": pooled["requests"],
            "frontier_usd_per_1k": pooled["usd_per_1k"],
            "local_usd_per_1k": round(local, 4),
            "ratio_if_gpu_fully_busy": round(pooled["usd_per_1k"] / local, 1),
            "break_even_rps": round(break_even_rps(gpu_usd_per_hour, pooled["usd_per_1k"]), 2),
            "break_even_requests_per_hour": round(
                break_even_rps(gpu_usd_per_hour, pooled["usd_per_1k"]) * 3600),
            "reading": "local is cheaper only while sustained load exceeds break_even_rps; "
                       "the ratio holds only at M1's throughput with no idle time",
            "not_included": ["adapter training GPU time", "idle rented time",
                             "the quality difference — GPT-4o-mini scores lower on these tasks"],
        },
        "footprint": footprint(base_bytes, adapter_bytes),
        "not_measured": {
            "gpu_memory_at_serving": "no nvidia-smi or vLLM memory figure was recorded",
            "frontier_latency": "the F8 run recorded tokens, not timings",
            "max_local_throughput": "no saturation sweep was run",
        },
    }


def hub_weight_sizes(pins_file: Path = PINS_FILE) -> tuple[int, dict[str, int]]:
    """Weight file sizes at each pinned revision — a metadata read, nothing downloaded."""
    from huggingface_hub import HfApi

    api = HfApi()
    components = json.loads(pins_file.read_text())["components"]

    def size(repo: str, revision: str, filename: str) -> int:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        return next(s.size for s in info.siblings if s.rfilename == filename)

    base = components["base_model"]
    base_bytes = size(base["repo"], base["revision"], BASE_WEIGHT_FILE)
    adapters = {task: size(e["repo"], e["revision"], e["weight_file"])
                for task, e in components.items() if task != "base_model" and e}
    return base_bytes, adapters


def main() -> int:
    from adapterops.router.frontier import PRICE_PER_1M

    rows = [json.loads(line) for line in CACHE_FILE.read_text().splitlines() if line.strip()]
    serving = json.loads(SERVING_RUN.read_text())
    base_bytes, adapter_bytes = hub_weight_sizes()
    out = derive(rows, serving, PRICE_PER_1M, base_bytes, adapter_bytes)
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")

    w = out["same_workload"]
    f = out["footprint"]
    print(f"frontier  ${w['frontier_usd_per_1k']:.4f} / 1K  ({w['pairs']} pairs, list price)")
    print(f"local     ${w['local_usd_per_1k']:.4f} / 1K  at {serving['throughput_rps']} rps, "
          f"${GPU_USD_PER_HOUR}/h assumed")
    print(f"break-even {w['break_even_rps']} rps sustained "
          f"({w['break_even_requests_per_hour']:,} requests/hour)")
    print(f"weights   {f['one_base_plus_adapters_bytes'] / 1e9:.2f} GB shared vs "
          f"{f['full_copy_per_task_bytes'] / 1e9:.2f} GB as full copies ({f['ratio']}x)")
    print(f"wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0
