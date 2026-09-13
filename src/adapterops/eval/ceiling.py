"""Frontier ceiling on the golden sets (M2's third column) — GPT-4o-mini, with F8's prompts and caps.

M2 scores each adapter against a prompted base model *and* a frontier ceiling on the same golden
set. The ceiling was only ever measured on the router pool (F8), whose rows are not the golden sets,
so that column was empty. This fills it with F8's model, prompts and token caps, so the ceiling here
and the escalation arm on the operating curve are the same system.

Drafting has no gold. Its replies are graded by GPT-4o through `judge-m2 --sides frontier`, on the
same 300 golden requests the adapter and prompted sides were graded on, so all three share a scale.

**Scored by the rule every other system is scored by** — exact label for intent and urgency, strict
spans for PII. GPT-4o-mini sometimes changes a label's case, which that rule counts as wrong, so a
case-insensitive accuracy is recorded beside it and marked indicative.

**Its own cache.** Golden items never enter `frontier_cache.jsonl`: `economics.py` prices F8's
workload from that file by pair-id prefix, and golden items there would be billed to a task
called `golden`.

    uv run python -m adapterops.eval.ceiling --project        # prices the run, no API call
    uv run python -m adapterops.eval.ceiling --spend-cap 0.25
    uv run adapterops judge-m2 --sides frontier --spend-cap 1.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from adapterops.eval.regression import GATED, TASKS, load_split, score
from adapterops.judge import label
from adapterops.router import frontier as fr

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_FILE = REPO_ROOT / "data" / "router" / "ceiling_cache.jsonl"
PREDICTIONS = REPO_ROOT / "runs" / "frontier__golden__predictions.parquet"
OUT_FILE = REPO_ROOT / "runs" / "frontier__golden.json"
PHASE3_FRONTIER_GRADES = "frontier:"
"""Item-id prefix of Phase 3's GPT-4o grades of GPT-4o-mini drafting replies — the same grading
job the drafting ceiling needs, so its measured tokens price this one."""


def plan() -> pd.DataFrame:
    frames = []
    for task in TASKS:
        texts, gold = load_split(task, "random")
        frames.append(pd.DataFrame({
            "pair_id": [f"golden-{task}-{i:04d}" for i in range(len(texts))],
            "task": task, "split": "random", "text": texts, "gold": gold,
        }))
    return pd.concat(frames, ignore_index=True)


def project(items: pd.DataFrame) -> dict:
    """Priced from recorded tokens: F8's per-task means, scaled up (never down) by how much longer
    golden texts run than the pool's, plus Phase 3's measured tokens per GPT-4o grade."""
    pool = pd.read_parquet(fr.OUT_FILE)[["pair_id", "task", "text"]]
    cache = fr.load_cache()
    pool = pool[pool.pair_id.isin(cache)]
    pool = pool.assign(prompt_tokens=[cache[p]["prompt_tokens"] for p in pool.pair_id],
                       completion_tokens=[cache[p]["completion_tokens"] for p in pool.pair_id])

    mini, total = {}, 0.0
    for task, g in items.groupby("task"):
        f8 = pool[pool.task == task]
        scale = max(1.0, g.text.str.len().mean() / f8.text.str.len().mean())
        usd = len(g) * (f8.prompt_tokens.mean() * scale * fr.PRICE_PER_1M["input"]
                        + f8.completion_tokens.mean() * fr.PRICE_PER_1M["output"]) / 1e6
        mini[task] = {"items": len(g), "length_scale": round(scale, 3), "usd": round(usd, 4)}
        total += usd

    grades = [r for r in label.load_cache().values()
              if r["item_id"].startswith(PHASE3_FRONTIER_GRADES) and r.get("prompt_tokens")]
    if not grades:
        msg = "no Phase 3 frontier grades in the judge cache to price the drafting grading from"
        raise ValueError(msg)
    n_draft = int((items.task == "drafting").sum())
    per_grade = (sum(r["prompt_tokens"] for r in grades) / len(grades) * label.PRICE_PER_1M["input"]
                 + sum(r["completion_tokens"] for r in grades) / len(grades)
                 * label.PRICE_PER_1M["output"]) / 1e6
    grading = n_draft * per_grade * label.TOKEN_MARGIN
    return {
        "gpt4o_mini_calls": {"per_task": mini, "usd": round(total, 4), "requests": len(items)},
        "gpt4o_grading": {"items": n_draft, "usd": round(grading, 4),
                          "priced_from": f"{len(grades):,} Phase 3 grades of GPT-4o-mini replies, "
                                         f"x{label.TOKEN_MARGIN} margin"},
        "total_usd": round(total + grading, 4),
        "total_requests": len(items) + n_draft,
    }


def summarise(items: pd.DataFrame, cache: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    frame = items.assign(
        prediction=[cache[p]["prediction"] for p in items.pair_id],
        prompt_tokens=[cache[p]["prompt_tokens"] for p in items.pair_id],
        completion_tokens=[cache[p]["completion_tokens"] for p in items.pair_id],
    )
    out: dict = {
        "model": fr.MODEL,
        "prompts": "router.frontier.build_prompt — the F8 escalation arm's",
        "max_tokens": fr.MAX_TOKENS,
        "usd_token_derived": round(float(
            (frame.prompt_tokens.sum() * fr.PRICE_PER_1M["input"]
             + frame.completion_tokens.sum() * fr.PRICE_PER_1M["output"]) / 1e6), 4),
        "per_task": {},
    }
    for task, g in frame.groupby("task", sort=False):
        metrics = score(task, g.text.tolist(), g.gold.tolist(), g.prediction.tolist())
        cell = {**metrics, "gated_metric": GATED[task],
                "at_token_cap": round(float((g.completion_tokens >= fr.MAX_TOKENS[task]).mean()), 4)}
        if task in ("intent", "urgency"):
            first = g.prediction.str.strip().str.split("\n").str[0].str.strip().str.casefold()
            cell["micro_accuracy_casefold"] = round(float(
                (first == g.gold.str.strip().str.casefold()).mean()), 4)
            cell["casefold_note"] = "indicative only — every system is gated on the exact label"
        if task == "drafting":
            cell["note"] = "graded by GPT-4o: runs/drafting__m2_gpt4o__frontier.json"
        out["per_task"][task] = cell
    return frame, out


def run(workers: int, spend_cap: float) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("  no OPENAI_API_KEY in the environment or .env")
        return 2

    items = plan()
    with fr.single_run():
        cached = fr.load_cache(CACHE_FILE)
        todo = items[~items.pair_id.isin(cached)]
        print(f"  {len(items):,} golden items · {len(items) - len(todo):,} cached · "
              f"{len(todo):,} to call · spend cap ${spend_cap:.2f}", flush=True)
        ledger = fr.Ledger(cap=spend_cap, request_cap=2 * len(items))
        results = (fr.call_batch((r for _, r in todo.iterrows()), ledger, workers,
                                 fr.intent_labels(), cache_file=CACHE_FILE) if len(todo) else [])
    print(f"  {len(results):,} calls this run · {ledger.requests:,} requests · ${ledger.usd:.4f}")

    cache = fr.load_cache(CACHE_FILE)
    missing = int((~items.pair_id.isin(cache)).sum())
    if ledger.stopped or missing:
        print(f"  {'STOPPED: ' + ledger.reason + '. ' if ledger.stopped else ''}"
              f"{missing} items uncalled — cached results are kept; re-run to resume")
        return 1

    frame, out = summarise(items, cache)
    frame.to_parquet(PREDICTIONS)
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["per_task"], indent=2))
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)} and {PREDICTIONS.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adapterops.eval.ceiling")
    ap.add_argument("--project", action="store_true", help="price the run; no API call")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spend-cap", type=float, default=0.25)
    args = ap.parse_args(argv)
    if args.project:
        print(json.dumps(project(plan()), indent=2))
        return 0
    return run(args.workers, args.spend_cap)


if __name__ == "__main__":
    raise SystemExit(main())
