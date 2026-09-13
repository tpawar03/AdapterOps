"""GPT-4o grades on golden drafting, adapter and prompted side by side — drafting's M2.

The distilled judge put the prompted base model ahead of the drafting adapter (4.70 vs 4.27), but
77% of the prompted replies stop at the token cap and the judge scored those *higher* than complete
ones. So the comparison goes to the teacher: the Phase 3 rubric, model and cache, on the same 300
golden requests for both sides. Phase 3 never graded golden replies — it graded the router and
mining pools — so the adapter side needs grades too, or there is nothing to pair against.

**This never writes `data/judge/judgments.parquet`.** That file is rewritten from its own plan by
`judge-label`; routing these items through it would drop the 2,200 Phase 3 grades. Item ids are
prefixed `m2-<side>:` so they share the cache without colliding.

    uv run adapterops judge-m2 --project
    uv run adapterops judge-m2 --spend-cap 1.0
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.judge import label
from adapterops.router.generate import MAX_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[3]
SIDES = {
    "adapter": "runs/regression__v1-baseline-1__predictions.parquet",
    "prompted": "runs/drafting__prompted-fewshot__predictions.parquet",
}
OUT_FILE = label.JUDGE_DIR / "m2_golden.parquet"
SUMMARY_FILE = REPO_ROOT / "runs" / "drafting__m2_gpt4o.json"
SUCCESS_MIN = 4                                   # the Phase 3 success rule, judge >= 4
AT_CAP_TOKENS = MAX_TOKENS["drafting"] - 8        # within a few tokens of the cap counts as cut


def plan_items(sides: Sequence[str] = tuple(SIDES)) -> pd.DataFrame:
    frames = []
    for side in sides:
        preds = pd.read_parquet(REPO_ROOT / SIDES[side])
        rows = preds[(preds.task == "drafting") & (preds.split == "random")].reset_index(drop=True)
        frames.append(pd.DataFrame({
            "item_id": [f"m2-{side}:golden-{i:03d}" for i in range(len(rows))],
            "source": side,
            "pair": range(len(rows)),
            "instruction": rows.text.astype(str).to_numpy(),
            "reply": rows.prediction.astype(str).str.strip().to_numpy(),
            "split": "m2_golden",
        }))
    by_side = [f.instruction.tolist() for f in frames]
    if any(b != by_side[0] for b in by_side[1:]):
        msg = "the sides were generated on different golden rows — nothing to pair"
        raise ValueError(msg)
    return pd.concat(frames, ignore_index=True)


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def summarise(graded: pd.DataFrame,
              judge: Callable[[Sequence[str], Sequence[str]], list[float]] | None = None,
              count_tokens: Callable[[str], int] | None = None) -> dict:
    out: dict = {"model": label.MODEL, "success_min": SUCCESS_MIN, "sides": {}}
    for side, g in graded.groupby("source"):
        ok = g[g.score.notna()]
        scores = ok.score.to_numpy(dtype=float)
        cell = {"n": len(g), "graded": len(ok), "gpt4o_mean": _r(scores.mean()),
                "gpt4o_success": _r((scores >= SUCCESS_MIN).mean())}
        if count_tokens is not None:
            at_cap = np.array([count_tokens(r) >= AT_CAP_TOKENS for r in ok.reply])
            cell["at_cap_share"] = _r(at_cap.mean())
            cell["gpt4o_mean_at_cap"] = _r(scores[at_cap].mean()) if at_cap.any() else None
            cell["gpt4o_mean_complete"] = _r(scores[~at_cap].mean()) if (~at_cap).any() else None
        if judge is not None:
            distilled = pd.Series(judge(ok.instruction.tolist(), ok.reply.tolist()), dtype=float)
            cell["distilled_mean"] = _r(distilled.mean())
            cell["distilled_vs_gpt4o_spearman"] = _r(
                distilled.corr(pd.Series(scores), method="spearman"))
        out["sides"][side] = cell
    if {"adapter", "prompted"} <= set(graded.source):
        wide = graded.pivot(index="pair", columns="source", values="score").dropna()
        diff = wide.prompted.astype(float) - wide.adapter.astype(float)
        out["paired"] = {"n": len(wide), "prompted_higher": _r((diff > 0).mean()),
                         "adapter_higher": _r((diff < 0).mean()), "tied": _r((diff == 0).mean()),
                         "mean_difference": _r(diff.mean())}
    tokens_in = float(graded.prompt_tokens.fillna(0).sum())
    tokens_out = float(graded.completion_tokens.fillna(0).sum())
    out["cost_usd_from_tokens"] = _r(tokens_in / 1e6 * label.PRICE_PER_1M["input"]
                                     + tokens_out / 1e6 * label.PRICE_PER_1M["output"])
    return out


def main(project: bool = False, sides: Sequence[str] = tuple(SIDES), workers: int = 4,
         spend_cap: float = 1.0, judge=None, count_tokens=None) -> int:
    items = plan_items(sides)
    if project:
        estimate = label.project_cost(items)
        print(json.dumps(estimate, indent=2))
        print(f"\n  projected ${estimate['total']['usd']:.2f} for "
              f"{estimate['total']['requests']:,} requests — no API call was made")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("  no OPENAI_API_KEY in the environment or .env")
        return 2

    with label.single_run():
        cached = label.load_cache()
        todo = items[~items.item_id.isin(cached)]
        print(f"  {len(items):,} items · {len(items) - len(todo):,} cached · "
              f"{len(todo):,} to grade · spend cap ${spend_cap:.2f}", flush=True)
        ledger = label.JudgeLedger(cap=spend_cap, request_cap=2 * len(items))
        results = label.call_batch(todo, ledger, workers) if len(todo) else []

    cache = label.load_cache()
    graded = items[items.item_id.isin(cache)].copy()
    for column in ("score", "raw", "prompt_tokens", "completion_tokens"):
        graded[column] = [cache[i].get(column) for i in graded.item_id]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT_FILE)

    failed = [r for r in results if r.get("error")]
    print(f"  {len(results) - len(failed):,} graded this run · {ledger.requests:,} requests · "
          f"${ledger.usd:.4f}")
    if ledger.stopped:
        print(f"  STOPPED: {ledger.reason}. Cached grades are kept; re-run to resume.")
        return 1
    if failed or len(graded) < len(items):
        print(f"  {len(items) - len(graded)} items ungraded — re-run to retry them")
        return 1

    if judge is None:
        from adapterops.judge.score import load_judge

        judge = load_judge()
    if count_tokens is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        def count_tokens(reply: str) -> int:
            return len(tok(reply).input_ids)

    summary = summarise(graded, judge, count_tokens)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0
