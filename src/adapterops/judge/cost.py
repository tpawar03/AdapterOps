"""Judge cost per 1,000 evaluations — distilled judge vs GPT-4o (PRD M5's cost line, D13).

**GPT-4o's side is priced from tokens already recorded** for every grade in the judgments cache, at
the list price the labelling run used. No new call.

**The distilled side is timed.** The 300 golden drafting replies from the v1 baseline run are scored
on this machine's CPU, after a warm-up batch, and the mean is checked against the score that run
recorded — so the timing is of the real scoring path, not a stand-in.

**The distilled side is reported in seconds, not dollars.** It needs no API and runs on a CPU the
project already had; a dollar figure would need a host price nobody paid. D13's conclusion stands
either way: at this project's volume every GPT-4o grade together cost a few dollars, so distillation
does not pay for itself here.

    uv run python -m adapterops.judge.cost
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from adapterops.judge import label

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_RUN = REPO_ROOT / "runs" / "regression__v1-baseline-1.json"
BASELINE_PREDICTIONS = REPO_ROOT / "runs" / "regression__v1-baseline-1__predictions.parquet"
OUT_FILE = REPO_ROOT / "runs" / "judge__cost.json"
WARMUP = 16


def gpt4o_cost(rows: Iterable[Mapping], price_per_1m: Mapping[str, float]) -> dict:
    """Token-derived cost per 1,000 grades, overall and by item-id prefix (the job that made them)."""
    frame = pd.DataFrame([r for r in rows
                          if r.get("error") is None and r.get("prompt_tokens") is not None])
    frame["job"] = frame.item_id.str.split(":").str[0]

    def priced(g: pd.DataFrame) -> dict:
        usd = (g.prompt_tokens.sum() * price_per_1m["input"]
               + g.completion_tokens.sum() * price_per_1m["output"]) / 1e6
        return {"grades": len(g), "usd": round(float(usd), 4),
                "usd_per_1k": round(float(usd / len(g) * 1000), 4),
                "mean_prompt_tokens": round(float(g.prompt_tokens.mean()), 1)}

    return {"overall": priced(frame),
            "by_job": {job: priced(g) for job, g in frame.groupby("job")}}


def main() -> int:
    import torch

    from adapterops.judge.score import load_judge

    preds = pd.read_parquet(BASELINE_PREDICTIONS)
    draft = preds[(preds.task == "drafting") & (preds.split == "random")]
    texts, replies = draft.text.astype(str).tolist(), draft.prediction.astype(str).tolist()
    recorded = json.loads(BASELINE_RUN.read_text())["per_split"]["drafting"]["random"]

    started = time.perf_counter()
    judge = load_judge()
    load_seconds = time.perf_counter() - started
    judge(texts[:WARMUP], replies[:WARMUP])

    started = time.perf_counter()
    scores = judge(texts, replies)
    seconds = time.perf_counter() - started
    mean = sum(scores) / len(scores)
    if abs(mean - recorded["judge_score_mean"]) > 1e-3:
        msg = (f"timed judge gives {mean:.4f}, the run recorded {recorded['judge_score_mean']} — "
               f"this is not the scoring path the gate used")
        raise ValueError(msg)

    out = {
        "decision": "D13",
        "distilled": {
            "model": "checkpoints/judge (DeBERTa-v3-base, F14)",
            "items": len(scores),
            "seconds": round(seconds, 2),
            "seconds_per_1k": round(seconds / len(scores) * 1000, 1),
            "load_seconds": round(load_seconds, 1),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "host": f"{platform.system()} {platform.machine()}",
            "api_usd": 0.0,
            "mean_score_check": {"timed": round(mean, 4), "recorded": recorded["judge_score_mean"]},
            "dollars": "not given — no host price was paid; see module docstring",
        },
        "gpt4o": {
            "model": label.MODEL,
            "price_per_1m": label.PRICE_PER_1M,
            **gpt4o_cost(label.load_cache().values(), label.PRICE_PER_1M),
        },
    }
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    d, g = out["distilled"], out["gpt4o"]["overall"]
    print(f"  distilled: {d['seconds_per_1k']} s per 1K on {d['host']} CPU, $0 API "
          f"(mean {d['mean_score_check']['timed']} matches the run)")
    print(f"  gpt-4o:    ${g['usd_per_1k']} per 1K over {g['grades']:,} recorded grades")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
