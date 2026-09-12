"""Measure how often escalation actually succeeds (F8's frontier arm, and F11's input).

The operating curve compares policies that escalate different pairs. That comparison is
only meaningful if escalation is worth something, and *how much* it is worth is a
measurement, not an assumption — assuming the frontier always succeeds flatters every
escalating policy, including whichever one the project would most like to win.
`baselines.py` therefore refuses to run without either this column or an explicitly stated
assumption.

**Scored by exactly the same code as the local path.** `score_pair` from `scoring.py`, the
same span scorer, the same token-F1. A frontier arm scored by a more generous rule is not
a comparison, and it is an easy mistake to make because the frontier's output format has
to be *prompted* rather than fine-tuned in.

**Which means the prompt has to hand over what the adapter learned.** The adapters absorbed
the label vocabulary and the output format during training; the frontier has not seen
either, so withholding them would measure formatting compliance rather than capability.
The 77 intent labels and the 19 PII labels go in the prompt.

**Generation and labelling are separated here too (D27).** This module writes raw
predictions and score components only. `frontier_success` is derived later, alongside the
local labels, so that drafting's proxy cut is the *same* number on both sides — deriving
it from each side's own median would hand the frontier a 50% success rate by construction.

    uv run adapterops frontier --limit 20      # cost projection on a sample first
    uv run adapterops frontier
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from adapterops.router.scoring import score_pair

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_FILE = REPO_ROOT / "data" / "router" / "pool.parquet"
OUT_FILE = REPO_ROOT / "data" / "router" / "frontier.parquet"
CACHE_FILE = REPO_ROOT / "data" / "router" / "frontier_cache.jsonl"

MODEL = "gpt-4o-mini"
PRICE_PER_1M = {"input": 0.15, "output": 0.60}
"""Recorded here so the cost in COST-LOG.md is derived from token counts rather than
read off a dashboard later."""

HARD_SPEND_CAP_USD = 1.50
"""A second backstop under the account's own $15 cap. The approved spend is ~$0.55; this
aborts the run rather than discovering an overrun afterwards."""

HARD_REQUEST_CAP = 8_000
"""Dollars are not the only exhaustible resource, which is the lesson of the first run.

The account has a 10,000 requests-per-day ceiling, and it was hit after roughly 3,500
useful calls — because a retry loop here was stacked on top of the SDK's own internal
retries, turning each failure into as many as twelve requests. $0.16 of tokens consumed a
whole day's request quota. A dollar cap could not have seen it coming."""

MAX_TOKENS = {"intent": 16, "urgency": 6, "pii": 256, "drafting": 400}

PII_LABELS = ("AGE BUILDINGNUM CITY CREDITCARDNUMBER DATE DRIVERLICENSENUM EMAIL GENDER "
              "GIVENNAME IDCARDNUM PASSPORTNUM SEX SOCIALNUM STREET SURNAME TAXNUM "
              "TELEPHONENUM TITLE ZIPCODE")


def intent_labels() -> str:
    labels = sorted(pd.read_parquet(REPO_ROOT / "data/intent/split_train.parquet")
                    .label_text.unique())
    return ", ".join(labels)


def build_prompt(task: str, text: str, intent_vocab: str) -> str:
    if task == "intent":
        return ("Classify the customer's banking request into exactly one intent label.\n"
                f"Reply with the label only, chosen from: {intent_vocab}\n\n"
                f"Request: {text}\nIntent:")
    if task == "urgency":
        return ("Classify the urgency of this support ticket as exactly one of: "
                "low, medium, high.\nReply with the word only.\n\n"
                f"Ticket: {text}\nUrgency:")
    if task == "pii":
        return ("List every piece of personal information in the text, one per line, as "
                "LABEL: value.\nCopy each value exactly as it appears in the text. Use "
                f"only these labels: {PII_LABELS}\nReply with the lines only.\n\n"
                f"Text: {text}\nFound:")
    return ("Write a helpful customer-support reply to this request. Reply with the "
            f"message only.\n\nRequest: {text}\nReply:")


class Ledger:
    """Running token spend, with a cap that stops the run rather than reporting it."""

    def __init__(self, cap: float = HARD_SPEND_CAP_USD,
                 request_cap: int = HARD_REQUEST_CAP) -> None:
        self.cap = cap
        self.request_cap = request_cap
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        self._lock = threading.Lock()
        self.stopped = False
        self.reason = ""

    def count_request(self) -> None:
        """Every attempt, successful or not — retries spend quota exactly like calls do."""
        with self._lock:
            self.requests += 1
            if self.requests > self.request_cap:
                self.stopped, self.reason = True, f"request cap {self.request_cap:,}"

    def halt(self, reason: str) -> None:
        with self._lock:
            self.stopped, self.reason = True, reason

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            if self.usd > self.cap:
                self.stopped, self.reason = True, f"spend cap ${self.cap:.2f}"

    @property
    def usd(self) -> float:
        return (self.input_tokens / 1e6 * PRICE_PER_1M["input"]
                + self.output_tokens / 1e6 * PRICE_PER_1M["output"])


def load_cache() -> dict[str, dict]:
    if not CACHE_FILE.exists():
        return {}
    out = {}
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["pair_id"]] = row
    return out


def call_batch(rows: Iterable[pd.Series], ledger: Ledger, workers: int,
               intent_vocab: str) -> list[dict]:
    """Concurrent calls, appending each result to the cache as it lands.

    Cached per pair rather than per run: 5,800 calls is long enough that a crash two
    thirds through would otherwise mean paying for the first two thirds twice.
    """
    from openai import OpenAI

    # Bounded on both axes. The first full run stalled for five minutes with no output:
    # the SDK's default is a 600-second timeout and its own internal retries, which on a
    # rate-limited burst stacks with the retry loop below into a wait long enough to look
    # like a hang. 60 seconds is generous for a 400-token completion.
    # max_retries=0: the loop below is the *only* retry owner. Leaving the SDK's default
    # in place multiplied every failure by its internal retries and burned a 10,000/day
    # request quota on ~3,500 useful calls.
    client = OpenAI(timeout=60.0, max_retries=0)
    write_lock = threading.Lock()
    done = {"n": 0}
    handle = CACHE_FILE.open("a", encoding="utf-8")

    def one(row: pd.Series) -> dict | None:
        if ledger.stopped:
            return None
        prompt = build_prompt(row.task, row.text, intent_vocab)
        for attempt in range(4):
            try:
                ledger.count_request()
                if ledger.stopped:
                    return None
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=MAX_TOKENS[row.task],
                    temperature=0.0,
                )
                break
            except Exception as exc:                   # noqa: BLE001 - retried, then recorded
                if "requests per day" in str(exc):
                    # A daily quota cannot be waited out inside a run, and retrying it can
                    # only consume more of tomorrow's. Stop the whole run.
                    ledger.halt("daily request quota exhausted")
                    return None
                if attempt == 3:
                    # Left out of the cache on purpose, so a resume retries it. The count
                    # is reported rather than silently absorbed.
                    return {"pair_id": row.pair_id, "prediction": "",
                            "error": type(exc).__name__}
                time.sleep(2 ** attempt)
        usage = r.usage
        ledger.add(usage.prompt_tokens, usage.completion_tokens)
        out = {
            "pair_id": row.pair_id,
            "prediction": (r.choices[0].message.content or "").strip(),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "error": None,
        }
        with write_lock:
            handle.write(json.dumps(out) + "\n")
            handle.flush()
            done["n"] += 1
            if done["n"] % 250 == 0:
                print(f"    {done['n']:,} calls · ${ledger.usd:.3f}", flush=True)
        return out

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [r for r in pool.map(one, rows) if r is not None]
    finally:
        handle.close()


def project_cost(pool: pd.DataFrame, sample: list[dict]) -> dict:
    """Extrapolate the full run from a measured sample, per task."""
    by_pair = {s["pair_id"]: s for s in sample if s.get("prompt_tokens")}
    frame = pool[pool.pair_id.isin(by_pair)].copy()
    frame["prompt_tokens"] = [by_pair[p]["prompt_tokens"] for p in frame.pair_id]
    frame["completion_tokens"] = [by_pair[p]["completion_tokens"] for p in frame.pair_id]

    total, per_task = 0.0, {}
    for task, g in frame.groupby("task"):
        n_full = int((pool.task == task).sum())
        cost = (g.prompt_tokens.mean() / 1e6 * PRICE_PER_1M["input"]
                + g.completion_tokens.mean() / 1e6 * PRICE_PER_1M["output"]) * n_full
        per_task[task] = {"pairs": n_full,
                          "mean_prompt_tokens": round(float(g.prompt_tokens.mean()), 1),
                          "mean_completion_tokens": round(float(g.completion_tokens.mean()), 1),
                          "projected_usd": round(cost, 4)}
        total += cost
    return {"per_task": per_task, "projected_usd": round(total, 4)}


def main(limit: int | None = None, workers: int = 6, purpose: str | None = None) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("  no OPENAI_API_KEY in the environment or .env")
        return 2

    pool = pd.read_parquet(POOL_FILE)
    if purpose:
        pool = pool[pool.purpose == purpose]

    cached = load_cache()
    todo = pool[~pool.pair_id.isin(cached)]
    if limit:
        todo = todo.groupby("task", group_keys=False).head(limit)
    print(f"  {len(pool):,} pairs · {len(cached):,} already cached · "
          f"{len(todo):,} to call", flush=True)
    if todo.empty and not cached:
        return 0

    ledger = Ledger()
    started = time.perf_counter()
    results = call_batch((r for _, r in todo.iterrows()), ledger, workers, intent_labels())
    elapsed = time.perf_counter() - started

    if limit:
        projection = project_cost(pool, results)
        print(json.dumps(projection, indent=2))
        print(f"\n  sample of {len(results)} cost ${ledger.usd:.4f} in {elapsed:.0f}s")
        print(f"  full run projects to ${projection['projected_usd']:.2f} — "
              f"re-run without --limit to proceed (the sample is cached, not re-billed)")
        return 0

    if ledger.stopped:
        write_outputs(pool)
        print(f"\n  STOPPED: {ledger.reason}. {ledger.requests:,} requests, "
              f"${ledger.usd:.4f} spent. Cached results are kept; re-run to resume.")
        return 1

    failed = [r for r in results if r.get("error")]
    write_outputs(pool)
    print(f"\n  {len(results) - len(failed):,} calls · {ledger.requests:,} requests · "
          f"${ledger.usd:.4f} · {elapsed:.0f}s "
          f"({ledger.input_tokens:,} in / {ledger.output_tokens:,} out)")
    if failed:
        counts = pd.Series([r["error"] for r in failed]).value_counts().to_dict()
        print(f"  {len(failed)} failed and were NOT cached — re-run to retry them: {counts}")
        return 1
    return 0


def write_outputs(pool: pd.DataFrame) -> Path:
    """Score the cached predictions and freeze them. No labels — those come later (D27)."""
    cached = load_cache()
    rows = pool[pool.pair_id.isin(cached)].copy()
    rows["prediction"] = [cached[p]["prediction"] for p in rows.pair_id]
    rows["error"] = [cached[p].get("error") for p in rows.pair_id]
    components = pd.DataFrame([
        score_pair(r.task, r.text, r.gold, r.prediction) for _, r in rows.iterrows()
    ], index=rows.index)
    out = pd.concat([rows, components.add_prefix("frontier_")], axis=1)
    out.to_parquet(OUT_FILE)
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)} ({len(out):,} rows)")
    return OUT_FILE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="pairs per task — measures token use and projects the full cost")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--purpose", choices=["router", "mining"], default=None)
    a = ap.parse_args()
    raise SystemExit(main(limit=a.limit, workers=a.workers, purpose=a.purpose))
