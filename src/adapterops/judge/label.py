"""GPT-4o judgments on drafting outputs (F13) — the teacher labels for the distilled judge.

Drafting is the one task whose router label is a stand-in: token-F1 against the Bitext
reference, cut at the pool median (D28). Its real metric is a judge, and this module
produces the supervision that judge is distilled from.

**Reference-free, deliberately.** The judge sees the customer's request and the reply, and
not the Bitext reference. The token-F1 proxy already measures overlap with the reference,
and the Phase 2 escalation finding showed what a reference-shaped metric does: it rewards
conformance to one dataset's house style, which fine-tuning transfers and a frontier model
does not. A reference-guided judge would rebuild that bias one level up. The cost is that
the judge rates quality without an answer key — acceptable here because Bitext's replies
are generic procedures, not account-specific facts.

**Template slots are declared, not left to the judge's taste.** 49% of the adapter's replies
and 42% of Bitext's own references contain slots like `{{Order Number}}` — the dataset's
convention for values filled in at send time. Only 12.5% of frontier replies do. Left
unstated, a judge could read a slot as an unfinished reply and mark the adapter down for
following the dataset's own format. The rubric states the policy, identically for every
reply.

**The frontier arm's safety patterns, and one thing that could not be reused as-is.**
Retries live in one place, every attempt counts against a request cap, a daily-quota 429
stops the run, a per-item cache means a resume is never re-billed, and a lock stops two
runs paying twice. The frontier arm's `Ledger` is subclassed rather than used directly: it
prices tokens as gpt-4o-mini, and reused unchanged it would price this run about 16x too
cheap — the spend cap would never fire.

**Cost is projected before anything is called.** `--project` counts prompt tokens locally
and prices them without an API request. The Qwen tokenizer stands in for GPT-4o's, with a
15% margin; the aim is a spend estimate to approve, not an invoice.

    uv run adapterops judge-label --project --include-frontier
    uv run adapterops judge-label --include-frontier --spend-cap 8
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from adapterops.router.frontier import Ledger

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORED_FILE = REPO_ROOT / "data" / "router" / "scored.parquet"
FRONTIER_FILE = REPO_ROOT / "data" / "router" / "frontier.parquet"
JUDGE_DIR = REPO_ROOT / "data" / "judge"
CACHE_FILE = JUDGE_DIR / "judgments_cache.jsonl"
LOCK_FILE = JUDGE_DIR / "judge.lock"
OUT_FILE = JUDGE_DIR / "judgments.parquet"

MODEL = "gpt-4o"
PRICE_PER_1M = {"input": 2.50, "output": 10.00}
"""gpt-4o list price when the PRD was costed (§12: ~$4 for ~1,200 labels). Spend is derived
from token counts at this price; the account's usage page is the authority."""

DEFAULT_SPEND_CAP_USD = 8.0
HARD_REQUEST_CAP = 4_000
MAX_COMPLETION_TOKENS = 120
LOCAL_BUDGET = 1_200
"""PRD §9: ~1,200 judgments. All 750 router-slice drafting pairs are included, so the judge
replaces the D28 proxy label on every pair the router trains and evaluates on; the other
450 come from the mining slice."""
CALIBRATION_HOLDOUT = 150
SEED = 20260909
TOKEN_MARGIN = 1.15
EST_OUTPUT_TOKENS = 60
CHAT_OVERHEAD_TOKENS = 12

RUBRIC = """You are grading a customer-support reply. Rate how well the REPLY serves the \
customer's REQUEST on a 1-5 scale.

5 - Fully resolves the request: correct, specific and actionable (clear steps, or exactly \
the right follow-up question), appropriate tone, no errors.
4 - Helpful and correct, with minor gaps: a missing step, somewhat generic, or mildly verbose.
3 - Partly helpful: on-topic but vague or generic, or missing important steps.
2 - Mostly unhelpful: misreads the request, is largely filler, or contains a notable error.
1 - Unhelpful or harmful: off-topic, wrong, an unjustified refusal, or unsafe.

Rules:
- Template slots written like {{Order Number}} or {{Account Type}} are placeholders that are \
filled in automatically when the reply is sent. Treat them as the correct value. Do not \
reward or penalise a reply for using them.
- Judge the reply on its own merits. There is no reference answer.
- Do not reward length for its own sake.

Respond with JSON only: {"score": <integer 1-5>, "reason": "<at most 25 words>"}"""


class JudgeLedger(Ledger):
    """The frontier arm's ledger, priced as gpt-4o.

    Its parent reads gpt-4o-mini prices from the frontier module. Unchanged, this run would
    be priced about 16x too cheap and the spend cap would never fire.
    """

    @property
    def usd(self) -> float:
        return (self.input_tokens / 1e6 * PRICE_PER_1M["input"]
                + self.output_tokens / 1e6 * PRICE_PER_1M["output"])


def build_messages(instruction: str, reply: str) -> list[dict]:
    """Request and reply only — the reference never reaches the judge."""
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"REQUEST:\n{instruction}\n\nREPLY:\n{reply}"},
    ]


def parse_judgment(raw: str) -> int | None:
    """An integer 1-5, or None. Anything ambiguous is None rather than guessed at."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or "score" not in obj:
        return None
    score = obj["score"]
    if isinstance(score, bool):
        return None
    if isinstance(score, float):
        if not score.is_integer():
            return None
        score = int(score)
    if isinstance(score, str):
        try:
            score = int(score.strip())
        except ValueError:
            return None
    if not isinstance(score, int):
        return None
    return score if 1 <= score <= 5 else None


def plan_items(include_frontier: bool = False, seed: int = SEED,
               include_remaining_mining: bool = False) -> pd.DataFrame:
    """Which replies get a GPT-4o grade.

    Local items are the adapter's drafting replies: every router-slice pair plus a seeded
    sample of mining pairs, with 150 held out for calibration. Frontier items are
    GPT-4o-mini's replies to the same router-slice requests, labelled `frontier_eval`: they
    exist for the fair escalation comparison and the §10 same-family check, and never
    enter judge training.
    """
    scored = pd.read_parquet(SCORED_FILE)
    drafting = scored[scored.task == "drafting"]
    router = drafting[drafting.purpose == "router"]
    mining_pool = drafting[drafting.purpose == "mining"]
    need = LOCAL_BUDGET - len(router)
    if not 0 <= need <= len(mining_pool):
        msg = (f"LOCAL_BUDGET {LOCAL_BUDGET} needs {need} mining pairs; "
               f"{len(mining_pool)} are available")
        raise ValueError(msg)

    local = (pd.concat([router, mining_pool.sample(n=need, random_state=seed)])
             .sort_values("pair_id").reset_index(drop=True))
    items = pd.DataFrame({
        "item_id": ("local:" + local.pair_id).to_numpy(),
        "source": "local",
        "pair_id": local.pair_id.to_numpy(),
        "purpose": local.purpose.to_numpy(),
        "instruction": local.text.astype(str).to_numpy(),
        "reply": local.prediction.astype(str).str.strip().to_numpy(),
        "proxy_token_f1": local.proxy_token_f1.to_numpy(),
    })
    items["split"] = "train"
    items.loc[items.sample(n=CALIBRATION_HOLDOUT, random_state=seed).index,
              "split"] = "calibration"

    if include_remaining_mining:
        # D38: the mining-slice drafting replies the original 1,200 did not sample, graded so the
        # drafting hard bucket can be rebuilt from judge failures across the whole slice. Their
        # split is `mining_only`, which judge training never reads — the judge already training
        # was planned from the original 1,200 and must stay reproducible from them.
        rest = mining_pool[~mining_pool.pair_id.isin(set(local.pair_id))].sort_values("pair_id")
        items = pd.concat([items, pd.DataFrame({
            "item_id": ("local:" + rest.pair_id).to_numpy(),
            "source": "local",
            "pair_id": rest.pair_id.to_numpy(),
            "purpose": "mining",
            "instruction": rest.text.astype(str).to_numpy(),
            "reply": rest.prediction.astype(str).str.strip().to_numpy(),
            "proxy_token_f1": rest.proxy_token_f1.to_numpy(),
            "split": "mining_only",
        })], ignore_index=True)

    if include_frontier:
        front = pd.read_parquet(FRONTIER_FILE)
        fd = (front[(front.task == "drafting") & (front.purpose == "router")]
              .sort_values("pair_id").reset_index(drop=True))
        items = pd.concat([items, pd.DataFrame({
            "item_id": ("frontier:" + fd.pair_id).to_numpy(),
            "source": "frontier",
            "pair_id": fd.pair_id.to_numpy(),
            "purpose": "router",
            "instruction": fd.text.astype(str).to_numpy(),
            "reply": fd.prediction.astype(str).str.strip().to_numpy(),
            "proxy_token_f1": fd.frontier_proxy_token_f1.to_numpy(),
            "split": "frontier_eval",
        })], ignore_index=True)
    return items


def project_cost(items: pd.DataFrame) -> dict:
    """Price the run from locally counted tokens. Makes no API call."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    rubric_tokens = len(tok(RUBRIC).input_ids)
    frame = items.assign(est_input_tokens=[
        rubric_tokens + CHAT_OVERHEAD_TOKENS
        + len(tok(f"REQUEST:\n{i}\n\nREPLY:\n{r}").input_ids)
        for i, r in zip(items.instruction, items.reply, strict=True)
    ])

    def price(group: pd.DataFrame) -> dict:
        tokens_in = math.ceil(group.est_input_tokens.sum() * TOKEN_MARGIN)
        tokens_out = len(group) * EST_OUTPUT_TOKENS
        usd = tokens_in / 1e6 * PRICE_PER_1M["input"] + tokens_out / 1e6 * PRICE_PER_1M["output"]
        return {"requests": len(group), "input_tokens": tokens_in,
                "output_tokens": tokens_out, "usd": round(usd, 2)}

    out: dict = {src: price(g) for src, g in frame.groupby("source")}
    out["total"] = price(frame)
    out["assumptions"] = {
        "model": MODEL,
        "price_per_1M": PRICE_PER_1M,
        "tokenizer": "Qwen2.5 as a stand-in for GPT-4o's, x1.15 margin",
        "output_tokens_per_call": EST_OUTPUT_TOKENS,
        "rubric_tokens": rubric_tokens,
    }
    return out


@contextlib.contextmanager
def single_run() -> Iterator[None]:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        msg = (f"another judge run holds {LOCK_FILE.name}. Two runs share one daily request "
               f"quota and pay twice for whatever they both reach. If none is active, "
               f"delete the file.")
        raise RuntimeError(msg) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def load_cache() -> dict[str, dict]:
    """Item id to cached judgment. First response wins, so a duplicate can never re-grade."""
    if not CACHE_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(row["item_id"], row)
    return out


def call_batch(items: pd.DataFrame, ledger: JudgeLedger, workers: int) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(timeout=60.0, max_retries=0)       # retries are owned below, once
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    done = {"n": 0}
    handle = CACHE_FILE.open("a", encoding="utf-8")

    def one(row: pd.Series) -> dict | None:
        if ledger.stopped:
            return None
        for attempt in range(4):
            try:
                ledger.count_request()
                if ledger.stopped:
                    return None
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=build_messages(row.instruction, row.reply),
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                break
            except Exception as exc:                   # noqa: BLE001 - retried, then recorded
                if "requests per day" in str(exc):
                    ledger.halt("daily request quota exhausted")
                    return None
                if attempt == 3:
                    return {"item_id": row.item_id, "error": type(exc).__name__}
                time.sleep(2 ** attempt)
        usage = response.usage
        ledger.add(usage.prompt_tokens, usage.completion_tokens)
        raw = (response.choices[0].message.content or "").strip()
        out = {"item_id": row.item_id, "raw": raw, "score": parse_judgment(raw),
               "prompt_tokens": usage.prompt_tokens,
               "completion_tokens": usage.completion_tokens, "error": None}
        with write_lock:
            handle.write(json.dumps(out) + "\n")
            handle.flush()
            done["n"] += 1
            if done["n"] % 100 == 0:
                print(f"    {done['n']:,} graded · ${ledger.usd:.3f}", flush=True)
        return out

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [r for r in pool.map(one, (row for _, row in items.iterrows()))
                    if r is not None]
    finally:
        handle.close()


def write_outputs(items: pd.DataFrame) -> Path:
    cache = load_cache()
    rows = items[items.item_id.isin(cache)].copy()
    for column in ("score", "raw", "prompt_tokens", "completion_tokens"):
        rows[column] = [cache[i].get(column) for i in rows.item_id]
    rows["parsed_ok"] = rows.score.notna()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT_FILE)
    return OUT_FILE


def main(project: bool = False, include_frontier: bool = False, limit: int | None = None,
         workers: int = 4, spend_cap: float = DEFAULT_SPEND_CAP_USD,
         include_remaining_mining: bool = False) -> int:
    items = plan_items(include_frontier=include_frontier,
                       include_remaining_mining=include_remaining_mining)
    if limit:
        items = items.groupby("source", group_keys=False).head(limit)

    if project:
        estimate = project_cost(items)
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

    with single_run():
        cached = load_cache()
        todo = items[~items.item_id.isin(cached)]
        print(f"  {len(items):,} items · {len(items) - len(todo):,} cached · "
              f"{len(todo):,} to grade · spend cap ${spend_cap:.2f}", flush=True)
        ledger = JudgeLedger(cap=spend_cap, request_cap=HARD_REQUEST_CAP)
        results = call_batch(todo, ledger, workers) if len(todo) else []
        path = write_outputs(items)

        failed = [r for r in results if r.get("error")]
        unparsed = sum(1 for r in results if not r.get("error") and r.get("score") is None)
        print(f"\n  {len(results) - len(failed):,} graded · {ledger.requests:,} requests · "
              f"${ledger.usd:.4f} · {unparsed} unparseable · wrote "
              f"{path.relative_to(REPO_ROOT)}")
        if ledger.stopped:
            print(f"  STOPPED: {ledger.reason}. Cached judgments are kept; re-run to resume.")
            return 1
        if failed:
            print(f"  {len(failed)} failed and were not cached — re-run to retry them")
            return 1
    return 0
