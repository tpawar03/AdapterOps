"""Out-of-domain training replies for the drafting judge (TODO.md §3, the judge's second attempt).

The mixed-generator judge fixed cross-generator grading (Spearman 0.28 → 0.81) but not out-of-domain grading
(0.20 → 0.36, interval crossing zero): the adapter's replies to unfamiliar tickets keep its fluent house style
while failing the ticket, and no training reply showed that failure. This supplies it.

**Tickets disjoint from every evaluation ticket.** 150 from ABCD's *train* split (15 per flow) and 150 from the CFPB
subset's *train* file (stratified by product), at most 200 words, sampled the same way as the out-of-domain
evaluation but from splits it never touched, and checked against its texts.

**Replies from the two generators the held-out set grades.** The served drafting adapter, locally with the serving
token cap, and GPT-4o-mini with the request path's frontier prompt. GPT-4o grades both with the judge's rubric.
Responses are cached; the run stops at `SPEND_CAP_USD`.

**Limit, stated before any result:** these tickets share their two sources with the held-out 120, so a pass shows
the judge learned those sources' failure, not that it grades any new domain.

    uv run adapterops judge-ood-train --sample
    uv run adapterops judge-ood-train --generate     # local, ~50 min, resumable
    uv run adapterops judge-ood-train --grade        # GPT-4o-mini replies + GPT-4o grades, capped
"""

from __future__ import annotations

import gzip
import hashlib
import json

import pandas as pd

from adapterops.eval import ood
from adapterops.judge.train import JUDGE_DIR, REPO_ROOT

CFPB_TRAIN = ood.DATA / "cfpb_mini_train.parquet"
CFPB_TRAIN_SHA256 = "a14a9afceb34e1098d05447a192f005658214463d4ea21d10aeaee9615c4020b"
TICKETS = JUDGE_DIR / "ood_train_tickets.parquet"
ADAPTER_REPLIES = JUDGE_DIR / "ood_train_adapter_replies.parquet"
CACHE = JUDGE_DIR / "ood_train_cache.jsonl"
GRADED = JUDGE_DIR / "ood_train_graded.parquet"
SPEND_CAP_USD = 3.00


def sample() -> pd.DataFrame:
    if hashlib.sha256(CFPB_TRAIN.read_bytes()).hexdigest() != CFPB_TRAIN_SHA256:
        msg = f"{CFPB_TRAIN.name} is not the pinned file"
        raise ValueError(msg)
    with gzip.open(ood._verified("abcd")) as fh:
        conversations = json.load(fh)["train"]
    abcd = ood.abcd_sample(conversations)
    abcd["id"] = abcd.id.str.replace("abcd:", "abcdtrain:", regex=False)
    cfpb = ood.cfpb_sample(pd.read_parquet(CFPB_TRAIN))
    cfpb["id"] = cfpb.id.str.replace("cfpb:", "cfpbtrain:", regex=False)
    tickets = pd.concat([abcd, cfpb], ignore_index=True)
    evaluation = set(pd.read_parquet(ood.SAMPLE_FILE).text)
    if shared := set(tickets.text) & evaluation:
        msg = f"{len(shared)} training tickets are also evaluation tickets"
        raise ValueError(msg)
    return ood.freeze(tickets, TICKETS)


def generate(checkpoint_every: int = 10) -> int:
    """The served drafting adapter's reply to every ticket, locally, resumable."""
    from adapterops.serve.pipeline import TransformersBackend, adapter_components, load_manifest

    tickets = pd.read_parquet(TICKETS)
    done = pd.read_parquet(ADAPTER_REPLIES) if ADAPTER_REPLIES.exists() else pd.DataFrame(columns=["id", "reply"])
    todo = tickets[~tickets.id.isin(set(done.id))]
    print(f"  {len(done)} done, {len(todo)} to go", flush=True)
    backend = TransformersBackend(adapter_components(load_manifest()))
    rows = done.to_dict("records")
    for n, ticket in enumerate(todo.itertuples(), start=1):
        rows.append({"id": ticket.id, "reply": backend.generate("drafting", ticket.text).text.strip()})
        if n % checkpoint_every == 0 or n == len(todo):
            pd.DataFrame(rows).to_parquet(ADAPTER_REPLIES, index=False)
            print(f"  {len(rows)}/{len(tickets)}", flush=True)
    return 0


def _key(model: str, messages: list[dict], max_tokens: int) -> str:
    return hashlib.sha256(json.dumps([model, messages, max_tokens], sort_keys=True).encode()).hexdigest()


def _cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, CACHE.read_text().splitlines()) if r.get("key")}


def _call(items: list[dict], spent: float) -> float:
    from dotenv import load_dotenv
    from openai import OpenAI

    from adapterops.judge.label import PRICE_PER_1M as GPT4O
    from adapterops.router.frontier import PRICE_PER_1M as MINI

    load_dotenv(REPO_ROOT / ".env")
    client, cache, calls = OpenAI(timeout=60.0, max_retries=2), _cache(), 0
    with CACHE.open("a", encoding="utf-8") as fh:
        for item in items:
            if item["key"] in cache:
                continue
            if spent >= SPEND_CAP_USD:
                print(f"  stopped at the ${SPEND_CAP_USD} cap")
                break
            r = client.chat.completions.create(model=item["model"], messages=item["messages"],
                                               max_tokens=item["max_tokens"], temperature=0)
            price = MINI if item["model"] == "gpt-4o-mini" else GPT4O
            spent += r.usage.prompt_tokens / 1e6 * price["input"] + r.usage.completion_tokens / 1e6 * price["output"]
            row = {"key": item["key"], "id": item["id"], "model": item["model"], "raw": r.choices[0].message.content or ""}
            fh.write(json.dumps(row) + "\n")
            cache[item["key"]] = row
            calls += 1
            if calls % 100 == 0:
                print(f"  {calls} calls, ${spent:.3f}", flush=True)
    return spent


def grade() -> int:
    from adapterops.judge.label import build_messages, parse_judgment
    from adapterops.router.frontier import MAX_TOKENS, build_prompt, intent_labels

    tickets = pd.read_parquet(TICKETS)
    adapter = pd.read_parquet(ADAPTER_REPLIES).set_index("id").reply
    if len(adapter) < len(tickets):
        print("  adapter generation has not finished")
        return 2
    vocab = intent_labels()
    mini_items = []
    for t in tickets.itertuples():
        messages = [{"role": "user", "content": build_prompt("drafting", t.text, vocab)}]
        mini_items.append({"id": t.id, "model": "gpt-4o-mini", "max_tokens": MAX_TOKENS["drafting"],
                           "messages": messages, "key": _key("gpt-4o-mini", messages, MAX_TOKENS["drafting"])})
    spent = _call(mini_items, 0.0)
    cache = _cache()
    mini = {i["id"]: cache[i["key"]]["raw"].strip() for i in mini_items if i["key"] in cache}
    grade_items = []
    for t in tickets.itertuples():
        for generator, reply in (("adapter", adapter.get(t.id)), ("gpt4o_mini", mini.get(t.id))):
            if not reply:
                continue
            messages = build_messages(t.text, reply)
            grade_items.append({"id": t.id, "generator": generator, "reply": reply, "model": "gpt-4o",
                                "max_tokens": 80, "messages": messages, "key": _key("gpt-4o", messages, 80)})
    spent = _call(grade_items, spent)
    cache = _cache()
    text = tickets.set_index("id").text
    rows = [{"item_id": f"oodtrain:{g['generator']}:{g['id']}", "generator": g["generator"],
             "origin": g["id"].split(":")[0], "instruction": text[g["id"]], "reply": g["reply"],
             "score": parse_judgment(cache[g["key"]]["raw"]), "calibration": False, "domain": "ood_train"}
            for g in grade_items if g["key"] in cache]
    frame = pd.DataFrame(rows)
    frame = frame[frame.score.notna()].astype({"score": float})
    frame.to_parquet(GRADED, index=False)
    print(frame.groupby("generator").score.agg(["count", "mean"]).round(2).to_string())
    print(f"  spend ${spent:.3f} · wrote {GRADED.relative_to(REPO_ROOT)}")
    return 0
