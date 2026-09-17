"""GPT-4o labels for the out-of-domain sample (TODO.md §3) — the only paid step, capped.

Neither source carries labels in the project's schema, so GPT-4o supplies them, with the same prompts the
project already used for GPT-4o-mini (`router/frontier.py`) plus one change: intent may answer NONE, because a
ticket about a delayed sofa has no Banking77 label and forcing one would score the adapter against a guess.

- **intent** — a Banking77 label, or NONE (all 300 tickets)
- **urgency** — low, medium or high (all 300 tickets)
- **drafting** — the judge rubric's 1-5 grade on 60 replies (30 per source), to check whether the distilled
  judge still tracks GPT-4o on text it has never seen

Responses are cached by prompt hash, so a rerun never pays twice. `project` prices the run before any call; the
run stops at `SPEND_CAP_USD` rather than reporting an overrun.

    uv run adapterops ood --project-cost
    uv run adapterops ood --label
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from adapterops.eval.ood import OUTPUTS_FILE, REPO_ROOT, SAMPLE_FILE, SEED
from adapterops.judge.label import PRICE_PER_1M, RUBRIC, build_messages, parse_judgment
from adapterops.router.frontier import build_prompt, intent_labels

MODEL = "gpt-4o"
CACHE = REPO_ROOT / "runs" / "ood__labels_cache.jsonl"
LABELS_FILE = REPO_ROOT / "runs" / "ood__labels.parquet"
SPEND_CAP_USD = 3.00
GRADED_PER_SOURCE = 30
CHARS_PER_TOKEN = 4.0
"""No tokenizer for GPT-4o is installed, so the projection uses characters / 4 with a 15% margin; the ledger
counts the tokens the API reports."""


def intent_prompt(text: str, vocab: str) -> str:
    return build_prompt("intent", text, vocab).replace(
        "Reply with the label only, chosen from:",
        "If no label fits the request, reply NONE. Otherwise reply with the label only, chosen from:")


def requests(sample: pd.DataFrame, outputs: pd.DataFrame | None) -> list[dict]:
    """Every call the run would make, as chat messages keyed by a stable hash."""
    vocab = intent_labels()
    items = []
    for row in sample.itertuples():
        items.append({"id": row.id, "task": "intent", "max_tokens": 16,
                      "messages": [{"role": "user", "content": intent_prompt(row.text, vocab)}]})
        items.append({"id": row.id, "task": "urgency", "max_tokens": 6,
                      "messages": [{"role": "user", "content": build_prompt("urgency", row.text, vocab)}]})
    graded = pd.concat([g.sample(min(GRADED_PER_SOURCE, len(g)), random_state=SEED)
                        for _, g in sample.groupby("source")])
    replies = {}
    if outputs is not None:
        drafts = outputs[outputs.task == "drafting"]
        replies = {r.id: json.loads(r.output).get("reply", "") for r in drafts.itertuples()}
    for row in graded.itertuples():
        reply = replies.get(row.id, "x" * 800)  # a 200-token placeholder until generation has run
        items.append({"id": row.id, "task": "drafting_grade", "max_tokens": 80,
                      "messages": build_messages(row.text, reply)})
    for item in items:
        item["key"] = hashlib.sha256(json.dumps([MODEL, item["messages"]], sort_keys=True).encode()).hexdigest()
    return items


def project(items: list[dict]) -> dict:
    chars_in = sum(len(m["content"]) for i in items for m in i["messages"])
    tokens_in = chars_in / CHARS_PER_TOKEN * 1.15
    tokens_out = sum(i["max_tokens"] for i in items)
    usd = tokens_in / 1e6 * PRICE_PER_1M["input"] + tokens_out / 1e6 * PRICE_PER_1M["output"]
    return {"model": MODEL, "requests": len(items),
            "by_task": pd.Series([i["task"] for i in items]).value_counts().to_dict(),
            "estimated_input_tokens": int(tokens_in), "max_output_tokens": tokens_out,
            "estimated_usd_upper": round(usd, 2), "spend_cap_usd": SPEND_CAP_USD, "price_per_1M": PRICE_PER_1M}


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {row["key"]: row for row in map(json.loads, CACHE.read_text().splitlines()) if row.get("key")}


def label(items: list[dict]) -> dict:
    """Call GPT-4o for every uncached item, one at a time (gpt-4o's 30,000 TPM binds before dollars)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(timeout=60.0, max_retries=2)
    cache = load_cache()
    spent = tokens_in = tokens_out = calls = 0
    with CACHE.open("a", encoding="utf-8") as fh:
        for item in items:
            if item["key"] in cache:
                continue
            if spent >= SPEND_CAP_USD:
                print(f"  stopped at the ${SPEND_CAP_USD} cap after {calls} calls")
                break
            response = client.chat.completions.create(model=MODEL, messages=item["messages"],
                                                      max_tokens=item["max_tokens"], temperature=0)
            usage = response.usage
            tokens_in += usage.prompt_tokens
            tokens_out += usage.completion_tokens
            spent = tokens_in / 1e6 * PRICE_PER_1M["input"] + tokens_out / 1e6 * PRICE_PER_1M["output"]
            calls += 1
            row = {"key": item["key"], "id": item["id"], "task": item["task"],
                   "raw": response.choices[0].message.content or ""}
            fh.write(json.dumps(row) + "\n")
            cache[item["key"]] = row
            if calls % 50 == 0:
                print(f"  {calls} calls, ${spent:.3f}", flush=True)
    return {"calls": calls, "input_tokens": tokens_in, "output_tokens": tokens_out, "usd": round(spent, 4)}


def parse(task: str, raw: str, vocab: set[str]) -> str | int | None:
    text = (raw or "").strip().splitlines()[0].strip().strip(".").strip() if raw and raw.strip() else ""
    if task == "intent":
        return "NONE" if text.upper() == "NONE" else (text if text in vocab else None)
    if task == "urgency":
        return text.lower() if text.lower() in ("low", "medium", "high") else None
    return parse_judgment(raw)


def main(project_only: bool = False) -> int:
    sample = pd.read_parquet(SAMPLE_FILE)
    outputs = pd.read_parquet(OUTPUTS_FILE) if OUTPUTS_FILE.exists() else None
    items = requests(sample, outputs)
    projection = project(items)
    print(json.dumps(projection, indent=1))
    if project_only:
        return 0
    if outputs is None or outputs.id.nunique() < len(sample):
        print("  generation has not finished — the drafting grades need every reply first")
        return 2
    ledger = label(items)
    vocab = set(intent_labels().split(", "))
    cache = load_cache()
    rows = [{"id": i["id"], "task": i["task"], "raw": cache[i["key"]]["raw"],
             "label": parse(i["task"], cache[i["key"]]["raw"], vocab)} for i in items if i["key"] in cache]
    pd.DataFrame(rows).astype({"label": "string"}).to_parquet(LABELS_FILE, index=False)
    (REPO_ROOT / "runs" / "ood__labels.json").write_text(json.dumps(
        {"projection": projection, "spend": ledger, "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
         "labels_file": str(LABELS_FILE.relative_to(REPO_ROOT)),
         "unparsed": int(sum(r["label"] is None for r in rows))}, indent=2) + "\n", encoding="utf-8")
    print(f"  {len(rows)} labels, spend ${ledger['usd']} — wrote {LABELS_FILE.relative_to(REPO_ROOT)}")
    return 0
