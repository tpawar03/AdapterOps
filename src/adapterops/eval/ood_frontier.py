"""GPT-4o-mini on the out-of-domain sample (TODO.md §3, before serving the out-of-domain gate).

The input check flags almost every far out-of-domain ticket for intent, urgency and drafting, but a flag needs an
action, and routing a flagged ticket to GPT-4o-mini is only worth it if GPT-4o-mini does better there. This answers
that on the same 300 tickets and against the same GPT-4o labels, with the prompts the request path already sends
(`router/frontier.py`), so the result describes the route that would actually be taken.

Two paid steps, both cached and capped: GPT-4o-mini answers every task (1,200 calls), and GPT-4o grades its replies
on the same 60 tickets it graded the adapter's, so the drafting comparison is like for like.

**A caveat stated up front:** the intent and urgency labels come from GPT-4o, a larger model of the same family, so
GPT-4o-mini may agree with them for reasons that are not correctness. The drafting grades share that bias. PII
needs no labels on ABCD, so it is the cleanest comparison.

    uv run adapterops ood --frontier-project
    uv run adapterops ood --frontier
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from adapterops.eval.ood import OUTPUTS_FILE, REPO_ROOT, SAMPLE_FILE
from adapterops.eval.ood_label import CACHE as LABEL_CACHE
from adapterops.eval.ood_label import LABELS_FILE
from adapterops.eval.ood_label import requests as label_requests
from adapterops.judge.label import PRICE_PER_1M as GPT4O_PRICE
from adapterops.judge.label import build_messages, parse_judgment
from adapterops.router.frontier import MAX_TOKENS, PRICE_PER_1M, build_prompt, intent_labels

MODEL = "gpt-4o-mini"
GRADER = "gpt-4o"
CACHE = REPO_ROOT / "runs" / "ood__frontier_cache.jsonl"
RUN_FILE = REPO_ROOT / "runs" / "ood__frontier.json"
SPEND_CAP_USD = 0.40
TASKS = ("intent", "urgency", "pii", "drafting")


def _key(model: str, messages: list[dict], max_tokens: int) -> str:
    return hashlib.sha256(json.dumps([model, messages, max_tokens], sort_keys=True).encode()).hexdigest()


def answer_items(sample: pd.DataFrame) -> list[dict]:
    vocab = intent_labels()
    items = []
    for row in sample.itertuples():
        for task in TASKS:
            messages = [{"role": "user", "content": build_prompt(task, row.text, vocab)}]
            items.append({"id": row.id, "task": task, "model": MODEL, "max_tokens": MAX_TOKENS[task],
                          "messages": messages, "key": _key(MODEL, messages, MAX_TOKENS[task])})
    return items


def graded_ids(sample: pd.DataFrame) -> list[str]:
    """The 60 tickets GPT-4o graded the adapter's replies on — the same seeded draw as `ood_label`."""
    return [i["id"] for i in label_requests(sample, None) if i["task"] == "drafting_grade"]


def grade_items(sample: pd.DataFrame, replies: dict[str, str]) -> list[dict]:
    text = sample.set_index("id").text
    items = []
    for ticket_id in graded_ids(sample):
        messages = build_messages(text[ticket_id], replies.get(ticket_id, "x" * 1200))
        items.append({"id": ticket_id, "task": "drafting_grade", "model": GRADER, "max_tokens": 80,
                      "messages": messages, "key": _key(GRADER, messages, 80)})
    return items


def price(model: str) -> dict:
    return PRICE_PER_1M if model == MODEL else GPT4O_PRICE


def project(items: list[dict]) -> dict:
    usd, by_model = 0.0, {}
    for item in items:
        tokens_in = sum(len(m["content"]) for m in item["messages"]) / 4.0 * 1.15
        cost = tokens_in / 1e6 * price(item["model"])["input"] + item["max_tokens"] / 1e6 * price(item["model"])["output"]
        usd += cost
        by_model[item["model"]] = by_model.get(item["model"], 0) + 1
    return {"requests": by_model, "estimated_usd_upper": round(usd, 3), "spend_cap_usd": SPEND_CAP_USD}


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {row["key"]: row for row in map(json.loads, CACHE.read_text().splitlines()) if row.get("key")}


def call(items: list[dict], spent: float = 0.0) -> float:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(timeout=60.0, max_retries=2)
    cache = load_cache()
    calls = 0
    with CACHE.open("a", encoding="utf-8") as fh:
        for item in items:
            if item["key"] in cache:
                continue
            if spent >= SPEND_CAP_USD:
                print(f"  stopped at the ${SPEND_CAP_USD} cap")
                break
            response = client.chat.completions.create(model=item["model"], messages=item["messages"],
                                                      max_tokens=item["max_tokens"], temperature=0)
            p = price(item["model"])
            spent += (response.usage.prompt_tokens / 1e6 * p["input"]
                      + response.usage.completion_tokens / 1e6 * p["output"])
            row = {"key": item["key"], "id": item["id"], "task": item["task"], "model": item["model"],
                   "raw": response.choices[0].message.content or ""}
            fh.write(json.dumps(row) + "\n")
            cache[item["key"]] = row
            calls += 1
            if calls % 100 == 0:
                print(f"  {calls} calls, ${spent:.3f}", flush=True)
    return spent


def compare(sample: pd.DataFrame, answers: dict[tuple[str, str], str], grades: dict[str, int | None]) -> dict:
    """GPT-4o-mini beside the served adapters, per source, on every task."""
    from sklearn.metrics import f1_score

    from adapterops.eval.pii_errors import coverage
    from adapterops.eval.spans import Span, parse_model_output, spans_from_values

    outputs = pd.read_parquet(OUTPUTS_FILE)
    labels = pd.read_parquet(LABELS_FILE)
    vocab = set(intent_labels().split(", "))
    served = {(r.id, r.task): json.loads(r.output) for r in outputs.itertuples()}
    gold = {(r.id, r.task): r.label for r in labels.itertuples() if pd.notna(r.label)}
    result = {}
    for source in ("abcd", "cfpb"):
        rows = sample[sample.source == source]
        block = {}
        fits = [i for i in rows.id if gold.get((i, "intent")) not in (None, "NONE")]
        mini_intent = {i: (answers.get((i, "intent")) or "").strip().split("\n")[0].strip() for i in rows.id}
        block["intent"] = {
            "tickets_where_a_label_fits": len(fits),
            "adapter_accuracy": round(sum(served[(i, "intent")].get("label") == gold[(i, "intent")] for i in fits) / len(fits), 4) if fits else None,
            "gpt4o_mini_accuracy": round(sum(mini_intent[i] == gold[(i, "intent")] for i in fits) / len(fits), 4) if fits else None,
            "gpt4o_mini_label_outside_the_77": round(sum(mini_intent[i] not in vocab for i in rows.id) / len(rows), 4),
        }
        urgent = [i for i in rows.id if (i, "urgency") in gold]
        mini_urgency = [(answers.get((i, "urgency")) or "").strip().lower().strip(".") for i in urgent]
        block["urgency"] = {
            "tickets": len(urgent),
            "served_tfidf_macro_f1": round(float(f1_score([gold[(i, "urgency")] for i in urgent],
                                                          [served[(i, "urgency")].get("label") for i in urgent],
                                                          average="macro", zero_division=0)), 4),
            "gpt4o_mini_macro_f1": round(float(f1_score([gold[(i, "urgency")] for i in urgent], mini_urgency,
                                                        average="macro", zero_division=0)), 4)}
        if source == "abcd":
            leaks = {"adapter": [0, 0], "gpt4o_mini": [0, 0]}
            spans_total = 0
            for r in rows.itertuples():
                g = [Span(x["start"], x["end"], x["label"]) for x in json.loads(r.pii_gold) if x["scope"] == "in"]
                if not g:
                    continue
                spans_total += len(g)
                adapter_pairs = [(s["label"], s["value"]) for s in served[(r.id, "pii")].get("spans", [])]
                mini_pairs = parse_model_output(answers.get((r.id, "pii")) or "")
                for name, pairs in (("adapter", adapter_pairs), ("gpt4o_mini", mini_pairs)):
                    partly, wholly = coverage(r.text, g, spans_from_values(r.text, pairs))
                    leaks[name][0] += partly
                    leaks[name][1] += wholly
            block["pii"] = {"in_scope_spans": spans_total,
                            **{f"{n}_wholly_unmasked": round(v[1] / spans_total, 4) for n, v in leaks.items()},
                            **{f"{n}_partly_unmasked": round(v[0] / spans_total, 4) for n, v in leaks.items()}}
        graded = [i for i in graded_ids(sample) if i in set(rows.id)]
        adapter_grades = [gold.get((i, "drafting_grade")) for i in graded]
        mini_grades = [grades.get(i) for i in graded]
        paired = [(float(a), float(m)) for a, m in zip(adapter_grades, mini_grades, strict=True)
                  if a is not None and m is not None]
        block["drafting"] = {"graded": len(paired),
                             "adapter_gpt4o_grade_mean": round(sum(a for a, _ in paired) / len(paired), 4) if paired else None,
                             "gpt4o_mini_gpt4o_grade_mean": round(sum(m for _, m in paired) / len(paired), 4) if paired else None,
                             "gpt4o_mini_better": sum(m > a for a, m in paired),
                             "adapter_better": sum(a > m for a, m in paired)}
        result[source] = block
    return result


def main(project_only: bool = False) -> int:
    sample = pd.read_parquet(SAMPLE_FILE)
    answers_needed = answer_items(sample)
    cache = load_cache()
    replies = {i["id"]: cache[i["key"]]["raw"] for i in answers_needed
               if i["task"] == "drafting" and i["key"] in cache}
    grades_needed = grade_items(sample, replies)
    projection = project([i for i in answers_needed + grades_needed if i["key"] not in cache])
    print(json.dumps(projection))
    if project_only:
        return 0
    spent = call(answers_needed)
    cache = load_cache()
    replies = {i["id"]: cache[i["key"]]["raw"] for i in answers_needed if i["task"] == "drafting" and i["key"] in cache}
    grades_needed = grade_items(sample, replies)
    spent = call(grades_needed, spent)
    cache = load_cache()
    answers = {(i["id"], i["task"]): cache[i["key"]]["raw"] for i in answers_needed if i["key"] in cache}
    grades = {i["id"]: parse_judgment(cache[i["key"]]["raw"]) for i in grades_needed if i["key"] in cache}
    result = {"model": MODEL, "grader": GRADER, "spend_usd_this_run": round(spent, 4),
              "label_cache": str(LABEL_CACHE.relative_to(REPO_ROOT)),
              "caveat": "intent, urgency and drafting are judged by GPT-4o; PII on ABCD needs no labels",
              "per_source": compare(sample, answers, grades)}
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["per_source"], indent=1))
    print(f"  spend ${spent:.4f} · wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
