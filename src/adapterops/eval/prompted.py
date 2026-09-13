"""Prompted baselines (M2) — the base model, no adapter, a strong few-shot prompt.

M2 is the comparison that decides whether fine-tuning earned its place, and only intent has
one — run inside the Colab notebook and never committed as code. This generalises it to all
four tasks and runs it through the same vLLM server that serves the adapters, addressing the
base model by name instead of an adapter name, so both sides share a serving stack.

**Demonstrations never come from the golden set** — drawn from `split_train` and asserted
disjoint from golden, not assumed.

**Decoding matches the adapter runs:** greedy, with each task's `router.generate.MAX_TOKENS`.

**Scoring reuses the regression run's scorers** (`eval.regression.score`), so M2 and F18
cannot disagree about what a metric means.

**Intent's recorded baseline disclosed its own weakness:** ten demonstrations covering ten of
77 classes. The `per-class` recipe gives one demonstration per class — the stronger comparison
that run said it should have been. It is written under a different system name, and the
recorded baseline is never overwritten.

**Long prompts are checked before a server is involved.** One demonstration per intent class
is roughly 3,000 tokens, and `scripts/phase2_serve.sh` configures a 1,536-token context. A
prompt that does not fit is rejected by the server one request at a time, so `budget()`
measures the longest prompt plus generation against the context and the run refuses to start
if it does not fit.

    uv run adapterops prompted --task urgency --budget-only --max-model-len 1536
    uv run adapterops prompted --task urgency --base-url http://localhost:8000
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from adapterops.eval import regression
from adapterops.train.qlora import COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 20260909

PII_LABELS = ("AGE BUILDINGNUM CITY CREDITCARDNUMBER DATE DRIVERLICENSENUM EMAIL GENDER "
              "GIVENNAME IDCARDNUM PASSPORTNUM SEX SOCIALNUM STREET SURNAME TAXNUM "
              "TELEPHONENUM TITLE ZIPCODE")

RECIPES = {
    # task: {recipe: (mode, k)}. per_class draws k per label; total draws k overall.
    "intent": {"fewshot": ("total", 10), "per-class": ("per_class", 1)},
    "urgency": {"fewshot": ("per_class", 4)},
    "pii": {"fewshot": ("total", 5)},
    "drafting": {"fewshot": ("total", 3)},
}

Generate = Callable[[Sequence[str]], list[str]]


def demonstrations(task: str, recipe: str = "fewshot") -> pd.DataFrame:
    """Seeded demonstrations from split_train, asserted disjoint from the golden set."""
    mode, k = RECIPES[task][recipe]
    text_col, gold_col = COLUMNS[task]
    train = pd.read_parquet(REPO_ROOT / "data" / task / "split_train.parquet")
    golden = set(pd.read_parquet(REPO_ROOT / "evals" / "golden" / f"{task}.parquet")[text_col])
    train = train[~train[text_col].isin(golden)]

    if mode == "per_class":
        shuffled = train.sample(frac=1.0, random_state=SEED)
        demos = shuffled.groupby(gold_col, group_keys=False).head(k)
    else:
        demos = train.sample(n=k, random_state=SEED)
    demos = demos.sort_values([gold_col, text_col]).reset_index(drop=True)

    leaked = set(demos[text_col]) & golden
    if leaked:
        msg = f"{task}: {len(leaked)} demonstrations are golden-set rows"
        raise RuntimeError(msg)
    return demos[[text_col, gold_col]].rename(columns={text_col: "text", gold_col: "gold"})


def header(task: str, recipe: str = "fewshot") -> str:
    demos = demonstrations(task, recipe)
    if task == "intent":
        _, gold_col = COLUMNS[task]
        labels = sorted(pd.read_parquet(REPO_ROOT / "data/intent/split_train.parquet")[gold_col]
                        .astype(str).unique())
        shots = "\n".join(f"Request: {r.text}\nIntent: {r.gold}" for r in demos.itertuples())
        return ("Classify the customer's banking request into exactly one intent label.\n"
                "Valid labels:\n" + ", ".join(labels) + "\n\nExamples:\n" + shots + "\n\n")
    if task == "urgency":
        shots = "\n\n".join(f"Ticket: {r.text}\nUrgency: {r.gold}" for r in demos.itertuples())
        return ("Classify the urgency of this support ticket as exactly one of: "
                "low, medium, high.\n\nExamples:\n" + shots + "\n\n")
    if task == "pii":
        shots = "\n\n".join(f"Text: {r.text}\nFound:\n{r.gold}" for r in demos.itertuples())
        return ("List every piece of personal information in the text, one per line, as "
                "LABEL: value. Copy each value exactly as it appears in the text. Use only "
                f"these labels: {PII_LABELS}\n\nExamples:\n" + shots + "\n\n")
    shots = "\n\n".join(f"Request: {r.text}\nReply: {r.gold}" for r in demos.itertuples())
    return "Write a helpful customer-support reply to this request.\n\nExamples:\n" + shots + "\n\n"


def query(task: str, text: str) -> str:
    return {
        "intent": f"Request: {text}\nIntent:",
        "urgency": f"Ticket: {text}\nUrgency:",
        "pii": f"Text: {text}\nFound:\n",
        "drafting": f"Request: {text}\nReply:",
    }[task]


def budget(task: str, recipe: str = "fewshot", max_model_len: int = 1536) -> dict:
    """Longest prompt plus generation, against the server's context. No server needed."""
    from transformers import AutoTokenizer

    from adapterops.router.generate import MAX_TOKENS

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    head = header(task, recipe)
    texts, _ = regression.load_split(task, "random")
    head_tokens = len(tok(head).input_ids)
    longest = max(len(tok(query(task, t)).input_ids) for t in texts)
    needed = head_tokens + longest + MAX_TOKENS[task]
    return {"task": task, "recipe": recipe, "header_tokens": head_tokens,
            "longest_query_tokens": longest, "max_new_tokens": MAX_TOKENS[task],
            "needed": needed, "max_model_len": max_model_len, "fits": needed <= max_model_len}


def label_set(task: str) -> set[str]:
    _, gold_col = COLUMNS[task]
    return set(pd.read_parquet(REPO_ROOT / "data" / task / "split_train.parquet")[gold_col]
               .astype(str))


def run(task: str, generate: Generate, recipe: str = "fewshot",
        collect: list[dict] | None = None) -> dict:
    head = header(task, recipe)
    texts, gold = regression.load_split(task, "random")
    predictions = generate([head + query(task, t) for t in texts])
    if len(predictions) != len(texts):
        msg = f"{task}: {len(predictions)} predictions for {len(texts)} golden rows"
        raise ValueError(msg)
    metrics = regression.score(task, texts, gold, predictions)
    if collect is not None:
        # The regression run's layout, so judge-score reads either file the same way.
        collect.extend({"task": task, "split": "random", "text": t, "gold": g,
                        "prediction": p}
                       for t, g, p in zip(texts, gold, predictions, strict=True))
    if task in ("intent", "urgency"):
        valid = label_set(task)
        first = [p.strip().split("\n")[0].strip() for p in predictions]
        metrics["exact_label_rate"] = round(sum(p in valid for p in first) / len(first), 4)
    demos = demonstrations(task, recipe)
    return {
        "task": task,
        "system": f"prompted-{recipe}",
        "split": f"evals/golden/{task}.parquet",
        "n": len(texts),
        "gated_metric": regression.GATED[task],
        "metrics": metrics,
        "setup": {"model": f"{BASE_MODEL}, no adapter", "recipe": recipe,
                  "demonstrations": len(demos), "demonstration_source": "split_train, "
                  "asserted disjoint from golden", "decoding": "greedy, max_new_tokens as "
                  "router.generate.MAX_TOKENS", "seed": SEED},
    }


def http_generate(base_url: str, task: str, concurrency: int = 8) -> Generate:
    from adapterops.router.generate import MAX_TOKENS, complete

    def generate(prompts: Sequence[str]) -> list[str]:
        def one(p: str) -> str:
            return complete(base_url, BASE_MODEL, p, MAX_TOKENS[task])["prediction"]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, prompts))

    return generate


def _shown(path: Path) -> Path:
    """Repo-relative when it can be; an absolute path otherwise, instead of raising."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def main(task: str, recipe: str = "fewshot", base_url: str = "http://localhost:8000",
         max_model_len: int = 1536, budget_only: bool = False, force: bool = False,
         save_predictions: bool = False) -> int:
    b = budget(task, recipe, max_model_len)
    print(f"  {task}/{recipe}: header {b['header_tokens']} + longest query "
          f"{b['longest_query_tokens']} + {b['max_new_tokens']} new = {b['needed']} tokens "
          f"against a {max_model_len} context — {'fits' if b['fits'] else 'DOES NOT FIT'}")
    if budget_only:
        return 0 if b["fits"] else 1
    if not b["fits"]:
        print(f"  refusing to run: serve with MAX_MODEL_LEN >= {b['needed']}")
        return 1

    out = RUNS_DIR / f"{task}__prompted-{recipe}.json"
    if out.exists() and not force:
        print(f"  {_shown(out)} exists — --force to replace a recorded baseline")
        return 1
    rows: list[dict] | None = [] if save_predictions else None
    result = run(task, http_generate(base_url, task), recipe, collect=rows)
    result["context_budget"] = b
    RUNS_DIR.mkdir(exist_ok=True)
    if rows is not None:
        # Drafting has no metric until the laptop-side judge scores these replies.
        predictions = RUNS_DIR / f"{task}__prompted-{recipe}__predictions.parquet"
        pd.DataFrame(rows).to_parquet(predictions)
        result["predictions_file"] = str(_shown(predictions))
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"  wrote {_shown(out)}")
    return 0
