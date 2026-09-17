"""Out-of-domain evaluation of the served system (TODO.md §3).

Every score in the project is on held-out data from the dataset each model trained on. PII was the one task tested
outside it, and it leaked 28% of spans before a retrain aimed at exactly that. This measures the other tasks on
tickets from sources none of the training datasets came from, through the same request path that serves them.

**Two sources, chosen to fail in different ways.**
- **ABCD** (ASAPP Research, MIT): online-shop customer-service dialogues. A new domain for every task. The
  customer's own words are the ticket, and each conversation records the customer's true details, so every
  name, email, phone, address and account ID the customer types becomes a PII gold span without labelling.
- **CFPB complaints** (US government, public domain, via a CC0 subset): real consumer complaints about banks,
  lenders and collectors — intent's own domain in real, messy wording. Personal data is pre-redacted as XXXX,
  so it tests PII only for false positives.

**Frozen before any output is read.** 150 tickets per source, stratified (15 per ABCD flow, 21 or 22 per CFPB
product), at most 200 words like the training data, seeded; written once to `evals/ood/sample.parquet` and
refused if a rebuild differs.

**Local, through the served manifest.** `generate` builds the real service (transformers backend, the v8
pins, the PII guard, the distilled judge, no frontier calls) and records each pair's output, confidence score
and whether the operating threshold would escalate it. Labels for intent and urgency come later from GPT-4o
(`label`), because neither source carries labels in the project's schema.

    uv run adapterops ood --sample      # freeze the sample
    uv run adapterops ood --generate    # ~1 h on this Mac, resumable
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "ood"
SOURCES = {
    "abcd": {"url": "https://github.com/asappresearch/abcd", "licence": "MIT",
             "revision": "6b8700ce67c6b37b062dd7a60abc76d7ef832a97", "file": "data/abcd_v1.1.json.gz",
             "local": DATA / "abcd_v1.1.json.gz",
             "sha256": "2bdf53ac359543dcdc38d55bc6513e78df120363f8f44870716e909f4606de15"},
    "cfpb": {"url": "https://huggingface.co/datasets/liri-uzh/cfpb-complaints-mini",
             "licence": "CC0-1.0 (CFPB narratives are US public domain)",
             "revision": "9add3032b668ffe3bbdf546788c9c19e489d84a6", "file": "cfpb_mini_test.parquet",
             "local": DATA / "cfpb_mini_test.parquet",
             "sha256": "d1ca76d3bc9226913f912be5357f48e93f361752fd5bbda9b3916ff8d4e5b950"},
}
SAMPLE_FILE = REPO_ROOT / "evals" / "ood" / "sample.parquet"
OUTPUTS_FILE = REPO_ROOT / "runs" / "ood__outputs.parquet"
PER_ABCD_FLOW = 15
PER_SOURCE = 150
MAX_WORDS = 200
SEED = 20260916
TASKS = ("intent", "urgency", "pii", "drafting")

IN_SCOPE = {"email": "EMAIL", "phone": "TELEPHONENUM", "street_address": "STREET", "city": "CITY",
            "zip_code": "ZIPCODE"}
"""ABCD scenario fields with a counterpart among the PII adapter's labels. The customer's name is added as
GIVENNAME and SURNAME in `abcd_gold`."""
OUT_OF_SCOPE = ("username", "account_id", "order_id", "pin_number", "password", "security_answer")
"""Identifiers a redaction step should arguably mask but the adapter has no label for — reported apart."""


def _verified(source: str) -> Path:
    spec = SOURCES[source]
    path = spec["local"]
    if not path.exists():
        msg = f"{path} missing: download {spec['file']} from {spec['url']} at {spec['revision']}"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        msg = f"{path.name} is not the pinned file ({digest[:12]} vs {spec['sha256'][:12]})"
        raise ValueError(msg)
    return path


def _add(gold: list[dict], text: str, value: str, label: str, scope: str, whole_word: bool = False) -> None:
    """Every case-insensitive occurrence of `value` not already covered by a gold span. `whole_word` keeps a
    name part from matching inside a longer token — "chloe" in chloe@example.com or chloezhang81."""
    value = (value or "").strip()
    if len(value) < 2:
        return
    pattern = re.escape(value)
    if whole_word:
        pattern = rf"(?<![\w@.]){pattern}(?![\w@])"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        i, j = match.span()
        if not any(g["start"] < j and i < g["end"] for g in gold):
            gold.append({"start": i, "end": j, "label": label, "scope": scope})


def abcd_gold(text: str, scenario: dict) -> list[dict]:
    personal, order = scenario.get("personal", {}), scenario.get("order", {})
    gold: list[dict] = []
    name = (personal.get("customer_name") or "").split()
    if name:
        # The full name first, split into its parts.
        full = re.escape(" ".join(name))
        for match in re.finditer(rf"(?<![\w@.]){full}(?![\w@])", text, flags=re.IGNORECASE):
            i, j = match.span()
            gold.append({"start": i, "end": i + len(name[0]), "label": "GIVENNAME", "scope": "in"})
            if len(name) > 1:
                gold.append({"start": j - len(name[-1]), "end": j, "label": "SURNAME", "scope": "in"})
    # Whole identifiers before name parts, so a name can never claim part of an email or username.
    for field, label in IN_SCOPE.items():
        _add(gold, text, personal.get(field) or order.get(field) or "", label, "in")
    for field in OUT_OF_SCOPE:
        _add(gold, text, personal.get(field) or order.get(field) or "", field.upper(), "out")
    if name:
        _add(gold, text, name[0], "GIVENNAME", "in", whole_word=True)
        if len(name) > 1:
            _add(gold, text, name[-1], "SURNAME", "in", whole_word=True)
    return sorted(gold, key=lambda g: g["start"])


def abcd_sample(conversations: list[dict]) -> pd.DataFrame:
    rows = []
    for c in conversations:
        text = " ".join(" ".join(t.split()) for speaker, t in c["original"] if speaker == "customer")
        if not text or len(text.split()) > MAX_WORDS:
            continue
        rows.append({"source": "abcd", "id": f"abcd:{c['convo_id']}", "stratum": c["scenario"]["flow"],
                     "source_label": c["scenario"]["subflow"], "text": text,
                     "pii_gold": json.dumps(abcd_gold(text, c["scenario"]))})
    frame = pd.DataFrame(rows)
    # An explicit loop, not groupby().apply(): pandas drops the grouping column from apply's result.
    parts = [frame[frame.stratum == flow].sample(min(PER_ABCD_FLOW, int((frame.stratum == flow).sum())),
                                                 random_state=SEED)
             for flow in sorted(frame.stratum.unique())]
    return pd.concat(parts).reset_index(drop=True)


def cfpb_sample(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame.text.astype(str).str.split().str.len() <= MAX_WORDS].reset_index(drop=True)
    labels = sorted(frame.label.unique())
    base, extra = divmod(PER_SOURCE, len(labels))
    parts = [frame[frame.label == label].sample(base + (i < extra), random_state=SEED)
             for i, label in enumerate(labels)]
    picked = pd.concat(parts)
    return pd.DataFrame({"source": "cfpb", "id": [f"cfpb:{i}" for i in picked.index],
                         "stratum": picked.label.to_numpy(), "source_label": picked.product_raw.to_numpy(),
                         "text": picked.text.astype(str).to_numpy(), "pii_gold": "[]"})


def build_sample() -> pd.DataFrame:
    with gzip.open(_verified("abcd")) as fh:
        conversations = json.load(fh)["test"]
    cfpb = pd.read_parquet(_verified("cfpb"))
    return pd.concat([abcd_sample(conversations), cfpb_sample(cfpb)], ignore_index=True)


def freeze(sample: pd.DataFrame, path: Path = SAMPLE_FILE) -> pd.DataFrame:
    """Write the sample once; afterwards return the frozen one, refusing a rebuild that differs."""
    if path.exists():
        frozen = pd.read_parquet(path)
        key = ["id", "text"]
        if not frozen[key].reset_index(drop=True).equals(sample[key].reset_index(drop=True)):
            msg = f"{path.name} is frozen and a rebuild differs from it — delete it deliberately to re-sample"
            raise ValueError(msg)
        return frozen
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(path, index=False)
    return sample


def pair_rows(ticket: dict, result: dict, threshold_key: str = "threshold") -> list[dict]:
    """One row per (ticket, task) from a `Service.handle` result."""
    rows = []
    for task, pair in result["pairs"].items():
        score, threshold = pair.get("score"), pair.get(threshold_key)
        rows.append({"id": ticket["id"], "source": ticket["source"], "task": task,
                     "output": json.dumps(pair.get("output")), "valid": pair.get("valid"),
                     "score": score, "threshold": threshold,
                     "would_escalate": score is not None and threshold is not None and score >= threshold,
                     "judge_score": pair.get("judge_score"), "latency_ms": pair.get("latency_ms"),
                     "local_error": pair.get("local_error"), "notes": json.dumps(pair.get("notes") or []),
                     "manifest_version": result.get("manifest_version")})
    return rows


def generate(checkpoint_every: int = 10) -> int:
    """Every sampled ticket through the served system, resumable: finished tickets are skipped."""
    from adapterops.serve.pipeline import build_service

    sample = pd.read_parquet(SAMPLE_FILE)
    done = pd.read_parquet(OUTPUTS_FILE) if OUTPUTS_FILE.exists() else pd.DataFrame(columns=["id"])
    finished = set(done.id)
    todo = sample[~sample.id.isin(finished)]
    print(f"  {len(finished) // len(TASKS) if len(finished) else 0} tickets done, {len(todo)} to go", flush=True)
    service = build_service(backend="transformers", frontier=False, judge=True)
    rows = done.to_dict("records")
    for n, ticket in enumerate(todo.to_dict("records"), start=1):
        rows += pair_rows(ticket, service.handle(ticket["text"], TASKS, ticket_id=ticket["id"]))
        if n % checkpoint_every == 0 or n == len(todo):
            pd.DataFrame(rows).to_parquet(OUTPUTS_FILE, index=False)
            print(f"  {len(finished) + n}/{len(sample)} tickets", flush=True)
    return 0


def main(sample: bool = False, run: bool = False) -> int:
    if sample:
        frozen = freeze(build_sample())
        spans = [json.loads(g) for g in frozen.pii_gold]
        record = {
            "sources": {k: {x: v for x, v in s.items() if x != "local"} for k, s in SOURCES.items()},
            "sample": {"file": str(SAMPLE_FILE.relative_to(REPO_ROOT)),
                       "sha256": hashlib.sha256(SAMPLE_FILE.read_bytes()).hexdigest(),
                       "tickets": frozen.source.value_counts().to_dict(), "max_words": MAX_WORDS, "seed": SEED,
                       "strata": frozen.groupby("source").stratum.nunique().to_dict()},
            "pii_gold_spans": {
                "in_scope": sum(g["scope"] == "in" for s in spans for g in s),
                "out_of_scope": sum(g["scope"] == "out" for s in spans for g in s),
                "abcd_tickets_with_in_scope_span": sum(any(g["scope"] == "in" for g in s) for s in spans)},
        }
        out = REPO_ROOT / "runs" / "ood__sample.json"
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record["sample"], indent=1))
        print(json.dumps(record["pii_gold_spans"]))
    if run:
        return generate()
    return 0
