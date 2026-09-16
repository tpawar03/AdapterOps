"""A PII training split with realistic formats (TODO.md §2, train PII on realistic formats).

The served adapter leaves 28% of in-scope spans wholly unmasked on Nemotron-PII's realistic formats and 14% on real
court text, and confidence routing escalates none of them (`runs/pii__realistic.json`). This adds Nemotron-PII
training documents to the served adapter's frozen mix, and freezes the split and its decision rule before any GPU time.

- **Additive.** The served adapter's 13,910 rows (`split_train_negatives`, checked against its sha256) stay as they are.
- **4,000 Nemotron-PII documents** from its train file (CC BY 4.0), 1,000 per locale × format, one of each uid's US
  and international rewrites, at most 200 words, and none whose prompt and answer exceed the 512-token training cap.
  The train and test files share no uid, so the 200-text evaluation sample is never trained on.
- **Targets in the adapter's schema.** The 16 categories with a counterpart become its labels; the rest stay untagged,
  since its label set has no place for them. A street address with a number at either end is split into BUILDINGNUM
  and STREET, ai4privacy's convention.
- **TAB decides.** Nemotron-PII's test sample now shares the training data's distribution, so a fall there proves
  little. TAB's real court text is never trained on, so the rule rests on it.

    uv run adapterops pii-realistic-split
    python -m adapterops.train.cli_train --task pii --train-split train_realistic --variant realistic \\
        --subsample 100000 --epochs 3
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from adapterops.eval.pii_realistic import FILES, MAX_WORDS, NEMOTRON_LABELS

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260915
BASE_RECORD = REPO_ROOT / "data" / "pii" / "split_train_negatives.json"
NEMOTRON_TRAIN = REPO_ROOT / "data" / "pii_realistic" / "nemotron_train.parquet"
NEMOTRON_TRAIN_SHA256 = "9cb1a88c339ab84ddbc35ccdde4f6bb713300f5585881f9e02b8db1e0dfdbece"
OUT = REPO_ROOT / "data" / "pii" / "split_train_realistic.parquet"
RECORD = REPO_ROOT / "data" / "pii" / "split_train_realistic.json"
PER_CELL = 1000
MAX_SEQ = 512
"""TASK_CONFIGS["pii"]["max_seq_length"]: a longer example would have its answer truncated in training."""
LEADING = re.compile(r"^(\d+[A-Za-z]?)\s+(\S.*)$")
TRAILING = re.compile(r"^(.*?[^\s,]),?\s+(\d+[A-Za-z]?)$")


def address_parts(value: str) -> list[tuple[str, str]]:
    if match := LEADING.match(value):
        return [("BUILDINGNUM", match.group(1)), ("STREET", match.group(2))]
    if match := TRAILING.match(value):
        return [("STREET", match.group(1)), ("BUILDINGNUM", match.group(2))]
    return [("STREET", value)]


def target(text: str, spans: list[dict]) -> str:
    """The adapter's answer for a Nemotron-PII document: mapped categories only, in text order."""
    lines = []
    for span in sorted(spans, key=lambda s: s["start"]):
        label = NEMOTRON_LABELS.get(span["label"])
        value = text[span["start"]:span["end"]].strip()
        if label is None or not value:
            continue
        parts = address_parts(value) if span["label"] == "street_address" else [(label, value)]
        lines += [f"{lb}: {v}" for lb, v in parts]
    return "\n".join(lines)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> tuple[pd.DataFrame, dict]:
    from transformers import AutoTokenizer

    from adapterops.train.qlora import PROMPTS

    record = json.loads(BASE_RECORD.read_text())
    if _sha(REPO_ROOT / record["file"]) != record["sha256"]:
        msg = f"{record['file']} is not the served adapter's frozen split"
        raise ValueError(msg)
    if _sha(NEMOTRON_TRAIN) != NEMOTRON_TRAIN_SHA256:
        msg = f"{NEMOTRON_TRAIN} is not the pinned Nemotron-PII train file"
        raise ValueError(msg)
    base = pd.read_parquet(REPO_ROOT / record["file"])
    train = pd.read_parquet(NEMOTRON_TRAIN)
    if shared := set(train.uid) & set(pd.read_parquet(FILES["nemotron"]).uid):
        msg = f"{len(shared)} uids are in both Nemotron-PII files; the evaluation sample could be trained on"
        raise ValueError(msg)

    base_meta = json.loads((REPO_ROOT / "manifests" / "system.json").read_text())["components"]["base_model"]
    tok = AutoTokenizer.from_pretrained(base_meta["repo"], revision=base_meta["revision"])
    train = train.sample(frac=1, random_state=SEED).drop_duplicates("uid")
    train = train[train.text.str.split().str.len() <= MAX_WORDS]
    picked, too_long = [], 0
    for _, cell in train.groupby(["locale", "document_format"]):
        kept = []
        for row in cell.itertuples():
            answer = target(row.text, ast.literal_eval(row.spans))
            tokens = (len(tok(PROMPTS["pii"].format(text=row.text), add_special_tokens=False)["input_ids"])
                      + len(tok(" " + answer, add_special_tokens=False)["input_ids"]) + 1)
            if tokens > MAX_SEQ:
                too_long += 1
                continue
            kept.append({"source_text": row.text, "target": answer, "tokens": tokens})
            if len(kept) == PER_CELL:
                break
        if len(kept) < PER_CELL:
            msg = f"only {len(kept)} documents fit the cap in one cell"
            raise ValueError(msg)
        picked += kept
    nemotron = pd.DataFrame(picked)
    rows = pd.concat([base[["source_text", "target", "kind"]],
                      nemotron[["source_text", "target"]].assign(kind="nemotron_document")],
                     ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    labels = pd.Series([line.split(":", 1)[0] for t in nemotron.target for line in t.splitlines()]).value_counts()
    facts = {
        "rows": rows.kind.value_counts().to_dict(),
        "nemotron_documents_with_empty_answer": int((nemotron.target == "").sum()),
        "nemotron_skipped_over_token_cap": too_long,
        "nemotron_tokens": {q: int(nemotron.tokens.quantile(q)) for q in (0.5, 0.95, 1.0)},
        "nemotron_label_counts": labels.to_dict(),
    }
    return rows, facts


def main(force: bool = False) -> int:
    if RECORD.exists() and not force:
        print(f"  {RECORD.relative_to(REPO_ROOT)} exists — the split is frozen; --force rebuilds it")
        return 1
    rows, facts = build()
    rows.to_parquet(OUT, index=False)
    threshold = json.loads((REPO_ROOT / "evals" / "GATE_THRESHOLDS.json").read_text())["gate"]["thresholds"]["pii"]
    RECORD.write_text(json.dumps({
        "purpose": "PII training split with Nemotron-PII's realistic formats added (TODO.md §2).",
        "seed": SEED,
        "base": {"file": json.loads(BASE_RECORD.read_text())["file"],
                 "sha256": json.loads(BASE_RECORD.read_text())["sha256"]},
        "nemotron_train": {"url": "https://huggingface.co/datasets/nvidia/Nemotron-PII", "licence": "CC BY 4.0",
                           "revision": "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed",
                           "file": "data/train-00000-of-00001.parquet", "sha256": NEMOTRON_TRAIN_SHA256},
        "file": str(OUT.relative_to(REPO_ROOT)),
        "sha256": _sha(OUT),
        "label_map": NEMOTRON_LABELS,
        **facts,
        "preregistered": {
            "decision_rule": (
                "Replace the served PII adapter only if both hold: (1) the candidate's random-split strict span F1, "
                "regression-scored beside the served adapter in one session, drops by no more than the enforced "
                f"gate threshold ({threshold}); and (2) on TAB's 200 frozen court-text paragraphs, never trained on, "
                "its rate of in-scope spans left wholly unmasked falls, with the paired bootstrap's 95% interval "
                "(10,000 resamples of texts, `adapterops pii-realistic-compare`) entirely below zero."),
            "reported_whatever_they_are": [
                "Nemotron-PII test sample leak rates (in-distribution for this split, so not decisive)",
                "false-positive rate on evals/pii_negatives/pii_negatives.parquet (928 texts)",
                "false-positive rate on the held-out ai4privacy validation sentences (491)",
                "hard-split span F1 and grouped span F1 (report-only)",
            ],
            "no_target_rate": ("No leak rate is set as a bar in advance: any fall TAB can distinguish from noise "
                               "counts, and any number chosen now would be invented."),
            "known_limits": [
                "Nemotron-PII is synthetic; only TAB is real text, and its categories are coarser than the adapter's.",
                ("Categories without a counterpart (company names, URLs, account numbers) are trained as untagged, "
                 "which may teach the adapter to skip ID-like values it would otherwise mask."),
            ],
        },
        "train_with": ("python -m adapterops.train.cli_train --task pii --train-split train_realistic "
                       "--variant realistic --subsample 100000 --epochs 3"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}: {facts['rows']}")
    print(f"  Nemotron-PII: {facts['nemotron_documents_with_empty_answer']} empty answers, "
          f"{facts['nemotron_skipped_over_token_cap']} skipped over the cap, tokens {facts['nemotron_tokens']}")
    return 0
