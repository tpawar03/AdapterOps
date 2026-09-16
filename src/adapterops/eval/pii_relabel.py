"""Relabel confusable PII spans by the cue word before them (TODO.md §1, the two label confusions).

Most of the served PII adapter's label errors are two confusions whose values cannot decide the label: "Male"
and "Female" appear under both GENDER and SEX, and 10-character codes appear under both IDCARDNUM and
DRIVERLICENSENUM. The context can: "gender" versus "sex", "driver licence" versus "id card". `relabel` keeps
every span's text and changes only the label of a span in one of those groups, to the label of the nearest cue
word before it — and only within the group.

`main` checks the rule on the training split's gold spans (how often the nearest cue gives the gold label), then
applies it to the served adapter's golden and hard predictions and reports label errors fixed, correct spans
broken and strict span F1 before and after.

    uv run adapterops pii-relabel
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from adapterops.eval.spans import Span, parse_model_output, score_spans, spans_from_values

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_FILE = REPO_ROOT / "runs" / "pii__relabel.json"
PREDICTIONS = REPO_ROOT / "runs" / "regression__a10-v5-pii-negatives__predictions.parquet"
WINDOW = 60
"""Characters before a span searched for a cue word."""

GROUPS = {
    "sex_gender": {
        "GENDER": re.compile(r"\bgender\b", re.IGNORECASE),
        "SEX": re.compile(r"\bsex\b", re.IGNORECASE),
    },
    "id_numbers": {
        "DRIVERLICENSENUM": re.compile(r"\bdriv(?:er|ing)'?s?\b|\blicen[cs]e\b", re.IGNORECASE),
        "PASSPORTNUM": re.compile(r"\bpassport\b", re.IGNORECASE),
        "SOCIALNUM": re.compile(r"\bsocial\b|\bsecurity number\b|\bssn\b|\binsurance number\b", re.IGNORECASE),
        "TAXNUM": re.compile(r"\btax\b|\btin\b", re.IGNORECASE),
        "IDCARDNUM": re.compile(r"\bid\b|\bidentity\b|\bnational id\b|\bid card\b", re.IGNORECASE),
    },
}
GROUP_OF = {label: group for group, cues in GROUPS.items() for label in cues}


def cue_label(text: str, span: Span) -> str | None:
    """The label of the cue word nearest before `span`, within the span's group; None without one."""
    group = GROUP_OF.get(span.label)
    if group is None:
        return None
    window_start = max(0, span.start - WINDOW)
    before = text[window_start:span.start]
    best, best_end = None, -1
    for label, pattern in GROUPS[group].items():
        for match in pattern.finditer(before):
            if match.end() > best_end:
                best, best_end = label, match.end()
    return best


def relabel(text: str, prediction: str) -> str:
    """The answer with each confusable span's label set by its nearest preceding cue word."""
    out, placed = [], []
    for label, value in parse_model_output(prediction):
        # Placing the pairs so far reproduces spans_from_values, which places left to right.
        now = spans_from_values(text, [*placed, (label, value)])
        new = cue_label(text, now[-1]) if len(now) > len(spans_from_values(text, placed)) else None
        placed.append((label, value))
        out.append(f"{new or label}: {value}")
    return "\n".join(out)


def cue_accuracy(frame: pd.DataFrame) -> dict:
    """On gold spans: how often the nearest cue gives the gold label, per group (a span with no cue counts)."""
    counts = {g: {"spans": 0, "cue_found": 0, "cue_right": 0} for g in GROUPS}
    for text, mask in zip(frame.source_text.astype(str), frame.privacy_mask, strict=True):
        for m in mask:
            span = Span(int(m["start"]), int(m["end"]), m["label"])
            group = GROUP_OF.get(span.label)
            if group is None:
                continue
            counts[group]["spans"] += 1
            predicted = cue_label(text, span)
            counts[group]["cue_found"] += predicted is not None
            counts[group]["cue_right"] += predicted == span.label
    return {g: {**c, "accuracy_when_found": round(c["cue_right"] / c["cue_found"], 4) if c["cue_found"] else None}
            for g, c in counts.items()}


def measure(frame: pd.DataFrame) -> dict:
    gold, before, after, fixed, broken = [], [], [], 0, 0
    for text, g, p in zip(frame.text, frame.gold, frame.prediction, strict=True):
        gs = spans_from_values(text, parse_model_output(g))
        ps = spans_from_values(text, parse_model_output(p))
        rs = spans_from_values(text, parse_model_output(relabel(text, p)))
        gold_set = set(gs)
        fixed += sum(1 for b, r in zip(ps, rs, strict=False) if b not in gold_set and r in gold_set)
        broken += sum(1 for b, r in zip(ps, rs, strict=False) if b in gold_set and r not in gold_set)
        gold.append(gs), before.append(ps), after.append(rs)
    return {"span_f1_before": score_spans(gold, before, strict=True)["f1"],
            "span_f1_after": score_spans(gold, after, strict=True)["f1"],
            "spans_fixed": fixed, "spans_broken": broken}


def main() -> int:
    train = pd.read_parquet(REPO_ROOT / "data" / "pii" / "split_train.parquet")
    predictions = pd.read_parquet(PREDICTIONS)
    result = {"window_chars": WINDOW, "cue_accuracy_on_training_gold": cue_accuracy(train), "splits": {}}
    for split in ("random", "hard"):
        frame = predictions[(predictions.task == "pii") & (predictions.split == split)].reset_index(drop=True)
        result["splits"][split] = measure(frame)
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  cue accuracy on training gold: {result['cue_accuracy_on_training_gold']}")
    for split, m in result["splits"].items():
        print(f"  {split}: F1 {m['span_f1_before']} → {m['span_f1_after']} · fixed {m['spans_fixed']} · "
              f"broken {m['spans_broken']}")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
