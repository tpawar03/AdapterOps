"""A second detector for what the PII adapter misses (TODO.md §1, the confident-error guard).

Confidence routing cannot flag a PII answer that leaves personal data unmasked: a missed span has no score.
The regex patterns from `eval/pii_baseline.py` can find some of what the adapter misses. `augment` adds a
pattern's span to the adapter's answer only where the adapter tagged nothing, so the adapter's own spans are
never changed.

`main` measures three label sets on the served adapter's golden and hard predictions — leaks removed, and
what the added spans cost in strict span F1 — and on customer text with numbers in it (every intent and
drafting golden text), where an added span is almost always a false positive.

    uv run adapterops pii-guard
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from adapterops.eval.pii_baseline import predict as pattern_spans
from adapterops.eval.pii_errors import coverage
from adapterops.eval.spans import parse_model_output, score_spans, spans_from_values

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_FILE = REPO_ROOT / "runs" / "pii__guard.json"
PREDICTIONS = REPO_ROOT / "runs" / "regression__a10-v5-pii-negatives__predictions.parquet"
LABEL_SETS = {
    "structured": ("EMAIL", "CREDITCARDNUMBER", "SOCIALNUM", "PASSPORTNUM", "TELEPHONENUM"),
    "structured_ids_dates": ("EMAIL", "CREDITCARDNUMBER", "SOCIALNUM", "PASSPORTNUM", "TELEPHONENUM",
                             "DATE", "TAXNUM", "DRIVERLICENSENUM", "IDCARDNUM"),
    "all_patterns": ("EMAIL", "CREDITCARDNUMBER", "SOCIALNUM", "PASSPORTNUM", "TELEPHONENUM",
                     "DATE", "TAXNUM", "DRIVERLICENSENUM", "IDCARDNUM", "ZIPCODE"),
}


def augment(text: str, prediction: str, labels: Iterable[str]) -> str:
    """The adapter's answer plus a `LABEL: value` line for each pattern match in `labels` that no span the
    adapter placed overlaps."""
    wanted = set(labels)
    placed = spans_from_values(text, parse_model_output(prediction))
    extra = [s for s in pattern_spans(text)
             if s.label in wanted and not any(s.start < p.end and p.start < s.end for p in placed)]
    lines = [prediction.rstrip("\n")] if prediction.strip() else []
    lines += [f"{s.label}: {text[s.start:s.end].strip()}" for s in extra if text[s.start:s.end].strip()]
    return "\n".join(lines)


def measure(frame: pd.DataFrame, labels: Iterable[str] | None) -> dict:
    labels = tuple(labels or ())
    gold, pred, added = [], [], 0
    for text, g, p in zip(frame.text, frame.gold, frame.prediction, strict=True):
        answer = augment(text, p, labels) if labels else p
        added += len(parse_model_output(answer)) - len(parse_model_output(p))
        gold.append(spans_from_values(text, parse_model_output(g)))
        pred.append(spans_from_values(text, parse_model_output(answer)))
    leaks = [coverage(t, g, p) for t, g, p in zip(frame.text, gold, pred, strict=True)]
    return {"spans_added": added,
            "span_f1_strict": score_spans(gold, pred, strict=True)["f1"],
            "docs_fully_masked": sum(partly == 0 for partly, _ in leaks),
            "gold_spans_partly_unmasked": sum(p for p, _ in leaks),
            "gold_spans_wholly_unmasked": sum(w for _, w in leaks)}


def false_additions(texts: list[str], labels: Iterable[str]) -> dict:
    flagged = [(t, augment(t, "", labels)) for t in texts]
    flagged = [(t, a) for t, a in flagged if a]
    return {"texts": len(texts), "texts_with_an_added_span": len(flagged),
            "examples": [{"text": t[:120], "added": a} for t, a in flagged[:8]]}


def main() -> int:
    predictions = pd.read_parquet(PREDICTIONS)
    customer_text = (list(pd.read_parquet(REPO_ROOT / "evals/golden/intent.parquet").text.astype(str))
                     + list(pd.read_parquet(REPO_ROOT / "evals/golden/drafting.parquet").instruction.astype(str)))
    result = {"predictions": str(PREDICTIONS.relative_to(REPO_ROOT)), "label_sets": LABEL_SETS, "splits": {},
              "customer_text_false_additions": {}}
    for split in ("random", "hard"):
        frame = predictions[(predictions.task == "pii") & (predictions.split == split)].reset_index(drop=True)
        result["splits"][split] = {"adapter_only": measure(frame, None)}
        for name, labels in LABEL_SETS.items():
            result["splits"][split][name] = measure(frame, labels)
    for name, labels in LABEL_SETS.items():
        result["customer_text_false_additions"][name] = false_additions(customer_text, labels)
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split, rows in result["splits"].items():
        for name, m in rows.items():
            print(f"  {split:6s} {name:22s} F1 {m['span_f1_strict']:.4f} · fully masked {m['docs_fully_masked']} · "
                  f"partly unmasked {m['gold_spans_partly_unmasked']} · wholly {m['gold_spans_wholly_unmasked']} · "
                  f"added {m['spans_added']}")
    for name, f in result["customer_text_false_additions"].items():
        print(f"  customer text {name:22s} {f['texts_with_an_added_span']} of {f['texts']} texts gain a span")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
