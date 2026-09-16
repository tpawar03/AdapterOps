"""Where the PII adapter's answers go wrong (TODO.md §1).

A PII answer counts as a success only when every span is exactly right, and by that rule about a quarter of
the served adapter's answers fail. That number cannot say what fails. This gives every emitted and every
gold span one category, so the failures can be read by cause:

- **exact** — right text, right label (the scorer's true positive);
- **label** — right text, wrong label;
- **boundary** — right label, overlapping but different text ("Main Street" for "123 Main Street");
- **boundary_and_label** — overlapping text, different label;
- **spurious** — a span in the text where there is no gold span;
- **invented** — a value that does not occur in the text at all;
- **repeated_value** — a value whose every occurrence an earlier span already claimed;
- **missed** — a gold span nothing overlaps.

**Invented values are invisible to the span F1.** `spans_from_values` drops a value it cannot find, so the
scorer never counts it as a false positive. It is counted here, because a consumer of the output would
still receive it.

Spans are placed exactly as the scorer places them, and pairs are matched greedily in the order above —
exact first — so the exact count reproduces the scorer's strict true positives, which is checked.

    uv run adapterops pii-errors
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from adapterops.eval.spans import Span, parse_model_output, score_spans, spans_from_values

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_FILE = REPO_ROOT / "runs" / "pii__errors.json"
SPANS_FILE = REPO_ROOT / "runs" / "pii__errors__spans.parquet"
SOURCES = {
    "served_v6": "runs/regression__a10-v5-pii-negatives__predictions.parquet",
    "previous_v5": "runs/regression__a10-v5-pii-original__predictions.parquet",
}
ERRORS = ("label", "boundary", "boundary_and_label", "spurious", "invented", "repeated_value", "missed")
DOC_CAUSES = ("label", "boundary", "boundary_and_label", "spurious", "missed")
"""The categories that make the scorer fail a document. Invented and repeated values never do."""
ID_NUMBERS = frozenset({"IDCARDNUM", "DRIVERLICENSENUM", "PASSPORTNUM", "SOCIALNUM", "TAXNUM"})
NAMES = frozenset({"GIVENNAME", "SURNAME"})


def coverage(text: str, gold: list[Span], pred: list[Span]) -> tuple[int, int]:
    """Gold spans with any character no predicted span covers, and gold spans no predicted span touches.

    What a redaction step would leak, whatever the labels: a name split differently between given name and
    surname, or an ID number under the wrong label, is still fully masked."""
    masked = [False] * len(text)
    for p in pred:
        for i in range(p.start, p.end):
            masked[i] = True
    partly = sum(1 for g in gold if not all(masked[g.start:g.end]))
    wholly = sum(1 for g in gold if not any(masked[g.start:g.end]))
    return partly, wholly


def locate(text: str, pairs: list[tuple[str, str]]) -> tuple[list[Span], list[tuple[str, str, str]]]:
    """`spans_from_values`, keeping the values it could not place and why."""
    used: list[tuple[int, int]] = []
    placed: list[Span] = []
    unplaced: list[tuple[str, str, str]] = []
    for label, value in pairs:
        if not value:
            continue
        start, found_anywhere, placed_it = 0, False, False
        while True:
            i = text.find(value, start)
            if i == -1:
                break
            found_anywhere = True
            if not any(a < i + len(value) and i < b for a, b in used):
                placed.append(Span(i, i + len(value), label))
                used.append((i, i + len(value)))
                placed_it = True
                break
            start = i + 1
        # Not a `while ... else`: the loop only ever ends by `break`, so an `else` would never run and every
        # unplaced value would vanish — which the first version did, reporting zero invented values.
        if not placed_it:
            unplaced.append((label, value, "repeated_value" if found_anywhere else "invented"))
    return placed, unplaced


def _overlap(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


MATCH_ORDER = (
    ("exact", lambda g, p: g == p),
    ("label", lambda g, p: (g.start, g.end) == (p.start, p.end)),
    ("boundary", lambda g, p: g.label == p.label and _overlap(g, p)),
    ("boundary_and_label", _overlap),
)


def classify(text: str, gold_text: str, prediction: str) -> list[dict]:
    """One row per gold or emitted span, each with exactly one category."""
    gold = spans_from_values(text, parse_model_output(gold_text))
    preds, unplaced = locate(text, parse_model_output(prediction))
    gold_left, pred_left, rows = list(gold), list(preds), []

    def row(kind: str, g: Span | None, p: Span | None) -> dict:
        return {"kind": kind,
                "gold_label": g.label if g else None, "gold_value": text[g.start:g.end] if g else None,
                "pred_label": p.label if p else None, "pred_value": text[p.start:p.end] if p else None}

    for kind, matches in MATCH_ORDER:
        for p in list(pred_left):
            g = next((g for g in gold_left if matches(g, p)), None)
            if g is not None:
                rows.append(row(kind, g, p))
                gold_left.remove(g)
                pred_left.remove(p)
    rows += [row("spurious", None, p) for p in pred_left]
    rows += [{"kind": reason, "gold_label": None, "gold_value": None, "pred_label": label, "pred_value": value}
             for label, value, reason in unplaced]
    rows += [row("missed", g, None) for g in gold_left]
    return rows


def classify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc, r in enumerate(frame.itertuples()):
        gold = spans_from_values(r.text, parse_model_output(r.gold))
        for item in classify(r.text, r.gold, r.prediction):
            rows.append({"doc": doc, "gold_spans": len(gold), "chars": len(r.text), **item})
        if not gold and not parse_model_output(r.prediction):
            rows.append({"doc": doc, "gold_spans": 0, "chars": len(r.text), "kind": "empty_agreed",
                         "gold_label": None, "gold_value": None, "pred_label": None, "pred_value": None})
    return pd.DataFrame(rows)


def _bins(values: pd.Series, edges: list[int], labels: list[str]) -> pd.Series:
    return pd.cut(values, bins=edges, labels=labels, right=True, include_lowest=True)


def summarise(frame: pd.DataFrame, spans: pd.DataFrame) -> dict:
    """One split: counts by category, the scorer check, per-label recall, causes of failed documents."""
    kinds = spans.kind.value_counts().to_dict()
    exact = kinds.get("exact", 0)
    gold = [spans_from_values(t, parse_model_output(g)) for t, g in zip(frame.text, frame.gold, strict=True)]
    pred = [spans_from_values(t, parse_model_output(p)) for t, p in zip(frame.text, frame.prediction, strict=True)]
    scored = score_spans(gold, pred, strict=True)
    if scored["tp"] != exact:
        msg = f"exact spans {exact} do not reproduce the scorer's strict true positives {scored['tp']}"
        raise ValueError(msg)
    leaks = [coverage(t, g, p) for t, g, p in zip(frame.text, gold, pred, strict=True)]
    confused = spans[spans.kind.isin(["label", "boundary_and_label"])]

    per_doc = spans.groupby("doc").kind.agg(lambda k: set(k) & set(DOC_CAUSES))
    failed = per_doc[per_doc.map(bool)]
    single = Counter(next(iter(c)) for c in failed if len(c) == 1)
    docs = spans.drop_duplicates("doc").set_index("doc")
    docs["perfect"] = ~docs.index.isin(failed.index)

    gold_rows = spans[spans.gold_label.notna()]
    per_label = {}
    for label, g in gold_rows.groupby("gold_label"):
        counts = g.kind.value_counts()
        per_label[label] = {"gold": len(g), "exact": int(counts.get("exact", 0)),
                            "recall": round(float(counts.get("exact", 0) / len(g)), 4),
                            **{k: int(counts.get(k, 0)) for k in ("label", "boundary", "boundary_and_label",
                                                                   "missed")}}
    confusions = Counter((r.gold_label, r.pred_label) for r in
                         spans[spans.kind.isin(["label", "boundary_and_label"])].itertuples())
    boundary = spans[spans.kind == "boundary"]

    def examples(kind: str, columns: list[str], n: int = 8) -> list[dict]:
        return spans[spans.kind == kind][columns].head(n).to_dict("records")

    span_bins = _bins(docs.gold_spans, [0, 2, 5, 9, 1000], ["1-2", "3-5", "6-9", "10+"])
    char_bins = pd.qcut(docs.chars, 4, labels=["shortest quarter", "second", "third", "longest quarter"],
                        duplicates="drop")
    return {
        "documents": len(docs),
        "documents_all_spans_right": int(docs.perfect.sum()),
        "document_success_rate": round(float(docs.perfect.mean()), 4),
        "strict_scorer": {k: scored[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "redaction": {
            "documents_all_pii_text_masked": sum(1 for partly, _ in leaks if partly == 0),
            "gold_spans_partly_unmasked": sum(partly for partly, _ in leaks),
            "gold_spans_wholly_unmasked": sum(wholly for _, wholly in leaks),
            "note": ("any predicted span masks the text it covers, whatever its label — what a redaction "
                     "step would leak, as distinct from what the strict scorer counts wrong"),
        },
        "label_errors_within_id_numbers": int(sum(
            r.gold_label in ID_NUMBERS and r.pred_label in ID_NUMBERS for r in confused.itertuples())),
        "label_errors_gender_sex": int(sum(
            {r.gold_label, r.pred_label} == {"GENDER", "SEX"} for r in confused.itertuples())),
        "boundary_errors_on_names": int((spans[spans.kind == "boundary"].gold_label.isin(NAMES)).sum()),
        "spans_by_category": {k: int(kinds.get(k, 0)) for k in ("exact", *ERRORS)},
        "gold_spans": len(gold_rows),
        "failed_documents": len(failed),
        "failed_documents_by_single_cause": dict(single.most_common()),
        "failed_documents_with_several_causes": int(sum(len(c) > 1 for c in failed)),
        "failed_documents_by_any_cause": dict(Counter(k for c in failed for k in c).most_common()),
        "per_label": dict(sorted(per_label.items(), key=lambda kv: kv[1]["recall"])),
        "label_confusions_gold_to_predicted": [
            {"gold": g, "predicted": p, "count": n} for (g, p), n in confusions.most_common(12)],
        "boundary_examples": boundary[["gold_label", "gold_value", "pred_value"]].head(12).to_dict("records"),
        "boundary_extends_gold": int((boundary.pred_value.str.len() > boundary.gold_value.str.len()).sum()),
        "boundary_shortens_gold": int((boundary.pred_value.str.len() < boundary.gold_value.str.len()).sum()),
        "spurious_by_label": spans[spans.kind == "spurious"].pred_label.value_counts().head(10).to_dict(),
        "spurious_examples": examples("spurious", ["pred_label", "pred_value"]),
        "missed_examples": examples("missed", ["gold_label", "gold_value"]),
        "invented_top_values": spans[spans.kind == "invented"].apply(
            lambda r: f"{r.pred_label}: {r.pred_value}", axis=1).value_counts().head(10).to_dict()
        if (spans.kind == "invented").any() else {},
        "success_by_gold_span_count": docs.groupby(span_bins, observed=True).perfect.agg(
            ["size", "mean"]).round(3).rename(columns={"size": "docs", "mean": "success"}).to_dict("index"),
        "success_by_text_length": docs.groupby(char_bins, observed=True).perfect.agg(
            ["size", "mean"]).round(3).rename(columns={"size": "docs", "mean": "success"}).to_dict("index"),
    }


def main() -> int:
    result, all_spans = {"sources": SOURCES, "splits": {}}, []
    for source, path in SOURCES.items():
        predictions = pd.read_parquet(REPO_ROOT / path)
        for split in ("random", "hard"):
            frame = predictions[(predictions.task == "pii") & (predictions.split == split)].reset_index(drop=True)
            spans = classify_frame(frame)
            result["splits"].setdefault(split, {})[source] = summarise(frame, spans)
            all_spans.append(spans.assign(source=source, split=split))
    for split, by_source in result["splits"].items():
        now, before = by_source["served_v6"], by_source["previous_v5"]
        by_source["served_minus_previous"] = {
            k: now["spans_by_category"][k] - before["spans_by_category"][k] for k in now["spans_by_category"]}
    RUN_FILE.parent.mkdir(exist_ok=True)
    RUN_FILE.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    pd.concat(all_spans, ignore_index=True).to_parquet(SPANS_FILE, index=False)
    for split, by_source in result["splits"].items():
        s = by_source["served_v6"]
        print(f"  {split}: {s['documents_all_spans_right']}/{s['documents']} documents right · "
              f"{s['spans_by_category']} · single causes {s['failed_documents_by_single_cause']}")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
