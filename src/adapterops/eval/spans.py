"""Span-level scoring for the PII task (PRD v2.5, changelog 24-25).

F3 was reframed from a binary flag to span detection because the data has no negatives —
see STATUS.md D21. The gated metric is therefore span-level F1, not macro-F1.

**Two things live here, and the second is the awkward one.**

`score_spans` is ordinary NER scoring: a predicted span counts as correct when it matches a
gold span. *Strict* requires identical offsets and label; *relaxed* accepts any character
overlap with the right label. Strict is the gate; relaxed is a diagnostic — a large gap
between them means the model finds the right things but draws boundaries badly, which is a
different problem from missing them.

`spans_from_values` converts the model's output — one `LABEL: value` line per detected
item — into offsets. `spans_from_masked` does the same for ai4privacy-style masked text
and **is kept only as evidence for why that format was rejected** (D22): masked text caps
strict F1 at **0.8973** even when fed ai4privacy's own ground truth, because the boundary
between adjacent spans is genuinely ambiguous. `Sarhat Shegë Böhmerle Cekci` masked as
`[GIVENNAME_1] [SURNAME_1]` has one space between the placeholders and three in the source;
nothing says which is the boundary. `LABEL: value` output caps at **0.9974**.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

PLACEHOLDER = re.compile(r"\[([A-Z][A-Z_]*?)(?:_\d+)?\]")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

    def overlaps(self, other: Span) -> bool:
        return self.label == other.label and self.start < other.end and other.start < self.end


def spans_from_masked(source: str, masked: str) -> tuple[list[Span], bool]:
    """Recover character spans by aligning masked output against the source.

    Returns (spans, recovered). `recovered` is False when the literal text between
    placeholders cannot be found in order in the source — which means the model altered
    text it should have copied, and no offsets can be trusted.
    """
    spans: list[Span] = []
    cursor = 0          # position in source
    last_end = 0        # end of the previous literal chunk in source
    pos = 0             # position in masked

    for m in PLACEHOLDER.finditer(masked):
        literal = masked[pos : m.start()]
        if literal:
            found = source.find(literal, cursor)
            if found == -1:
                return [], False
            last_end = found + len(literal)
        else:
            last_end = cursor
        # the span begins where the literal ended; its extent is decided by the next literal
        span_start = last_end
        next_literal_start = m.end()
        next_m = PLACEHOLDER.search(masked, next_literal_start)
        tail = masked[next_literal_start : next_m.start() if next_m else len(masked)]
        if tail:
            nxt = source.find(tail, span_start)
            if nxt == -1:
                return [], False
            span_end = nxt
        else:
            span_end = len(source)
        spans.append(Span(span_start, span_end, m.group(1)))
        cursor = span_end
        pos = next_literal_start

    return spans, True


def spans_from_values(source: str, pairs: list[tuple[str, str]]) -> list[Span]:
    """Locate each `(label, value)` the model emitted, left to right.

    A value already claimed by an earlier span is skipped, so a name appearing twice maps
    to two spans rather than both landing on the first occurrence. A value absent from the
    source is dropped — it is a hallucination, and counting it as a span with invented
    offsets would credit the model for inventing text.
    """
    used: list[tuple[int, int]] = []
    out: list[Span] = []
    for label, value in pairs:
        if not value:
            continue
        start = 0
        while True:
            i = source.find(value, start)
            if i == -1:
                break
            if not any(a < i + len(value) and i < b for a, b in used):
                out.append(Span(i, i + len(value), label))
                used.append((i, i + len(value)))
                break
            start = i + 1
    return out


def parse_model_output(text: str) -> list[tuple[str, str]]:
    """Parse `LABEL: value` lines. Unparseable lines are dropped, not guessed at."""
    pairs = []
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        label = label.strip()
        if sep and label and label.replace("_", "").isalpha() and label.isupper():
            pairs.append((label, value.strip()))
    return pairs


def score_spans(
    gold: list[list[Span]], pred: list[list[Span]], *, strict: bool = True
) -> dict:
    """Micro precision/recall/F1 over all documents, plus a per-label breakdown."""
    if len(gold) != len(pred):
        msg = f"{len(gold)} gold documents but {len(pred)} predicted"
        raise ValueError(msg)

    tp = fp = fn = 0
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for g_doc, p_doc in zip(gold, pred, strict=True):
        remaining = list(g_doc)
        for p in p_doc:
            hit = next(
                (g for g in remaining if (g == p) if strict) if strict
                else (g for g in remaining if g.overlaps(p)),
                None,
            )
            if hit is not None:
                remaining.remove(hit)
                tp += 1
                per_label[p.label][0] += 1
            else:
                fp += 1
                per_label[p.label][1] += 1
        for g in remaining:
            fn += 1
            per_label[g.label][2] += 1

    def prf(t: int, f_p: int, f_n: int) -> dict:
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "tp": t, "fp": f_p, "fn": f_n}

    out = prf(tp, fp, fn)
    out["mode"] = "strict" if strict else "relaxed"
    out["per_label"] = {
        lbl: prf(*counts) for lbl, counts in sorted(per_label.items())
    }
    return out
