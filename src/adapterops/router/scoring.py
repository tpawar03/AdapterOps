"""Turn adapter outputs into the router's training label (F7).

The router's label is *computed*: run a (ticket, task) pair through its adapter, score the
output, and the pair is a success or a failure. Two things about that are worth doing
deliberately.

**The expensive half and the revisable half are kept apart.** Generation costs GPU
minutes; deciding what counts as success costs nothing and is a judgement call that may
well change. So the scored table stores score *components* — per-document precision,
recall and F1 for PII, a proxy similarity for drafting, the raw prediction for everything —
and `label()` derives the binary from them here, on a laptop. Changing the success rule is
then a re-run of this module, not a re-run of the GPU.

**The four rules are not equally sound, and the weakest one is named.**

| task | success | free parameters |
|---|---|---|
| intent | predicted label equals gold | none |
| urgency | predicted priority equals gold | none |
| PII | every gold span found, exact boundaries and label, nothing invented | none |
| drafting | non-zero token-F1 against the reference, at or above the pool median | **one, and it is arbitrary** |

PII uses per-document strict F1 = 1.0 rather than a threshold, which is both
parameter-free and the operationally meaningful bar: a document with one span missed is a
document that leaked. The alternative considered was recall = 1.0 — treating a false
positive as over-redaction and a false negative as a breach — which is arguably the better
compliance rule and is a one-line change here because the components are stored.

Drafting is the weak one. Its real label is the distilled judge, which does not exist until
Phase 3, so Phase 2 uses token-F1 against the Bitext reference reply and cuts at the pool
median. That cut is arbitrary and is marked as such in the output. It also carries the one
guard a median cut needs: a median is degenerate when the distribution is, and a run where
every reply scored zero would put the cut at zero and label total failure as total success.
Sharing no tokens with the reference is therefore a disqualifier in its own right rather
than a second threshold to tune. Two things follow, both
of which are done rather than promised: the router's operating curve is reported with and
without drafting pairs, so it is visible how much of the result rests on the weak label;
and when the judge arrives, its agreement with this proxy is measured and reported.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from adapterops.eval.spans import (
    Span,
    parse_model_output,
    score_spans,
    spans_from_values,
)

WORD = re.compile(r"[a-z0-9']+")

PROXY_LABELLED_TASKS = ("drafting",)
"""Tasks whose success label is a stand-in rather than the real metric. Reported
separately everywhere, and replaced in Phase 3."""


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style bag-of-tokens F1. The drafting proxy — crude on purpose and labelled."""
    pred = Counter(WORD.findall(prediction.lower()))
    ref = Counter(WORD.findall(reference.lower()))
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    p = overlap / sum(pred.values())
    r = overlap / sum(ref.values())
    return 2 * p * r / (p + r)


def first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def score_pair(task: str, text: str, gold: str, prediction: str) -> dict:
    """Score one pair into components. No thresholds applied — that is `label`'s job."""
    if task in ("intent", "urgency"):
        pred = first_line(prediction)
        return {"pred_clean": pred, "exact": float(pred == gold.strip())}

    if task == "pii":
        gold_spans = spans_from_values(text, parse_model_output(gold))
        pred_spans = spans_from_values(text, parse_model_output(prediction))
        strict = score_spans([gold_spans], [pred_spans], strict=True)
        relaxed = score_spans([gold_spans], [pred_spans], strict=False)
        return {
            "pred_clean": prediction.strip(),
            "span_precision": strict["precision"],
            "span_recall": strict["recall"],
            "span_f1": strict["f1"],
            "span_f1_relaxed": relaxed["f1"],
            "gold_spans": len(gold_spans),
            "pred_spans": len(pred_spans),
        }

    if task == "drafting":
        return {"pred_clean": prediction.strip(),
                "proxy_token_f1": token_f1(prediction, gold)}

    msg = f"unknown task {task!r}"
    raise ValueError(msg)


def label(scored: pd.DataFrame) -> pd.DataFrame:
    """Derive the binary `success` column, plus `label_is_proxy` marking the weak rule.

    Runs on the whole table at once because the drafting cut is defined relative to that
    task's own distribution — a per-row function could not see it.
    """
    out = scored.copy()
    out["success"] = pd.NA
    out["label_rule"] = ""

    for task in out.task.unique():
        rows = out.task == task
        if task in ("intent", "urgency"):
            out.loc[rows, "success"] = out.loc[rows, "exact"] == 1.0
            out.loc[rows, "label_rule"] = "predicted label equals gold"
        elif task == "pii":
            out.loc[rows, "success"] = out.loc[rows, "span_f1"] >= 1.0
            out.loc[rows, "label_rule"] = "per-document strict span F1 = 1.0"
        elif task == "drafting":
            proxy = out.loc[rows, "proxy_token_f1"]
            cut = proxy.median()
            # A median cut is degenerate when the distribution is: if every reply scores
            # zero, the median is zero and `>= cut` marks total failure as total success.
            # Found by a test that fed empty predictions in. Sharing no tokens at all with
            # the reference is not a success at any cut, so that is a precondition rather
            # than a second threshold to tune.
            out.loc[rows, "success"] = (proxy > 0) & (proxy >= cut)
            degenerate = "" if cut > 0 else "  DEGENERATE: median is zero, "
            out.loc[rows, "label_rule"] = (
                f"PROXY: token-F1 vs reference > 0 and >= pool median ({cut:.4f}) — "
                f"arbitrary cut, replaced by the judge in Phase 3.{degenerate}")

    out["success"] = out["success"].astype(bool)
    out["label_is_proxy"] = out.task.isin(PROXY_LABELLED_TASKS)
    return out


def success_rates(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-task base rates. The number the router has to beat by reading the text."""
    g = labelled.groupby("task")
    return pd.DataFrame({
        "pairs": g.size(),
        "success_rate": g.success.mean().round(4),
        "label_is_proxy": g.label_is_proxy.first(),
        "rule": g.label_rule.first(),
    })


FAILURE_BUCKETS = {
    "intent": ("wrong_label",),
    "urgency": ("wrong_label",),
    "pii": ("no_output", "boundary", "missed", "invented", "mixed"),
    "drafting": ("off_reference",),
}


def failure_type(row) -> str | None:
    """Bucket a failed pair by *how* it failed (F31). None for a success.

    The buckets exist so the hard-cases split is capped per failure mode rather than per
    task: PII fails in four distinguishable ways, and an uncapped mine would fill the split
    with whichever one is most common and call it a hard-cases set.

    `boundary` is separated from `missed` on purpose — it is the difference between a model
    that cannot find a span and one that finds it and draws the edges wrong, and the PII
    baseline work already showed those are different problems with different fixes.
    """
    if bool(row["success"]):
        return None
    task = row["task"]
    if task in ("intent", "urgency"):
        return "wrong_label"
    if task == "drafting":
        return "off_reference"
    if row.get("pred_spans", 0) == 0:
        return "no_output"
    if row.get("span_f1_relaxed", 0.0) > row.get("span_f1", 0.0) + 0.01:
        return "boundary"
    missed = row.get("span_recall", 1.0) < 1.0
    invented = row.get("span_precision", 1.0) < 1.0
    if missed and not invented:
        return "missed"
    if invented and not missed:
        return "invented"
    return "mixed"


def gold_spans_for(text: str, gold: str) -> list[Span]:
    """Exposed for the GPU script, which needs gold spans without re-deriving the parse."""
    return spans_from_values(text, parse_model_output(gold))
