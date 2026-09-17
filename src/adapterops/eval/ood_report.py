"""Scores for the out-of-domain evaluation (TODO.md §3), against the in-domain numbers they test.

Reads the served system's outputs (`ood.generate`) and GPT-4o's labels (`ood_label`), and reports per source:

- **intent** — how often no Banking77 label fits (the served adapter must still answer one), accuracy where one
  does, and whether confidence routing would escalate the wrong answers rather than the right ones;
- **urgency** — the served model's macro F1 against GPT-4o's labels (served by TF-IDF since manifest v8);
- **PII** — on ABCD, spans left wholly or partly unmasked with the same scorer as the realistic-format test; on
  CFPB, where personal data is pre-redacted, how many texts gain a span and how many spans land on redaction marks;
- **drafting** — the distilled judge's mean, and on 60 replies GPT-4o's grade beside the judge's, so the judge's
  agreement on unseen text is measured rather than assumed.

    uv run adapterops ood --report
"""

from __future__ import annotations

import json
import re
import statistics

import pandas as pd

from adapterops.eval.ood import OUTPUTS_FILE, REPO_ROOT, SAMPLE_FILE
from adapterops.eval.ood_label import LABELS_FILE

RUN_FILE = REPO_ROOT / "runs" / "ood.json"
REDACTION = re.compile(r"^(?:X{2,}|XX/XX/XXXX|XX/XX/\d{2,4}|\{\$[\d,.]+\})$")
IN_DOMAIN = REPO_ROOT / "runs" / "regression__a10-v8-variance-served.json"


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def intent_block(outputs: pd.DataFrame, labels: pd.DataFrame) -> dict:
    served = outputs[outputs.task == "intent"].set_index("id")
    gold = labels[labels.task == "intent"].set_index("id").label
    frame = served.join(gold.rename("gold"), how="inner")
    frame = frame[frame.gold.notna()]
    frame["predicted"] = [json.loads(o).get("label") for o in frame.output]
    fits = frame[frame.gold != "NONE"]
    right = fits.predicted == fits.gold

    # A label outside the 77 is rejected as malformed and falls back to the frontier: routed away, like an
    # escalation, but by the validity check rather than by confidence.
    frame["invalid"] = ~frame.valid.astype(bool)
    frame["routed_away"] = frame.would_escalate.astype(bool) | frame.invalid

    def share(column: str, index) -> float | None:
        part = frame.loc[index]
        return _rate(int(part[column].sum()), len(part))

    none = frame.index[frame.gold == "NONE"]
    return {
        "tickets": len(frame),
        "no_banking77_label_fits": _rate(len(none), len(frame)),
        "invented_label_outside_the_77": _rate(int(frame.invalid.sum()), len(frame)),
        "accuracy_where_a_label_fits": _rate(int(right.sum()), len(fits)),
        "escalated_on_confidence": {"when_right": share("would_escalate", fits.index[right]),
                                    "when_wrong": share("would_escalate", fits.index[~right]),
                                    "when_no_label_fits": share("would_escalate", none)},
        "routed_away_confidence_or_fallback": {"when_right": share("routed_away", fits.index[right]),
                                               "when_wrong": share("routed_away", fits.index[~right]),
                                               "when_no_label_fits": share("routed_away", none)},
        "top_predictions_when_no_label_fits": frame.loc[none].predicted.value_counts().head(5).to_dict(),
    }


def urgency_block(outputs: pd.DataFrame, labels: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    served = outputs[outputs.task == "urgency"].set_index("id")
    gold = labels[labels.task == "urgency"].set_index("id").label
    frame = served.join(gold.rename("gold"), how="inner")
    frame = frame[frame.gold.notna()]
    predicted = [json.loads(o).get("label") for o in frame.output]
    return {"tickets": len(frame),
            "macro_f1": round(float(f1_score(frame.gold, predicted, average="macro", zero_division=0)), 4),
            "micro_accuracy": round(float(accuracy_score(frame.gold, predicted)), 4),
            "gold_distribution": frame.gold.value_counts().to_dict(),
            "predicted_distribution": pd.Series(predicted).value_counts().to_dict()}


def pii_block(sample: pd.DataFrame, outputs: pd.DataFrame, source: str) -> dict:
    from adapterops.eval.pii_realistic import score
    from adapterops.serve.pipeline import operating_threshold

    served = outputs[(outputs.task == "pii")].drop(columns=["source"], errors="ignore").set_index("id")
    rows = sample[sample.source == source].set_index("id").join(served, how="inner")
    spans = [json.loads(o).get("spans", []) for o in rows.output]
    if source == "abcd":
        frame = pd.DataFrame({
            "text": rows.text.to_numpy(), "gold": rows.pii_gold.to_numpy(),
            "prediction": ["\n".join(f"{s['label']}: {s['value']}" for s in doc) for doc in spans],
            "mean_logprob": [-float(s) if pd.notna(s) else None for s in rows.score]})
        result = score(frame, operating_threshold("confidence"))
        return {k: v for k, v in result.items() if k != "per_label"} | {
            "per_label": dict(list(result["per_label"].items())[:10])}
    placeholder = [sum(bool(REDACTION.match(s["value"].strip())) for s in doc) for doc in spans]
    return {"texts": len(rows),
            "texts_with_any_span": sum(bool(doc) for doc in spans),
            "spans": sum(len(doc) for doc in spans),
            "spans_on_redaction_marks": sum(placeholder),
            "spans_on_other_text": sum(len(doc) for doc in spans) - sum(placeholder),
            "labels": pd.Series([s["label"] for doc in spans for s in doc]).value_counts().head(8).to_dict()}


def drafting_block(outputs: pd.DataFrame, labels: pd.DataFrame, source: str, sample: pd.DataFrame) -> dict:
    ids = set(sample[sample.source == source].id)
    served = outputs[(outputs.task == "drafting") & outputs.id.isin(ids)].set_index("id")
    judge = served.judge_score.dropna().astype(float)
    grades = labels[(labels.task == "drafting_grade") & labels.id.isin(ids)].set_index("id").label.dropna()
    paired = pd.concat([judge.rename("judge"), grades.astype(float).rename("gpt4o")], axis=1, join="inner")
    block = {"replies": len(served), "judge_mean": round(float(judge.mean()), 4) if len(judge) else None,
             "graded_by_gpt4o": len(paired)}
    if len(paired) >= 5:
        block |= {"gpt4o_mean": round(float(paired.gpt4o.mean()), 4),
                  "judge_mean_on_graded": round(float(paired.judge.mean()), 4),
                  "spearman_judge_vs_gpt4o": round(float(paired.judge.corr(paired.gpt4o, method="spearman")), 4),
                  "gpt4o_grade_3_or_lower": _rate(int((paired.gpt4o <= 3).sum()), len(paired))}
    return block


def latency(outputs: pd.DataFrame) -> dict:
    out = {}
    for task, g in outputs.groupby("task"):
        ms = sorted(float(x) for x in g.latency_ms.dropna())
        if ms:
            out[task] = {"p50_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1)}
    return out


def main() -> int:
    sample = pd.read_parquet(SAMPLE_FILE)
    outputs = pd.read_parquet(OUTPUTS_FILE)
    labels = pd.read_parquet(LABELS_FILE)
    in_domain = json.loads(IN_DOMAIN.read_text())["per_split"]
    result = {"in_domain_reference": {"run": str(IN_DOMAIN.relative_to(REPO_ROOT)),
                                      "intent_accuracy": in_domain["intent"]["random"]["micro_accuracy"],
                                      "urgency_macro_f1_tfidf": 0.54,
                                      "drafting_judge_mean": in_domain["drafting"]["random"]["judge_score_mean"]},
              "sources": {}, "latency_local_mps": latency(outputs),
              "note": "Latency is this Mac through transformers, not the A10 through vLLM; compare shape, not size."}
    for source in ("abcd", "cfpb"):
        ids = set(sample[sample.source == source].id)
        o, lab = outputs[outputs.id.isin(ids)], labels[labels.id.isin(ids)]
        result["sources"][source] = {"intent": intent_block(o, lab), "urgency": urgency_block(o, lab),
                                     "pii": pii_block(sample, o, source),
                                     "drafting": drafting_block(o, lab, source, sample)}
    RUN_FILE.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["sources"], indent=1, default=str)[:4000])
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
