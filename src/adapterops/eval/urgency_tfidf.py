"""TF-IDF + logistic regression for urgency, behind the gate's own scorer (TODO.md §2).

The urgency adapter loses to TF-IDF: 0.4275 macro F1 against 0.5465 on the golden set, and its margin
over prompting is inside its own run-to-run spread. That leaves a decision — try the untested LoRA
fixes, or serve the cheaper model behind the same interface — and a decision needs the baseline
measured the way the gate measures a release, not ad hoc.

**The configuration is fixed here, not tuned.** Word 1-2 grams, min_df 2, sublinear term frequencies,
balanced class weights, one fit on the frozen training split. Cross-validation on the training split
is reported for sanity; nothing is selected on golden or hard.

**The rule is written before the scores are read** (`RULE`): TF-IDF replaces the adapter for urgency
only if it beats it on golden by more than urgency's own gate threshold, and is not worse on the hard
split by more than that threshold. The threshold is the one the gate already enforces, so no new
number is invented here.

    uv run adapterops urgency-tfidf
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from adapterops.eval.regression import GATED, load_split, score

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_FILE = REPO_ROOT / "runs" / "urgency__tfidf.json"
SERVED_RUN = REPO_ROOT / "runs" / "regression__a10-v7-pii-served.json"
"""The served adapters in the most recent session — urgency's side of the comparison."""
SEED = 20260909
RULE = ("Serve TF-IDF for urgency instead of the adapter only if its golden macro F1 exceeds the "
        "served adapter's by more than urgency's enforced gate threshold, and its hard-split macro "
        "F1 is not below the adapter's by more than that threshold. Latency, model size and the "
        "CC BY-NC training data are reported either way; the licence does not change, since both "
        "models learn from the same dataset.")


def model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def cross_validated(texts, labels) -> dict:
    """5-fold macro F1 on the training split — a sanity check, not a selection."""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model(), texts, labels, cv=5, scoring="f1_macro")
    return {"folds": 5, "macro_f1_mean": round(float(scores.mean()), 4),
            "macro_f1_min": round(float(scores.min()), 4), "macro_f1_max": round(float(scores.max()), 4)}


def threshold(root: Path = REPO_ROOT) -> float | None:
    gate = json.loads((root / "evals" / "GATE_THRESHOLDS.json").read_text())["gate"]
    return gate["thresholds"].get("urgency")


def decide(tfidf: dict, adapter: dict, gate: float) -> dict:
    """The frozen rule, applied. Both conditions must hold."""
    better_on_golden = tfidf["random"] - adapter["random"] > gate
    not_worse_on_hard = tfidf["hard"] - adapter["hard"] >= -gate
    return {"gate_threshold": gate,
            "golden_gain": round(tfidf["random"] - adapter["random"], 4),
            "hard_change": round(tfidf["hard"] - adapter["hard"], 4),
            "beats_adapter_on_golden_by_more_than_the_gate": bool(better_on_golden),
            "not_worse_on_hard_than_the_gate_allows": bool(not_worse_on_hard),
            "serve_tfidf_for_urgency": bool(better_on_golden and not_worse_on_hard)}


def shared_failure_rows(pipeline) -> dict | None:
    """The fair third comparison: golden items that GPT-4o-mini gets wrong (`evals/hard/shared_failures.parquet`).

    The hard-cases split was mined from the first adapter's own failures, so the adapter scores near 0 there and
    any different model scores higher by construction (D32). These items were mined from a model neither system
    is, so both are measured on the same footing."""
    split_file = REPO_ROOT / "evals" / "hard" / "shared_failures.parquet"
    predictions_file = REPO_ROOT / "runs" / "regression__a10-v7-pii-served__predictions.parquet"
    if not split_file.exists() or not predictions_file.exists():
        return None
    shared = pd.read_parquet(split_file)
    shared = shared[shared.task == "urgency"].reset_index(drop=True)
    served = pd.read_parquet(predictions_file)
    served = served[(served.task == "urgency") & (served.split == "random")].reset_index(drop=True)
    rows = served.iloc[list(shared.source_row)]
    metric = GATED["urgency"]
    texts, gold = list(rows.text), list(rows.gold)
    return {"n": len(rows),
            "tfidf": score("urgency", texts, gold, list(pipeline.predict(texts)))[metric],
            "adapter": score("urgency", texts, gold, list(rows.prediction))[metric],
            "source": "evals/hard/shared_failures.parquet, urgency rows"}


def main() -> int:
    train = pd.read_parquet(REPO_ROOT / "data" / "urgency" / "split_train.parquet")
    texts, labels = train.text.astype(str).tolist(), train.priority.astype(str).tolist()
    pipeline = model()
    started = time.perf_counter()
    pipeline.fit(texts, labels)
    fit_seconds = time.perf_counter() - started

    scores, predictions = {}, {}
    for split in ("random", "hard"):
        split_texts, gold = load_split("urgency", split)
        started = time.perf_counter()
        predicted = list(pipeline.predict(split_texts))
        elapsed = time.perf_counter() - started
        predictions[split] = predicted
        scores[split] = {**score("urgency", split_texts, gold, predicted),
                         "ms_per_text": round(1000 * elapsed / len(split_texts), 3)}

    metric = GATED["urgency"]
    served = json.loads(SERVED_RUN.read_text())["per_split"]["urgency"]
    tfidf_metric = {s: scores[s][metric] for s in scores}
    adapter_metric = {s: served[s][metric] for s in scores}
    result = {
        "question": "Should urgency be served by TF-IDF instead of the adapter?",
        "rule_written_before_scoring": RULE,
        "configuration": {"vectorizer": "word 1-2 grams, min_df 2, sublinear tf",
                          "classifier": "logistic regression, balanced class weights",
                          "seed": SEED, "training_rows": len(texts), "fit_seconds": round(fit_seconds, 1)},
        "cross_validation_on_training_split": cross_validated(texts, labels),
        "tfidf": scores,
        "adapter_same_session": {"run": str(SERVED_RUN.relative_to(REPO_ROOT)), **adapter_metric},
        "shared_failure_items": shared_failure_rows(pipeline),
        "decision": decide(tfidf_metric, adapter_metric, threshold() or 0.0),
    }
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split in scores:
        print(f"  {split:6s} TF-IDF {metric} {tfidf_metric[split]:.4f} · adapter "
              f"{adapter_metric[split]:.4f} · {scores[split]['ms_per_text']} ms/text")
    if (shared := result["shared_failure_items"]):
        print(f"  shared TF-IDF {metric} {shared['tfidf']:.4f} · adapter {shared['adapter']:.4f} "
              f"(n={shared['n']}, mined from GPT-4o-mini's failures)")
    d = result["decision"]
    print(f"  golden gain {d['golden_gain']:+.4f} against the {d['gate_threshold']} gate · hard "
          f"{d['hard_change']:+.4f} → serve TF-IDF: {d['serve_tfidf_for_urgency']}")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
