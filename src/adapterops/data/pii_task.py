"""Construct the PII binary task, and measure how confounded it is (F3).

The mirrored ai4privacy split has **zero negatives** — every row contains PII — so a
binary "PII present" flag needs negatives from elsewhere. Bitext supplies them: same
customer-support domain, no real PII.

That import brings a confound. If the two sources differ in style, a model can score well
by recognising *which corpus a string came from* rather than whether it contains PII. So
this module both builds the task and probes the confound, and the probe is the deliverable
as much as the dataset is.

**The probe:** train a plain TF-IDF + logistic-regression classifier to separate the two
classes using the positives' *masked* text — PII replaced by `[TYPE_n]` placeholders, and
then those placeholders stripped. If PII is what distinguishes the classes, a classifier
that cannot see any PII should be near chance. Whatever it scores above chance is
source-style leakage, not PII signal.

Run:  uv run adapterops pii-task
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260909

PLACEHOLDER = re.compile(r"\{\{([^}]*)\}\}")      # Bitext:      {{Order Number}}
MASK_TOKEN = re.compile(r"\[[A-Z_]+_?\d*\]")      # ai4privacy:  [GIVENNAME_1]


def _normalise(text: str) -> str:
    """Remove both corpora's template markers — they are pure source tells."""
    text = PLACEHOLDER.sub(lambda m: m.group(1), text)
    return MASK_TOKEN.sub(" ", text).strip()


def build() -> tuple[pd.DataFrame, dict]:
    pii = pd.read_parquet(REPO_ROOT / "data/pii/train.parquet")
    bitext = pd.read_parquet(REPO_ROOT / "data/drafting/train.parquet")

    # Mitigation 1: use Bitext `response` (mean ~634 chars), not `instruction` (~47),
    # since ai4privacy averages ~354 — length alone must not be the signal.
    neg_raw = bitext.response.drop_duplicates()
    n = min(len(pii), len(neg_raw))
    pos = pii.sample(n=n, random_state=SEED)
    neg = neg_raw.sample(n=n, random_state=SEED)

    frame = pd.DataFrame({
        "source_text": list(pos.source_text) + list(neg),
        "masked_text": list(pos.masked_text) + list(neg),
        "has_pii": ["yes"] * n + ["no"] * n,
        "origin": ["ai4privacy"] * n + ["bitext"] * n,
    }).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    stats = {
        "rows": len(frame),
        "balance": frame.has_pii.value_counts().to_dict(),
        "mean_chars": {
            "positives": round(float(pos.source_text.str.len().mean()), 1),
            "negatives": round(float(neg.str.len().mean()), 1),
        },
    }
    return frame, stats


def probe_confound(frame: pd.DataFrame) -> dict:
    """Can a classifier separate the classes with all PII removed? It should not be able to."""
    blind = frame.masked_text.map(_normalise)          # no PII, no template markers
    sighted = frame.source_text.map(_normalise)        # PII intact

    def fit_score(x: pd.Series) -> float:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, frame.has_pii, test_size=0.25, random_state=SEED, stratify=frame.has_pii)
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50_000),
            LogisticRegression(max_iter=1000))
        model.fit(x_tr, y_tr)
        return float(accuracy_score(y_te, model.predict(x_te)))

    def length_only() -> float:
        x = blind.str.len().to_frame("n")
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, frame.has_pii, test_size=0.25, random_state=SEED, stratify=frame.has_pii)
        model = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
        return float(accuracy_score(y_te, model.predict(x_te)))

    blind_acc = fit_score(blind)
    return {
        "chance": 0.5,
        "pii_blind_accuracy": round(blind_acc, 4),
        "pii_visible_accuracy": round(fit_score(sighted), 4),
        "length_only_accuracy": round(length_only(), 4),
        "leakage_above_chance": round(blind_acc - 0.5, 4),
        "interpretation": (
            "pii_blind_accuracy is the confound. A classifier that cannot see any PII "
            "should sit near 0.5; everything above that is source-style leakage. Compare "
            "any trained adapter's score against pii_blind_accuracy, not against chance."
        ),
    }


SURROGATE_A = {
    "GIVENNAME": "the customer", "SURNAME": "the account holder", "TITLE": "the client",
    "DATE": "the agreed date", "EMAIL": "the address on file", "CITY": "the local branch",
    "TELEPHONENUM": "the number on file", "STREET": "the delivery address",
    "AGE": "the stated age", "BUILDINGNUM": "the premises", "ZIPCODE": "the postal area",
    "IDCARDNUM": "the reference on file",
}
SURROGATE_B = {
    "GIVENNAME": "this individual", "SURNAME": "said party", "TITLE": "the addressee",
    "DATE": "a scheduled day", "EMAIL": "a contact route", "CITY": "that locality",
    "TELEPHONENUM": "a listed line", "STREET": "the given location", "AGE": "an age on record",
    "BUILDINGNUM": "the site", "ZIPCODE": "the postcode zone", "IDCARDNUM": "an identifier held",
}


def _redact(text: str, spans, table: dict, default: str) -> str:
    for span in sorted(spans, key=lambda s: -s["start"]):
        text = text[: span["start"]] + table.get(span["label"], default) + text[span["end"] :]
    return text


def probe_same_corpus() -> dict:
    """Second construction: negatives made from the *same* documents by replacing each PII
    span with a type-matched surrogate. No corpus confound is possible. But a model may
    instead learn the surrogate vocabulary, so it is trained on one surrogate set and
    tested on a disjoint one — if accuracy collapses, it never learned PII at all."""
    df = pd.read_parquet(REPO_ROOT / "data/pii/train.parquet").sample(n=10_000, random_state=SEED)
    train_part, test_part = df.iloc[:7500], df.iloc[7500:]

    def frame_for(part, table, default):
        neg = [_redact(t, list(m), table, default)
               for t, m in zip(part.source_text, part.privacy_mask, strict=True)]
        return pd.DataFrame({"text": list(part.source_text) + neg,
                             "y": ["yes"] * len(part) + ["no"] * len(part)})

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50_000),
        LogisticRegression(max_iter=1000))
    train = frame_for(train_part, SURROGATE_A, "the details on file")
    model.fit(train.text, train.y)

    seen = frame_for(test_part, SURROGATE_A, "the details on file")
    unseen = frame_for(test_part, SURROGATE_B, "information retained")
    acc_seen = float(accuracy_score(seen.y, model.predict(seen.text)))
    acc_unseen = float(accuracy_score(unseen.y, model.predict(unseen.text)))
    return {
        "accuracy_seen_surrogates": round(acc_seen, 4),
        "accuracy_unseen_surrogates": round(acc_unseen, 4),
        "drop": round(acc_seen - acc_unseen, 4),
        "interpretation": (
            "Collapse toward chance on unseen surrogates means the classifier learned the "
            "surrogate phrases, not PII — so this construction is invalid too."
        ),
    }


def main() -> int:
    frame, stats = build()
    out = REPO_ROOT / "data/pii/binary_task.parquet"
    frame.to_parquet(out)
    confound = probe_confound(frame)

    same_corpus = probe_same_corpus()
    report = {
        "verdict": (
            "Both constructions of a binary PII task are invalid on this data. "
            "Cross-corpus negatives are separable with all PII removed; same-corpus "
            "surrogate negatives collapse to chance on unseen surrogates. See STATUS.md."
        ),
        "stats": stats,
        "cross_corpus_probe": confound,
        "same_corpus_probe": same_corpus,
    }
    (REPO_ROOT / "data/pii/CONFOUND.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"  built {out.relative_to(REPO_ROOT)}  ({stats['rows']:,} rows, balanced)")
    print(f"  mean chars: positives {stats['mean_chars']['positives']}  "
          f"negatives {stats['mean_chars']['negatives']}")
    print("\n  confound probe")
    print("    chance                 0.5000")
    print(f"    length only            {confound['length_only_accuracy']:.4f}")
    print(f"    PII-blind (the number) {confound['pii_blind_accuracy']:.4f}"
          f"   <- leakage {confound['leakage_above_chance']:+.4f} above chance")
    print(f"    PII visible            {confound['pii_visible_accuracy']:.4f}")
    print("\n  same-corpus surrogate probe")
    print(f"    seen surrogates        {same_corpus['accuracy_seen_surrogates']:.4f}")
    print(f"    unseen surrogates      {same_corpus['accuracy_unseen_surrogates']:.4f}"
          f"   <- drop {same_corpus['drop']:.4f}")
    print("\n  verdict: both constructions invalid — see data/pii/CONFOUND.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
