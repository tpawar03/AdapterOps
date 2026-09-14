"""A PII training split that teaches the adapter an empty answer (changelog 62).

On 928 texts with no personal data the served PII adapter reported some in every one, because every
training document contained PII (D21) and an empty answer was never a target. This builds the split
that changes that, and freezes it before any GPU time is spent.

**Negatives come from the same documents, one sentence at a time.** Changelog 24 rejected the two
obvious constructions: PII-free text borrowed from another corpus is separable by corpus alone
(0.9999 with all PII removed), and PII replaced by surrogates teaches the surrogates. So each
training document is split into sentences, and a sentence is a negative only when no recorded span
touches it *and* it passes the strict PII-free screen (`eval/pii_negatives.pii_free`) — the
dataset's labels are trusted, then checked. Its target is empty.

**Positives are matched to them.** Span-free sentences are shorter (median 66 characters against
101), so negatives alone would teach "short sentence, empty answer". Each negative is paired with a
PII sentence from the same length bin, keeping only sentences that wholly contain every span they
touch, with those spans as the target. A length-only classifier on the matched pairs sits at chance.

**What the documents keep.** The served adapter was trained on `load_split`'s seeded 8,000-row
subsample of `split_train`; those exact rows stay, with their targets, so the change is additive.

**A known limit, measured rather than hoped away.** With PII removed, a word classifier still
separates matched positives from negatives well above chance — mostly on words that sit beside PII
("email", "call", "address", "name"), which is legitimate context, but also because span-free sentences
lean toward sign-offs ("thank you", "look forward"). An adapter could learn "pleasantries are empty"
and still flag a banking question. The frozen 928-text set, which has no sign-offs, is what tests
that; the validation split's span-free sentences are the in-distribution check.

    uv run adapterops pii-negatives-split
    python -m adapterops.train.cli_train --task pii --train-split train_negatives --variant negatives \\
        --subsample 100000 --epochs 3
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.eval.pii_negatives import VAL_SET_FILE, pii_free

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260909
SOURCE = REPO_ROOT / "data" / "pii" / "split_train.parquet"
VAL_SOURCE = REPO_ROOT / "data" / "pii" / "split_val.parquet"
OUT = REPO_ROOT / "data" / "pii" / "split_train_negatives.parquet"
RECORD = REPO_ROOT / "data" / "pii" / "split_train_negatives.json"
DOCUMENTS = 8000
"""The served adapter's training rows: `load_split("pii", "train", 8000)` samples these with SEED."""
SENTENCE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
"""A sentence ends at punctuation followed by whitespace or the end of the text — not at every full
stop, which would cut `dana@example.com` at `.com` and break PII sentences into fragments."""
MIN_CHARS = 15
LENGTH_BINS = 20
GATE_THRESHOLD_TASK = "pii"


def sentences(frame: pd.DataFrame) -> pd.DataFrame:
    """Every sentence of every document, with the spans it touches and whether it contains them all."""
    rows = []
    for doc, (text, mask) in enumerate(zip(frame.source_text.astype(str), frame.privacy_mask,
                                           strict=True)):
        spans = sorted(({"label": m["label"], "start": int(m["start"]), "end": int(m["end"]),
                         "value": str(m["value"])} for m in mask), key=lambda s: s["start"])
        for match in SENTENCE.finditer(text):
            start, end = match.start(), match.end()
            sentence = text[start:end].strip()
            if len(sentence) < MIN_CHARS:
                continue
            touching = [s for s in spans if s["start"] < end and start < s["end"]]
            rows.append({
                "doc": doc,
                "source_text": sentence,
                "spans": len(touching),
                "contained": all(start <= s["start"] and s["end"] <= end for s in touching),
                "target": "\n".join(f"{s['label']}: {s['value']}" for s in touching),
                "chars": len(sentence),
                "blind": re.sub(r"\s+", " ", _remove(text[start:end], start, touching)).strip(),
            })
    return pd.DataFrame(rows)


def _remove(sentence: str, offset: int, spans: list[dict]) -> str:
    for s in sorted(spans, key=lambda s: s["start"], reverse=True):
        a, b = max(s["start"] - offset, 0), min(s["end"] - offset, len(sentence))
        sentence = sentence[:a] + sentence[b:]
    return sentence


def negatives_from(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame.spans == 0) & frame.source_text.map(pii_free)].reset_index(drop=True)


def length_matched(negatives: pd.DataFrame, positives: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One positive per negative from the same length bin, as far as the bin's pool allows."""
    edges = np.quantile(pd.concat([negatives.chars, positives.chars]), np.linspace(0, 1, LENGTH_BINS + 1))

    def bin_of(chars: pd.Series) -> np.ndarray:
        return np.clip(np.searchsorted(edges, chars.to_numpy(), side="right") - 1, 0, LENGTH_BINS - 1)

    neg_bins, pos_bins = bin_of(negatives.chars), bin_of(positives.chars)
    picks = []
    for b, count in pd.Series(neg_bins).value_counts().sort_index().items():
        pool = positives[pos_bins == b]
        if len(pool):
            picks.append(pool.sample(min(int(count), len(pool)), random_state=seed))
    return pd.concat(picks).reset_index(drop=True) if picks else positives.iloc[:0]


def probe(negatives: pd.DataFrame, positives: pd.DataFrame, seed: int = SEED) -> dict:
    """Changelog 24's test on the matched pairs: with PII removed, how separable are they, and by what."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    frame = pd.concat([negatives.assign(y=0), positives.assign(y=1)], ignore_index=True)
    x_tr, x_te, y_tr, y_te = train_test_split(frame, frame.y, test_size=0.25, random_state=seed,
                                              stratify=frame.y)
    words = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=50_000),
                          LogisticRegression(max_iter=2000)).fit(x_tr.blind, y_tr)
    length = LogisticRegression(max_iter=1000).fit(x_tr[["chars"]], y_tr)
    vec, lr = words.named_steps["tfidfvectorizer"], words.named_steps["logisticregression"]
    names, coef = np.array(vec.get_feature_names_out()), lr.coef_[0]
    return {
        "pii_blind_word_accuracy": round(float(accuracy_score(y_te, words.predict(x_te.blind))), 4),
        "length_only_accuracy": round(float(accuracy_score(y_te, length.predict(x_te[["chars"]]))), 4),
        "chance": 0.5,
        "words_marking_pii_sentences": names[np.argsort(coef)[-12:]][::-1].tolist(),
        "words_marking_span_free_sentences": names[np.argsort(coef)[:12]].tolist(),
    }


def build() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    train = pd.read_parquet(SOURCE)
    documents = train.sample(n=DOCUMENTS, random_state=SEED).reset_index(drop=True)
    sent = sentences(train)
    negatives = negatives_from(sent)
    positives = length_matched(negatives, sent[(sent.spans > 0) & sent.contained])
    rows = pd.concat([
        documents[["source_text", "target"]].assign(kind="document"),
        negatives[["source_text", "target"]].assign(kind="negative_sentence"),
        positives[["source_text", "target"]].assign(kind="positive_sentence"),
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_sentences = sentences(pd.read_parquet(VAL_SOURCE))
    val_negatives = negatives_from(val_sentences)[["source_text"]].rename(
        columns={"source_text": "text"}).assign(source="ai4privacy_val")
    facts = {
        "rows": rows.kind.value_counts().to_dict(),
        "negative_documents": int(negatives.doc.nunique()),
        "median_chars": {"negative_sentence": _median(negatives.chars),
                         "positive_sentence": _median(positives.chars),
                         "document": _median(documents.source_text.str.len())},
        "probe_on_matched_pairs": probe(negatives, positives),
        "val_negative_sentences": len(val_negatives),
    }
    return rows, facts, val_negatives


def _median(values: pd.Series) -> int | None:
    return int(values.median()) if len(values) else None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(force: bool = False) -> int:
    if RECORD.exists() and not force:
        print(f"  {RECORD.relative_to(REPO_ROOT)} exists — the split is frozen; --force rebuilds it")
        return 1
    rows, facts, val_negatives = build()
    rows.to_parquet(OUT, index=False)
    VAL_SET_FILE.parent.mkdir(parents=True, exist_ok=True)
    val_negatives.to_parquet(VAL_SET_FILE, index=False)
    thresholds = json.loads((REPO_ROOT / "evals" / "GATE_THRESHOLDS.json").read_text())["gate"]
    RECORD.write_text(json.dumps({
        "purpose": "PII training split with PII-free sentences and empty answers (changelog 62).",
        "seed": SEED,
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "file": str(OUT.relative_to(REPO_ROOT)),
        "sha256": _sha(OUT),
        "held_out_negatives": {"file": str(VAL_SET_FILE.relative_to(REPO_ROOT)),
                               "sha256": _sha(VAL_SET_FILE),
                               "from": str(VAL_SOURCE.relative_to(REPO_ROOT))},
        **facts,
        "preregistered": {
            "decision_rule": (
                "Replace the served PII adapter only if the candidate's random-split strict span "
                "F1, regression-scored beside the served adapter in one session, drops by no more "
                f"than the enforced gate threshold ({thresholds['thresholds'][GATE_THRESHOLD_TASK]}). "
                "The gate decides; no new threshold is introduced."),
            "reported_whatever_they_are": [
                "false-positive rate on evals/pii_negatives/pii_negatives.parquet (928 texts)",
                "false-positive rate on the held-out ai4privacy validation sentences",
                "hard-split span F1 (report-only, as for every task)",
            ],
            "no_target_rate": ("No false-positive rate is set as a success bar in advance: the "
                               "served adapter's is 100%, and any number chosen now would be invented."),
            "known_limit": ("Span-free sentences lean toward sign-offs; the 928-text set, which has "
                            "none, tests whether the adapter learned that instead of PII."),
        },
        "train_with": ("python -m adapterops.train.cli_train --task pii --train-split train_negatives "
                       "--variant negatives --subsample 100000 --epochs 3"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}: {facts['rows']}")
    print(f"  held-out negatives: {len(val_negatives)} → {VAL_SET_FILE.relative_to(REPO_ROOT)}")
    print(f"  probe on matched pairs: {facts['probe_on_matched_pairs']['pii_blind_word_accuracy']} "
          f"PII-blind, {facts['probe_on_matched_pairs']['length_only_accuracy']} length-only")
    return 0
