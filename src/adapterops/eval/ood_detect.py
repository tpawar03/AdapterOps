"""An out-of-domain check in front of the adapters (TODO.md §3, follow-up to `runs/ood.json`).

On 300 tickets from sources no model trained on, the adapters were confidently wrong: confidence routing escalated
none of intent's wrong answers. A model's own log-probability cannot say "this is not what I was trained on", so this
asks the question before the model answers, from the input alone.

**One check per task.** A ticket can be in-domain for PII and out of domain for intent, so each task compares the
ticket with its own training texts: TF-IDF character 3-5 grams, and the mean cosine similarity of the ticket's
5 nearest training texts. A length-only check is reported beside it — intent and drafting trained on ~10-word queries,
so length alone may carry most of the signal, and a similarity check that does not beat it is not worth serving.

**Thresholds never see an out-of-domain ticket.** Each task's threshold is the 5th percentile of its own validation
split, so about 5% of in-domain text is flagged by construction; the golden set is a second in-domain check of that.
The 300 tickets only evaluate: how many are flagged, and — the question that matters — whether the flagged ones are
the ones the served system got wrong.

    uv run adapterops ood --detect
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from adapterops.eval.ood import OUTPUTS_FILE, REPO_ROOT, SAMPLE_FILE
from adapterops.train.qlora import COLUMNS

RUN_FILE = REPO_ROOT / "runs" / "ood__detector.json"
TRAIN_SPLIT = {"intent": "train", "urgency": "train", "pii": "train_realistic", "drafting": "train"}
"""The split each served model trained on — PII's is the realistic-format mix manifest v7 serves."""
FLAG_QUANTILE = 0.05
NEIGHBOURS = 5
TASKS = ("intent", "urgency", "pii", "drafting")


def texts(task: str, split: str) -> list[str]:
    column = COLUMNS[task][0]
    if split == "golden":
        return pd.read_parquet(REPO_ROOT / "evals" / "golden" / f"{task}.parquet")[column].astype(str).tolist()
    return pd.read_parquet(REPO_ROOT / "data" / task / f"split_{split}.parquet")[column].astype(str).tolist()


class Similarity:
    """Mean cosine similarity of the k nearest training texts, on TF-IDF character n-grams."""

    def __init__(self, train: list[str], k: int = NEIGHBOURS) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                          max_features=200_000)
        self.train = self.vectorizer.fit_transform(train)

    def score(self, items: list[str], batch: int = 512) -> np.ndarray:
        out = []
        for i in range(0, len(items), batch):
            sims = (self.vectorizer.transform(items[i:i + batch]) @ self.train.T).toarray()
            top = np.partition(sims, -self.k, axis=1)[:, -self.k:]
            out.append(top.mean(axis=1))
        return np.concatenate(out) if out else np.array([])


def length(items: list[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in items], dtype=float)


def thresholds(val_similarity: np.ndarray, val_length: np.ndarray) -> dict:
    """Flag below the 5th percentile of in-domain similarity, or above the 95th percentile of in-domain length."""
    return {"similarity_below": float(np.quantile(val_similarity, FLAG_QUANTILE)),
            "length_above": float(np.quantile(val_length, 1 - FLAG_QUANTILE))}


def _rate(mask: np.ndarray) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def usefulness(flagged: pd.Series, wrong: pd.Series) -> dict:
    """Whether a check flags the tickets the served system got wrong more often than the ones it got right."""
    frame = pd.concat([flagged.rename("flagged"), wrong.rename("wrong")], axis=1, join="inner").dropna()
    if frame.empty:
        return {}
    wrong_rows, right_rows = frame[frame.wrong.astype(bool)], frame[~frame.wrong.astype(bool)]
    return {"n": len(frame), "wrong": len(wrong_rows),
            "flagged_when_wrong": _rate(wrong_rows.flagged.astype(bool).to_numpy()),
            "flagged_when_right": _rate(right_rows.flagged.astype(bool).to_numpy())}


def served_errors(sample: pd.DataFrame) -> dict[str, pd.Series]:
    """Per task, whether the served system was wrong on each out-of-domain ticket, from the evaluation's own data."""
    from adapterops.eval.ood_label import LABELS_FILE
    from adapterops.eval.pii_errors import coverage
    from adapterops.eval.spans import Span, spans_from_values

    outputs = pd.read_parquet(OUTPUTS_FILE)
    labels = pd.read_parquet(LABELS_FILE)
    errors = {}
    for task in ("intent", "urgency"):
        served = outputs[outputs.task == task].set_index("id")
        gold = labels[labels.task == task].set_index("id").label
        joined = served.join(gold.rename("gold"), how="inner")
        joined = joined[joined.gold.notna() & (joined.gold != "NONE")]
        predicted = joined.output.map(lambda o: json.loads(o).get("label"))
        errors[task] = (predicted != joined.gold) | ~joined.valid.astype(bool)
    grades = labels[labels.task == "drafting_grade"].set_index("id").label.dropna().astype(float)
    errors["drafting"] = grades <= 3
    pii = outputs[outputs.task == "pii"].set_index("id")
    abcd = sample[sample.source == "abcd"].set_index("id")
    leak = {}
    for ticket_id, row in abcd.iterrows():
        gold = [Span(g["start"], g["end"], g["label"]) for g in json.loads(row.pii_gold) if g["scope"] == "in"]
        if not gold or ticket_id not in pii.index:
            continue
        pred = spans_from_values(row.text, [(s["label"], s["value"])
                                            for s in json.loads(pii.loc[ticket_id].output).get("spans", [])])
        leak[ticket_id] = coverage(row.text, gold, pred)[0] > 0
    errors["pii"] = pd.Series(leak)
    return errors


SHIFT_POPULATIONS = {"in_distribution": "router_eval__judged.parquet", "shifted": "router_shift_eval__judged.parquet"}
"""The router study's recorded pairs (F36): a within-task shift split with per-pair success labels and the adapter's
own log-probability, so the check can be measured on benign shift and compared with confidence routing."""
UNCHANGED_SINCE_LABELLED = ("intent", "drafting")
"""The pairs were labelled against the adapters of that time. Intent and drafting still serve those weights; PII
was retrained twice and urgency replaced by TF-IDF, so for those two only flag rates are meaningful."""


def shift_block(task: str, similarity: Similarity, limits: dict) -> dict:
    from adapterops.serve.pipeline import operating_threshold

    threshold = operating_threshold("confidence")
    out = {}
    for name, file in SHIFT_POPULATIONS.items():
        pairs = pd.read_parquet(REPO_ROOT / "data" / "router" / file)
        pairs = pairs[pairs.task == task].reset_index(drop=True)
        items = pairs.text.astype(str).tolist()
        flagged = (similarity.score(items) < limits["similarity_below"]) | (length(items) > limits["length_above"])
        confidence = (-pairs.mean_logprob.astype(float) >= threshold).to_numpy()
        block = {"pairs": len(pairs), "flag_rate": _rate(flagged), "confidence_escalation_rate": _rate(confidence)}
        if task in UNCHANGED_SINCE_LABELLED:
            wrong = ~pairs.success.astype(bool).to_numpy()
            both = flagged | confidence
            block |= {"wrong": int(wrong.sum()),
                      "flag": {"when_wrong": _rate(flagged[wrong]), "when_right": _rate(flagged[~wrong])},
                      "confidence": {"when_wrong": _rate(confidence[wrong]), "when_right": _rate(confidence[~wrong])},
                      "either": {"when_wrong": _rate(both[wrong]), "when_right": _rate(both[~wrong])}}
        else:
            block["note"] = "labels describe a model no longer served; flag rate only"
        out[name] = block
    return out


def main() -> int:
    sample = pd.read_parquet(SAMPLE_FILE)
    errors = served_errors(sample)
    result = {"method": {"signal": f"mean cosine of {NEIGHBOURS} nearest training texts, TF-IDF char 3-5 grams",
                         "baseline": "word count", "flag_quantile_on_validation": FLAG_QUANTILE,
                         "training_splits": TRAIN_SPLIT},
              "per_task": {}}
    for task in TASKS:
        similarity = Similarity(texts(task, TRAIN_SPLIT[task]))
        val = texts(task, "val")
        limits = thresholds(similarity.score(val), length(val))
        golden = texts(task, "golden")
        ood_text = sample.text.tolist()
        sim_ood, len_ood = similarity.score(ood_text), length(ood_text)
        flags = {"similarity": sim_ood < limits["similarity_below"], "length": len_ood > limits["length_above"]}
        flags["either"] = flags["similarity"] | flags["length"]
        golden_flags = {"similarity": similarity.score(golden) < limits["similarity_below"],
                        "length": length(golden) > limits["length_above"]}
        golden_flags["either"] = golden_flags["similarity"] | golden_flags["length"]
        block = {"thresholds": {k: round(v, 4) for k, v in limits.items()},
                 "in_domain_golden_flag_rate": {k: _rate(v) for k, v in golden_flags.items()},
                 "flag_rate": {}, "usefulness": {}}
        for source in ("abcd", "cfpb"):
            mask = (sample.source == source).to_numpy()
            block["flag_rate"][source] = {k: _rate(v[mask]) for k, v in flags.items()}
        if task in errors:
            for name, mask in flags.items():
                block["usefulness"][name] = usefulness(pd.Series(mask, index=sample.id), errors[task])
        block["router_populations"] = shift_block(task, similarity, limits)
        result["per_task"][task] = block
        print(f"  {task:9s} golden flagged {block['in_domain_golden_flag_rate']} · "
              f"abcd {block['flag_rate']['abcd']} · cfpb {block['flag_rate']['cfpb']}")
    RUN_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {RUN_FILE.relative_to(REPO_ROOT)}")
    return 0
