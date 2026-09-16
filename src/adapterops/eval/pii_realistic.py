"""The served PII adapter on realistic formats and real text (TODO.md §3, PII on realistic formats).

The adapter was trained and tested on ai4privacy's synthetic spans. Two outside sets test what that hides:

- **Nemotron-PII** (NVIDIA, CC BY 4.0): synthetic, but US and international formats in forms, invoices and emails,
  55 categories. Categories with a counterpart among the adapter's 19 labels are in scope; the rest are reported apart.
- **TAB** (Norsk Regnesentral, MIT): real European Court of Human Rights judgments, each mention marked as needing
  masking (DIRECT, QUASI) or not. In scope: people, dates, places and codes that need masking.

Labels differ between schemas, so the headline is label-agnostic: in-scope gold spans left partly or wholly unmasked
by any predicted span, and the share of masked characters that land on annotated personal data. Each sample is 200
texts of at most 200 words — the training set's longest — drawn with a fixed seed. Generation runs locally with the
served revision and is cached, so re-scoring does not regenerate.

    uv run adapterops pii-realistic
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from adapterops.eval.pii_errors import coverage
from adapterops.eval.spans import Span, parse_model_output, spans_from_values

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "pii_realistic"
FILES = {"nemotron": DATA / "nemotron_test.parquet", "tab": DATA / "tab_echr_test.json"}
SOURCES = {
    "nemotron": {"url": "https://huggingface.co/datasets/nvidia/Nemotron-PII", "licence": "CC BY 4.0",
                 "revision": "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed", "file": "data/test-00000-of-00001.parquet"},
    "tab": {"url": "https://github.com/NorskRegnesentral/text-anonymization-benchmark", "licence": "MIT",
            "revision": "558e09e26d6b36f5f78440074e6a233946d98bd9", "file": "echr_test.json"},
}
def paths(name: str | None = None) -> tuple[Path, Path]:
    """Predictions and run file for the served adapter, or for a named local candidate."""
    suffix = f"__{name}" if name else ""
    return (REPO_ROOT / "runs" / f"pii__realistic{suffix}__predictions.parquet",
            REPO_ROOT / "runs" / f"pii__realistic{suffix}.json")


PREDICTIONS, RUN_FILE = paths()
GOLDEN_RUN = REPO_ROOT / "runs" / "regression__a10-v5-pii-negatives.json"
PER_SOURCE = 200
MAX_WORDS = 200
SEED = 0

NEMOTRON_LABELS = {
    "first_name": "GIVENNAME", "last_name": "SURNAME", "date": "DATE", "date_of_birth": "DATE", "email": "EMAIL",
    "phone_number": "TELEPHONENUM", "fax_number": "TELEPHONENUM", "city": "CITY", "street_address": "STREET",
    "postcode": "ZIPCODE", "credit_debit_card": "CREDITCARDNUMBER", "ssn": "SOCIALNUM", "national_id": "IDCARDNUM",
    "tax_id": "TAXNUM", "age": "AGE", "gender": "GENDER",
}
"""Nemotron-PII categories with a counterpart among the adapter's labels. street_address includes the building
number, which the adapter tags apart; masking coverage counts it either way."""
TAB_TYPES = ("PERSON", "DATETIME", "LOC", "CODE")
TAB_MASK = ("DIRECT", "QUASI")


def nemotron_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """50 texts from each locale × format cell with at least one in-scope span. Each uid has a US and an
    international rewrite of one document; one of the two, chosen at random, is kept so no document counts twice."""
    frame = frame.sample(frac=1, random_state=SEED).drop_duplicates("uid")
    frame = frame[frame.text.str.split().str.len() <= MAX_WORDS]
    spans = frame.spans.map(ast.literal_eval)
    keep = spans.map(lambda s: any(m["label"] in NEMOTRON_LABELS for m in s))
    frame, spans = frame[keep], spans[keep]
    cells = frame.groupby(["locale", "document_format"])
    picked = pd.concat([g.sample(PER_SOURCE // cells.ngroups, random_state=SEED) for _, g in cells])
    gold = [json.dumps([{"start": m["start"], "end": m["end"], "label": m["label"],
                         "scope": "in" if m["label"] in NEMOTRON_LABELS else "out"} for m in spans[i]])
            for i in picked.index]
    return pd.DataFrame({"source": "nemotron", "id": picked.uid.to_numpy(),
                         "stratum": (picked.locale + "/" + picked.document_format).to_numpy(),
                         "text": picked.text.to_numpy(), "gold": gold})


def tab_scope(mention: dict) -> str:
    if mention["identifier_type"] not in TAB_MASK:
        return "no_mask"
    return "in" if mention["entity_type"] in TAB_TYPES else "out"


def paragraphs(doc: dict) -> list[dict]:
    """A document's paragraphs of 20 to MAX_WORDS words holding an in-scope mention, offsets made relative."""
    mentions = doc["annotations"][min(doc["annotations"])]["entity_mentions"]
    out = []
    for match in re.finditer(r"[^\n]+", doc["text"]):
        a, b = match.span()
        if not 20 <= len(match.group().split()) <= MAX_WORDS:
            continue
        gold = [{"start": m["start_offset"] - a, "end": m["end_offset"] - a, "label": m["entity_type"],
                 "scope": tab_scope(m)} for m in mentions if a <= m["start_offset"] and m["end_offset"] <= b]
        if any(g["scope"] == "in" for g in gold):
            out.append({"source": "tab", "id": f"{doc['doc_id']}:{a}", "stratum": doc["doc_id"],
                        "text": match.group(), "gold": json.dumps(gold)})
    return out


def tab_sample(docs: list[dict]) -> pd.DataFrame:
    """At most three paragraphs per judgment, so no single case dominates."""
    frame = pd.DataFrame([p for doc in docs for p in paragraphs(doc)])
    return (frame.sample(frac=1, random_state=SEED).groupby("stratum").head(3)
            .head(PER_SOURCE).reset_index(drop=True))


def _chars(spans) -> set[int]:
    return {i for s in spans for i in range(s.start, s.end)}


def score(frame: pd.DataFrame, threshold: float, mapping: dict[str, str] | None = None) -> dict:
    per_label: dict[str, list[int]] = {}
    fully = masked = on_personal = on_no_mask = exact = label_right = escalated = leaky = leaky_escalated = 0
    for row in frame.itertuples():
        gold = json.loads(row.gold)
        pred = spans_from_values(row.text, parse_model_output(row.prediction))
        doc_partly = 0
        for g in gold:
            if g["scope"] == "no_mask":
                continue
            span = Span(g["start"], g["end"], g["label"])
            partly, wholly = coverage(row.text, [span], pred)
            counts = per_label.setdefault(f"{g['scope']}:{g['label']}", [0, 0, 0])
            counts[0], counts[1], counts[2] = counts[0] + 1, counts[1] + partly, counts[2] + wholly
            if g["scope"] == "in":
                doc_partly += partly
                match = next((p for p in pred if (p.start, p.end) == (span.start, span.end)), None)
                if match is not None:
                    exact += 1
                    label_right += mapping is not None and match.label == mapping.get(span.label)
        predicted = _chars(pred)
        personal = _chars(Span(g["start"], g["end"], "") for g in gold if g["scope"] != "no_mask")
        no_mask = _chars(Span(g["start"], g["end"], "") for g in gold if g["scope"] == "no_mask")
        masked += len(predicted)
        on_personal += len(predicted & personal)
        on_no_mask += len((predicted & no_mask) - personal)
        fully += doc_partly == 0
        goes_up = row.mean_logprob is not None and -row.mean_logprob >= threshold
        escalated += goes_up
        leaky += doc_partly > 0
        leaky_escalated += doc_partly > 0 and goes_up

    def rate(a, b):
        return round(a / b, 4) if b else None

    scope = {s: [sum(c[i] for k, c in per_label.items() if k.startswith(s)) for i in range(3)] for s in ("in", "out")}
    return {
        "texts": len(frame),
        "docs_fully_masked_in_scope": rate(fully, len(frame)),
        "in_scope_spans": scope["in"][0],
        "in_scope_partly_unmasked": rate(scope["in"][1], scope["in"][0]),
        "in_scope_wholly_unmasked": rate(scope["in"][2], scope["in"][0]),
        "out_of_scope_wholly_unmasked": rate(scope["out"][2], scope["out"][0]),
        "masked_chars_on_personal_data": rate(on_personal, masked),
        "masked_chars_on_no_mask_mentions": rate(on_no_mask, masked),
        "in_scope_boundary_exact": rate(exact, scope["in"][0]),
        "label_right_when_boundary_exact": rate(label_right, exact) if mapping is not None else None,
        "escalated_at_confidence_threshold": rate(escalated, len(frame)),
        "texts_with_a_leak": leaky,
        "texts_with_a_leak_escalated": leaky_escalated,
        "per_label": {k: {"spans": c[0], "partly_unmasked": c[1], "wholly_unmasked": c[2]}
                      for k, c in sorted(per_label.items(), key=lambda kv: -kv[1][0])},
    }


def predict(texts: list[str], batch: int, adapter_dir: str | None = None) -> list[dict]:
    """The served PII revision (or a local checkpoint), generated locally with the router rescore's chunked,
    cache-releasing loop."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from adapterops.router.rescore import generate

    components = json.loads((REPO_ROOT / "manifests" / "system.json").read_text())["components"]
    base, pin = components["base_model"], components["adapters"]["pii"]
    path = adapter_dir or snapshot_download(pin["repo"], revision=pin["revision"],
                             allow_patterns=["adapter_model.safetensors", "adapter_config.json"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"], dtype=dtype)
    model = PeftModel.from_pretrained(model, path, adapter_name="pii").to(device).eval()
    return generate(tok, model, device, "pii", "pii", texts, batch)


def main(batch: int = 8, adapter_dir: str | None = None, name: str | None = None) -> int:
    from adapterops.serve.pipeline import operating_threshold

    predictions_file, run_file = paths(name)
    for source, path in FILES.items():
        if not path.exists():
            s = SOURCES[source]
            msg = f"{path} missing: download {s['file']} from {s['url']} at {s['revision']}"
            raise FileNotFoundError(msg)
    sample = pd.concat([nemotron_sample(pd.read_parquet(FILES["nemotron"])),
                        tab_sample(json.loads(FILES["tab"].read_text()))], ignore_index=True)
    cached = pd.read_parquet(predictions_file) if predictions_file.exists() else None
    if cached is not None and cached.text.tolist() == sample.text.tolist():
        frame = cached
    else:
        frame = pd.concat([sample, pd.DataFrame(predict(sample.text.tolist(), batch, adapter_dir))], axis=1)
        frame.to_parquet(predictions_file, index=False)

    threshold = operating_threshold("confidence")
    golden = json.loads(GOLDEN_RUN.read_text())["per_split"]["pii"]["random"]
    result = {
        "adapter": ({"path": adapter_dir} if adapter_dir else
                    json.loads((REPO_ROOT / "manifests" / "system.json").read_text())["components"]["adapters"]["pii"]),
        "sources": {n: {**s, "sha256": hashlib.sha256(FILES[n].read_bytes()).hexdigest()} for n, s in SOURCES.items()},
        "sample": {"per_source": PER_SOURCE, "max_words": MAX_WORDS, "seed": SEED,
                   "nemotron": "equal cells of locale × format, at least one in-scope span",
                   "tab": "paragraphs of 20+ words with an in-scope DIRECT/QUASI mention, at most 3 per judgment"},
        "in_scope": {"nemotron": NEMOTRON_LABELS, "tab": {"types": TAB_TYPES, "identifier_types": TAB_MASK}},
        "confidence_threshold": threshold,
        "reference_golden": {k: golden.get(k) for k in ("docs_fully_masked", "gold_spans_partly_unmasked",
                                                        "gold_spans_wholly_unmasked")},
        "nemotron": score(frame[frame.source == "nemotron"], threshold, NEMOTRON_LABELS),
        "tab": score(frame[frame.source == "tab"], threshold),
    }
    from adapterops.eval.pii_guard import augment
    from adapterops.serve.pipeline import PII_GUARD_LABELS

    guarded = frame.assign(prediction=[augment(t, p, PII_GUARD_LABELS) for t, p in zip(frame.text, frame.prediction)])
    for source, mapping in (("nemotron", NEMOTRON_LABELS), ("tab", None)):
        # The request path adds the pattern guard's spans; this is what a served request would mask.
        result[source]["with_request_path_guard"] = {
            k: v for k, v in score(guarded[guarded.source == source], threshold, mapping).items()
            if k in ("docs_fully_masked_in_scope", "in_scope_partly_unmasked", "in_scope_wholly_unmasked",
                     "masked_chars_on_personal_data")}
    nem = frame[frame.source == "nemotron"]
    result["nemotron"]["by_stratum"] = {
        s: {k: v for k, v in score(g, threshold, NEMOTRON_LABELS).items() if k in
            ("texts", "docs_fully_masked_in_scope", "in_scope_wholly_unmasked", "masked_chars_on_personal_data")}
        for s, g in nem.groupby("stratum")}
    run_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ref = result["reference_golden"]
    print(f"  golden reference: fully masked {ref['docs_fully_masked']} · wholly unmasked {ref['gold_spans_wholly_unmasked']}")
    for source in ("nemotron", "tab"):
        r = result[source]
        print(f"  {source:8s} fully masked {r['docs_fully_masked_in_scope']} · in-scope partly {r['in_scope_partly_unmasked']}"
              f" wholly {r['in_scope_wholly_unmasked']} · masked chars on personal data {r['masked_chars_on_personal_data']}"
              f" · label right {r['label_right_when_boundary_exact']} · escalated {r['escalated_at_confidence_threshold']}"
              f" · leaky texts escalated {r['texts_with_a_leak_escalated']}/{r['texts_with_a_leak']}")
    print(f"  wrote {run_file.relative_to(REPO_ROOT)}")
    return 0


def leaks_per_text(frame: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Per text: in-scope gold spans left wholly unmasked, and in-scope gold spans."""
    wholly, spans = [], []
    for row in frame.itertuples():
        gold = [Span(g["start"], g["end"], g["label"]) for g in json.loads(row.gold) if g["scope"] == "in"]
        wholly.append(coverage(row.text, gold, spans_from_values(row.text, parse_model_output(row.prediction)))[1])
        spans.append(len(gold))
    return wholly, spans


def paired_bootstrap(served: list[int], candidate: list[int], spans: list[int], resamples: int = 10_000,
                     seed: int = SEED) -> dict:
    """The change in wholly-unmasked rate, candidate minus served, with a 95% interval from resampling texts."""
    import numpy as np

    s, c, n = (np.asarray(x, dtype=float) for x in (served, candidate, spans))
    idx = np.random.default_rng(seed).integers(0, len(n), size=(resamples, len(n)))
    diffs = (c[idx].sum(axis=1) - s[idx].sum(axis=1)) / n[idx].sum(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"served_wholly_unmasked": round(float(s.sum() / n.sum()), 4),
            "candidate_wholly_unmasked": round(float(c.sum() / n.sum()), 4),
            "difference": round(float((c.sum() - s.sum()) / n.sum()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "resamples": resamples}


def compare(candidate: str) -> int:
    """The realistic-format half of the frozen rule in data/pii/split_train_realistic.json: the candidate's TAB
    leak rate must fall, with the interval's upper end below zero. The gate on golden is the other half."""
    served, cand = pd.read_parquet(paths()[0]), pd.read_parquet(paths(candidate)[0])
    if served.text.tolist() != cand.text.tolist():
        msg = "the served and candidate runs did not score the same texts"
        raise ValueError(msg)
    result: dict = {"candidate": candidate, "sources": {}}
    for source in ("tab", "nemotron"):
        s_wholly, spans = leaks_per_text(served[served.source == source])
        c_wholly, _ = leaks_per_text(cand[cand.source == source])
        result["sources"][source] = paired_bootstrap(s_wholly, c_wholly, spans)
    result["tab_leak_falls"] = result["sources"]["tab"]["ci95"][1] < 0
    result["note"] = ("TAB decides: it is real text and never trained on. Nemotron-PII's sample shares the new "
                      "training data's distribution, so its fall is reported, not decisive.")
    out = REPO_ROOT / "runs" / f"pii__realistic__compare__{candidate}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for source, r in result["sources"].items():
        print(f"  {source:8s} wholly unmasked {r['served_wholly_unmasked']} → {r['candidate_wholly_unmasked']} "
              f"({r['difference']:+.4f}, 95% CI {r['ci95']})")
    print(f"  TAB leak falls: {result['tab_leak_falls']} · wrote {out.relative_to(REPO_ROOT)}")
    return 0
