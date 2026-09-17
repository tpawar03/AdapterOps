"""How often the PII adapter finds personal data in text that has none (changelog 53).

The PII adapter's 0.946 span F1 was measured on documents that all contain PII: the ai4privacy split
has no PII-free documents (D21), so neither training nor the golden set ever showed the adapter an
empty answer. Its first live tickets returned `AGE: 3, SEX: M` for a question about a late card.
This measures that failure instead of quoting examples of it.

**The negative set is PII-free by construction, not by judgement.** A text enters only if it has
nothing any of the adapter's 19 labels could legitimately point at:

- no digit (AGE, BUILDINGNUM, DATE, ZIPCODE and every number-like label need one);
- no `@` (EMAIL) and no template slot such as `{{Order Number}}`;
- no title (TITLE) and no gendered word (GENDER, SEX);
- no capitalised word after the first in a sentence (GIVENNAME, SURNAME, CITY, STREET), with "I"
  and all-caps acronyms allowed.

The screen is deliberately strict — it throws away clean text to keep out ambiguous text — and it
runs over the intent (Banking77) and drafting (Bitext) golden sets, where it keeps 707 and 221
texts. Urgency's tickets are full of names and numbers; 7 of 300 survive, so they are left out.

**Two counts, because the failure has two shapes.** An emitted line is anything the adapter
returns; on this set every one is a false positive. A grounded span is an emitted value that
occurs in the text — what a redaction step would actually mask. A line whose value is not in the
text at all is personal data the adapter invented, which a downstream consumer would store.

The regex-and-spaCy baseline (`eval/pii_baseline.py`) runs on the same texts for comparison.

    uv run adapterops pii-false-positives
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from adapterops.eval.spans import parse_model_output

REPO_ROOT = Path(__file__).resolve().parents[3]
SET_DIR = REPO_ROOT / "evals" / "pii_negatives"
SET_FILE = SET_DIR / "pii_negatives.parquet"
VAL_SET_FILE = SET_DIR / "ai4privacy_val_sentences.parquet"
"""Span-free, screened sentences from the PII validation split — built by `pii-negatives-split`."""
SET_RECORD = SET_DIR / "PII_NEGATIVES.json"
RUN_FILE = REPO_ROOT / "runs" / "pii__false_positives.json"
OUTPUTS_FILE = REPO_ROOT / "runs" / "pii__false_positives__outputs.parquet"
SOURCES = {"intent": ("evals/golden/intent.parquet", "text"),
           "drafting": ("evals/golden/drafting.parquet", "instruction")}

TITLE = re.compile(r"\b(mr|mrs|ms|miss|dr|sir|madam|mister|prof)\b\.?", re.IGNORECASE)
GENDERED = re.compile(r"\b(he|she|him|her|his|hers|male|female|man|woman|boy|girl|husband|wife|son|"
                      r"daughter|mother|father|mom|dad|brother|sister)\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
Generate = Callable[[Sequence[str]], list[str]]


def pii_free(text: str) -> bool:
    """True only when no PII label in the adapter's vocabulary could point at anything in the text."""
    text = str(text)
    if re.search(r"\d", text) or "@" in text or "{{" in text or "<" in text:
        return False
    if TITLE.search(text) or GENDERED.search(text):
        return False
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        words = WORD.findall(sentence)
        if any(w[0].isupper() and w != "I" and not w.isupper() for w in words[1:]):
            return False
    return True


def negative_set(root: Path | None = None) -> pd.DataFrame:
    root = REPO_ROOT if root is None else root
    rows = []
    for source, (path, column) in SOURCES.items():
        frame = pd.read_parquet(root / path)
        for i, text in enumerate(frame[column].astype(str)):
            if pii_free(text):
                rows.append({"source": source, "source_row": i, "text": text})
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


def freeze(force: bool = False) -> pd.DataFrame:
    """Write the set once and pin it; later runs read the frozen file, so the bar cannot drift."""
    if SET_FILE.exists() and not force:
        return pd.read_parquet(SET_FILE)
    frame = negative_set()
    SET_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(SET_FILE, index=False)
    SET_RECORD.write_text(json.dumps({
        "purpose": "PII-free texts for the PII adapter's false-positive rate (changelog 53).",
        "rules": [line.strip("- ").strip() for line in (__doc__ or "").split("\n")
                  if line.startswith("- ")],
        "sources": {k: v[0] for k, v in SOURCES.items()},
        "counts": frame.source.value_counts().to_dict(),
        "sha256": hashlib.sha256(SET_FILE.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return frame


def grounded_in(text: str, value: str) -> bool:
    """Whether a value occurs in the text as whole words. A substring test would call `SEX: M`
    grounded in "My card" — a redaction step masks words, not the letter inside one."""
    return bool(value) and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text) is not None


def score(text: str, output: str) -> dict:
    pairs = parse_model_output(output)
    grounded = sum(grounded_in(text, value) for _, value in pairs)
    return {"lines": len(pairs), "grounded": grounded, "invented": len(pairs) - grounded,
            "labels": [label for label, _ in pairs]}


def summarise(frame: pd.DataFrame, column: str = "output") -> dict:
    scored = pd.DataFrame([score(t, o) for t, o in zip(frame.text, frame[column], strict=True)])
    joined = pd.concat([frame.reset_index(drop=True), scored], axis=1)

    def block(part: pd.DataFrame) -> dict:
        n = len(part)
        labels = Counter(label for row in part.labels for label in row)
        return {
            "texts": n,
            "texts_with_any_line": int((part.lines > 0).sum()),
            "false_positive_rate": round(float((part.lines > 0).mean()), 4) if n else None,
            "texts_with_grounded_span": int((part.grounded > 0).sum()),
            "grounded_rate": round(float((part.grounded > 0).mean()), 4) if n else None,
            "texts_with_invented_value": int((part.invented > 0).sum()),
            "lines": int(part.lines.sum()),
            "lines_per_flagged_text": (round(float(part.lines[part.lines > 0].mean()), 2)
                                       if (part.lines > 0).any() else None),
            "labels": dict(labels.most_common()),
        }

    return {"all": block(joined),
            "per_source": {s: block(g) for s, g in joined.groupby("source", sort=True)}}


def adapter_generator(batch: int = 16, adapter_dir: str | None = None) -> tuple[Generate, dict]:
    """The pinned PII adapter — or a local checkpoint in its place — batched, with the serving prompt
    and token cap."""
    import torch

    from adapterops.router.generate import MAX_TOKENS
    from adapterops.serve.pipeline import TransformersBackend, adapter_components, load_manifest
    from adapterops.train.qlora import PROMPTS

    manifest = load_manifest()
    components = adapter_components(manifest)
    if adapter_dir:
        components["pii"] = {"path": adapter_dir}
    backend = TransformersBackend(components)
    tok, model, device = backend._load()
    model.set_adapter("pii")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(texts: Sequence[str]) -> list[str]:
        out = []
        for i in range(0, len(texts), batch):
            chunk = [PROMPTS["pii"].format(text=t) for t in texts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_TOKENS["pii"], do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            print(f"\r  {len(out)}/{len(texts)}", end="", flush=True)
        print()
        return out

    pii = manifest["components"]["adapters"]["pii"]
    served = ({"adapter": adapter_dir, "revision": None, "manifest_version": None}
              if adapter_dir else
              {"adapter": pii["repo"], "revision": pii["revision"],
               "manifest_version": manifest["version"]})
    return generate, {**served, "device": device, "batch": batch,
                      "max_new_tokens": MAX_TOKENS["pii"]}


def vllm_generator(base_url: str, workers: int = 16) -> tuple[Generate, dict]:
    """The served PII adapter through vLLM, with the serving prompt and token cap — the numbers the
    laptop runs only approximate."""
    from concurrent.futures import ThreadPoolExecutor

    from adapterops.router.generate import MAX_TOKENS
    from adapterops.serve.pipeline import VLLMBackend, load_manifest

    backend = VLLMBackend(base_url, timeout=300.0)

    def generate(texts: Sequence[str]) -> list[str]:
        with ThreadPoolExecutor(workers) as pool:
            return [g.text for g in pool.map(lambda t: backend.generate("pii", t), texts)]

    manifest = load_manifest()
    pii = manifest["components"]["adapters"]["pii"]
    return generate, {"adapter": pii["repo"], "revision": pii["revision"],
                      "manifest_version": manifest["version"], "device": f"vllm {base_url}",
                      "max_new_tokens": MAX_TOKENS["pii"]}


def baseline_predictor() -> tuple[Callable[[str], list], str]:
    from adapterops.eval import pii_baseline

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        return (lambda text: pii_baseline.predict_hybrid(text, nlp)), "regex + spaCy en_core_web_sm"
    except (ImportError, OSError):
        return pii_baseline.predict, "regex only — spaCy model not installed"


SETS = ("golden_screened", "ai4privacy_val")


def main(limit: int | None = None, batch: int = 16, force_set: bool = False,
         generate: Generate | None = None, baseline: Callable[[str], list] | None = None,
         set_name: str = "golden_screened", adapter_dir: str | None = None,
         name: str | None = None, base_url: str | None = None) -> int:
    if set_name == "golden_screened":
        frame, set_file = freeze(force=force_set), SET_FILE
    elif set_name == "ai4privacy_val":
        set_file = VAL_SET_FILE
        if not set_file.exists():
            print(f"  {set_file} missing — build it with `adapterops pii-negatives-split`")
            return 2
        frame = pd.read_parquet(set_file)
    else:
        msg = f"unknown set {set_name!r}; expected one of {SETS}"
        raise ValueError(msg)
    if limit:
        frame = frame.groupby("source", group_keys=False).head(limit).reset_index(drop=True)
    info: dict = {}
    if generate is None and base_url:
        generate, info = vllm_generator(base_url)
    elif generate is None:
        generate, info = adapter_generator(batch, adapter_dir)
    run_file, outputs_file = RUN_FILE, OUTPUTS_FILE
    if name:
        run_file = RUN_FILE.with_name(f"pii__false_positives__{name}.json")
        outputs_file = RUN_FILE.with_name(f"pii__false_positives__{name}__outputs.parquet")
    baseline_name = "supplied"
    if baseline is None:
        baseline, baseline_name = baseline_predictor()

    started = time.perf_counter()
    frame = frame.assign(output=generate(list(frame.text)))
    seconds = time.perf_counter() - started
    frame = frame.assign(baseline=["\n".join(f"{s.label}: {t[s.start:s.end]}" for s in baseline(t))
                                   for t in frame.text])

    adapter, base = summarise(frame, "output"), summarise(frame, "baseline")
    flagged = frame[[bool(parse_model_output(o)) for o in frame.output]]
    result = {
        "question": "How often does the PII adapter report personal data in text that has none?",
        "set": {"name": set_name, "file": str(set_file.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(set_file.read_bytes()).hexdigest(),
                "texts": len(frame), "limit_per_source": limit},
        "adapter": {**info, "seconds": round(seconds, 1), **adapter},
        "baseline": {"name": baseline_name, **base},
        "examples": [{"source": r.source, "text": r.text, "output": r.output.strip()}
                     for r in flagged.head(15).itertuples()],
        "notes": ("Every emitted line is a false positive on this set. A grounded span's value "
                  "occurs in the text; an invented value does not. Generated on this machine with "
                  "the serving prompt and token cap, greedy, so exact outputs can differ from vLLM's."),
    }
    run_file.parent.mkdir(exist_ok=True)
    run_file.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    frame.to_parquet(outputs_file, index=False)
    a = adapter["all"]
    print(f"  adapter: {a['texts_with_any_line']}/{a['texts']} texts with a line "
          f"({a['false_positive_rate']:.1%}), {a['texts_with_grounded_span']} grounded, "
          f"{a['texts_with_invented_value']} invented · baseline: "
          f"{base['all']['texts_with_any_line']}/{base['all']['texts']} "
          f"({base['all']['false_positive_rate']:.1%})")
    print(f"  wrote {run_file.relative_to(REPO_ROOT)}")
    return 0
