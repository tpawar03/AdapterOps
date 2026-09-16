"""Hugging Face model cards for the four served adapters, rendered from committed runs (PRD §9).

The Hub cards were the unfilled template. PRD §9 asks for eval scores at upload time; the adapters
were uploaded before most scores existed, so the cards are written now — every number read from a
run file, the revision those numbers describe named in the card, and each adapter's caveats beside
its scores rather than in a repository the reader may never open.

**A card commit moves `main` without moving weights.** Serving resolves adapters by pinned revision
(changelog 29), so the commit changes nothing that is served; `verify-pins` reports it as
`main_moved_same_weights`.

    uv run python -m adapterops.manifest.cards            # render to manifests/cards/
    uv run python -m adapterops.manifest.cards --push     # upload each as the repo's README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapterops.eval.regression import GATED, TASKS
from adapterops.router.generate import MAX_TOKENS
from adapterops.train.qlora import PROMPTS, TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "manifests" / "cards"
REPO_URL = "https://github.com/tpawar03/AdapterOps"

WHAT = {
    "intent": "Classifies a banking customer's request into one of Banking77's 77 intent labels.",
    "urgency": "Classifies a support ticket's urgency as low, medium or high.",
    "pii": "Lists the personal information in a text as `LABEL: value` lines, over 19 labels.",
    "drafting": "Writes a customer-support reply to a request.",
}
DATASETS = {
    "intent": ("mteb/banking77", "mit"),
    "urgency": ("Tobi-Bueck/customer-support-tickets", "cc-by-nc-4.0"),
    "pii": ("ai4privacy/pii-masking-openpii-1m", "cc-by-4.0"),
    "drafting": ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "cdla-sharing-1.0"),
}
PROMPTED = {
    "intent": "runs/intent__prompted-per-class.json",
    "urgency": "runs/urgency__prompted-fewshot.json",
    "pii": "runs/pii__prompted-fewshot.json",
}
TRAIN_ROWS = {
    "intent": ("runs/intent__train.json", ("train_rows",)),
    "urgency": ("runs/urgency__adapter.json", ("training", "rows")),
    "pii": ("runs/pii-negatives__train.json", ("train_rows",)),
    "drafting": ("runs/drafting__train.json", ("train_rows",)),
}


def _json(path: str) -> dict | None:
    file = REPO_ROOT / path
    return json.loads(file.read_text()) if file.exists() else None


def load() -> dict:
    return {
        "pins": _json("manifests/adapters.json")["components"],
        "baseline": [_json("runs/regression__v1-baseline-1.json"),
                     _json("runs/regression__v1-baseline-2.json")],
        "prompted": {t: _json(p) for t, p in PROMPTED.items()},
        "urgency_adapter": _json("runs/urgency__adapter.json"),
        "m2": _json("runs/drafting__m2_gpt4o.json"),
        "serving": _json("runs/m1_serving.json"),
        "ceiling": _json("runs/frontier__golden.json"),
        "ceiling_drafting": _json("runs/drafting__m2_gpt4o__frontier.json"),
        "train_rows": {t: _dig(_json(p), keys) for t, (p, keys) in TRAIN_ROWS.items()},
        # PII since manifest v6 (D50): scored beside the adapter it replaced, in one session.
        "pii_v6": _json("runs/regression__a10-v5-pii-negatives.json"),
        "pii_previous": _json("runs/regression__a10-v5-pii-original.json"),
        "fp": _json("runs/pii__false_positives__negatives.json"),
        "fp_previous": _json("runs/pii__false_positives.json"),
        "fp_val": _json("runs/pii__false_positives__negatives-val.json"),
        "fp_val_previous": _json("runs/pii__false_positives__served-val.json"),
    }


def _dig(data: dict, keys: tuple[str, ...]):
    for k in keys:
        data = data[k]
    return data


def caveats(task: str, ctx: dict) -> list[str]:
    if task == "intent":
        return ["Gated on exact-label accuracy; macro-F1 over 77 classes is indicative only.",
                "English retail-banking phrasing only (Banking77)."]
    if task == "urgency":
        tfidf = ctx["urgency_adapter"]["context"]["tfidf_logreg_baseline"]["macro_f1"]
        return [(f"**Negative result.** Loses to TF-IDF + logistic regression ({tfidf:.3f} "
                 "macro-F1), and its margin over a prompted base model is within its own "
                 "run-to-run spread."),
                ("**Non-commercial.** Trained on CC BY-NC 4.0 data by Tobi Bueck "
                 "(`Tobi-Bueck/customer-support-tickets`); this adapter inherits the restriction.")]
    if task == "pii":
        out = ["Trained and evaluated on synthetic spans only. Not a compliance control.",
               "Strict scoring requires each value to match its span exactly."]
        fp = ctx.get("fp")
        if fp:
            a = fp["adapter"]["all"]
            out.insert(0, (f"Trained with PII-free sentences and empty answers so it can report nothing; on "
                           f"{a['texts']} PII-free texts it still reports a span in {a['texts_with_any_line']}. "
                           "The previous revision, trained only on documents containing PII, reported one "
                           "in every text."))
        return out
    sides = ctx["m2"]["sides"]
    return [(f"The distilled judge (the gate metric) tracks GPT-4o on this adapter's replies "
             f"(Spearman {sides['adapter']['distilled_vs_gpt4o_spearman']:.2f}) but not on another "
             f"generator's ({sides['prompted']['distilled_vs_gpt4o_spearman']:.2f}). Compare "
             "models on GPT-4o grades."),
            "Replies can contain template slots such as `{{Order Number}}`, from the Bitext data.",
            "Share-alike: trained on CDLA-Sharing-1.0 data."]


def _f(v: float | None) -> str:
    return "not measured" if v is None else f"{v:.4f}"


def pii_rows(ctx: dict) -> list[str] | None:
    """PII's rows since manifest v6: the served adapter beside the one it replaced, in one session, plus
    the false-positive rates that motivated the change. None when those runs are absent."""
    now_run, before_run = ctx.get("pii_v6"), ctx.get("pii_previous")
    if not now_run or not before_run:
        return None
    metric = GATED["pii"]
    now, before = now_run["per_split"]["pii"], before_run["per_split"]["pii"]
    replaced = (ctx["pins"]["pii"].get("replaces") or {}).get("revision", "")[:8]
    prompted = ctx["prompted"]["pii"]
    ceiling = (ctx["ceiling"] or {}).get("per_task", {}).get("pii", {})
    rows = [
        "| system | split (n) | metric | score |", "|---|---|---|---|",
        f"| this adapter | golden ({now['random']['n']}) | {metric} | {now['random'][metric]:.4f} |",
        (f"| previous adapter `{replaced}`, same session | golden ({before['random']['n']}) | {metric} | "
         f"{before['random'][metric]:.4f} |"),
        (f"| base model, {prompted['setup']['demonstrations']} demonstrations | golden ({prompted['n']}) | "
         f"{metric} | {prompted['metrics'][metric]:.4f} |"),
        f"| GPT-4o-mini (frontier reference) | golden | {metric} | {_f(ceiling.get(metric))} |",
    ]
    for field, name in (("docs_fully_masked", "documents with all personal text masked"),
                        ("gold_spans_wholly_unmasked", "gold spans left wholly unmasked")):
        if now["random"].get(field) is not None:
            rows.append(f"| this adapter | golden ({now['random']['n']}) | {name} | "
                        f"{now['random'][field]:.4f} |")
    for label, key in (("PII-free texts", "fp"), ("held-out PII-free sentences", "fp_val")):
        mine, theirs = ctx.get(key), ctx.get(f"{key}_previous")
        if mine and theirs:
            a, b = mine["adapter"]["all"], theirs["adapter"]["all"]
            rows += [(f"| this adapter | {label} ({a['texts']}) | texts with a reported span | "
                      f"{a['texts_with_any_line']} |"),
                     (f"| previous adapter | {label} ({b['texts']}) | texts with a reported span | "
                      f"{b['texts_with_any_line']} |")]
    rows.append(f"| this adapter | hard cases ({now['hard']['n']}), report-only | {metric} | "
                f"{now['hard'][metric]:.4f} |")
    return rows


def eval_rows(task: str, ctx: dict) -> list[str]:
    if task == "pii" and (rows := pii_rows(ctx)):
        return rows
    metric = GATED[task]
    a, b = (run["per_split"][task] for run in ctx["baseline"])
    ceiling = (ctx["ceiling"] or {}).get("per_task", {}).get(task, {})
    rows = ["| system | split (n) | metric | score |", "|---|---|---|---|"]
    if task == "drafting":
        sides = ctx["m2"]["sides"]
        frontier = (ctx["ceiling_drafting"] or {}).get("sides", {}).get("frontier", {})
        rows += [
            (f"| this adapter | golden ({sides['adapter']['n']}) | GPT-4o grade, 1–5 | "
            f"{sides['adapter']['gpt4o_mean']:.4f} |"),
            (f"| base model, prompted | golden ({sides['prompted']['n']}) | GPT-4o grade, 1–5 | "
            f"{sides['prompted']['gpt4o_mean']:.4f} |"),
            (f"| GPT-4o-mini (frontier reference) | golden | GPT-4o grade, 1–5 | "
            f"{_f(frontier.get('gpt4o_mean'))} |"),
            (f"| this adapter, run 1 / run 2 | golden ({a['random']['n']}) | distilled judge | "
            f"{a['random'][metric]:.4f} / {b['random'][metric]:.4f} |"),
        ]
    else:
        prompted = ctx["prompted"][task]
        rows += [
            (f"| this adapter, run 1 / run 2 | golden ({a['random']['n']}) | {metric} | "
            f"{a['random'][metric]:.4f} / {b['random'][metric]:.4f} |"),
            (f"| base model, {prompted['setup']['demonstrations']} demonstrations | "
            f"golden ({prompted['n']}) | {metric} | {prompted['metrics'][metric]:.4f} |"),
            (f"| GPT-4o-mini (frontier reference) | golden | {metric} | "
            f"{_f(ceiling.get(metric))} |"),
        ]
        if task == "urgency":
            tfidf = ctx["urgency_adapter"]["context"]["tfidf_logreg_baseline"]
            rows.append(f"| TF-IDF + logistic regression | golden (300) | macro_f1 | "
                        f"{tfidf['macro_f1']:.4f} |")
    hard_metric = "distilled judge" if task == "drafting" else metric
    rows.append(f"| this adapter | hard cases ({a['hard']['n']}), report-only | {hard_metric} | "
                f"{a['hard'][metric]:.4f} |")
    return rows


def render(task: str, ctx: dict) -> str:
    pin = ctx["pins"][task]
    dataset, licence = DATASETS[task]
    lat = ctx["serving"]["per_adapter"][task]
    cfg = TrainConfig()
    lines = [
        "---",
        f"base_model: {ctx['pins']['base_model']['repo']}",
        "library_name: peft",
        f"license: {licence}",
        "datasets:", f"- {dataset}",
        "language:", "- en",
        "tags:", "- lora", "- qlora", "- adapterops",
        "---", "",
        f"# adapterops-{task}", "",
        WHAT[task], "",
        (f"Part of [AdapterOps]({REPO_URL}): four LoRA adapters over one Qwen2.5-1.5B base, served "
         "together with vLLM multi-LoRA. Portfolio project — no real users or customer data."), "",
        (f"**The scores below describe revision `{pin['revision']}`** (adapter weights sha256 "
         f"`{pin['weight_sha256'][:16]}…`), the revision the project serves. Load that revision "
         "rather than `main`."), "",
        "## Prompt", "",
        "```", PROMPTS[task].rstrip("\n"), "```", "",
        (f"Raw text, no chat template. Greedy decoding, at most {MAX_TOKENS[task]} new tokens. "
         "Replace `{text}` with the input."), "",
        "## Evaluation", "",
        ("Golden sets are frozen random held-out splits; every system below was run on the same "
         "items. The hard-cases split is mined from this adapter's own failures, so it is "
         "report-only and sits near zero by construction for classification."), "",
        *eval_rows(task, ctx), "",
        (f"Latency with all four adapters served at once on one A10 (vLLM, concurrency "
         f"{ctx['serving']['concurrency']}): P50 {lat['p50_ms']:,.0f} ms · P95 "
         f"{lat['p95_ms']:,.0f} ms"
         + (", measured with the previous revision." if pin.get("replaces") else ".")), "",
        "## Caveats", "",
        *(f"- {c}" for c in caveats(task, ctx)), "",
        "## Training", "",
        (f"QLoRA (4-bit NF4) on `{ctx['pins']['base_model']['repo']}`, LoRA rank {cfg.lora_r}, "
         f"alpha {cfg.lora_alpha}, on all attention and MLP projections; prompt tokens masked from "
         f"the loss. {ctx['train_rows'][task]:,} training rows from `{dataset}` ({licence})."), "",
        f"Full decision log, results and negative findings: [{REPO_URL}]({REPO_URL}).", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adapterops.manifest.cards")
    ap.add_argument("--push", action="store_true", help="upload each card as the repo's README.md")
    args = ap.parse_args(argv)

    ctx = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = OUT_DIR / f"{task}.md"
        path.write_text(render(task, ctx))
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        for task in TASKS:
            pin = ctx["pins"][task]
            info = api.upload_file(
                path_or_fileobj=str(OUT_DIR / f"{task}.md"), path_in_repo="README.md",
                repo_id=pin["repo"], repo_type="model",
                commit_message=f"Model card: eval scores for pinned revision {pin['revision'][:8]}")
            print(f"  pushed {pin['repo']} -> {info.oid[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
