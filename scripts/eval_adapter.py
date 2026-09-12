"""Score a trained adapter on its frozen golden set.

Handles both metric families the project uses, so evaluation is a committed artifact
rather than a script pasted into a terminal:

  classification (intent, urgency) -> micro-accuracy, macro-F1, exact-label rate
  span extraction (pii)            -> strict and relaxed span F1 against a known ceiling

Writes runs/<task>__<system>.json.

  python scripts/eval_adapter.py --task urgency --adapter checkpoints/urgency
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapterops.eval.spans import (
    Span,
    parse_model_output,
    score_spans,
    spans_from_values,
)
from adapterops.router.generate import MAX_TOKENS
from adapterops.train.qlora import COLUMNS, PROMPTS

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
# One cap, shared with the router pool run. This script used to keep its own copy with PII at
# 160 tokens — 23 of 300 golden documents are longer, holding 18.3% of golden spans, and the
# committed 0.9190 was generated under it. The Phase 2 fix raised router.generate's copy and
# never reached this one.
MAX_NEW = MAX_TOKENS
PII_CEILING = 0.9942


def generate(model, tok, prompts: list[str], max_new: int, batch: int) -> list[str]:
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=768).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for g in gen]
        print(f"\r  {i + len(chunk)}/{len(prompts)}", end="", flush=True)
    print()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["intent", "urgency", "pii"])
    ap.add_argument("--adapter", required=True, help="local path or Hub id")
    ap.add_argument("--system", default="adapter")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto"),
        args.adapter,
    ).eval()

    golden = pd.read_parquet(ROOT / f"evals/golden/{args.task}.parquet")
    text_col, label_col = COLUMNS[args.task]
    raw = generate(model, tok, [PROMPTS[args.task].format(text=t) for t in golden[text_col]],
                   MAX_NEW[args.task], args.batch)

    result: dict = {"task": args.task, "system": args.system, "adapter": args.adapter,
                    "n": len(golden), "split": f"evals/golden/{args.task}.parquet"}

    if args.task == "pii":
        gold = [[Span(s["start"], s["end"], s["label"]) for s in m] for m in golden.privacy_mask]
        pred = [spans_from_values(t, parse_model_output(o))
                for t, o in zip(golden[text_col], raw, strict=True)]
        result["gated_metric"] = "span_f1_strict"
        result["ceiling_strict"] = PII_CEILING
        for strict in (True, False):
            r = score_spans(gold, pred, strict=strict)
            result[r["mode"]] = {k: r[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
        result["pct_of_ceiling"] = round(result["strict"]["f1"] / PII_CEILING, 4)
    else:
        pred = [o.strip().split("\n")[0].strip() for o in raw]
        gold = golden[label_col].astype(str).tolist()
        labels = set(pd.read_parquet(ROOT / f"data/{args.task}/split_train.parquet")[label_col]
                     .astype(str))
        result["gated_metric"] = "macro_f1" if args.task == "urgency" else "micro_accuracy"
        result["metrics"] = {
            "micro_accuracy": round(accuracy_score(gold, pred), 4),
            "macro_f1": round(f1_score(gold, pred, average="macro", zero_division=0), 4),
            "exact_label_rate": round(sum(p in labels for p in pred) / len(pred), 4),
        }

    out = ROOT / f"runs/{args.task}__{args.system}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
