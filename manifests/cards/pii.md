---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: cc-by-4.0
datasets:
- ai4privacy/pii-masking-openpii-1m
language:
- en
tags:
- lora
- qlora
- adapterops
---

# adapterops-pii

Lists the personal information in a text as `LABEL: value` lines, over 19 labels.

Part of [AdapterOps](https://github.com/tpawar03/AdapterOps): four LoRA adapters over one Qwen2.5-1.5B base, served together with vLLM multi-LoRA. Portfolio project — no real users or customer data.

**The scores below describe revision `3b38a28479215897ab927f00b78da99a57813469`** (adapter weights sha256 `8bf54f7d6bd24ea8…`), the revision the project serves. Load that revision rather than `main`.

## Prompt

```
List every piece of personal information in the text, one per line, as LABEL: value.
Text: {text}
Found:
```

Raw text, no chat template. Greedy decoding, at most 384 new tokens. Replace `{text}` with the input.

## Evaluation

Golden sets are frozen random held-out splits; every system below was run on the same items. The hard-cases split is mined from this adapter's own failures, so it is report-only and sits near zero by construction for classification.

| system | split (n) | metric | score |
|---|---|---|---|
| this adapter, run 1 / run 2 | golden (300) | span_f1_strict | 0.9462 / 0.9453 |
| base model, 5 demonstrations | golden (300) | span_f1_strict | 0.5700 |
| GPT-4o-mini (frontier reference) | golden | span_f1_strict | 0.6663 |
| this adapter | hard cases (150), report-only | span_f1_strict | 0.8520 |

Latency with all four adapters served at once on one A10 (vLLM, concurrency 16): P50 858 ms · P95 2,259 ms.

## Caveats

- Trained and evaluated on synthetic spans only. Not a compliance control.
- Strict scoring requires each value to match its span exactly.

## Training

QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16, alpha 32, on all attention and MLP projections; prompt tokens masked from the loss. 8,000 training rows from `ai4privacy/pii-masking-openpii-1m` (cc-by-4.0).

Full decision log, results and negative findings: [https://github.com/tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps).
