---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
datasets:
- mteb/banking77
language:
- en
tags:
- lora
- qlora
- adapterops
---

# adapterops-intent

Classifies a banking customer's request into one of Banking77's 77 intent labels.

Part of [AdapterOps](https://github.com/tpawar03/AdapterOps): four LoRA adapters over one Qwen2.5-1.5B base, served together with vLLM multi-LoRA. Portfolio project — no real users or customer data.

**The scores below describe revision `a7eb75ec67386805b1026aeacffa4f6c27d2de7e`** (adapter weights sha256 `6d1a57e9aa23e560…`), the revision the project serves. Load that revision rather than `main`.

## Prompt

```
Classify the customer's banking request into one intent label.
Request: {text}
Intent:
```

Raw text, no chat template. Greedy decoding, at most 12 new tokens. Replace `{text}` with the input.

## Evaluation

Golden sets are frozen random held-out splits; every system below was run on the same items. The hard-cases split is mined from this adapter's own failures, so it is report-only and sits near zero by construction for classification.

| system | split (n) | metric | score |
|---|---|---|---|
| this adapter, run 1 / run 2 | golden (770) | micro_accuracy | 0.9286 / 0.9299 |
| base model, 77 demonstrations | golden (770) | micro_accuracy | 0.5727 |
| GPT-4o-mini (frontier reference) | golden | micro_accuracy | 0.6870 |
| this adapter | hard cases (57), report-only | micro_accuracy | 0.0000 |

Latency with all four adapters served at once on one A10 (vLLM, concurrency 16): P50 82 ms · P95 136 ms.

## Caveats

- Gated on exact-label accuracy; macro-F1 over 77 classes is indicative only.
- English retail-banking phrasing only (Banking77).

## Training

QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16, alpha 32, on all attention and MLP projections; prompt tokens masked from the loss. 8,495 training rows from `mteb/banking77` (mit).

Full decision log, results and negative findings: [https://github.com/tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps).
