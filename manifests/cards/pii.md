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

**The scores below describe revision `e0bde68fb34693d10611e6888eb50c78825c86c7`** (adapter weights sha256 `fec1371a12133f24…`), the revision the project serves. Load that revision rather than `main`.

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
| this adapter | golden (300) | span_f1_strict | 0.9442 |
| previous adapter `3b38a284`, same session | golden (300) | span_f1_strict | 0.9470 |
| base model, 5 demonstrations | golden (300) | span_f1_strict | 0.5700 |
| GPT-4o-mini (frontier reference) | golden | span_f1_strict | 0.6663 |
| this adapter | golden (300) | documents with all personal text masked | 0.8433 |
| this adapter | golden (300) | gold spans left wholly unmasked | 0.0079 |
| this adapter | PII-free texts (928) | texts with a reported span | 5 |
| previous adapter | PII-free texts (928) | texts with a reported span | 928 |
| this adapter | held-out PII-free sentences (491) | texts with a reported span | 0 |
| previous adapter | held-out PII-free sentences (491) | texts with a reported span | 491 |
| this adapter | hard cases (150), report-only | span_f1_strict | 0.8785 |

Latency with all four adapters served at once on one A10 (vLLM, concurrency 16): P50 858 ms · P95 2,259 ms, measured with the previous revision.

## Caveats

- Trained with PII-free sentences and empty answers so it can report nothing; on 928 PII-free texts it still reports a span in 5. The previous revision, trained only on documents containing PII, reported one in every text.
- Trained and evaluated on synthetic spans only. Not a compliance control.
- Strict scoring requires each value to match its span exactly.

## Training

QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16, alpha 32, on all attention and MLP projections; prompt tokens masked from the loss. 13,910 training rows from `ai4privacy/pii-masking-openpii-1m` (cc-by-4.0).

Full decision log, results and negative findings: [https://github.com/tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps).
