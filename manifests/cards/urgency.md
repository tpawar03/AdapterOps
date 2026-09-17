---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: cc-by-nc-4.0
datasets:
- Tobi-Bueck/customer-support-tickets
language:
- en
tags:
- lora
- qlora
- adapterops
---

# adapterops-urgency

Classifies a support ticket's urgency as low, medium or high.

Part of [AdapterOps](https://github.com/tpawar03/AdapterOps): four LoRA adapters over one Qwen2.5-1.5B base, served together with vLLM multi-LoRA. Portfolio project — no real users or customer data.

**No longer served.** Since a manifest promotion this task is answered by a TF-IDF + logistic regression model (`models/urgency-tfidf/model.joblib`, sha256 `0aa15ca8b4f63b21…`), which beats this adapter on the golden set, on items mined from GPT-4o-mini's failures, and across three training seeds — see `runs/urgency__tfidf.json` in [the repository](https://github.com/tpawar03/AdapterOps).

**The scores below describe revision `53d1006e1c4cc863ab1d5db56cb5049938c14888`** (adapter weights sha256 `c821ef1dd3ccedaa…`), the last revision the project served. Load that revision rather than `main`.

## Prompt

```
Classify the urgency of this support ticket.
Ticket: {text}
Urgency:
```

Raw text, no chat template. Greedy decoding, at most 6 new tokens. Replace `{text}` with the input.

## Evaluation

Golden sets are frozen random held-out splits; every system below was run on the same items. The hard-cases split is mined from this adapter's own failures, so it is report-only and sits near zero by construction for classification.

| system | split (n) | metric | score |
|---|---|---|---|
| this adapter, run 1 / run 2 | golden (300) | macro_f1 | 0.4096 / 0.4223 |
| base model, 12 demonstrations | golden (300) | macro_f1 | 0.3962 |
| GPT-4o-mini (frontier reference) | golden | macro_f1 | 0.3824 |
| TF-IDF + logistic regression | golden (300) | macro_f1 | 0.5465 |
| this adapter | hard cases (150), report-only | macro_f1 | 0.0068 |

Latency with all four adapters served at once on one A10 (vLLM, concurrency 16): P50 51 ms · P95 60 ms.

## Caveats

- **Negative result.** Loses to TF-IDF + logistic regression (0.546 macro-F1), and its margin over a prompted base model is within its own run-to-run spread.
- **Non-commercial.** Trained on CC BY-NC 4.0 data by Tobi Bueck (`Tobi-Bueck/customer-support-tickets`); this adapter inherits the restriction.

## Training

QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16, alpha 32, on all attention and MLP projections; prompt tokens masked from the loss. 9,879 training rows from `Tobi-Bueck/customer-support-tickets` (cc-by-nc-4.0).

Full decision log, results and negative findings: [https://github.com/tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps).
