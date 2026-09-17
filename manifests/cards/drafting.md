---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: cdla-sharing-1.0
datasets:
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
language:
- en
tags:
- lora
- qlora
- adapterops
---

# adapterops-drafting

Writes a customer-support reply to a request.

Part of [AdapterOps](https://github.com/tpawar03/AdapterOps): four LoRA adapters over one Qwen2.5-1.5B base, served together with vLLM multi-LoRA. Portfolio project — no real users or customer data.

**The scores below describe revision `3778c895d3501f4e8d513252e8e217da4287e720`** (adapter weights sha256 `2e3eeb7e0aac6087…`), the revision the project serves. Load that revision rather than `main`.

## Prompt

```
Write a helpful customer-support reply to this request.
Request: {text}
Reply:
```

Raw text, no chat template. Greedy decoding, at most 448 new tokens. Replace `{text}` with the input.

## Evaluation

Golden sets are frozen random held-out splits; every system below was run on the same items. The hard-cases split is mined from this adapter's own failures, so it is report-only and sits near zero by construction for classification.

| system | split (n) | metric | score |
|---|---|---|---|
| this adapter | golden (300) | GPT-4o grade, 1–5 | 4.2367 |
| base model, prompted | golden (300) | GPT-4o grade, 1–5 | 2.8567 |
| GPT-4o-mini (frontier reference) | golden | GPT-4o grade, 1–5 | 4.5167 |
| this adapter, run 1 / run 2 | golden (300) | distilled judge | 4.4388 / 4.4447 |
| this adapter | hard cases (113), report-only | distilled judge | 3.6686 |

Latency with all four adapters served at once on one A10 (vLLM, concurrency 16): P50 1,249 ms · P95 3,000 ms.

## Caveats

- The distilled judge (the gate metric) is trained on replies from this adapter, the prompted base model and GPT-4o-mini, in and out of domain. Its Spearman with GPT-4o is 0.76 on this adapter's replies, 0.81 across generators and 0.84 on out-of-domain tickets. A judge score is still not a GPT-4o grade: compare models on GPT-4o grades.
- Replies can contain template slots such as `{{Order Number}}`, from the Bitext data.
- Share-alike: trained on CDLA-Sharing-1.0 data.

## Training

QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct`, LoRA rank 16, alpha 32, on all attention and MLP projections; prompt tokens masked from the loss. 6,000 training rows from `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (cdla-sharing-1.0).

Full decision log, results and negative findings: [https://github.com/tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps).
