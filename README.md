# AdapterOps

A multi-task LoRA adapter service: four task-specific adapters over one Qwen2.5-1.5B
base, served together from a single GPU footprint, routed between local inference and a
frontier API by a learned cost-quality policy, and gated by a regression check proven
against two deliberately injected failures.

Portfolio project. No real users, no real customer data.

## The documents

| File | What it is |
|---|---|
| [`adapter-service-prd.md`](adapter-service-prd.md) | The spec — v2.3. What gets built and why, with `§`/`F`/`M` identifiers everything else references. |
| [`BUILD-PLAN.md`](BUILD-PLAN.md) | Execution order: day-1 verification gate, per-phase task lists, hour budgets, and the points to stop and reassess. |
| [`STATUS.md`](STATUS.md) | Live status, decision log with rejected alternatives, trade-offs, and findings. |
| [`COST-LOG.md`](COST-LOG.md) | Every charge against the $50 ceiling. |

## Layout

```
src/adapterops/     train · eval · router · judge · serve · manifest
evals/golden/       random held-out splits (intent 770, others 300)
evals/hard/         failure-mined split, from Phase 4
manifests/          versioned system manifests
runs/               committed regression-run results
data/               mirrored dataset snapshots (PRD §9)
notebooks/          Colab training notebooks
demo/               Gradio app
```

## Setup

```bash
uv sync                  # base — works on macOS and Linux
uv sync --extra gpu      # on the rented CUDA box: adds vLLM + bitsandbytes
uv run adapterops --help
```

Training runs on free Colab/Kaggle T4s; all reportable latency numbers come from a
rented A10G. See PRD §7 — numbers from a shared free tier are not reportable.

## Status

Not started. Phase 0 begins with the eight day-1 verification checks in `BUILD-PLAN.md`;
several can re-scope the project, so they come before any code.

## Data sources and licences

| task | source | licence | notes |
|---|---|---|---|
| intent | `mteb/banking77` | MIT | parquet mirror of Banking77 |
| urgency | `Tobi-Bueck/customer-support-tickets` | **CC BY-NC 4.0** | **non-commercial** — see below |
| pii | `ai4privacy/pii-masking-openpii-1m` | CC BY 4.0 | synthetic spans, English rows only |
| drafting | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | CDLA-Sharing-1.0 | share-alike; attribution required |

### Non-commercial restriction on the urgency adapter

The urgency data is licensed **CC BY-NC 4.0**. The adapter trained on it
(`Tanny03/adapterops-urgency`) is a derivative work and inherits that restriction:

- **Not for commercial use.** Demonstration, evaluation and portfolio use only.
- **Attribution required** — dataset by Tobi Bueck, `Tobi-Bueck/customer-support-tickets`.

This is the only non-permissive source in the project. The other three adapters carry no
such restriction.

It was a deliberate trade. PRD §9 originally named a CC0 Kaggle dataset
(`albertobircoci/support-ticket-priority-dataset-50k`), which turned out to contain **no
ticket text at all** — it is tabular, and its `description_length` column is an integer
with the description itself absent. The choice was therefore an NC-licensed source that
works, or no urgency adapter. See PRD changelog 27.
