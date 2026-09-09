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
