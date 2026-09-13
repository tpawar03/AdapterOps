# AdapterOps

![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Tanny03/adapterops-demo)

AdapterOps is a multi-task LoRA adapter service for customer-support tickets. Four task-specific
adapters — intent classification, urgency, PII detection and reply drafting — share one
Qwen2.5-1.5B base model and are served together from a single GPU with vLLM, instead of running a
separate model or frontier-API call per task.

It was built to answer the questions that decide whether a small fine-tuned model is worth running:
does fine-tuning beat prompting the same model, when should a request escalate to a frontier API, and
has a new release regressed before it ships? So the service comes with its evaluation: frozen golden
sets, a regression gate whose thresholds are derived from measured run-to-run variance, a versioned
manifest with a proven rollback, a routing study against GPT-4o-mini, and a distilled judge for the one
open-ended task. Results — including the negative ones — are in [`STATUS.md`](STATUS.md) and
[`runs/DASHBOARD.md`](runs/DASHBOARD.md). Portfolio project: no real users or customer data.

[![Recorded demo: adapter and GPT-4o-mini outputs on the intent golden set](docs/demo.png)](https://huggingface.co/spaces/Tanny03/adapterops-demo)

## Setup & Usage

### Prerequisites

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)** — dependencies are locked in `uv.lock`.
- **macOS or Linux** for evaluation, routing, the judge, the dashboard and the demo; all run on CPU.
- **Linux with an NVIDIA GPU (CUDA)** only for training adapters and serving with vLLM. The adapters
  were trained on free Colab/Kaggle T4s and a rented A10.
- **`OPENAI_API_KEY`** (optional) — only for commands that call GPT-4o-mini or GPT-4o: `frontier`,
  `judge-label`, `judge-m2` and `python -m adapterops.eval.ceiling`.
- **A Hugging Face login** (optional) — only to push adapters, model cards or the demo Space.

### Installation

```bash
git clone https://github.com/tpawar03/AdapterOps.git
cd AdapterOps
uv sync
```

On the GPU machine, add vLLM and bitsandbytes:

```bash
uv sync --extra gpu
```

For the API-calling commands, put the key in a `.env` file at the repository root:

```bash
echo "OPENAI_API_KEY=your-key" > .env
```

### Usage Guide

Everything runs through one CLI; `uv run adapterops --help` lists every command.

Inspect the system — CPU only, no API key:

```bash
uv run adapterops manifest show     # served adapter revisions, frozen eval splits, gate state
uv run adapterops verify-pins       # has any pinned adapter moved on the Hugging Face Hub?
uv run adapterops economics         # cost per 1K requests and weight footprint -> runs/economics.json
uv run adapterops dashboard         # six-metric results dashboard -> runs/DASHBOARD.md
```

Run the demo:

```bash
uv run python demo/app.py           # live Gradio app; downloads the pinned base model and adapters
uv run python demo/build_static.py  # recorded demo page -> demo/_static/ (--push publishes it)
```

Train, serve and gate a release — on the GPU machine:

```bash
uv run adapterops train --task intent --hub-repo <user>/adapterops-intent
bash scripts/phase2_serve.sh        # vLLM: the base model plus all four pinned adapters
uv run adapterops regress --base-url http://localhost:8000 --name candidate \
    --baseline runs/regression__v1-baseline-1.json
uv run adapterops manifest promote --note "what changed" --regression runs/regression__candidate.json
uv run adapterops manifest rollback --to 2
```

`manifest promote` refuses a candidate whose drop on any enforced task exceeds its gate threshold;
`manifest rollback` restores an earlier manifest version. See the comments in
`scripts/phase2_serve.sh` for setting up the serving environment.

### Running Tests

```bash
uv run pytest                       # full suite, CPU only, no API key or network
uv run ruff check src tests demo    # lint
```

One test reproduces the distilled judge's calibration scores and needs the locally trained judge in
`checkpoints/`, which is not in git; it skips when the checkpoint is absent.

## Tech Stack

| Layer | Technologies |
|---|---|
| Language and tooling | Python 3.12, uv, pytest, Ruff |
| Base model and fine-tuning | Qwen2.5-1.5B-Instruct · QLoRA (4-bit NF4) with Hugging Face Transformers, PEFT, bitsandbytes and Accelerate |
| Serving | vLLM with multi-LoRA — four adapters on one base model, one GPU |
| Router and judge | DeBERTa-v3 (small for the router, base for the judge) · scikit-learn baselines |
| Frontier models | OpenAI API — GPT-4o-mini as the escalation target, GPT-4o for judge labels |
| Data | pandas and Parquet, Hugging Face Datasets, spaCy for the PII baseline |
| Registry and hosting | Hugging Face Hub — pinned adapter revisions, model cards and a static Space |
| Demo | Gradio for the live app · static HTML and JavaScript for the public page |
| Compute | Colab and Kaggle T4 GPUs for training · a rented A10 for training and serving |

**Data licences:** Banking77 via `mteb/banking77` (MIT) · `Tobi-Bueck/customer-support-tickets` by Tobi Bueck (CC BY-NC 4.0 — the urgency adapter is non-commercial) · `ai4privacy/pii-masking-openpii-1m` (CC BY 4.0) · `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (CDLA-Sharing-1.0); sources in `data/MANIFEST.json`.
