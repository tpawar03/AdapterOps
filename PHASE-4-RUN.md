# Phase 4–5 GPU session — everything left that needs a GPU, in one bill

Every item below needs adapter inference or QLoRA training, and each one alone would be a
separate rental. Bundled, it is one session of roughly an hour.

| # | Deliverable | Milestone | Output |
|---|---|---|---|
| 1 | Shuffled-label intent adapter, trained and pushed | F21 → M7 | `runs/intent-shuffled__train.json`, Hub `Tanny03/adapterops-intent-shuffled` |
| 2 | Two baseline regression runs on the unchanged manifest | F33 | `runs/regression__v1-baseline-{1,2}.json` + predictions |
| 3 | Golden PII re-scored at 384 tokens | the 0.9190 caveat | inside baseline-1 |
| 4 | Prompted baselines for urgency, PII, drafting and intent per-class | M2 | `runs/<task>__prompted-*.json` |
| 5 | All four under-trained checkpoints scored on both splits | M11 | `runs/regression__all-m11.json` |
| 6 | The shuffled adapter scored against baseline | M7 detect | `runs/regression__intent-shuffled.json` |

**Budget: $1.50.** An A10-class card at about $0.75/hr, with roughly 15 minutes of training,
three model loads, four regression runs and four prompted baselines. **Get it approved before
renting** — the spend rule applies — and terminate the instance the moment the last run prints.

---

## Before renting — on the laptop

```bash
uv run adapterops verify-pins
uv run adapterops verify-pins --pins manifests/candidates/all-m11.json
git push origin main
```

Every line must read `ok`. The session serves what these files pin; a pin that has moved means
a score gets attributed to the wrong weights, which has already happened once (changelog 29).

---

## On the box

The box cannot push to GitHub — results come back by `scp` at the end. Install into a venv,
never `--user` (GATE-0.5.md). Training needs a Hugging Face token with **write** scope, for
pushing the shuffled adapter.

```bash
git clone https://github.com/tpawar03/AdapterOps && cd AdapterOps
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install vllm pandas pyarrow requests huggingface_hub python-dotenv scikit-learn \
            peft bitsandbytes datasets accelerate sentencepiece
export PYTHONPATH=src HF_TOKEN=<write-scoped token>
```

### 1 · Train the shuffled-label adapter (no server running — they share the GPU)

```bash
python -m adapterops.train.cli_train --task intent --train-split train_shuffled \
    --variant shuffled --hub-repo Tanny03/adapterops-intent-shuffled
python -m adapterops.cli pin-candidate --task intent \
    --repo Tanny03/adapterops-intent-shuffled --name intent-shuffled
```

`--variant` is not optional here: without it the run would overwrite the real intent adapter's
checkpoint, its M11 checkpoint and `runs/intent__train.json`, and the CLI refuses.

### 2–4 · Serve the real manifest, with a context long enough for M2

Shell 1:

```bash
MAX_MODEL_LEN=4096 ./scripts/phase2_serve.sh
```

The Phase 2 server ran at 1,536 tokens. Three of the prompted baselines do not fit in that:
urgency needs 1,639, PII 1,850, and intent per-class 2,195.

Shell 2:

```bash
python -m adapterops.cli regress --name v1-baseline-1 --save-predictions
python -m adapterops.cli regress --name v1-baseline-2 --save-predictions
python -m adapterops.cli prompted --task urgency --max-model-len 4096
python -m adapterops.cli prompted --task pii --max-model-len 4096
python -m adapterops.cli prompted --task drafting --max-model-len 4096
python -m adapterops.cli prompted --task intent --recipe per-class --max-model-len 4096
```

`--save-predictions` keeps every reply. Drafting is scored by the distilled judge on the laptop
afterwards; without the replies, that would mean renting the GPU again to regenerate them.

### 5 · Serve all four under-trained checkpoints at once (M11)

Stop shell 1, then:

```bash
PINS=manifests/candidates/all-m11.json MAX_MODEL_LEN=4096 ./scripts/phase2_serve.sh
```

```bash
python -m adapterops.cli regress --name all-m11 \
    --baseline runs/regression__v1-baseline-1.json --save-predictions
```

One server for four checkpoints, because the regression run scores each task independently.

### 6 · Serve the shuffled adapter (M7 detect)

Stop shell 1, then:

```bash
PINS=manifests/candidates/intent-shuffled.json MAX_MODEL_LEN=4096 ./scripts/phase2_serve.sh
```

```bash
python -m adapterops.cli regress --name intent-shuffled \
    --baseline runs/regression__v1-baseline-1.json
```

### Pass / fail, fixed before the numbers

| # | Criterion | Pass |
|---|---|---|
| 1 | Shuffled adapter still emits valid labels | intent `exact_label_rate` high — it learned the format, not the mapping |
| 2 | Shuffled adapter is caught on the random split | intent drop far above any plausible threshold (near the 1/77 floor) |
| 3 | Baseline runs agree | inference spread small; if not, that is itself the F33 finding |
| 4 | Every prompted baseline fits | no run refuses on the context check |
| 5 | M11 has a verdict per task | one of: hard flags / random passes · both flag · neither flags (§11) |

---

## Afterwards — copy results back, then terminate

From the laptop:

```bash
BOX=ubuntu@<ip>; KEY=~/Downloads/AdapterOps.pem
scp -i $KEY "$BOX:AdapterOps/runs/regression__*" runs/
scp -i $KEY "$BOX:AdapterOps/runs/*__prompted-*.json" runs/
scp -i $KEY "$BOX:AdapterOps/runs/intent-shuffled__train.json" runs/
scp -i $KEY "$BOX:AdapterOps/manifests/candidates/intent-shuffled.json" manifests/candidates/
```

Verify the files, then **terminate the instance.**

---

## On the laptop — no GPU

```bash
uv run adapterops derive-thresholds \
    --run-a runs/regression__v1-baseline-1.json --run-b runs/regression__v1-baseline-2.json
```

M7, detect → block — the gate must refuse the shuffled candidate:

```bash
uv run adapterops manifest promote --pins manifests/candidates/intent-shuffled.json \
    --regression runs/regression__intent-shuffled.json --note "F21 shuffled-label candidate"
```

M7, rollback — force the bad release through, then undo it:

```bash
uv run adapterops manifest promote --pins manifests/candidates/intent-shuffled.json \
    --regression runs/regression__intent-shuffled.json --note "forced, to prove rollback" --force
uv run adapterops manifest rollback
```

`--force` records what it overrode in the manifest, so the forced release is visible in the
history rather than indistinguishable from a legitimate one.
