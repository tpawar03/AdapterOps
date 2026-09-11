# Gate 0.5 — can vLLM serve many adapters from one base?

**The question:** does concurrent multi-LoRA serving work on affordable hardware?

**Why it is here and not in week 6:** the project's headline claim is *many adapters, one
base*. If vLLM cannot hold two adapters and answer requests against both at once on a
rented A10G, the claim is wrong and the PRD needs rewriting. BUILD-PLAN front-loads this
so a "no" costs one week rather than six.

**First real spend.** Everything before this cost $0.00.

---

## Pass / fail criteria

Decide these *before* seeing the numbers.

| # | Criterion | Pass |
|---|---|---|
| 1 | Both adapters load and appear in `/v1/models` | both names present |
| 2 | **Adapters are actually distinct** | non-zero disagreements between the two |
| 3 | Both answer under concurrent interleaved load | no errors at concurrency 16 |
| 4 | Trained adapter's quality survives the serving path | within ~0.02 of the offline 0.9312 |
| 5 | Latency is usable | P95 under ~2 s at concurrency 16 |
| 6 | Fits the budget | A10G-class card, ≲ $1/hr |

**Criterion 2 is the one that can quietly fool you.** If vLLM resolved both adapter names
to the same weights, every other number would look healthy and the gate would read as
passed. The probe sends the same prompts to the trained adapter and the M11 under-trained
one; they differ in quality, so they must disagree on some inputs. Zero disagreements is a
FAIL even if everything else is green.

Criterion 5's threshold is a judgement call, not a measurement — record the real P50/P95
whatever they are. A slow but working gate is still a pass on the *claim*; it just changes
what §7 can promise.

---

## Before renting anything

- [ ] Adapters are on the Hub: `Tanny03/adapterops-intent`, `Tanny03/adapterops-intent-undertrained-m11`
- [ ] Both repos are public, or `HF_TOKEN` is available on the box
- [ ] `evals/golden/intent.parquet` is committed (it is — comes with the clone)
- [ ] Offline reference number to beat: **0.9312** micro-accuracy
- [ ] Decide the spend ceiling for this session and write it here: **$____**

## On the box

Use a **Lambda Stack 24.04** image (Ubuntu 24.04 → Python 3.12). Ubuntu 22.04 ships
Python 3.10 and this project requires ≥ 3.12. Plain Ubuntu images have no CUDA or torch
and cost you paid minutes to build.

**Install into a venv, not `--user`.** Lambda Stack ships scipy, scikit-learn, pandas and
ml_dtypes compiled against numpy 1.x. vLLM pulls numpy 2, and each of those then dies on an
ABI mismatch in turn — four separate failures before the server starts. A venv does not
inherit `/usr/lib/python3/dist-packages`, so none of it applies.

```bash
git clone --depth 1 https://github.com/tpawar03/AdapterOps.git && cd AdapterOps
python3 --version              # expect 3.12.x
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install vllm pandas pyarrow requests
```

Terminal 2 must `source ~/venv/bin/activate` as well.

**Shell 1 — serve.** Blocks; wait for `Application startup complete`.

```bash
./scripts/gate05_serve.sh
```

**Shell 2 — probe.**

The package lives under `src/`, so it is not importable from the repo root without help.
`PYTHONPATH=src` is the one-line fix — no install step, nothing to go stale.

```bash
cd AdapterOps && PYTHONPATH=src python -m adapterops.serve.gate05_probe --n 200 --concurrency 16
```

Writes `runs/gate05.json`.

## Record, then tear down

- [ ] Commit `runs/gate05.json`
- [ ] Add the line to `COST-LOG.md`: provider, instance type, wall-clock, charge
- [ ] **Destroy the instance** — not stop, destroy. Check the dashboard shows no running
      resources. A forgotten GPU at $0.75/hr is $18/day and would eat a third of the
      $50 budget in silence.

---

## If it fails

Which criterion failed decides what happens next, and none of these are disasters — they
are findings, and they are the reason this gate exists.

| Failure | What it means | Next |
|---|---|---|
| 2 — adapters identical | vLLM is not routing per-request | Check `--max-loras` ≥ 2 and that each `--lora-modules` entry has a distinct name. If it truly cannot, the claim changes to *swapped* rather than *concurrent* adapters. |
| 3 — errors under load | Concurrency limit lower than hoped | Record the level it does sustain; that becomes the honest number. |
| 4 — quality drops | Serving path differs from the eval path | Suspect the prompt template or `max_lora_rank` < 16 silently truncating. |
| 5 — latency too high | Claim survives, §7's promise does not | Re-baseline §7 against the measured P95 and say so in the changelog. |
| 6 — needs a bigger card | Affordability claim weakens | Report the card it does need. "Works on an A100" is a different product than "works on an A10G". |

A failure here is worth writing up. A project that front-loaded its riskiest assumption and
reported the answer honestly is a better interview story than one that never tested it.
