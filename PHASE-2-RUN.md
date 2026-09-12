# Phase 2 GPU session — one box, three deliverables

**The question this session answers:** what does each adapter actually get right, on 5,800
pairs it has never seen? Everything downstream of Phase 2 — the router, the operating
curve, the hard-cases split, M4 — is waiting on that one table.

It is bundled with two things Phase 1 still owes, because the marginal cost of both is zero
once the box is up:

| # | Deliverable | Output |
|---|---|---|
| 1 | **M1** — four adapters served concurrently on one base | `runs/m1_serving.json` |
| 2 | Per-adapter P50/P95 under interleaved load (PRD §7) | same file |
| 3 | The scored router pool, with log-probabilities for F9 | `data/router/scored.parquet` |

**Budget: $1.00 approved.** An A10-class card at ~$0.75/hr, projected 30–40 minutes
including model load and adapter download. Kill the instance the moment it prints the
summary — an idle box is the likeliest way to lose the ceiling (BUILD-PLAN standing
rituals).

---

## Before you rent anything

```bash
uv run adapterops verify-pins
```

Every line must read `ok`. The session serves whatever `manifests/adapters.json` pins, and
a pin that no longer resolves to the weights it was pinned to means any score this run
produces is attributed to the wrong adapter — which is the changelog-29 incident, and it
has already happened once on the PII repo.

---

## On the box

Lambda Stack and most ML images ship system packages built against numpy 1.x, and vLLM
pulls numpy 2. Install into a venv, never `--user`; a venv does not inherit
`/usr/lib/python3/dist-packages`, so the ABI mismatch cannot arise (GATE-0.5.md).

```bash
git clone https://github.com/tpawar03/AdapterOps && cd AdapterOps
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install vllm pandas pyarrow requests huggingface_hub python-dotenv scikit-learn
```

**Shell 1 — serve.** Downloads each pinned adapter at its revision, then blocks:

```bash
./scripts/phase2_serve.sh
```

**Shell 2 — smoke test first.** 100 pairs, under a minute, and it runs the distinctness
probe:

```bash
source ~/venv/bin/activate && cd AdapterOps && export PYTHONPATH=src
python -m adapterops.router.generate --limit 25
```

Then the full run:

```bash
python -m adapterops.router.generate --concurrency 16
```

---

## Pass / fail, decided before the numbers arrive

| # | Criterion | Pass |
|---|---|---|
| 1 | All four adapters appear in `/v1/models` | four names |
| 2 | **The four are actually distinct** | the probe aborts the run if any two agree on every prompt |
| 3 | No errors under interleaved concurrency | `fallback_rate` 0.0000 |
| 4 | Quality survives the serving path | intent success rate within ~0.02 of the offline 0.9312 |
| 5 | Latency usable | P95 under ~2 s per adapter at concurrency 16 |
| 6 | Drafting is not being truncated | `truncated` near zero; 400 new tokens covers the p90 reference |

**Criterion 2 is the one that quietly fools you**, and it is the same trap Gate 0.5 caught.
If vLLM resolves several adapter names to one set of weights, throughput, latency and error
rate all look healthy and every number downstream is meaningless. Gate 0.5's discriminator
was two adapters of different quality disagreeing. Here it is sharper: the four were tuned
on four different output formats, so one prompt sent to all four names must produce four
different answers. The probe runs first and aborts before spending anything.

**Criterion 6 has a real failure mode.** Drafting references reach 2,147 characters. A
truncated reply scores as a quality failure under the token-F1 proxy, and that failure
would be indistinguishable from a bad adapter. If `truncated` is material, raise
`MAX_TOKENS["drafting"]` and re-run drafting alone rather than reinterpreting the number.

---

## Afterwards

```bash
git add runs/m1_serving.json data/router/scored.parquet && git commit && git push
```

Then **terminate the instance**, and back on the laptop:

```bash
uv run adapterops router-dataset     # train/eval/shift + hard-case mine, exclusions verified
uv run adapterops router-train       # F10, DeBERTa-v3-small
uv run adapterops router-report      # F11/M3, the operating curve
```

`router-report` also needs the frontier arm finished — it stopped at 3,499 of 5,800 pairs
on a daily request quota. Resume it with the router slice first, since that is what the
curve is computed over:

```bash
uv run adapterops frontier --purpose router
uv run adapterops frontier
```

Nothing already cached is re-billed.
