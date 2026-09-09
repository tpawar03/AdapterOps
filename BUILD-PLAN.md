# Build plan — AdapterOps

Execution layer for `adapter-service-prd.md` (v2.3). The PRD is the spec: what gets
built and why, with §/F/M identifiers. This file is the order of operations: what to
do on which day, how long to give it, and the points at which you stop and reassess
rather than push on.

**Budget:** 180 hours · 9 weeks · 20 hrs/week · $50 ceiling.
**Critical path:** day-1 verification → vLLM multi-LoRA proven → everything else.

Nothing in weeks 2–9 is worth starting until Phase 0 answers one question: *can this
machine train, serve and score an adapter end to end on compute you can afford?*

---

## Day 1 — verification before any code (3 h)

Eight facts the PRD flags as unverified. All are cheap to check and any of them can
reshape the project. Do these before writing a line.

| # | Check | Where | If it fails |
|---|---|---|---|
| 1 | Qwen2.5-1.5B-Instruct license permits portfolio publication | HF model card | Swap to Llama-3.2-1B or Phi-3.5-mini; §12 lists both with their tradeoffs |
| 2 | `PolyAI/banking77` license | HF dataset card | Intent adapter needs a new source; everything downstream of 77 classes changes |
| 3 | Banking77 test split really holds ~3,080 rows | `load_dataset` + `len()` | The 770-example intent golden set (§11) needs resizing |
| 4 | Kaggle ticket-priority dataset still available + license | Kaggle page | Urgency adapter needs a new source |
| 5 | ai4privacy variant permits this use — the 1m and 200k variants differ | HF dataset card | Use the permissive variant, or cut the PII adapter to three adapters |
| 6 | Bitext CDLA-Sharing 1.0 attribution requirements | HF dataset card | Add required attribution to README now, not at the end |
| 7 | RunPod vs Vast.ai A10G spot price and availability today | Both dashboards | Whichever is cheaper; record the rate in the cost log |
| 8 | OpenAI API key works, billing cap set to $15 | Platform settings | Hard cap protects the $50 ceiling from a runaway loop |

**Then, same day:** mirror all four datasets locally and commit them (§9 — this is a
Phase 0 exit criterion and guards against the takedown risk in §14). Create
`COST-LOG.md` with the first entry.

> **Gate 0.** If two or more of checks 1–6 fail, stop and re-scope before building.
> Adapters are cheap to swap on paper and expensive to swap in week 4.

---

## Repo scaffold — done

```
AdapterOps/
├── pyproject.toml                  uv project; base deps + marker-gated `gpu` extra
├── uv.lock                         committed — reproducibility (STATUS.md D15)
├── .python-version                 3.12
├── README.md
├── adapter-service-prd.md          spec (synced — see adapter-service-prd-build/README.md)
├── adapter-service-prd-build/      PRD build tooling
├── BUILD-PLAN.md                   this file
├── STATUS.md                       status, decision log, interview material
├── COST-LOG.md                     every session, every charge (F24, P0)
├── data/                           mirrored dataset snapshots, committed
├── evals/
│   ├── golden/                     random splits: intent 770, others 300
│   └── hard/                       mined split, from Phase 4
├── manifests/                      versioned system manifests (F16)
├── runs/                           committed regression-run results
├── src/adapterops/
│   ├── cli.py                      `uv run adapterops ...`
│   ├── train/                      QLoRA training, one entrypoint, task as arg
│   ├── eval/                       harness: adapter vs prompted vs frontier
│   ├── serve/                      vLLM multi-LoRA launcher + FastAPI
│   ├── router/
│   ├── judge/
│   └── manifest/                   load, pin, promote, roll back
├── tests/
├── notebooks/                      Colab training notebooks
└── demo/                           Gradio app
```

```bash
uv sync                  # base — macOS and Linux
uv sync --extra gpu      # rented CUDA box: adds vLLM + bitsandbytes
uv run adapterops --help
```

Write `src/eval/` to take a split path as an argument from the very first version.
The hard-cases split in Phase 4 is then a second call, not a refactor.

---

## Phase 0 — thin slice (30 h · weeks 1–1.5)

**Goal: retire the one risk that can invalidate the headline claim.** Not to build
well — to find out if the architecture is possible at this budget.

| Task | h |
|---|---|
| Day-1 verification + dataset mirroring + repo scaffold | 4 |
| Banking77 → training format; 70/15/15 with the 770 golden set held out | 3 |
| QLoRA intent adapter on Colab T4 — get *one* run finishing, quality irrelevant | 6 |
| Push adapter to HF Hub as a revision, with a model card | 2 |
| Rent GPU; stand up vLLM; load the single adapter; hit it | 5 |
| **Load a second adapter alongside it** — even a junk copy — to prove multi-LoRA | 3 |
| Eval harness v1: adapter vs. prompted baseline on the golden set | 5 |
| P95 latency + cost measured on the rented box, written down | 2 |

> **Gate 0.5 — the project's real decision point.** Does vLLM serve two adapters
> concurrently on affordable hardware?
> - **Yes** → continue as specified.
> - **No** → take the §14 fallback *now*: sequential adapter loading, and rewrite the
>   headline as "4 adapters, one base model, one registry" before building three more.
>   Do not spend weeks 2–9 hoping it resolves itself.

Budget ≤ $8 of GPU here. Kill the instance between sessions — spot pricing is per
second and an idle box is the most likely way to lose the $50.

---

## Phase 1 — remaining adapters (30 h · weeks 1.5–3)

Mechanical once Phase 0 works. Same pipeline, three more datasets.

| Task | h |
|---|---|
| Urgency adapter: prep, train, eval | 6 |
| PII adapter: prep, train, eval | 6 |
| Drafting adapter: prep, train, eval (generative — scoring is cruder until Phase 3) | 7 |
| **Save an early checkpoint from one run** — this is the M11 subtle regression, and it costs nothing now and a full retrain later | 0.5 |
| All four served concurrently; smoke test hitting each (M1) | 3 |
| Latency + cost benchmark on rented GPU, all four | 3 |
| Freeze and commit the golden sets; scorecards written | 4 |

Two things to get right here because they are expensive to fix later:

- **Freeze the golden sets and never touch them again.** A split that drifts makes
  every later comparison meaningless.
- **Measure baseline variance.** Run the same eval twice against the same adapter and
  record the spread. That number sets the regression thresholds (§15, open question 2)
  and there is no other way to get it.

> **Gate 1.** If adapters lose to the prompted baseline on both quality *and* cost,
> invoke the §14 "whole approach underwhelms" fallback: reframe around Phase 4/5 and
> keep going. M7 tests the pipeline, not the models — it survives bad adapters.

---

## Phase 2 — router (40 h · weeks 3–5)

The largest phase, and the one where the discipline matters most.

| Task | h |
|---|---|
| Ticket pool → run every (ticket, task) pair through the adapters | 5 |
| Harness scores each pair; **persist failure records tagged by component** (F29) | 4 |
| Reserve hard-case candidates and **exclude them from router training** (F31) | 3 |
| Train DeBERTa-v3-small with task as an input feature (F10) | 6 |
| Baselines: always-cheap, always-frontier (F8) | 3 |
| **Confidence baseline (F9)** — the one that matters | 4 |
| Operating curve across ≥5 thresholds, all policies on one chart (F11, M3) | 6 |
| Within-task shift set: TF-IDF cluster + length deciles (F36) | 5 |
| Shift evaluation, reported per task (M4) | 4 |

**Order matters:** build the confidence baseline *before* you train the router. If the
free signal is already good, you learn it while the learned router is still cheap to
descope — and you avoid the motivated reasoning that comes from having already spent
six hours training the thing you're comparing against.

The F29/F31 exclusion is the single most important line of code in this phase. Get it
wrong and every router number is inflated and you will not notice.

> **Gate 2.** If the learned router does not beat confidence: that is the reported
> finding (§14, N1 is nice-to-have). Do not tune until it wins — the honest negative
> is the stronger interview answer and the PRD already commits to it.

---

## Phase 3 — judge (20 h · weeks 5–6) — *first phase to cut*

| Task | h |
|---|---|
| ~1,200 GPT-4o judgments on drafting outputs (F13) | 4 |
| Pick DeBERTa-v3-base vs Qwen2.5-0.5B on training speed (§15) | 2 |
| Train distilled judge (F14) | 6 |
| Calibration: Spearman/Pearson + ±1 agreement on 150 held out (F15, M5) | 4 |
| Wire the judge into drafting-task scoring | 4 |

Report the correlation whatever it is. §10 already concedes the cost argument doesn't
hold at this scale — the pattern and the number are the deliverable.

If you are behind by the end of Phase 2, **cut this phase entirely.** It removes one
headline claim and nothing structural.

---

## Phase 4 — manifest, splits, dashboard (40 h · weeks 6–8)

| Task | h |
|---|---|
| Manifest format; pins adapters + router + judge + **eval-split versions** (F16) | 5 |
| Manifest drives what is served; edit it, serving changes (M6) | 4 |
| Mine hard cases from Phase 2 records, bucketed by failure type (F31) | 5 |
| **Frontier adjudication screen; report quarantine rate per task** (F30) | 5 |
| Regression script: both splits, on demand, results committed to `runs/` (F18) | 6 |
| 6-metric dashboard, two quality rows never blended (F17, F32) | 6 |
| **Two baseline runs against an unchanged manifest** to measure hard-split variance | 3 |
| Derive and commit the gating threshold from that variance (F33) | 2 |
| Promotion blocking (F19) + rollback (F20) | 4 |

The two baseline runs are not optional padding. Without them the hard-cases gate has
no defensible threshold, and §11 explicitly forbids inventing one.

> **Gate 4.** If hard-split variance is wider than a regression worth catching, the
> split stays **report-only** and you say so in the README. That is a legitimate
> outcome, not a failure — see §14.

---

## Phase 5 — failure demos + polish (20 h · week 9) — *never cut*

| Task | h |
|---|---|
| Shuffled-label regressed adapter (F21) | 2 |
| **Record detect → block → rollback end to end (M7)** | 4 |
| M11: score the Phase 1 under-trained checkpoint on both splits (F34) | 3 |
| Gradio demo (F22) | 5 |
| README with architecture, results, and the negative findings (F23) | 5 |
| Deploy demo; verify the URL from a clean browser (M8) | 1 |

M7 is the project's distinguishing claim and M11 is what turns the hard-cases split
from an assertion into a measurement. If the schedule collapses, this phase is what
survives.

---

## Standing rituals

- **Cost log every session** (F24, P0). Date, provider, duration, charge, running
  total. Stop and reassess at $25.
- **Kill the GPU when you stop.** Most likely way to blow the budget.
- **Commit after every working step.** The repo is the evidence; a reviewer reading
  the history is reading the process.
- **Write negative results down as they happen**, not from memory in week 9.

## Cut order, if behind

`F27/F28 (P2)` → `F37 misroute diagnostic` → `Phase 3 judge` → `F26 rules baseline` →
`M11 subtle regression`. **Phase 5 and the two-split evaluation are never cut.**

## Where the hours go

| Phase | Hours | Share |
|---|---|---|
| 0 — thin slice | 30 | 17% |
| 1 — adapters | 30 | 17% |
| 2 — router | 40 | 22% |
| 3 — judge | 20 | 11% |
| 4 — manifest + splits | 40 | 22% |
| 5 — failure demos | 20 | 11% |
| **Total** | **180** | |

Evaluation and release machinery (Phases 2, 4, 5) take 100 of 180 hours. Training the
four adapters takes 60. That ratio is the project's actual argument.
