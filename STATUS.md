# Project status & decision log — AdapterOps

Running record for the Multi-Task Adapter Service. Three jobs: track where the project
actually is, capture why each non-obvious choice was made while the reasoning is still
fresh, and keep interview-ready material from having to be reconstructed from memory in
six months.

**Rule for this file: nothing goes in that isn't true yet.** Results sections stay empty
until measured. A status file that quietly drifts into aspiration is worse than none —
and the whole project is an argument about honest evaluation.

**This file carries no dates.** Position in the build is tracked by phase, which is the
unit that actually means something here; a date stamp only records when the file was last
touched, which is not the same as whether it is current. Sequence lives in §2's D-numbers
and in the phase column of §4.

---

## 1. Status at a glance

| | |
|---|---|
| **Phase** | Phase 0 — repo + Python environment scaffolded; verification not begun |
| **Spec** | PRD v2.3, published + in repo |
| **Hours logged** | 0 / 180 |
| **Spend** | $0.00 / $50.00 |
| **Repo** | `~/Desktop/AdapterOps`, git `main`, **no commits yet**. uv project `adapterops`, Python 3.12.13, 244 packages locked, base env installs clean on macOS. |
| **Blocking** | Day-1 verification (BUILD-PLAN.md) — 8 unverified facts, any can re-scope the project |
| **Done so far** | Repo scaffold, `pyproject.toml` with base/gpu split, `uv.lock`, package skeleton, `COST-LOG.md`, ruff clean |
| **Next action** | Verify licenses + Banking77 test-split size, mirror datasets, create `COST-LOG.md` |

### Milestone tracker

| ID | Milestone | State |
|---|---|---|
| M1 | 4 adapters served concurrently | not started |
| M2 | Adapters vs prompted baseline | not started |
| M3 | Router operating curve vs 3 baselines | not started |
| M4 | Within-task distribution shift | not started |
| M5 | Judge calibration reported | not started |
| M6 | Manifest drives serving | not started |
| M7 | **detect → block → rollback proven** | not started |
| M8 | Live demo + public repo | not started |
| M9 | Spend ≤ $50 | on track ($0) |
| M10 | Hard-cases split mined + adjudicated | not started |
| M11 | **Gate sensitivity measured** | not started |

---

## 2. Decision log

Each entry: what was decided, what was rejected, why, and the interview answer it
supports. Entries D1–D5 come from the v1→v2 scope pass; D6–D13 from the v2.2 pass that
found real defects in v2.1's own design.

---

### D1 · Six adapters cut to four
**Rejected:** keeping sentiment and topic classifiers.
**Why:** adapters 5 and 6 were the same code path with a different CSV. No claim in the
project depended on 6 over 4, and sentiment's labels were derived rather than
purpose-built. Saved ~1.5 weeks.
**Interview angle:** scope discipline — distinguishing work that adds evidence from work
that adds volume. "Six" is a bigger number that proves nothing extra.

---

### D2 · Router operates on (ticket, task) pairs, not tickets
**Rejected:** one routing decision per ticket.
**Why:** with four adapters per ticket, a per-ticket success label is undefined when
adapter A succeeds and adapter B fails. The original spec never said which it meant.
**Interview angle:** finding an ill-defined label before it becomes an unexplainable
training result. Most routing questions expect "route the request" — the follow-up
"route *what*, exactly?" is where the real design sits.

---

### D3 · Confidence-based routing is a P0 baseline
**Rejected:** comparing the learned router only against always-cheap / always-frontier.
**Why:** the adapter's own softmax confidence is nearly free and is the strongest
realistic alternative. Omitting it would have made the learned router look good against
two strawmen.
**Interview angle:** the single best "how do you evaluate?" answer in the project.
Choosing the baseline most likely to beat you, on purpose. The PRD also states up front
that losing to it is a publishable finding, not a failure — that commitment was made
*before* seeing results, which is what makes it credible.

---

### D4 · On-demand regression runs, not nightly CI
**Rejected:** the original "nightly GitHub Actions regression" design.
**Why:** GitHub Actions free runners are CPU-only and cannot run GPU inference against
the served adapters. The design was not buildable on free infrastructure.
**Trade-off accepted:** no unattended monitoring. Documented in the README rather than
quietly dropped.
**Interview angle:** reading an infrastructure constraint correctly instead of
discovering it in week 6. Also a clean answer to "what would you do with more budget?"

---

### D5 · Demo and benchmark are separate artifacts
**Rejected:** one hosted app serving as both the demo and the source of benchmark numbers.
**Why:** vLLM multi-LoRA on a free HF Space is a likely blocker, not an open question.
Benchmarks run on rented GPU with recorded evidence; the demo runs a lighter path.
**Interview angle:** knowing which numbers are reportable. Latency measured on a shared,
throttled free tier is not a latency measurement.

---

### D6 · Two eval splits, never blended
**Rejected:** a single random golden set.
**Why:** a model can improve on average while getting worse on the cases that matter. One
blended number cannot show that; two numbers can.
**Interview angle:** the strongest evaluation-design story here. Also the setup for D7
and D8, which are what make it actually valid.

---

### D7 · Hard-cases items excluded from router training — *the leak*
**Rejected:** v2.1's design, which fed the same failure records into both the router's
training set and the hard-cases eval split.
**Why:** the router would have trained on its own eval items. Every router number
measured through that split would have been inflated, and nothing in the pipeline would
have flagged it.
**Interview angle:** the best "tell me about a bug you caught in your own design" answer
in the project — a data leak found by reading the data flow, before any code existed.
Worth being able to draw: the exclusion is an explicit edge in the architecture diagram
precisely because it is the kind of thing that gets silently dropped in implementation.

---

### D8 · Frontier adjudication screen before an item enters the hard split
**Rejected:** gating on unscreened mined failures.
**Why:** "examples the model got wrong" is not a neutral sample of hard examples — it is
systematically enriched for rows whose *gold label is wrong*, because a competent model
disagrees with bad labels. Banking77 has genuinely overlapping intent classes; crowd
-labeled priority data carries annotator disagreement. Gating on that set means blocking
releases for failing to reproduce annotation errors.
**Mitigation:** items where the frontier model also disputes the label are quarantined,
not gated. The quarantine rate is reported per task.
**Interview angle:** strong signal about understanding label noise as a first-class
problem. The by-product — a measured label-noise rate for two public datasets — is more
interesting than most of the modelling. Under $1 of API to produce.

---

### D9 · Hard-split gate: report for two runs, then threshold from observed variance
**Rejected:** v2.1's "any drop blocks promotion".
**Why:** a ~100-example split whose items sit at the decision boundary will swing several
points between identical runs. Zero-tolerance on that set recreates exactly the
unfalsifiable gate that raising the golden set from 100 to 300 was meant to eliminate.
The two versions also contradicted each other — §11 said it gates, §15 listed it as open.
**Interview angle:** a threshold you can defend because you measured the noise floor
first. Directly answers "how did you pick that number?" — the question most eval answers
fail on.

---

### D10 · Golden sets sized per task; intent raised to 770 on micro-accuracy
**Rejected:** a blanket 300 for every task.
**Why:** Banking77 has **77 classes**. 300 examples is under 4 per class — macro-F1 there
is dominated by which four examples were drawn, and per-class F1 is not reportable at
all. Correct for urgency (few classes), PII (binary) and drafting (1–5 score); wrong for
intent. Banking77's test split has ~3,080 rows, so 770 (≈10/class) is nearly free.
**Interview angle:** knowing which metric a dataset's shape will support. This survived
two revisions unnoticed, which is itself the point: uniform-looking numbers hide
task-specific invalidity.

---

### D11 · Cross-source shift test redesigned — the original was confounded
**Rejected:** "train the router on Kaggle tickets, test on Banking77 queries."
**Why:** task identity is a router *input feature*, and each task is bound to one source
— intent labels exist only in Banking77, urgency only in the Kaggle set. Splitting by
source also splits by task, so the test measures behaviour on an unseen task value. It
would have failed for a reason unrelated to distribution shift, and the failure would
have been reported as a shift result.
**Replacement:** hold task constant; TF-IDF-cluster each task's own pool, hold out one
cluster plus the top/bottom length deciles.
**Interview angle:** spotting a confound in your own experiment design. A large,
meaningless number is more dangerous than no number.

---

### D12 · Two injected regressions, not one
**Rejected:** proving the gate with only a shuffled-label adapter.
**Why:** a grossly broken model is caught by the random set alone, so it proves the
rollback machinery works (M7) but says nothing about whether the hard split earns its
place. A second, *subtle* regression — an early checkpoint from a run that happens anyway
— measures that. All three outcomes are publishable, including "the hard split adds
nothing here."
**Interview angle:** testing your test. Also the epistemics: **running** it is a must-hit,
**passing** it is not, because passing isn't under the builder's control.

---

### D13 · Judge kept although its stated cost rationale doesn't hold
**Rejected:** claiming distillation saves money here.
**Why:** at ~2,000 eval items per run, calling the frontier judge directly costs cents.
The industry argument is about volume this project doesn't have. The judge is retained to
demonstrate the pattern and produce a calibration number — and the writeup says exactly
that.
**Trade-off:** it is also the first phase to cut if the schedule slips.
**Interview angle:** naming where a technique is *not* justified. Interviewers hear
"I distilled an evaluator to cut costs" constantly; "I distilled one, and at my volume
the cost argument was false, so here's the real reason" is the differentiated answer.

---

### D14 · Generated HTML gitignored; vendored assets committed
**Rejected:** committing the 3.6 MB generated file, or gitignoring `vendor/`.
**Why:** the generated file would add a fresh multi-megabyte blob to history on every PRD
edit; the vendored library and fonts change only on a version bump and are what make the
build reproducible offline. Asymmetric costs, asymmetric treatment.
**Interview angle:** minor, but a clean example of reasoning about repo hygiene from
change frequency rather than from file size alone.

---

### D15 · uv with a packaged `src/` layout and a committed lockfile
**Rejected:** `requirements.txt` + a hand-managed venv; a flat `src/train/`, `src/eval/`
script layout; gitignoring `uv.lock`.
**Why:** this project has **three** environments, not one — macOS arm64 for development,
eval, router, judge and the demo; a rented CUDA box for vLLM serving and benchmarks; and
Colab's own preinstalled stack for training. A single flat requirements file breaks on
the Mac the moment vLLM appears in it. The CUDA-only packages therefore sit in a
marker-gated extra (`sys_platform == 'linux'`), which keeps `uv lock` resolvable
everywhere while the GPU box gets the real stack via `uv sync --extra gpu`.

The lockfile is committed for the same reason `vendor/` is committed and §9 mirrors the
datasets: reproducibility against upstream drift.

**Trade-off:** a packaged layout means the code must be installed (`uv sync`) before it
runs, rather than being loose scripts. Worth it — no `sys.path` juggling, and
`adapterops.eval` is importable from a Colab notebook the same way it is locally.

**Interview angle:** ties the environment story to the project's thesis. A regression
gate that cannot reproduce its own dependency set cannot attribute a regression to the
model — the dependency could have moved instead. Pinning is part of the rollback claim,
not housekeeping adjacent to it.

---

## 3. Trade-offs consciously accepted

| Trade-off | Chosen | Cost of the choice |
|---|---|---|
| Unattended monitoring vs $50 ceiling | On-demand runs | No continuous monitoring story; must be explained, not hidden |
| Gate sensitivity vs false alarms | Threshold from measured variance | Gate may prove too noisy to use; then it is report-only |
| Adapter count vs depth per adapter | 4 adapters, deeper evaluation | Less impressive headline number |
| Judge distillation vs direct API calls | Distil anyway, state the real reason | Spends a week on a component whose stated rationale is void at this scale |
| Frontier parity vs cost/latency | Documented tradeoff, not parity | Cannot claim "matches GPT-4o"; must argue the curve instead |
| Repo size vs offline reproducibility | Commit 3.4 MB of vendored assets | Heavier clone |
| Real data realism vs privacy | Synthetic PII spans only | No real-world messiness in the PII task |

---

## 4. Observations & findings

Filled in as results arrive. **Empty is the correct state today.**

### Measurements pending
| What | Value | Recorded |
|---|---|---|
| vLLM multi-LoRA works on rented A10G? | — | Phase 0 |
| Adapter vs prompted baseline, per task | — | Phase 1 |
| P95 latency, per adapter, rented GPU | — | Phase 1 |
| Golden-set run-to-run variance | — | Phase 1 |
| Learned router vs confidence baseline | — | Phase 2 |
| Within-task shift degradation | — | Phase 2 |
| Judge correlation + ±1 agreement | — | Phase 3 |
| **Label-noise quarantine rate, per task** | — | Phase 4 |
| Hard-split run-to-run variance | — | Phase 4 |
| **M11: does the hard split catch what the random set misses?** | — | Phase 5 |

### Findings log
> Append entries as things are learned — especially the surprising and the negative.
> Order by phase, not by date. Format: **phase · what happened · what it means ·
> whether it changes the plan.**

*(none yet — Phase 0 not started)*

---

## 5. Interview material

### The stories, ranked by strength

1. **The data leak I caught before writing code** (D7). Failure records were feeding both
   router training and the eval split. Found by tracing the data flow on paper.
2. **Why my hard-case eval set was measuring label noise** (D8). Failure-mined sets are
   enriched for mislabeled rows; the fix costs under a dollar and produces a label-quality
   measurement as a by-product.
3. **The baseline I added to make my own work look worse** (D3). Confidence routing is
   nearly free and might beat the learned router — committed in advance to reporting that.
4. **77 classes, 300 examples** (D10). Metric validity as a function of dataset shape.
5. **The confounded shift test** (D11). Task identity was collinear with data source.
6. **Testing the test** (D12). Two injected regressions; sensitivity measured, not assumed.

### Mapping to common questions

| Question | Use |
|---|---|
| "Tell me about a fine-tuning project" | The architecture, then pivot fast to evaluation — 100 of 180 hours went there |
| "How did you evaluate it?" | D6 two splits → D3 baseline choice → D9 threshold from measured variance |
| "Tell me about a bug you found in your own work" | D7, then D11 |
| "How do you know the model is good?" | D9 + D12 — the honest answer is "I measured the smallest regression I can detect" |
| "What would you do differently?" | D4 (needs an always-on GPU) and D13 (judge unjustified at this volume) |
| "Where did fine-tuning not help?" | Whatever Phase 1 shows — commit to answering from data, not from hope |
| "How do you handle disagreement about a design?" | The v2.1 → v2.2 pass: nine defects found in my own prior spec, each logged with its reason |

### Framing to avoid
- Don't lead with "I fine-tuned four adapters." Every candidate has a fine-tuning story;
  almost none has a rollback proof or a measured gate sensitivity.
- Don't claim the router beats the baseline until it does. The PRD commits to reporting
  the negative — that commitment is worth more than the win would be.
- Don't inflate the judge's cost rationale (D13). Naming where a technique isn't
  justified reads as senior; repeating the marketing line doesn't.

---

## 6. Maintaining this file

Updating this file is part of finishing a piece of work, not a chore batched at the end
of a phase. The value here is the rejected alternatives and the negative results, and
both are exactly what memory loses first — a decision log written a month late produces a
tidy narrative of choices that were never actually weighed.

| Trigger | Update |
|---|---|
| End of a working session | §1 — hours, spend, phase |
| A non-obvious choice made | §2 — next D-number, with the rejected alternative and why |
| A measurement produced | §4 — replace the `—` with the real number |
| A surprising or negative result | §4 findings log, tagged by phase |
| A milestone completed | §1 tracker — flip the M-state |
| Any spend | §1 **and** `COST-LOG.md` |

Three rules that make the difference between a log and a story:

- **No dates.** Next decision takes the next D-number; findings take a phase.
- **Never** move a §4 row from pending to a number you did not measure. Unknown stays `—`.
- **Record the negatives with the same care as the wins.** They are the strongest
  interview material in §5, and the ones most likely to go unwritten.
