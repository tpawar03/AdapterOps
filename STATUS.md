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
| **Phase** | **Phase 2 complete, and its results are negative and recorded.** M1 passed on real hardware; the router was trained and the operating curve measured. The Phase 4 manifest, promotion blocking and rollback were built ahead while waiting on paid runs. Phase 3 (judge) is next. |
| **Spec** | PRD v2.6, published + in repo |
| **Hours logged** | 0 / 180 |
| **Spend** | **$2.16** / $50.00 |
| **Repo** | [tpawar03/AdapterOps](https://github.com/tpawar03/AdapterOps) — **public**, local clone at `~/Desktop/AdapterOps`, `main` pushed and tracking. uv project `adapterops`, Python 3.12.13, base env installs clean on macOS. |
| **Blocking** | Nothing. Two open inputs, neither blocking: the **actual GPU charge** for the Phase 2 session (the cost log is missing it), and the frontier arm's last mining-slice pairs, running unattended against the daily request quota. |
| **Done so far** | All 8 day-1 checks · 4 datasets mirrored · all eval splits frozen · four adapters trained and published · intent **0.9312** (+0.4091 over prompted) · PII **0.9190** strict · urgency **loses to TF-IDF** · Gate 0.5 · **M1 PASS**, 5,800 requests and 0 errors · router pool scored · **525 hard cases mined** · router measured as **0.02–0.03 over a task-name lookup, across six runs** · escalation measured as a **quality loss** on every task · system manifest v1 with blocking and rollback |
| **Next action** | **Phase 3 judge.** GPT-4o grading approved (1,950 grades, $5 cap) and starting with a small sample; then F14 training on whichever base the §15 benchmark favours, and F15 calibration. |

### Milestone tracker

| ID | Milestone | State |
|---|---|---|
| M1 | 4 adapters served concurrently | **PASS** — 5,800 reqs, 0 errors, 23.6 rps |
| M2 | Adapters vs prompted baseline | intent ✓ (+0.4091). urgency and pii have **non-prompted** baselines only (TF-IDF, regex+NER) — those do not close M2 |
| M3 | Router operating curve vs 3 baselines | **measured — and it is a negative.** Escalation lowers quality on this benchmark |
| M4 | Within-task distribution shift | **no measurable degradation** — local 0.656 → 0.652, router AUC 0.707 → 0.716 · consistent with a router that barely reads the text |
| M5 | Judge calibration reported | not started |
| M6 | Manifest drives serving | manifest format + promote/rollback built · serving reads it after the GPU run |
| M7 | **detect → block → rollback proven** | **detect (F18) → block proven in code** on the frozen splits, serving mocked · rollback tested · remains: real serving + the F21 shuffled-label adapter (GPU) |
| M8 | Live demo + public repo | repo public ✓ · demo not started |
| M9 | Spend ≤ $50 | on track ($2.16) |
| M10 | Hard-cases split mined + adjudicated | **done** — 525 mined → **507 retained**, 18 quarantined · intent label-noise rate **24.0%** · screen applied only where it beats chance (D36) |
| M11 | **Gate sensitivity measured** | checkpoint captured · scoring in Phase 5 |

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

### D16 · Intent data switched from `PolyAI/banking77` to `mteb/banking77`
**Rejected:** keeping the canonical PolyAI repo and pinning `datasets<3` to retain
loading-script support.
**Why:** `PolyAI/banking77` ships a `banking77.py` loading script and no parquet. The
`datasets` 5.0.1 in this project's lockfile refuses it outright — *"Dataset scripts are
no longer supported"* — and the Hub has no `refs/convert/parquet` branch for it to fall
back on. Pinning an old `datasets` to keep a script alive would have traded a permanent
dependency ceiling for a dataset that has a clean drop-in mirror.

`mteb/banking77` is parquet-native, **MIT** licensed, and carries the same 77 classes
with `text` / `label` / `label_text` columns.

**Cost of the switch:** row counts differ slightly from canonical — 9,993 train and
3,076 test against 10,003 / 3,080, so 14 rows fewer overall. Immaterial to a 770-example
golden set, but the PRD's stated "13,083 available" is now off by 14 and its §9 source
row is wrong until corrected.

**Interview angle:** a dependency-driven dataset migration caught on day 1 rather than
in week 4. Also the concrete payoff of the day-1 verification block existing at all —
this was check 3, and it failed for a reason nobody would have predicted from the PRD.

---

### D17 · Mirror the PII source capped and seeded, not whole
**Rejected:** mirroring all 143,515 English rows, as the other two sources are mirrored
whole.
**Why:** the full English mirror is **147 MB of parquet** — 96% of a 153 MB `data/`
directory — for a task PRD §9 subsamples to ~3K rows. Capping at 20,000 train / 5,000
validation under a fixed seed (`SEED = 20260909`, recorded in `data/MANIFEST.json`) still
leaves ~7x headroom over the training subsample and the Phase 2 ticket pool, and brings
the mirror to **26.6 MB**. Intent (372 KB) and drafting (5.6 MB) are mirrored whole
because there is nothing to gain by trimming them.

`mbert_tokens` / `mbert_token_classes` are dropped as well — unused by this project and
by far the largest columns.

**Trade-off:** the mirror is no longer a faithful copy of upstream, so the manifest has
to carry the filter, the cap, the seed and the Hub revision or the sample is not
reproducible. It does, and re-running the mirror produces byte-identical files.

**Interview angle:** treating an eval/data snapshot as a versioned artifact with a
recorded provenance chain — source revision, filter, seed, per-file sha256 — rather than
"I downloaded a dataset."

---

### D18 · Repo moved from the Desktop to `~/code/AdapterOps`
**Rejected:** leaving it on the Desktop.

**Correction — the original rationale for this move was wrong.** It was made on the
belief that iCloud sync applied the `UF_HIDDEN` flag that silently disabled the editable
install, and that the move removed a recurring cause. That was tested afterwards and
disproved: a uv venv created on the Desktop has unflagged `.pth` files and works. The
flag on the two affected venvs came from some past event that was never identified.
`chflags -R nohidden .venv` was a sufficient fix and the move was not required.

**What still argues for `~/code`, on weaker grounds:** the Desktop is iCloud-synced —
`~/Desktop` and the iCloud Desktop folder list identical contents — so a 1.1 GB venv, a
27 MB data mirror and a git object store would all be continuously syncing. `.gitignore`
does not exclude anything from iCloud. That is a storage-and-churn preference, not a
correctness fix, and the repo would work in either place.

**Verified after the move:** git history and all staged files intact, remote unchanged,
PRD build byte-identical, all five mirrored-data sha256s match the manifest, ruff clean.

**Interview angle:** not the move — the debugging chain, and the correction.
`ModuleNotFoundError` → `.pth` present and correct → silently skipped → `site.addpackage`
ignores hidden files → macOS flag. Four layers between symptom and cause. Then a
plausible-sounding explanation for the flag that turned out to be wrong when finally
tested, after it had already been acted on. The lesson worth telling is the second part:
a fix that works is not evidence that the diagnosis was right.

---

### D19 · Moved back to `~/Desktop/AdapterOps`
**Rejected:** staying at `~/code/AdapterOps`.
**Why:** D18's move was made on a diagnosis that turned out to be wrong — the `UF_HIDDEN`
flag was never caused by the location, and `chflags -R nohidden .venv` was the actual fix.
Once that collapsed, the only argument left for `~/code` was avoiding iCloud sync of the
venv and git store, which is a storage-churn preference. Weighed against having the work
covered by iCloud backup, the preference lost.

**What is genuinely accepted by living on the Desktop:** iCloud syncs everything in the
folder regardless of `.gitignore` — the ~1.1 GB venv, the 27 MB data mirror and the whole
`.git` object store. That is bandwidth and iCloud storage, plus a small corruption risk
for a git object store inside a continuously syncing directory. Worth revisiting only if
sync actually misbehaves.

**Verified after the move back:** HEAD `cd31720`, all 11 staged files, remote unchanged,
PRD build md5 identical (`760badfc1…`), all five mirrored-data sha256s match, ruff clean,
and the `.pth` files came back **unflagged** — which is itself the final confirmation that
the location was never the cause.

**Interview angle:** the pair D18/D19 is worth more than either alone — an action taken on
an untested diagnosis, the diagnosis later falsified, and the action reversed rather than
retrofitted with a better-sounding justification.

---

### D20 · `UV_PROJECT_ENVIRONMENT=venv` set globally; both projects migrated off `.venv`
**Rejected:** `chflags -R nohidden` after every sync (treats the symptom, fails silently
when forgotten); a `.nosync` suffix (tested — does not exclude from this sync); moving
projects off the Desktop (tried in D18, reverted in D19); turning off iCloud Desktop &
Documents sync (removes the backup the user wants).

**What was done:** `export UV_PROJECT_ENVIRONMENT="venv"` in `~/.zshenv` — chosen over
`.zshrc` so non-interactive shells and IDE-spawned processes inherit it. Every uv project
on this machine now creates `venv/` instead of `.venv/`. AdapterOps and IncidentIQ both
had their venvs removed and rebuilt under the new name, and both `.gitignore` files
updated.

**Why this is the right shape of fix:** it removes the *precondition* rather than the
cause. Whatever agent sets `UF_HIDDEN` — a 24-minute probe failed to reproduce it at all,
so the iCloud attribution stays unproven — it only ever touches dot-prefixed names. A venv without a leading dot
is outside the behaviour entirely, so the fix holds even if the attribution is wrong.

**Known residue, accepted:** `.git` is still flagged in every Desktop project. Git does
not care, and no tool in this stack reads `.pth` files from `.git`. The venv is also
still synced to iCloud (~1 GB) — `.nosync` was tested and does not prevent it, and macOS
offers no per-folder exclusion for Desktop & Documents sync.

**Verified:** AdapterOps — `venv/` unflagged, 0 of 3 `.pth` hidden, CLI and deep imports
work. IncidentIQ — was 33,748 flagged entries with its own package unimportable; now
0 flagged and `import incidentiq` succeeds.

**Interview angle:** the whole arc, D18 → D19 → D20. An untested diagnosis acted on, the
action reversed when it was falsified, then a real investigation that found the actual
discriminator (dot-prefix, not location), and a fix chosen specifically so that it works
whether or not the remaining inference is right.

---

### D21 · F3 reframed from a binary PII flag to span detection
**Rejected:** cross-corpus negatives from Bitext (the option originally chosen); same-corpus
negatives built by surrogate substitution; shipping the binary task with the confound
merely disclosed.

**Why:** both constructions were built and measured before any training code was written,
and both failed. Bitext negatives — a classifier that cannot see any PII separates the
classes at **0.9999** against 0.5 chance, so the adapter would be scoring corpus identity.
Same-corpus surrogates — trained on one surrogate vocabulary, tested on a disjoint one,
**0.9918 → 0.5102**, so the adapter would be scoring my twelve phrases. Span detection
invents no negatives at all: `source_text`, `masked_text` and `privacy_mask` offsets come
straight from the data.

**Cost:** F3's gated metric becomes span-level F1 rather than macro-F1, which touches M2
and §11. The golden set stays 300 documents — at a median of six spans each that is ~1,800
scored spans, a denser signal than 300 binary labels, so changelog 15's sizing argument is
unaffected.

**Why not just disclose the confound and ship it:** the adapter would have reported ~0.99
on a scorecard. A number that looks excellent and means nothing is worse than no number,
and it would have sat next to three honest ones.

**Interview angle:** the strongest evidence-driven story in the project so far. Both probes
are reproducible — `uv run adapterops pii-task`, results in `data/pii/CONFOUND.json` — so
this is a claim with an artifact behind it rather than a recollection. Pairs with D7 (the
leak) and D8 (label noise) as the evaluation-design cluster in §5.

---

### D22 · The PII adapter emits `LABEL: value` lines, not masked text
**Rejected:** generating ai4privacy-style masked text (`[GIVENNAME_1]` in place of a name)
and recovering offsets by aligning it against the source — the format the dataset ships,
and the obvious choice.

**Why:** it caps the gated metric. Feeding ai4privacy's **own ground-truth masked text**
through offset recovery scores **0.8973 strict span F1**, not 1.0. A perfect model would
score 0.897. The cause is not a bug: `Sarhat Shegë Böhmerle Cekci` masked as
`[GIVENNAME_1] [SURNAME_1]` has one space between the placeholders and three in the source,
and nothing indicates which is the boundary. `WJ@gmail.com` splits at the wrong dot for the
same reason. Relaxed (overlap) F1 is 0.9994, which confirms the labels are right and only
the boundaries are lost.

`LABEL: value` output recovers offsets by locating each value in the source: **0.9974
strict** on the same 1,000 documents. The residual 0.26% is values appearing more than once
where left-to-right assignment picks the wrong occurrence.

**Cost:** the training target is no longer the column the dataset ships, so `masked_text`
has to be converted to `LABEL: value` lines at training time. `spans_from_masked` is kept
in `eval/spans.py` as the evidence for this decision, with a test asserting it still scores
below 0.95 — if someone "fixes" it, the test explains why it cannot be fixed.

**Interview angle:** the output format silently determined the metric's ceiling. Measuring
the ceiling *before* training — by running ground truth through the scorer and checking it
scores 1.0 — cost one command and would otherwise have surfaced as an unexplained 10-point
shortfall blamed on the model.

---

### D23 · Urgency moves to an NC-licensed source, because the CC0 one has no text
**Rejected:** the PRD's own `albertobircoci/support-ticket-priority-dataset-50k` (CC0);
dropping urgency and shipping three adapters.

**Why:** the Kaggle dataset contains **no ticket text**. Every string column is a short
categorical (`'Wed'`, `'Small'`, `'media'`, max 15 chars) and `description_length` is an
*integer* — the description was measured and discarded. It is a tabular dataset where
`priority` is predicted from `error_rate_pct`, `downtime_min` and `security_incident_flag`,
which gradient-boosted trees would do better and cheaper. It also does not fit the
architecture: the router dispatches ticket *text*, and there is none here.

`Tobi-Bueck/customer-support-tickets` has subject and body, three priority classes, and
11,922 usable English rows after filtering. Its licence is **cc-by-nc-4.0**.

**Cost:** the first non-permissive source in the project. The derived adapter inherits the
restriction — non-commercial, attribution required — and that is disclosed in the README,
`data/MANIFEST.json` (which now records `licence` and `licence_permissive` per source), and
the adapter card. Three of four adapters remain unrestricted.

**What this exposes about day-1 check 4:** it verified the dataset *existed* and was CC0.
It never opened the file. Existence and licensing were checked; **suitability was assumed**,
and the assumption survived several sessions of treating a missing Kaggle token as the
blocker. The token was never the blocker and could not have been — the dataset is public
and downloads anonymously.

**Interview angle:** a verification step that checks the wrong property is worse than no
check, because it produces confidence. "Licence verified, availability verified" read as
"source verified" for weeks. The fix is cheap and now in the mirror: load the data and look
at its columns as part of verification, not after.

---

### D24 · Drafting's frozen train/val leak is recorded and routed around, not re-frozen
**Rejected:** re-running `adapterops splits --force` to fix it.
**Why:** the drafting split is group-aware where the *golden* set is carved out — 989
instructions repeat up to 8x, and a row-wise golden holdout would have left copies of
golden texts in train. That guard works. But the remainder is then split into train and
val row-wise, so the same protection was never applied one level down: **467 of 3,982 val
rows (11.7%) share an `instruction` with `split_train`.**

Consequence, stated at its real size: training uses `load_best_model_at_end` on
`eval_loss`, so drafting's checkpoint was selected against a val set that is 11.7%
memorised. The bias is mild — it inflates every checkpoint's val loss roughly equally, so
the *ranking* the selection depends on is largely preserved — and it touches **no reported
number**, because every score in §4 comes from the golden sets, which are clean.

Re-freezing would move `split_train`, which invalidates the drafting adapter's provenance
and costs a retrain to restore. That is a paid run to fix a bias that changes no published
figure, so the split stays and the number is written down instead. The exclusion is
applied where it actually bites — the router pool drops all 467 rows (D25).

**Interview angle:** the same defect at two levels, guarded at one. A group-aware split is
not a property of a function call; it is a property that has to hold at *every* cut of the
data, and the second cut was made by different code three lines later. Also a worked
example of pricing a fix: the honest move here is the measurement, not the retrain.

---

### D25 · Router pool drawn from the val splits, filtered against adapter training, unstratified
**Rejected:** drawing the pool from each task's full mirror, and stratifying it on the
task label.
**Why:** the router's label is computed — run a pair through its adapter, score it, and
the pair is a success or a failure. So the pool decides what "success rate" means. A row
the adapter was fine-tuned on is answered from memory, and a router trained on those
learns a success rate that does not exist at serving time. The pool therefore comes from
`split_val.parquet` — held out from both the adapter and the golden set — with two further
filters whose removal counts are recorded in `evals/ROUTER_POOL.json`: texts that also
appear in `split_train` (467, all drafting, per D24) and texts that repeat inside val
itself (16).

**Not stratified**, unlike the golden sets, and for the opposite reason. The golden sets
are stratified because their gated metrics demand it. Here the label *is* adapter success,
and stratifying on the task label would reshape the base rate the router exists to learn.

**Interview angle:** "what is your training data for the router?" has a boring answer and
an interesting one. The interesting one is that the pool's provenance defines the label,
so every leak is a silent inflation rather than an error — which is why the freeze verifies
the exclusions by recomputing them from the source files rather than trusting the builder.

---

### D26 · The shift split reports evidence that it shifts, not just that it was built
**Rejected:** freezing the F36 split on the strength of "cluster holdout plus length
deciles" and reporting median length per side as the sanity check.
**Why:** the median is the wrong statistic here and would have passed a broken split. The
shift side is the union of the *top and bottom* length deciles, so its two tails nearly
cancel: drafting's median moves 48 → 49 characters and intent's moves 48 → 45. Read alone,
those numbers say the split does nothing.

What the manifest records instead is the length spread on each side — intent's p10/p90
goes 35/73 → 25/157, so the tails widen as designed — and the share of the shift side's
vocabulary that never appears in-distribution: **25.1% drafting, 29.8% urgency, 41.4%
intent, 56.0% PII**. The vocabulary count is deliberately computed with a crude regex
rather than the TF-IDF analyzer that produced the clusters, so the evidence does not come
from the same code as the thing it is evidence for.

**Interview angle:** the M4 number is only interpretable if the split is known to shift
something. Without this, a router that holds up under "shift" has two explanations —
it generalises, or the split was inert — and no way to tell them apart. Measuring the
split itself costs nothing and removes one of them *before* the result arrives, which is
the only time that removal is credible.

---

### D27 · Generation and the success rule are separated, so the threshold is free to change
**Rejected:** scoring pairs inside the GPU run and writing out a binary `success` column.
**Why:** "did the adapter succeed on this pair" is a judgement call, and three of the four
rules have a defensible alternative. PII uses per-document strict F1 = 1.0 — every span
found, exact boundaries, nothing invented — but *recall* = 1.0 is arguably the better
compliance rule, since a false positive is over-redaction and a false negative is a
breach. Deciding that inside the GPU run would price a one-line change at a GPU session.

So the run writes score *components* — per-document precision, recall, strict and relaxed
F1, the raw prediction, the log-probabilities — and `router/scoring.py:label()` derives the
binary on a laptop. The rule is a CPU re-run; the generation is not.

**Interview angle:** the cheap decision and the expensive one were tangled, and separating
them costs nothing at design time. It is also what makes the drafting proxy tolerable —
Phase 3 swaps the judge in and re-derives, without re-generating 3,000 completions.

---

### D28 · Drafting's router label is a named proxy, and its cost is measured rather than argued
**Rejected:** leaving drafting out of the router pool until the Phase 3 judge exists.
**Why:** task identity is an input feature to the router (F10), and F36 requires every task
on both sides of the shift split. A router that has never seen `task=drafting` reproduces
exactly the confound D11 was written to remove. Drafting has to be in the pool.

But its real metric is the judge, which does not exist yet. So Phase 2 labels it by
token-F1 against the Bitext reference reply, cut at the pool median — **the weakest label
in the project, and marked `label_is_proxy` in every output**. Two things follow, both
built rather than promised: the operating curve is reported with and without drafting
pairs, so it is visible how much of the result rests on the weak label; and when the judge
arrives, its agreement with this proxy is measured and published.

**Interview angle:** the useful move is not defending the proxy, it is bounding it. "One
of four tasks has a stand-in label, here is the result with and without it, and here is the
agreement number once the real metric existed" is a complete answer. Quietly using the
proxy and reporting one blended curve is the same work with none of the credibility.

---

### D29 · The hard-cases exclusion is an allocation, not a subtraction — after two failed attempts
**Rejected:** twice, before landing. The record of both is the point of this entry.

BUILD-PLAN calls the F31 exclusion the single most important line of code in Phase 2: get
it wrong and every router number is inflated with nothing to announce it. The exclusion
itself is one line. *Where* it is applied took three goes.

**Attempt 1 — mine hard cases from the whole pool, subtract from router training.** The
literal reading of PRD §11's "held-out router eval set (disjoint from hard cases)". It
destroys the eval set. Hard cases are adapter failures capped at ~150 per task; holding
them out of a 150-row eval slice takes nearly every failure with them. On synthetic data at
a 22% failure rate, the router eval set went from **22.5% failures to 2.5%** — an operating
curve measured on a population with almost nothing left to escalate, where every policy
converges for reasons that have nothing to do with routing.

**Attempt 2 — split first, mine only from the training portion.** Fixes eval, breaks
training. The 150-per-task cap is *larger* than the failure count a ~400-row training slice
contains, so reserving hard cases left router training at a **1.00 success rate**. One
class. It would have trained without error and predicted a constant.

**What both missed** is that failures are the scarce resource, and three consumers need
disjoint shares of them: the router learns from failures, the router's eval measures on
failures, and the hard-cases split is made of failures. That is an allocation problem, not
an ordering problem. The pool now draws two slices at freeze time — 750 `router` rows and
700 `mining` rows per task, 5,800 pairs — and `dataset.py` has no subtraction step at all.
The exclusion holds because the populations were never the same rows.

**Consequence, reported rather than engineered around:** 150 hard cases per task requires
~150 failures per task, and intent's adapter scores 0.9312. At a ~7% failure rate, 700
mining rows yield roughly 50. The intent hard-cases split will be small, the manifest
records `cap_bound: false` and the measured failure rate beside it, and the honest reading
is that **the size of a failure-mined split is a statement about the adapter**, not a
target to hit.

**Interview angle:** the best answer to "tell me about a bug in your own work" in the
project, because neither failed version was detectable by inspection. Both produced a
dataset of the right shape with plausible per-task counts; attempt 1's only symptom was
policies that agreed suspiciously well, and attempt 2's was a router that predicted one
class. Both were found by giving synthetic data a known failure rate and asserting it
survived to the far end — which is a cheap test to write and the only thing that would have
caught either.

---

### D30 · Eval splits are pinned in the manifest beside the models, and moving one blocks promotion
**Rejected:** pinning models only, and treating the splits as a fixed property of the repo.
**Why:** a score is a statement about a model *measured against a bar*. Pin the model and
let the bar move, and a changed number has two explanations with no way to separate them —
which is the failure the whole two-split design exists to prevent. So `manifests/system.json`
pins the four golden sets, the router pool and the shift split by sha256 alongside the
adapter revisions, and a promotion whose splits moved is **blocked even when a regression
run is attached**: a regression number computed across a moved bar does not license the
promotion, it is the thing that needs explaining.

**Interview angle:** "what does your manifest pin?" separates two answers. Pinning model
weights is the obvious half and everyone does it. Pinning the *evaluation* is what makes a
historical score re-derivable, and it costs one dictionary.

---

### D31 · The first manifest is exempt from the gate, because a regression needs something to regress from
**Rejected:** the first implementation, which blocked v1.
**Why:** `blocking_reasons` refuses a promotion whose components moved without a regression
run attached. On v1 every component "moves" — from `None` — so the baseline manifest could
never be promoted, and the gate could never be satisfied by any input. It read as the gate
working.

The fix is not a special case so much as the definition: a regression is a comparison
against a previous version, and v1 has none.

**Interview angle:** a gate that cannot be satisfied is indistinguishable from a strict gate
until someone tries to pass it. Worth pairing with F33's report-only period — both are the
same lesson about gates that are theoretically correct and practically unusable.

---

### D32 · The oracle ranks by gain from escalating, not by whether the adapter failed
**Rejected:** the first implementation, which escalated every pair the adapter got wrong.
**Why:** that is an upper bound only if the frontier always succeeds, and it does not.
Escalating a pair the frontier also fails costs quality and buys nothing. A full rehearsal
of the post-GPU chain caught it in the clearest possible way: the **confidence policy beat
the oracle** at two budgets, and reported capturing **1.39** of the available headroom — a
share above 1.0, which is not a number that can exist.

The oracle now ranks by `frontier_success - success`: +1 where escalation rescues a
failure, −1 where it breaks a success, 0 where it changes nothing. It also refuses to run
without a measured `frontier_success` column, because without one "escalate every failure"
cannot be known to be a bound.

**Interview angle:** the bug is interesting because the broken version is the intuitive one
— "the oracle knows which ones you got wrong" — and it only misbehaves once the escalation
target is fallible, which is exactly the realistic case. The test that now covers it is
parameterised on a frontier that sometimes fails; the original passed because the synthetic
frontier always won.

---

### D33 · The task feature stays — one ablation said drop it, three seeds said it does nothing
**Rejected:** removing `task` from the router input on the strength of a single ablation run.
**Why:** the first no-task run scored higher than the first with-task run on all four tasks,
and I reported that as "removing the task name made the router better at every task." A
second seed of the *identical* configuration reversed intent and urgency. Intent's eval
slice is 90 rows, and its ROC-AUC has a **standard deviation of 0.119 across seeds** — so a
per-task comparison between two single runs there is closer to a coin flip than a finding.

Three seeds per variant (`runs/router__seed_study.json`):

| | with task (mean ± sd) | no task (mean ± sd) | verdict |
|---|---|---|---|
| eval, overall | 0.7197 ± 0.007 | 0.7081 ± 0.013 | within noise |
| eval, intent | 0.5414 ± 0.119 | 0.4102 ± 0.041 | within noise |
| shift, overall | 0.7202 ± 0.004 | 0.7117 ± 0.002 | small, favours task |
| shift, intent | 0.5302 ± 0.020 | 0.4741 ± 0.010 | small, favours task |

F10 stands as specified. What held in **all six runs** is the finding that matters: the
router scores 0.708–0.720 against a task-name lookup table's **0.6883**.

**Interview angle:** I made a claim from one pair of runs and retracted it within the hour —
which is D9's rule (thresholds from measured variance) applied to my own conclusion rather
than to a gate. The consequence reaches past the router: any *per-task* gate on slices this
size would be dominated by seed noise, which is worth knowing before F33 derives thresholds.

---

### D34 · The judge is reference-free, and its rubric states a policy for template slots
**Rejected:** a reference-guided judge that sees the Bitext reply alongside the one it grades.
**Why:** the drafting proxy already measures overlap with the reference, and the Phase 2
escalation finding showed what a reference-shaped metric rewards — conformance to one
dataset's house style, which fine-tuning transfers and a frontier model does not. A judge
that grades against the reference would rebuild that bias one level up. The cost is grading
without an answer key, which is acceptable because Bitext's replies are generic procedures
rather than account-specific facts.

The slot policy was found in the data, not assumed. **49%** of the adapter's replies and
**42%** of Bitext's own references contain slots like `{{Order Number}}`; only **12.5%** of
GPT-4o-mini's replies do. Left to its own taste, a judge could read a slot as an unfinished
reply and mark the adapter down for following the dataset's format. The rubric says to treat
slots as the correct value, and says it identically for every reply.

*Also caught while building it:* the frontier arm's `Ledger` prices tokens as gpt-4o-mini.
Reused unchanged for a gpt-4o run it would under-count spend about 16x, and the spend cap
would never fire. It is subclassed with gpt-4o prices, and a test trips a $1 cap with $2.50
of tokens.

**Interview angle:** "how did you validate your LLM judge?" usually gets a correlation
number. The better answer starts earlier — what the judge is *allowed to see*, and which
conventions in the data would silently bias it — because a well-correlated judge that
inherited the benchmark's house style is measuring the wrong thing precisely.

---

### D35 · The judge is DeBERTa-v3-base, by §15's rule — a rule that could not see the option §10 named
**Rejected:** Qwen2.5-0.5B with QLoRA, the alternative §10 lists.
**Why:** §15 fixed the decision rule before any measurement: whichever base trains faster on
available compute. Available compute without new spend is this laptop's CPU. There is no
CUDA, so QLoRA's 4-bit path (bitsandbytes) cannot run at all, and MPS had already run out of
memory on DeBERTa. The comparison that *could* be made, on real judge inputs at batch 8 and
max length 512, median of six steps after two warmup:

| | seconds / step | projected real run | trainable params |
|---|---|---|---|
| DeBERTa-v3-base, full fine-tune | **2.74** | ~18 min | 184M |
| Qwen2.5-0.5B, LoRA r=16, fp32 | 3.38 | ~22 min | 2.2M |

DeBERTa takes about 19% less time per step. The rule picks it, and the gap is not close
enough to call a tie. Two things are stated rather than implied. First, **QLoRA never ran** —
this is not evidence that it loses, only that it was unavailable. Second, the rule measures
speed and nothing else; F15 calibration is where judge quality gets measured, and a poor
calibration would be a finding against the rule, not grounds for choosing differently in
hindsight.

**Interview angle:** a decision rule written in advance is only as good as the options it
can actually evaluate. This one could not evaluate the option the spec named. Reporting
"DeBERTa trains faster than QLoRA Qwen" would have been tidy and false.

---

### D36 · Hard cases are screened for label noise by independent agreement, not by a second miss
**Rejected:** PRD §9's literal rule — quarantine a mined item when the frontier model also
disagrees with the gold label.
**Why:** on the 525 mined candidates that rule quarantines **64%** of intent's hard cases and
**64%** of urgency's. The reason is selection. An item became a hard case *because* the adapter
failed on it, so the set is disproportionately difficult, and a second model missing it too is
exactly what difficulty predicts. The rule cannot tell a hard item from a mislabelled one.

The rule used quarantines an item when an independent model (gpt-4o-mini) gives **the same
alternative answer as the adapter**, against gold. Two models converging on one specific answer
the gold rejects is what label noise produces and difficulty does not. And that agreement is
checked against chance before it is trusted:

| task | classes | both wrong, same answer | uniform chance | applied |
|---|---|---|---|---|
| intent | 77 | **37.5%** | 1.3% | yes — about 29× chance |
| urgency | 3 | 60.4% | 50.0% | no — barely above chance |

Uniform chance is a floor, since confusable neighbouring intents raise the true rate, but a
29× gap is not closed by that. On urgency the same rule is barely distinguishable from a coin
flip, so it is reported and not applied. PII (the frontier gets 2.7% of documents fully right)
and drafting (no label to dispute) are declared not applicable.

Result: **18 of 75 intent hard cases quarantined (24.0%)**, 507 retained. It cost nothing — it
reuses the answers the escalation arm had already paid for. What it catches reads like the real
thing: *"Can you explain the various transaction times shown on my statement?"*, labelled
`transaction_charged_twice`, answered `transfer_timing` by both models.

**Interview angle:** the PRD's rule is the intuitive one, and it would have quarantined two
thirds of the split and published that as a label-noise rate. The better question is not
"does a second model disagree" but "does it disagree *the same way*" — and then whether that
agreement beats chance for the number of classes. That last check is what kept the rule off
urgency, where its output would have looked identical and meant nothing.

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
| iCloud backup vs sync churn | Live on the Desktop (D19) | ~1 GB of venv + git still syncing; no per-folder exclusion exists |
| Real data realism vs privacy | Synthetic PII spans only | No real-world messiness in the PII task |

---

## 4. Observations & findings

Filled in as results arrive. **Empty is the correct state today.**

### Measurements pending
| What | Value | Recorded |
|---|---|---|
| Day-1 license + availability checks (7 of 8) | **all pass** — see findings | Phase 0 |
| OpenAI API reachable, spend cap set | **auth ok, 129 models; cap $15** | Phase 0 |
| Datasets mirrored + provenance recorded | **4 of 4** · 28.9 MB, reproducible | Phase 1 |
| Eval splits frozen (F35) | **intent 770 (10/class), drafting 300** | Phase 0 |
| Majority-class floor, intent | **0.0130 micro-accuracy** | Phase 0 |
| Intent adapter trained | **done** — 3 epochs, 27 min, free T4 | Phase 0 |
| **Intent adapter, golden set** | **0.9312 micro-accuracy** · macro-F1 **0.9302** · exact-label 1.000 | Phase 0 |
| **M2: adapter vs prompted baseline** | **+0.4091** (0.9312 vs 0.5221) — fine-tuning justified | Phase 0 |
| Run-to-run variance, 2 independent runs | **±0.0039 micro-accuracy** (3 of 770) — all of it from training | Phase 0 |
| Eval harness determinism | **bit-identical** — baseline reproduced to 15 dp across VMs | Phase 0 |
| Adapters published | `Tanny03/adapterops-intent` + `-undertrained-m11` | Phase 0 |
| **Gate 0.5 — concurrent multi-LoRA** | **PASS** — 2 adapters, 400 reqs @ 16, 159 rps | Phase 0 |
| Serving latency, A10 | **P50 81ms · P95 155ms** @ concurrency 16 | Phase 0 |
| PII span scorer ceiling | **0.9942** on the frozen golden set (vs 0.8973 for masked text) | Phase 1 |
| PII splits frozen | golden 300 docs / **2,394 spans**, train 17,000 | Phase 1 |
| **PII adapter (8K rows, 3 epochs)** | **0.9190 strict** / 0.9477 relaxed — 92.4% of ceiling, rev `5405e954` | Phase 1 |
| PII baseline (regex+spaCy) | **0.5006 strict** — adapter wins by **+0.418** | Phase 1 |
| **Urgency adapter** | **0.470 micro / 0.421 macro — LOSES to TF-IDF (0.547/0.547)** | Phase 1 |
| Drafting adapter | trained, published; judge-scored in Phase 3 | Phase 1 |
| Adapters on the Hub | intent, pii, drafting (+ 3 M11 checkpoints) | Phase 1 |
| M11 under-trained checkpoint retained | **saved at step 119 of 798** | Phase 0 |
| PII binary-task confound | **0.9999 cross-corpus / 0.5102 unseen-surrogate** | Phase 0 |
| vLLM multi-LoRA works on rented A10G? | — | Phase 0 |
| Adapter vs prompted baseline, per task | — | Phase 1 |
| P95 latency, per adapter, rented GPU | — | Phase 1 |
| Golden-set run-to-run variance | — | Phase 1 |
| Learned router vs confidence baseline | — | Phase 2 |
| Within-task shift degradation | — | Phase 2 |
| Judge correlation + ±1 agreement | — | Phase 3 |
| **Router pool frozen (F7)** | **5,800 pairs** — 750 router + 700 mining per task, exclusions verified zero | Phase 2 |
| **Within-task shift split frozen (F36)** | shift side **32-40%** of the router slice · every task both sides | Phase 2 |
| Shift-side vocabulary unseen in-distribution | **25.1% / 29.8% / 41.4% / 56.0%** (drafting / urgency / intent / PII) | Phase 2 |
| Adapter revisions pinned (F16) | **4 + base**, PII restore verified byte-identical to the scored revision | Phase 2 |
| **M1 — four adapters concurrent** | **PASS** · 5,800 reqs · **0 errors** · 23.6 rps · 246 s | Phase 2 |
| Per-adapter P95, A10 | urgency **60 ms** · intent **136 ms** · PII **2,259 ms** · drafting **3,001 ms** | Phase 2 |
| Router pool scored (F7) | 5,800 pairs · success: intent **0.891**, PII **0.724**, urgency **0.503**, drafting 0.501 | Phase 2 |
| Hard-cases mined (F31) | **525** · 150 each for urgency/PII/drafting, **75 intent** (`cap_bound: false`) | Phase 2 |
| **Router (F10), ranking quality** | canonical retrain ROC-AUC **0.7069** eval / **0.7164** shift (first run 0.7064 / 0.7129, **0.996 correlated with a task-name lookup**) | Phase 2 |
| Task-prior-only baseline | ROC-AUC **0.6883** — the router adds **+0.018** over knowing only the task | Phase 2 |
| Router seed study, 3 seeds × 2 variants | task feature: **no reliable effect** · router beats the task prior by **0.02–0.03 in all six runs** · intent eval AUC sd **0.119** | Phase 2 |
| **Frontier arm quality (F8)** | in-distribution **0.278** vs local **0.656** · shift **0.313** vs local **0.652** — escalation *lowers* quality in both | Phase 2 |
| **Judge labelling projected (F13)** | **$3.38** for 1,950 GPT-4o grades (1,200 adapter + 750 frontier) · counted locally, no API call · **approved at a $5 cap** | Phase 3 |
| Judge base benchmark (§15), CPU | DeBERTa-v3-base **2.74 s/step** (~18 min) vs Qwen2.5-0.5B LoRA **3.38 s/step** (~22 min) · QLoRA not runnable here (no CUDA) | Phase 3 |
| GPT-4o judge sample | **10/10** parsed · **$0.0123** actual vs $0.0200 projected — the local projection runs conservative | Phase 3 |
| Judge training path (F14), smoke on a planted signal | **works** after target scaling · Spearman **0.859** · within-±1 **1.00** — the first smoke passed at Spearman −0.45 | Phase 3 |
| Router training path verified end to end | **works**, on synthetic labels · ~14 s / 25 steps on laptop MPS — **F10 needs no GPU** | Phase 2 |
| System manifest v1 promoted (F16) | 4 adapters + base + 6 splits pinned · gate `report_only` | Phase 4 |
| Frontier escalation arm measured (F8) | **3,954 of 5,800 pairs** · $0.21 · blocked on a daily request quota, resumes free | Phase 2 |
| Drafting val rows sharing an instruction with train | **467 of 3,982 (11.7%)** — D24 | Phase 1 |
| **Label-noise quarantine rate, per task** | intent **24.0%** (18/75) · urgency measured, not applied (60.4% vs 50.0% chance) · PII and drafting not applicable | Phase 4 |
| Hard-cases split frozen (F31) | **507 retained** — drafting 150, PII 150, urgency 150, intent 57 · 18 quarantined · **$0** | Phase 4 |
| Hard-split run-to-run variance | — | Phase 4 |
| **M11: does the hard split catch what the random set misses?** | — | Phase 5 |

### Findings log
> Append entries as things are learned — especially the surprising and the negative.
> Order by phase, not by date. Format: **phase · what happened · what it means ·
> whether it changes the plan.**

**Phase 4 · The PRD's adjudication rule would have quarantined two thirds of the hard split.**
§9 quarantines a mined item when the frontier model also disagrees with the gold label. On the
525 mined candidates that is **64.0%** of intent's and **64.0%** of urgency's hard cases —
which, reported as the per-task label-noise rate M10 asks for, would have said nearly two
thirds of those labels were wrong.

*Why it cannot be right:* the candidates were selected *because* the adapter failed on them.
The set is disproportionately difficult, so a second model failing too is what difficulty
predicts. The rule measures hardness and names it noise.

*What replaced it* (D36): quarantine when the second model gives the **same alternative
answer** as the adapter, and only where that beats chance for the number of classes. Intent:
37.5% against a 1.3% floor — applied, **18 of 75 quarantined (24.0%)**. Urgency: 60.4%
against 50.0% — reported, not applied.

*The caveat that stays attached:* two language models can share a confusion — adjacent
intents such as `transfer_timing` and `pending_transfer` are genuinely close — and uniform
chance does not model that. So 24.0% is strong evidence that those labels are *contestable*,
not proof that they are wrong. That is still the useful property: a gate should not block a
release for disagreeing with a contestable label.

**Phase 3 · A smoke test passed a judge that had learned nothing.** The first end-to-end
smoke of F14 reported **PASS** with a calibration Spearman of **−0.45** and a bias of
**−2.38**. Its criterion checked that a correlation *existed*, not that it was positive —
so a judge worse than useless cleared it.

*Why it learned nothing:* training began near **MSE 9.5**. A regression head initialises
near zero while grades run 1–5, so the first steps go to learning the average grade, and
over 22 steps that was the whole of what it learned.

*Fixed twice.* Grades now train as `(score − 3) / 2` and map back afterwards, so the offset
is not something to discover. And the smoke criterion now requires **Spearman > 0.5 on a
planted signal**. Re-run: Spearman **0.859**, Pearson 0.876, within-±1 **1.00**, bias −0.04.

*Means:* a test's pass criterion is itself a claim, and "it produced a number" is not the
claim that matters. It is D31 inverted — that was a gate too strict for anything to pass,
this was a check too weak for anything to fail — and both read as working until the
output was looked at rather than the verdict.

**Phase 2 · The committed operating curve measured M3 and M4 as one number.** The report
printed a single `router_eval` population of **1,439 pairs** — which is the 392
in-distribution eval pairs *plus* the 1,047 shift pairs. The population split keys on a
`side` column that lives in the router's eval splits and not in the scored pool, and the
join merged only `router_p_fail`, so the split never fired.

*Why no test caught it:* the synthetic fixture put `side` on the local frame, where the real
schema never has it. The test exercised the code path with a shape the production data does
not have. The regression test now drops `side` from the fixture to match the real schema.

*Means:* small in effect — the blended 0.304 vs 0.653 becomes 0.278 vs 0.656 in-distribution
and 0.313 vs 0.652 under shift, and escalation lowers quality in both — but it matters in
kind, because M4's entire purpose is to be read *separately* from M3. Separated, M4 shows no
degradation under shift at all.

**Phase 2 · The router learned which task it was looking at, and almost nothing else.**
It scores **0.7064 ROC-AUC** on the held-out eval set and **0.7129** under shift, which
reads like a working component. The per-task breakdown says otherwise: drafting 0.512,
intent **0.361** — *below chance* — PII 0.617, urgency 0.602.

*The explanation, tested rather than assumed.* The four tasks fail at very different rates
— intent 0.10, PII 0.27, drafting 0.49, urgency 0.49 — so a model that reads only the task
name already ranks failures well across the pooled set. A pure task-prior lookup, with no
text at all, scores **0.6883**. The trained router scores 0.7064 and its predictions
correlate **0.996** with that lookup. It added **0.018 AUC** over knowing nothing but which
task it was.

*Means:* the pooled number is an artifact of aggregating four tasks with different base
rates, and reporting it alone would have been the single most misleading number the project
could produce. The per-task rows are the result; the headline is not.

*This was the risk the design named in advance.* `pool.py` says an unbalanced pool "would
let it learn a per-task prior instead of reading the text", and the pool *is* balanced at
750 per task — but balancing the **rows** does not balance the **failure rates**, and the
shortcut lives in the latter. Balance was necessary and not sufficient.

*Resolved by a seed study, not by the first ablation run* (D33). Removing the task name has
no reliable effect: across three seeds per variant every eval difference sits inside the
noise, and the only shift differences beyond two standard deviations are small and favour
*keeping* the task feature. The shortcut is not an artifact of the task token — the text
itself carries task identity (PII documents are long, intent queries are short banking
phrases), so the model can recover the prior either way.

**Phase 2 · Escalating to the frontier makes quality *worse*, on every task.** The operating
curve's whole premise is that escalation buys quality at a cost. Measured, it does not: the frontier arm scores **0.278** against local's **0.656** in-distribution and **0.313**
against **0.652** under shift, and every policy's quality falls as budget rises. (First
reported as 0.304 vs 0.653, a figure that blended the two populations — see the finding
below. The conclusion does not move.) `rescued` (local fails, frontier succeeds) runs 0.9-15%; `broken` (local
succeeds, frontier fails) runs 27-63%.

| task | local | frontier | rescued | broken |
|---|---|---|---|---|
| intent | 0.891 | 0.647 | 3.6% | 28.0% |
| PII | 0.724 | 0.101 | 0.9% | 63.2% |
| urgency | 0.503 | 0.387 | 15.3% | 26.9% |
| drafting | 0.501 | 0.084 | 2.1% | 43.9% |

*What this is not.* **Not "a 1.5B adapter beats GPT-4o-mini."** Stating it that way would be
the overclaim this project exists to avoid. The PII breakdown shows why: the frontier's
span **recall is 0.547** against the adapter's 0.943, while its relaxed-minus-strict gap is
only 0.070 — so the regions it does find are mostly right, and it simply *does not list*
~45% of the items. ai4privacy counts TITLE, AGE, GENDER and DATE as personal information; a
model applying a commonsense notion of "personal information" skips them. At a per-document
rule of *every* span exact, 0.55 recall over ~7.6 spans per document arithmetically floors
the score near zero — which is exactly the 0.101 observed.

*What it is.* **These benchmarks reward conformance to a dataset's labelling conventions,
which fine-tuning transfers and prompting does not.** The adapter learned what ai4privacy
counts as PII and how Bitext phrases a reply. A frontier model's greater general capability
is invisible to a metric defined by those conventions.

*Means:* the cost-quality argument in §8 does not hold on this benchmark as scored, and the
honest curve shows escalation as a quality *loss*. A fair frontier comparison needs either
few-shot exemplars carrying the conventions, or metrics that do not reward exact conformance
— the relaxed span F1 and the Phase 3 judge are both already built, which makes this a
measurable follow-up rather than a caveat.

**Phase 2 · The post-GPU chain was rehearsed end to end on a synthetic scored pool, and it
found three things.** `router-dataset` → `router-train` → `router-report` had never run in
sequence, and all three were due to run for the first time on the far side of a paid GPU
session. The rehearsal synthesises a scored pool at the *measured* per-task failure rates —
intent 0.07, urgency 0.53, PII 0.45, drafting 0.50 — and runs the real chain against it in
a temp directory.

*First, the oracle was not an upper bound* (D32). Confidence beat it, and headroom read 1.39.

*Second, the router cannot train on this machine's MPS.* It OOM'd at micro-batch 16 and
again at 8, with ~12 GB of the 24 GB free — MPS shares unified memory and reports the
machine's total commitments, and DeBERTa's disentangled attention carries two extra
attention matrices per layer. The router now selects CUDA when present and **CPU
otherwise, skipping MPS deliberately**; a 141M model over ~200 steps costs minutes there,
and CPU is the deployment target anyway (PRD §7 budgets the router at <50 ms on CPU).

*Third, D29's prediction held numerically.* The hard-cases mine reserved the full 150 for
urgency, PII and drafting, and **51 for intent, with `cap_bound: false`** at a 7.3% failure
rate. D29 predicted "roughly 50" from the adapter's 0.9312 accuracy before any of this ran.

*Means:* the chain works, and three bugs that would each have cost a rented-GPU cycle were
paid for in laptop minutes instead. The rehearsal is the cheapest thing in the project per
defect found.

**Phase 2 · The operating curve would have compared two policies on two different
populations.** `report.py` runs for the first time *after* a paid GPU session, so it was
exercised against synthetic runs beforehand. The dry run found that the learned router only
has predictions for its held-out eval splits, while the curve was being computed over the
whole 3,000-pair router slice — so the router would have been ranked on missing values
while confidence was ranked on everything. Two policies, two populations, one chart, and no
error anywhere.

*Fixed by naming the population rather than assuming it.* The report now emits a block per
population — `router_in_distribution` (M3) and `router_shift` (M4) when a router exists,
and a `router_slice__no_router_yet` block labelled as such when one does not — and every
policy inside a block is scored on identical rows.

*Also checked while there, because it is the invariant D28 rests on:* both arms are cut at
the same drafting threshold, taken from the local run. A test reproduces the frontier's
drafting successes by hand from that one number, and separately asserts the rate is **not**
0.5000 — which is the hallmark of each arm having cut at its own median.

*The dry run also showed the curve is non-monotonic*, peaking mid-budget and falling back to
the frontier's own quality at budget 1.0. That is correct and is the entire cost-quality
argument in one line: escalating everything is worse than escalating selectively, because
the frontier loses on pairs the adapter already gets right.

**Phase 2 · The router trained to `nan`, and neither the device nor the data was at
fault.** A smoke run of the DeBERTa router on synthetic labels completed normally and
reported a train loss of **753.5**, `grad_norm: nan`, and `eval_loss: nan`. A run that
finishes and reports nan is worse than one that crashes: on a rented box it would have been
read as a bad hyper-parameter and retried.

*The bisection, in order.* Not MPS — CPU produced identical NaNs. Not the data — a single
forward pass gave a loss of 0.71 and a gradient norm of 2.09. Not the gradients — every one
was finite after backward. What was non-finite was **105 of ~200 parameters, after a single
AdamW step**. SGD on the same gradients was clean.

*The cause.* `microsoft/deberta-v3-small` ships **fp16** weights, and transformers 5 honours
a checkpoint's dtype rather than upcasting the way earlier versions did. AdamW's second
moment is `(1-β₂)·g²`; at g ≈ 10⁻², that is ≈ 10⁻⁷, which is subnormal in fp16, and dividing
back out overflows the format's 65,504 ceiling. One step, and most of the model is inf.
Loading with an explicit `dtype=torch.float32` fixes it — the same five steps then run
0.688 → 0.666.

*Means:* "the library picks a sensible dtype" stopped being true between major versions, and
the failure it produces is silent. The guard in `router/train.py` now raises if any
parameter arrives as anything but fp32, and a **test asserts the upstream default is still
fp16** rather than asserting the workaround — so if a future release upcasts again, the test
says the guard is redundant instead of quietly protecting nothing.

*Cost avoided:* this would have surfaced on the rented box, after the GPU session, as a
router that trained to nan.

*Also measured while there:* 25 training steps took **14 seconds** on the laptop's MPS. The
real router — ~1,200 rows, 3 epochs, ~225 steps — is roughly two minutes. **F10 costs $0**
and does not need the rented box at all.

**Phase 2 · $0.16 of tokens consumed a whole day's request quota, and the spend cap could
not have seen it.** The frontier escalation arm stopped at 3,499 of 5,800 pairs on a
**requests-per-day** ceiling — 10,000/day, exhausted — having spent **$0.16** against a
$1.50 guard that never came close to firing.

*Cause, and it is mine rather than the provider's.* The client was constructed with the
SDK's default internal retries still enabled *and* wrapped in a retry loop of its own. Each
failure therefore cost up to twelve requests instead of one, so a burst of rate-limiting
fed itself: more retries, more quota consumed, more rate-limiting. The visible symptom was
a process that looked alive and wrote nothing for five minutes.

*Means:* a cost guard measured in dollars is guarding one of at least three exhaustible
resources — dollars, requests, and tokens-per-minute — and it is not the one that binds
first on a small account. The run now counts every *attempt*, retries in exactly one place,
and treats a daily quota as a stop rather than something to retry, because retrying a daily
cap can only spend more of tomorrow's.

*Cheap by accident:* the per-pair cache meant nothing was lost and nothing will be paid for
twice. That was written for crash-resumption and turned out to matter for a reason it was
not written for.

**Phase 2 · A median threshold labelled total failure as total success.** The drafting
router label cuts at the pool median token-F1. A test that fed *empty* predictions through
the scorer — meant only to check that failed requests are not silently dropped — came back
with every drafting pair marked a success.

The cause is that a median is degenerate exactly when the distribution is. With every reply
scoring zero, the median is zero, and `>= median` is true for all of them. The rule would
have reported a 100% success rate on a task where the adapter produced nothing, and on a
GPU run that outcome would have looked like a quiet, plausible number rather than an error.

*Fixed as a precondition, not a second threshold:* a reply sharing no tokens at all with
the reference is not a success at any cut. Adding a tunable floor would have traded one
arbitrary constant for two.

*Means:* relative thresholds carry an absolute failure mode, and it surfaces only under
inputs nobody writes a test for on purpose. This one was found by a test aimed at something
else entirely — which is the argument for testing the degenerate input rather than the
representative one.

**Phase 2 · The drafting split leaks at the second cut, and the pool test found the
third.** Building the router pool surfaced two things neither the splits code nor its
tests were looking for.

*First:* `split_val` and `split_train` for drafting share 467 instructions (11.7%). The
group-aware guard was written for the golden holdout and never applied to the train/val
cut made three lines later (D24). No reported number moves — every score comes from the
golden sets — but drafting's best-checkpoint selection ran against a partly memorised val
set, and the router pool would have inherited the whole problem.

*Second, and this one was caught by a test rather than by reading:* the first frozen pool
contained the same drafting instruction twice. Val itself repeats 92 instructions, and the
draw took both copies of one. Two identical texts either side of the router's own
train/eval split is the same leak one level down — so the pool now drops in-split
duplicates too, and the freeze re-verifies all three exclusions by recomputing them from
the source files instead of trusting the builder that just applied them.

*Means:* the useful generalisation is not "check for leaks" but **check at every cut**. One
group-aware split does not make a pipeline group-aware; each new slice of the data is a new
chance to break it, and the check has to be independent of the code that did the slicing.

**Phase 1 · The PII adapter earns its place; the regex baseline says why.** A
pattern-and-lexicon baseline scores **0.4299 strict** against the adapter's **0.9190** —
a 0.489 gap, the opposite verdict to urgency.

The per-label breakdown is the useful part. **EMAIL: 0.995** — regex essentially solves it.
**ZIPCODE: 0.166** — five digits look like an age, a building number, a tax number or part
of a phone number. Those two are the same phenomenon from opposite ends: a shape nothing
else shares, versus a shape everything shares. The four name-like labels (GIVENNAME,
SURNAME, CITY, STREET — 33% of spans) score zero because patterns cannot reach them at all.

*Means:* PII detection is only marginally a pattern-matching problem. One label of nineteen
is genuinely solved by regex; the rest need something that reads context. That is a
specific claim about where the adapter's value sits, rather than "the model is better".

*The caveat was then closed.* The first version left ~33% of spans (names, cities,
streets) unattempted, making 0.489 an upper bound rather than a figure. Adding spaCy
`en_core_web_sm` lifts those labels to 0.38-0.43 F1 and the baseline to **0.5006 strict** —
so the adapter's real advantage is **+0.418**, not +0.489.

*Where NER falls short is itself informative.* Relaxed F1 rose much more than strict
(0.4632 → 0.5939 against 0.4299 → 0.5006): spaCy finds the right **regions** but the
boundaries and the GIVENNAME/SURNAME split are wrong, because gold sometimes treats two
words as one given name and a token-position heuristic cannot know that. Precision also
**fell**, 0.715 → 0.599 — NER buys recall at a cost in precision, which is worth stating
alongside the F1 gain rather than reporting only the improvement.

**Phase 1 · The urgency adapter loses to bag-of-words, and that is the finding.** It
scores **0.470 micro / 0.421 macro-F1** on the 300-row stratified golden set. TF-IDF plus
logistic regression — seconds to train on a laptop — reaches **0.5467 / 0.5465**. Chance
on three balanced classes is 0.3333.

*What it is not.* Not a format failure: `exact_label_rate` is 1.000, every output is a
valid class. Not label noise: across 9,879 unique tickets, **zero** identical texts carry
different priorities. Not unlearnable: TF-IDF is 21 points above chance.

*What it is.* The text-to-priority signal is genuinely weak. Even TF-IDF is near-random on
`medium` (F1 0.52), and its best class is `high` at 0.59. And there is corroborating
evidence from the source we rejected: the CC0 Kaggle dataset predicted priority from
`customer_tier`, `error_rate_pct`, `downtime_min` and `security_incident_flag` — account
and incident **metadata, not wording**. Its designers did not treat priority as a text
problem either. D23 rejected that dataset for having no text; its schema was telling us
something about the task that we only confirmed empirically 35 GPU-minutes later.

*Means:* F2 as specified — urgency from ticket text alone — has a low ceiling, and the
adapter is not even reaching it. Whether the shortfall against TF-IDF is fixable (rank,
learning rate, or the ~2 supervised tokens per example after prompt masking) or whether a
1.5B decoder is simply the wrong tool for a 3-class problem with weak lexical signal is
**untested**.

*Why this is worth keeping rather than quietly fixing:* it is the second place the project
has found a baseline beating the sophisticated approach, after the router's confidence
baseline in §5. A portfolio that reports those is more credible than one where every
number is good.

**Phase 1 · `epochs: 2` was over-generalised from one task, and it cost a retrain.** The
intent adapter overfitted at epoch 3 in three consecutive runs, so 2 became the global
default. PII and drafting then both finished with validation loss **still falling** and
`best_checkpoint` at the final step — under-trained, not over-trained. PII scored **0.8708
strict against a 0.9942 ceiling**, with recall (0.8475) well below precision (0.8954):
missing spans, not inventing them.

*Means:* intent is 77-way classification over 34-token sequences; PII is extraction of ~7
spans over 160-token sequences. They do not share a convergence profile, and one task's
overfitting point is not evidence about another's. The epoch count belongs in
`TASK_CONFIGS` alongside sequence length and batch size, not as a global default.

*What saved it:* the escalation rule written into `TrainConfig.train_subsample` before
training — *well below ceiling AND train loss still falling → data-limited* — fired
correctly and turned an ambiguous result into a decision. Retraining at 8,000 rows /
3 epochs took strict F1 **0.8708 → 0.9182** and recall **0.8475 → 0.8977**, for ~$0.88.

*And the same rule then said stop.* Val loss moved 0.01226 → 0.01192 across the final
epoch — 2.8%, against 38% for epoch 1→2 — with train loss at 0.003. The curve is flat, so
data and epochs are no longer the binding constraint. The remaining 7.6 points to the
ceiling would need LoRA rank raised from 16, which is a different experiment and not
obviously worth ~$1 at this margin. **Having a stopping rule written down before the
numbers arrived is what made both calls uncontroversial.**

**Phase 1 · Per-task training configs, because one size silently truncates.** PII
sequences run median 160 / p95 429 / max 802 tokens against intent's 34. At the shared
`max_seq_length=128` most PII targets would have been **cut mid-span**, teaching the model
to stop early — a failure that shows up as poor recall and looks like a model problem.

`TASK_CONFIGS` now carries per-task sequence length, batch size and accumulation. PII gets
512 tokens at micro-batch 2 (effective batch still 32), sized from the same memory model
that explained the intent OOM: peak loss memory is batch x seq x 151,936 vocab upcast to
fp32 with a gradient, so batch 4 at p95 would be ~2.1 GB — the figure that already OOM'd a
T4.

*Training volume is now a measurement, not a citation.* PRD §9 says "~3K subsample", but
that figure assumed free-tier-only training — a constraint Gate 0.5 weakened, since it cost
$0.44 of a $20 GPU allocation and proved the rented path. The reason to start at 3,000 is
different and better: 3,000 documents is already ~21,000 span examples over a fixed
19-label vocabulary, it costs 30 free minutes, and **it produces the number that decides
whether more data would help at all**. The escalation rule is written into the config —
if the result sits well below the 0.9942 ceiling *and* train loss is still falling, the
task is data-limited and the full 17,000 is worth ~$1 of rented A10; if it lands near the
ceiling, more rows buy a longer run and nothing else. The split stays frozen at 17,000 and
`train_subsample` caps consumption, so scaling up is a config change, not a re-freeze.

*Also applied:* `epochs` now defaults to **2**, not 3 — validation loss bottomed at epoch 2
in every intent run under two different loss definitions.

**Phase 1 · The PII scorer's ceiling was measured before any training.** Running
ai4privacy's ground truth through the span scorer should score 1.0. Masked-text recovery
scored **0.8973**; `LABEL: value` recovery scored **0.9974**. *Means:* the output format
caps the gated metric, and the obvious format (the one the dataset ships) caps it 10 points
low. *Changes the plan:* D22 — the adapter emits `LABEL: value` lines, built directly from
`privacy_mask` (which carries label, value and offsets), so `masked_text` is not used at
all. Ceiling on the frozen golden set: **0.9942 strict**.

**Phase 0 · Gate 0.5 passed — the headline claim survives contact with hardware.** vLLM
0.29.0 served one Qwen2.5-1.5B base with two LoRA adapters on a single A10: 400 interleaved
requests at concurrency 16, no errors, **159 req/s**, **P50 81 ms / P95 155 ms**.

*The check that mattered:* the two adapters disagreed on **43 of 200** prompts, and scored
0.940 against 0.775. Had vLLM silently resolved both names to one adapter, every other
number would have looked healthy and the gate would have read as passed. The M11
under-trained checkpoint — captured for an unrelated Phase 5 purpose — was the
discriminator.

*Stronger evidence still:* the two adapters' latencies are within **1.5 ms** of each other
at both P50 and P95. A swap-per-request implementation would show a penalty on alternation.
There is none, which means they are genuinely batched together — `PunicaWrapperGPU` in the
startup log names the kernel doing it.

*Means:* "many adapters, one base" is measured, not assumed, and the §7 latency budget has
enormous headroom. The KV cache reported room for **545×** the tested concurrency, so 16 is
nowhere near the limit — the actual ceiling is unmeasured and the honest claim is bounded
by what was tested.

**Phase 0 · First real number: the intent adapter reaches 0.9234 micro-accuracy** on the
frozen 770-example golden set, against a 0.0130 majority-class floor. `exact_label_rate`
is 0.9935 — only 5 of 770 outputs were not a valid label — so failures are wrong intents
rather than malformed generation, and the output format was learned cleanly. Macro-F1 is
0.8686; with exactly 10 examples per class the gap to micro-accuracy means precision
varies across classes rather than a few classes collapsing.

*Caveat, recorded rather than buried:* epoch 3 overfit — val loss rose 0.8600 → 0.8809
while train loss fell to 0.7185. `load_best_model_at_end` was not set, so the saved
adapter was the final epoch, not the best, and `save_total_limit=1` had already deleted
the epoch-2 checkpoint.

*Both flaws are now fixed in `train/qlora.py`* — best-checkpoint selection on `eval_loss`,
and prompt tokens masked to `-100` so the loss scores only the label. Masking required
swapping `DataCollatorForLanguageModeling` for `DataCollatorForSeq2Seq`: the LM collator
rebuilds labels from `input_ids` and silently discards any masking applied upstream.

*Consequence:* the 0.9234 weights were lost when the Colab runtime recycled before the
Hub push, and the current code would not reproduce them anyway. `runs/intent__adapter.json`
is marked stale. **The adapter must be retrained (~27 min, free) and re-measured**, and
the new number is the one that counts. Push to the Hub before anything else next time.

*Means:* nothing about M2 yet. **0.9234 is meaningless until the prompted baseline is
measured** — the same base model with a strong few-shot prompt and no adapter. That
comparison, not the floor, is what decides whether fine-tuning earned its place.

**Phase 0 · A binary PII task cannot be validly constructed from this data — measured
twice, failed twice.** The mirrored ai4privacy split contains **zero negatives**: every
one of 20,000 rows has at least one PII span, median six. A binary "PII present" flag
therefore needs negatives from somewhere, and both available constructions were built and
probed (`uv run adapterops pii-task`, results in `data/pii/CONFOUND.json`).

*Construction 1 — negatives from Bitext.* A classifier that cannot see any PII (positives
replaced by their masked text, all template markers stripped from both sides) separates
the classes at **0.9999**, against 0.5 chance. Length alone reaches 0.779. The task is
almost entirely "which corpus is this", not "does this contain PII". Mitigations were
applied first — Bitext *responses* rather than *instructions*, to close a 47-vs-354
character gap — and made no material difference.

*Construction 2 — negatives from the same documents*, each PII span replaced by a
type-matched surrogate, so no corpus confound is possible. Trained on one surrogate
vocabulary and tested on a disjoint one: **0.9918 → 0.5102**, a drop of 0.4816. The
classifier had learned the twelve surrogate phrases, not PII.

*Means:* F3 as written — "PII/compliance flag" as binary classification — produces an
adapter whose score measures an artifact. Any number it reports would be meaningless, and
would look excellent.

*Changes the plan:* yes — F3 is now span detection (D21), PRD v2.5 changelog 24-25.

**Phase 0 · Two credential-handling near-misses, both fixed.** The OpenAI key was stored
in `.env` wrapped in literal double quotes — 166 characters sent for a 164-character key —
which OpenAI rejected as `invalid_api_key` while echoing the quotes back. Diagnosed by
inspecting the file's shape rather than its contents. More importantly, the fix produced a
`.env.bak`, and `.gitignore` listed only `.env`, which **does not match `.env.bak`** — a
second copy of a live key would have been committable to a public repo. *Means:* the
pattern, not the filename, is what protects a secret. *Changes the plan:* `.gitignore` now
covers `.env` and `.env.*`; `python-dotenv` replaces ad-hoc `split('=')` parsing so quotes
and comments are handled properly.

**Phase 0 · `uv` editable installs were silently inert on this Mac — `site` skips
hidden `.pth` files.** `uv run adapterops` failed with `ModuleNotFoundError` even though
`adapterops.pth` existed in site-packages with the correct path. Root cause: every file
under `.venv` carries the macOS `UF_HIDDEN` flag, and CPython's `site.addpackage` skips
hidden `.pth` files *silently* — no error, no warning. A byte-identical `.pth` created by
hand was honoured; the flagged one was not. `chflags -R nohidden .venv` fixes it.
*Means:* two venvs on this machine had dead editable installs — this one and
`IncidentIQ`'s. **Cause, established on the second investigation:** on this Mac every
*dot-prefixed* file and directory inside the iCloud-synced Desktop carries `UF_HIDDEN`,
recursively — `.git`, `.venv`, `.gradle`, `.idea`, `.vscode`, `.claude`, `.gitignore`,
`.python-version`, across eight unrelated projects — while no non-dot item does
(`src`, `data`, `pyproject.toml`, `STATUS.md`: all clean). `FXICloudDriveDesktop` is `1`
and the iCloud store contains those dot-items. Items created minutes earlier are *not*
yet flagged, which is why an immediate check on a fresh venv looked like a disproof and
was not. *Changes the plan:* yes — see D20. `chflags -R nohidden <venv>` remains the
one-line rescue if it is ever seen again.

**Phase 0 · What was NOT observed — the probe came back negative.** A probe ran 24
samples over 24 minutes: a dot-dir, a dot-file, a normal dir, a normal file and a
`.nosync` dir on the Desktop, against a dot-dir control in `~/Downloads`. **Nothing was
flagged, in any sample, including the Desktop dot-dir.** So the flagging was not
reproduced and the trigger is unknown; a "syncs then flags within minutes" model is ruled
out at this timescale.

What the evidence still supports is a *correlation*, not a mechanism: every dot-prefixed
item on the Desktop older than today is flagged, nothing non-dot is, and nothing created
today is — which is consistent with a trigger on a much longer cycle (daily, login, or a
backup pass) but does not establish one. The attribution to iCloud is therefore
**unproven inference**, and is recorded here as such rather than as a cause.

**This does not weaken the fix.** D20 removes the precondition — a venv with no leading
dot is outside the behaviour whatever produces it. Separately disproved along the way: a
`.nosync` suffix does not exclude anything from this sync — the probe's `skip.nosync`
contents are in the iCloud store.

**Cheap open experiment:** the probe directories are left in place at
`~/Desktop/_flagprobe` and `~/Downloads/_flagprobe`. If the Desktop dot-items are flagged
on a later check and the `~/Downloads` control is not, the trigger is long-cycle and
location-bound, and the iCloud attribution firms up. `ls -ldO ~/Desktop/_flagprobe/.dotdir`
answers it in one command.

**Phase 0 · `PolyAI/banking77` cannot be loaded by modern `datasets`.** It is
script-based (`banking77.py`, no parquet) and `datasets` 5.0.1 rejects loading scripts
entirely; the Hub has no auto-converted parquet branch for it. *Means:* the PRD's named
intent source is unusable as specified. *Changes the plan:* yes — switched to
`mteb/banking77` (D16). PRD corrected in v2.4, changelog 21.

**Phase 0 · `ai4privacy/pii-masking-openpii-1m` is multilingual and far larger than
assumed.** 1,143,397 train + 284,746 validation rows, with `language` and `region`
columns covering at least English and French. *Means:* the PRD's "~3K subsample" is still
right, but the subsample **must filter to English** or it silently violates non-goal 11.
*Changes the plan:* a filter step in data prep, not a scope change. PRD v2.4, changelog 22.

**Phase 0 · All four licenses are permissive — better than assumed.** Qwen2.5-1.5B
Apache-2.0; `mteb/banking77` MIT; Kaggle ticket-priority **CC0 public domain**;
`ai4privacy/pii-masking-openpii-1m` CC-BY-4.0 (declared as `other` + `license_name:
cc-by-4.0`, with the README confirming CC-BY-4.0); Bitext CDLA-Sharing-1.0. *Means:* the
PRD's flagged risk that the ai4privacy variant might restrict portfolio use does not
materialise, and no adapter has to be cut. *Changes the plan:* no — but Bitext's
share-alike attribution still has to go in the README.

**Phase 0 · Bitext confirmed at 26,872 rows**, loads cleanly, columns
`flags / instruction / category / intent / response`. Matches the PRD's 26.8K.

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
