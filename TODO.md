# To do — every known limitation, as work

Each item names the limitation it addresses, what closing it takes, and what it costs. Items are grouped by
**when they can run**, not only by topic: free work first, then work to bundle into one GPU session (measure
the free things before booking the GPU, so a fix can share the session), then work waiting on data or a
decision, then limitations accepted by design. Results go to `STATUS.md` as usual; tick an item only when a
committed run or change closes it.

Cost key: **free** (this laptop, no spend) · **GPU** (rented A10 at ~$0.75/h) · **API** (OpenAI spend) ·
**public** (publishes something) · **data** (needs a dataset the project does not have) · **you** (needs your
account, keys or decision).

---

## 1 · Now — free, on this laptop

- [x] **State the scope limitation in the README.** Nothing shows the adapters or the router generalise
  beyond the four training datasets (banking queries, IT tickets, synthetic PII documents, templated retail
  replies); no real traffic was run. *Free.*
- [x] **PII error analysis.** Done — `adapterops pii-errors`, `runs/pii__errors.json`. On the 300 golden
  documents the served adapter is strictly right on 210 but fully masks the personal text in 253; errors are
  mostly ID-number and GENDER/SEX labels (39 spans) and names split differently (53 of 59 boundary errors);
  19 gold spans (0.8%) are left wholly unmasked. See STATUS.
- [x] **Report leakage beside span F1.** Done: PII regression scoring records it, the served run is backfilled
  (84.3% of golden documents fully masked, 0.79% of spans wholly unmasked), and the dashboard and model card show
  it. The Hub card is not re-pushed. Strict span F1 counts a harmless relabel the same as a leak. Add the
  redaction numbers — documents fully masked, gold spans partly and wholly unmasked — to regression runs, the
  dashboard and the PII model card, beside the gated metric. *Free.*
- [x] **Check the two label confusions against the text.** Done: ambiguous in the data. "Male"/"Female" have no
  cue word in 908 of 1,727 training spans; without a cue an ID's format gives its label about half the time; 139
  of 168 confused spans have no cue within 20 characters, and relabelling by cue word breaks more than it fixes
  (6 fixed, 14–27 broken; `adapterops pii-relabel`). PII runs now report `span_f1_grouped` beside strict (with
  the name item below): served v6 golden 0.9442 → 0.9862, hard 0.8785 → 0.9656. The gate stays on strict. IDCARDNUM vs DRIVERLICENSENUM (and the other ID
  types) and GENDER vs SEX account for most label errors. Test whether the gold label is recoverable from the
  text at all — value formats, cue words — or is noise the adapter cannot learn, and adjudicate as for D30.
  *Free.*
- [x] **Check the name-splitting convention.** Done: inconsistent. Invented three-word names are split
  given-given-surname 2,372 times and given-surname-surname 1,755 times in training. `span_f1_grouped` joins a
  name's adjacent parts into one span. 53 of 59 boundary errors are multi-word names split differently
  between GIVENNAME and SURNAME. Measure how consistently the dataset splits such names, and whether a
  name-level score (the union of both labels) is the fairer measure. *Free.*
- [x] **A guard for confidently wrong answers (PII).** Done: `adapterops pii-guard`; the request path now adds
  pattern-matched identifiers and dates the PII answer left untagged. Golden: wholly unmasked spans 19 → 12,
  strict F1 0.9442 → 0.9447; 0 of 1,070 customer texts gain a false span. No effect on the hard split. Confidence routing cannot flag PII answers that are wrong
  but confident. Aim a cheap check at what leaks — the 54 gold spans left partly unmasked — rather than at
  strict failures: value format validation, spans not found in the text, label-vocabulary checks. Measure how
  many leaks it catches and how many right answers it would wrongly escalate. *Free.*
- [x] **Per-task thresholds.** Tested, not adopted: `adapterops router-per-task` picks thresholds on training
  pairs; held out they tie the pooled threshold in distribution (0.000, 95% CI ±0.013) and gain +0.007 under
  shift (CI −0.002 to +0.015). The pooled threshold stays. The single 0.394 confidence threshold was calibrated on the pooled pairs.
  Measure per-task operating curves on the recorded pools and whether per-task thresholds beat the pooled one
  at the same 20% budget. *Free.*
- [x] **A margin guard for urgency.** Done: promotion blocks any candidate below the prompted base model's
  recorded golden score (intent, urgency, PII). Urgency's gate threshold (0.0381) is wider than its 0.0134 margin over
  the prompted baseline, so a release that lost that edge would pass. Add a promotion check that reports —
  or blocks on — a candidate falling below the prompted baseline, separate from the variance gate. *Free.*
- [x] **Frontier request budgeting in the request path.** Done: `serve-api --frontier-per-minute 400
  --frontier-per-day 8000` by default; pairs over budget stay local and are counted. Not yet load-tested. At 387 GPT-4o-mini calls a minute the account's
  10,000-requests-a-day cap lasts ~26 minutes. Add a per-minute and per-day frontier budget to `serve-api`,
  with pairs over budget answered locally and counted, and test it. *Free.*
- [x] **Hard split from several models' failures.** Done: `adapterops hard-shared` froze
  `evals/hard/shared_failures.parquet` (540 golden items) from failures of systems M11 never compares:
  GPT-4o-mini, plus the v6 PII adapter for PII. It flags M11 on all four tasks (drops intent +0.27, urgency +0.11,
  PII +0.22, drafting +0.18, each over 3× the baseline spread), where the first split scored M11 higher on three.
  Intent, urgency and drafting still rest on one source; a second needs the prompted base on golden (GPU or a slow
  laptop run). The first hard split stays pinned. The hard split was mined from the first adapter's own
  failures, so it rewards any different model and cannot gate. Rebuild it from failures several systems share
  (served adapters, M11 checkpoints, GPT-4o-mini on the golden items), frozen before any gated model, and
  re-run M11's sensitivity check against it. *Free* with recorded outputs; *GPU* if new predictions are needed.
- [x] **Push the v7 PII model card and redeploy the Space.** Done on retry after the Hub outage: PII card
  commit `b4eaf06d` (all four cards pushed) and Space `fb88c223`; `verify-pins` still clean.
- [ ] **Record billed costs.** Every cost figure is derived from an assumed $0.75/h. Add the actual Lambda and
  OpenAI charges per session to STATUS. *You* (invoices).

## 2 · Next GPU session — bundle these

- [ ] **Re-measure on vLLM what was only measured on the laptop:** the PII false-positive rates (928 and 491
  texts), the router PII re-check, and the int8 accuracy comparison. *GPU*, ~30 min.
- [ ] **PII training spread for the served configuration.** Its 0.0081 gate threshold still comes from the
  previous configuration's retrain. Retrain the 13,910-row configuration once more and regression-score it
  beside the served adapter in one session. *GPU*, ~2 h.
- [ ] **More than one retrain per task.** *Prepared, waiting on a GPU:* training takes `--seed` now, and
  `training-variance --runs A B C ...` reports a range plus a standard deviation from three or more runs
  (the range stays the gate's floor, and the 3× multiplier does not move without a decision).
  `scripts/gpu_variance_seeds_session.sh` retrains every task at further seeds and scores each set
  beside the served adapters in one session: `SEEDS="11 22"` ≈ 7 h for a three-value estimate,
  `SEEDS="11"` ≈ 3.5 h for a second range. Recorded spreads come from two runs at one seed, which
  measure nondeterminism and library drift rather than an equivalent retrain. *GPU.*
- [ ] **Urgency against TF-IDF.** *Measured under a frozen rule (`adapterops urgency-tfidf`), and the rule
  says serve TF-IDF:* golden macro F1 **0.5400 against the adapter's 0.4212** in the same session
  (+0.1188, gate 0.0381), on the shared-failure items 0.4442 against 0.3332, and 0.06 ms per text
  against the adapter's ~60 ms. **Waiting on your decision:** serve TF-IDF for urgency behind the same
  interface (free, but the manifest pins Hub revisions and a local sklearn model needs a component
  type), or first try the untested LoRA fixes — rank, learning rate, more supervised tokens per
  example — at *GPU* ~1 h per attempt. *You* + *free* or *GPU*.
- [x] **Train PII on realistic formats.** Done: the candidate passes both halves of the frozen rule — golden strict F1 drop 0.0000 (gate 0.0081) and TAB leaks 14.31% → 8.32%, CI [−0.0897, −0.0323]. Nemotron-PII 28.31% → 3.99%. Held-out false positives went 0 → 4 of 491 (report-only). **Not yet served:** publishing the adapter and promoting a manifest are public steps, awaiting approval. *Was prepared as:* `adapterops pii-realistic-split` froze
  `data/pii/split_train_realistic.parquet` (sha be3aaee0: the served 13,910 rows + 4,000 Nemotron-PII documents)
  with its decision rule — golden gate, and TAB's leak rate falling with a paired-bootstrap interval below zero
  (`pii-realistic-compare`). Run `scripts/gpu_pii_realistic_session.sh` after pushing the split. Served v6 leaks 28% of in-scope spans on Nemotron-PII and 14% on real
  court text, and confidence routing escalates none of them. Pre-register a retrain that adds Nemotron-PII's train
  split (CC BY 4.0; map the 16 matching categories, keep the uid-disjoint test sample out) to the D50 mix, and
  require golden strict F1 inside the 0.0081 gate while `runs/pii__realistic.json`'s leak rates fall. *GPU*, ~2 h.
- [ ] **PII's 5 remaining false positives.** Add negatives with misspellings and odd tokens ("Atm", "Chevk")
  and re-measure, inside the same decision rule as D50. *GPU*, ~2 h; low priority.
- [ ] **P95 latency for PII and drafting.** Both miss §7's 500 ms (2,259 ms and 3,000 ms). Measure the
  options — tighter token caps, streaming the first token, speculative decoding — or decide per-task targets
  and record the change in the PRD rather than moving the bar silently. *GPU.*
- [ ] **Sustained load with GPT-4o-mini answering.** Only 5 minutes were run. Run a longer test once frontier
  budgeting exists, within the daily request cap. *GPU* + *API*.
- [ ] **int8 in serving.** int8 was compared for accuracy on CPU only. Serve a weight-only int8 base in vLLM
  and measure latency, memory and accuracy. *GPU*; low priority — the project makes no int8 claim.

## 3 · Waiting on data, keys or a decision

- [ ] **Out-of-domain evaluation.** Assemble and label a few hundred tickets from a source none of the four
  datasets came from; run them through the request path; measure per-task accuracy, how often answers fall
  outside the label set, and whether the confidence rule escalates the wrong answers. *Data* + *API* for
  labels + human spot-checks. The routing threshold's out-of-domain check depends on this.
- [x] **PII on realistic formats.** Done: `adapterops pii-realistic`, served v6 locally on 200 Nemotron-PII texts
  (CC BY 4.0) and 200 real ECHR paragraphs from TAB (MIT). In-scope spans wholly unmasked: 28.3% and 14.3%
  (17.8% and 12.4% with the request-path guard), against 0.79% on golden; texts fully masked 42.5% and 48%. Masked
  characters stay on personal data (98.7%, 93.6%). No empty or capped answers; leaks grow with length. **No leaking
  text escalates** (highest score 0.12 and 0.32, threshold 0.39). Fix queued in §2. PII was trained and tested on synthetic spans. Find a permissively
  licensed PII set with real-world formats (names, IDs, addresses from several regions) and measure recall and
  precision. *Data.*
- [ ] **A judge that tracks GPT-4o on any generator.** The distilled judge tracks GPT-4o on adapter-like
  replies (Spearman 0.74) but not on another generator's (0.33). Grade a mixed-generator set with GPT-4o,
  retrain or recalibrate, and re-measure both correlations. *API* (~$2–4) + *free* training.
- [ ] **Langfuse tracing against a live project.** Tracing has only been tested against a mock transport.
  *You* (Langfuse keys in `.env`), then free.
- [ ] **Unattended regression runs.** Regression checks are on-demand because GitHub's free runners have no
  GPU. Options: a scheduled cloud GPU job started by CI, or a self-hosted runner. *You* (budget decision) +
  *GPU*.
- [ ] **Urgency's non-commercial licence.** The urgency adapter inherits CC BY-NC from its dataset. Decide
  to keep it, or find a permissively licensed urgency dataset and retrain. *You* + *data* + *GPU*.
- [ ] **A live public demo.** The public Space shows recorded outputs because Hugging Face requires PRO to
  host Gradio. Decide whether a live demo is worth paying for or hosting elsewhere. *You* + *public*.
- [ ] **Load beyond one A10.** Throughput and cost were measured on a single A10. Only worth measuring if a
  deployment target is chosen. *You* + *GPU*.

## 4 · Accepted by design — documented, not planned

- **English only** (PRD non-goal 11).
- **Public or synthetic data only; no real users or customer data** (portfolio project).
- **The learned router** lost to the adapter's own confidence, and three pre-registered fixes also lost (D42).
  Kept as a recorded negative result; revisit only with new data from the out-of-domain evaluation.
- **Costs are derived for a load pattern, not a deployment:** local is cheaper than GPT-4o-mini only above
  3.64 requests per second sustained.
