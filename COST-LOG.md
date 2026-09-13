# Cost log

F24 (P0). Every session, every charge. Hard ceiling **$50** (M9); stop and reassess
at **$25** (§14). Update this and STATUS.md §1 together.

**Running total: ~$10 / $50.00** — as reported; the Phase 2 and Phase 4 GPU sessions are not itemised

| Phase | Provider | What | Duration | Charge | Total |
|---|---|---|---|---|---|
| 0 | HuggingFace | Mirror 3 of 4 datasets (26.6 MB) | — | $0.00 | $0.00 |
| 0 | OpenAI | `models.list()` auth check — free endpoint | — | $0.00 | $0.00 |
| 0 | Colab | Intent adapter QLoRA training, free T4 | 27 min | $0.00 | $0.00 |
| 0 | Lambda | **Gate 0.5** — A10 24GB, vLLM multi-LoRA | ~35 min | ~$0.44 | ~$0.44 |
| 1 | Lambda | Train PII + drafting adapters, A10 | ~50 min | ~$0.63 | ~$1.07 |
| 1 | Lambda | Retrain PII, 8K rows / 3 epochs + eval | ~70 min | ~$0.88 | ~$1.95 |
| 2 | OpenAI | Frontier escalation arm (F8), `gpt-4o-mini` — all 5,800 pairs; resumer finished | ~5 h wall, daily-quota polling | $0.33 | $2.28 |
| 3 | OpenAI | GPT-4o judge grades (F13) — 1,950 replies, 1,200 adapter + 750 frontier; token-derived at list price | ~75 min | $2.40 | $4.68 |
| 3 | OpenAI | GPT-4o grades for the remaining 250 mining-slice drafting replies (D38) | ~3 min | $0.30 | $4.98 |
| 4 | Lambda | **Phase 4–5 session** (`PHASE-4-RUN.md`), A10 — F21 shuffled adapter, 2 F33 baselines, 4 M2 prompted baselines, M11, M7, 2 re-runs · approved at $1.50 | ~1 h+ | in the GPU row below | — |
| 4 | OpenAI | GPT-4o grades on golden drafting for M2 (`judge-m2`) — 300 adapter + 300 prompted replies · approved at $1.50 after a $1.28 projection · token-derived at list price | ~25 min | $0.95 | $5.93 |
| 2 + 4 | Lambda | **GPU sessions for Phase 2 (M1 serving, router pool) and Phase 4–5, not itemised.** The account total was reported as about $10; this row is the remainder over everything itemised above | — | **~$4.07** (derived) | **~$10** |

## Budget envelope (PRD §12)

| Line | Allocated |
|---|---|
| GPU rental (RunPod / Vast.ai A10G spot) | $20 |
| API — GPT-4o judge labels ≈$4, GPT-4o-mini escalation/ceiling ≈$6, adjudication <$1 | $10 |
| Contingency demo hosting (paid HF Space) | $9 |
| Buffer for failed runs | $11 |
| **Ceiling** | **$50** |

## Rules

- **One run at a time.** Two frontier runs were once started concurrently, in two shells.
  Nothing corrupted, but both drew on the same daily request quota and 199 pairs were
  called — and billed — twice, for $0.017. The run now takes a lock file.
- **Dollars are not the only exhaustible resource.** The frontier run stopped at 3,499 of
  5,800 pairs on a *requests-per-day* ceiling (10,000/day), having spent $0.16. A spend cap
  cannot see that coming; the run now counts requests as well as tokens.
- **Tokens per minute binds for long inputs.** `gpt-4o` on this account allows 30,000 TPM.
  The M2 grading sent ~660-token prompts on 4 workers: 465 grades took 1,202 requests before
  its request cap stopped it. One worker finished the last 135 in 166. Rejected requests are not
  billed, but they spend the daily request quota.
- Log **before** the money is spent where possible; a forgotten session is how the
  ceiling gets breached.
- **Kill the GPU when you stop.** Spot billing is per second and an idle box is the
  most likely way to lose the budget.
- ~~Set a hard billing cap on the OpenAI account (day-1 check 8).~~ **Done — $15 cap set.**
  This is the backstop, not the budget: §12 allocates ~$10 to API across all three uses.
