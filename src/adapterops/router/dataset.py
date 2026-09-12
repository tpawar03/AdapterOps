"""Assemble the router's train and eval sets, with the F31 exclusion (D7).

BUILD-PLAN calls the hard-cases exclusion the single most important line of code in this
phase: get it wrong and every router number is inflated and nothing announces it. Two
versions of it were written, and the first two failed for opposite reasons (D29).

**Attempt 1 — mine from everything, subtract from router training.** The literal reading
of PRD §11. It quietly destroys the *eval* set: hard cases are adapter failures capped at
~150 per task, and holding them out of a 150-row eval slice takes nearly all the failures
with them. On synthetic data at a 22% failure rate the eval set went from 22.5% failures to
2.5% — an operating curve measured on a population with almost nothing left to escalate,
where every policy converges for a reason unrelated to routing.

**Attempt 2 — split first, mine from the training portion only.** Fixes eval and breaks
training. The cap is larger than the number of failures a 400-row training slice contains,
so reserving hard cases left the router's training set at a **1.00 success rate** — no
failures at all, nothing for the router to learn from.

**What both attempts missed** is that failures are the scarce resource and three consumers
need disjoint shares: the router learns from them, the router's eval measures on them, and
the hard-cases split is made of them. That is an allocation problem, and it is solved in
`pool.py` — every pooled pair carries a `purpose`, and mining rows are drawn separately at
freeze time.

So this module has no subtraction step at all. Router rows split into train / eval /
shift-eval; hard candidates come from mining rows. The exclusion holds because the two
populations were never the same rows, which is the only version of it that cannot be
skipped by accident.

    uv run adapterops router-dataset
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from adapterops.router.scoring import FAILURE_BUCKETS, failure_type

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORED_FILE = REPO_ROOT / "data" / "router" / "scored.parquet"
SHIFT_FILE = REPO_ROOT / "data" / "router" / "shift.parquet"
OUT_DIR = REPO_ROOT / "data" / "router"
MANIFEST = REPO_ROOT / "evals" / "ROUTER_DATASET.json"

SEED = 20260909
EVAL_FRACTION = 0.20
HARD_CAP_PER_TASK = 150
"""PRD §9: ~150 mined per task, ~100 retained after the Phase 4 adjudication screen."""


def split_in_distribution(rows: pd.DataFrame, eval_fraction: float = EVAL_FRACTION,
                          seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified on (task, success) so train and eval carry the same base rate.

    Without the stratification the eval set's failure rate drifts from training's by a few
    points on a 150-row-per-task split, and a policy comparison read against a different
    base rate is not reading the same problem.
    """
    held_index = [
        i
        for _, group in rows.groupby(["task", "success"], sort=True)
        for i in group.sample(frac=eval_fraction, random_state=seed).index
    ]
    held = rows.loc[held_index]
    return rows.drop(index=held.index), held


def mine_hard_candidates(mining: pd.DataFrame, cap: int = HARD_CAP_PER_TASK,
                         seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Capped, bucket-balanced adapter failures from the mining slice (F31).

    The cap will not always bind, and that is a measurement rather than a shortfall: 150
    hard cases per task requires ~150 failures per task, and an adapter scoring 0.93 on
    intent does not produce them from 700 pairs. Where the cap does not bind, the size of
    the hard-cases split is a statement about the adapter, and `cap_bound` records which
    case each task is in.
    """
    failures = mining[~mining.success.astype(bool)].copy()
    failures["failure_type"] = [failure_type(r) for _, r in failures.iterrows()]

    picked, report = [], {}
    for task, g in failures.groupby("task"):
        buckets = [b for b in FAILURE_BUCKETS[task] if (g.failure_type == b).any()]
        per_bucket = max(1, cap // max(len(buckets), 1))
        take = pd.concat([
            g[g.failure_type == b].sample(n=min(per_bucket, int((g.failure_type == b).sum())),
                                          random_state=seed)
            for b in buckets
        ]) if buckets else g.head(0)
        # Buckets rarely divide the cap evenly; top up from what is left, largest first,
        # so an under-full bucket does not cost the split rows it could have had.
        if len(take) < min(cap, len(g)):
            rest = g.drop(index=take.index).sample(frac=1.0, random_state=seed)
            take = pd.concat([take, rest.head(min(cap, len(g)) - len(take))])
        picked.append(take)
        report[task] = {
            "failures_in_mining_slice": len(g),
            "mining_rows": int((mining.task == task).sum()),
            "failure_rate": round(float(len(g) / max((mining.task == task).sum(), 1)), 4),
            "reserved": len(take),
            "cap": cap,
            "cap_bound": len(take) >= cap,
            "by_bucket": take.failure_type.value_counts().to_dict(),
        }

    return (pd.concat(picked) if picked else mining.head(0)), report


def build(scored: pd.DataFrame, shift: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    router = scored[scored.purpose == "router"].merge(
        shift[["pair_id", "side"]], on="pair_id", validate="one_to_one")
    mining = scored[scored.purpose == "mining"].copy()

    shift_eval = router[router.side == "shift"].copy()
    train, eval_set = split_in_distribution(router[router.side == "in_distribution"].copy())
    hard, hard_report = mine_hard_candidates(mining)

    splits = {"train": train, "eval": eval_set, "shift_eval": shift_eval,
              "hard_candidates": hard}
    stats = {
        name: {
            "rows": len(frame),
            "success_rate": round(float(frame.success.mean()), 4) if len(frame) else None,
            "per_task": {t: int(n) for t, n in frame.task.value_counts().items()},
        }
        for name, frame in splits.items()
    }
    return splits, {"splits": stats, "hard_candidates": hard_report}


def verify(splits: dict[str, pd.DataFrame]) -> dict:
    """The checks that would catch the leak. Every count here must be zero."""
    ids = {name: set(frame.pair_id) for name, frame in splits.items()}
    return {
        "hard_candidates_in_train": len(ids["hard_candidates"] & ids["train"]),
        "hard_candidates_in_eval": len(ids["hard_candidates"] & ids["eval"]),
        "hard_candidates_in_shift_eval": len(ids["hard_candidates"] & ids["shift_eval"]),
        "train_failures_missing": int(splits["train"].success.all()),
        "train_in_eval": len(ids["train"] & ids["eval"]),
        "train_in_shift_eval": len(ids["train"] & ids["shift_eval"]),
        "eval_in_shift_eval": len(ids["eval"] & ids["shift_eval"]),
    }


def main(force: bool = False) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — the router dataset is frozen.")
        return 1
    if not SCORED_FILE.exists():
        print(f"  no {SCORED_FILE.relative_to(REPO_ROOT)} — the pool has not been scored.")
        print("  Run scripts/phase2_serve.sh and adapterops.router.generate on a GPU box.")
        return 2

    splits, stats = build(pd.read_parquet(SCORED_FILE), pd.read_parquet(SHIFT_FILE))
    checks = verify(splits)
    if any(checks.values()):
        msg = f"router dataset leaks and was not written: {checks}"
        raise RuntimeError(msg)

    files = {}
    for name, frame in splits.items():
        path = OUT_DIR / f"router_{name}.parquet"
        frame.reset_index(drop=True).to_parquet(path)
        files[name] = {"file": str(path.relative_to(REPO_ROOT)), "rows": len(frame),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"  {name:16s} {len(frame):>5,} rows · "
              f"success {stats['splits'][name]['success_rate']}")

    MANIFEST.write_text(json.dumps({
        "purpose": "Router train/eval assembly with the F31 hard-cases exclusion (D7/D29).",
        "regenerate": "uv run adapterops router-dataset --force",
        "seed": SEED,
        "exclusion": "structural, not subtractive — router and mining rows are drawn as "
                     "disjoint slices at pool-freeze time (D29). Nothing is removed here.",
        "exclusions_verified": checks,
        "files": files,
        **stats,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  all exclusion checks zero · froze {MANIFEST.relative_to(REPO_ROOT)}")
    return 0
