"""Router training pool (F7) — the (ticket, task) pairs Phase 2 scores through the adapters.

The router's label is computed, not annotated: run a pair through its adapter, score the
output, and the pair is a success or a failure. That makes the pool's *provenance* the
only thing standing between Phase 2 and a set of numbers that look fine and mean nothing,
because both ways of getting it wrong inflate the result and neither raises an error.

**Golden-set exclusion.** A pool row that is also a golden-set row puts an eval item into
the router's training data. Measured here rather than assumed: the count goes in the
manifest even when it is zero.

**Adapter-training exclusion.** The subtler one. The adapters were fine-tuned on
`data/<task>/split_train.parquet`; on a row they were trained on they answer from memory,
so the success rate the router learns is not the success rate it will meet at serving
time. The pool is therefore drawn from `split_val.parquet` — held out from both the
adapter and the golden set — and rows whose text *also* appears in the training split are
dropped on top of that. That last filter is not belt-and-braces on drafting: it removes
467 rows, because the frozen drafting split is group-aware only where golden was carved
out (D24).

A third filter drops texts that repeat inside the source split itself. It exists because
the first draw took both copies of a duplicated drafting instruction: two identical texts
either side of the router's own train/eval split is the same leak one level down.

**Not stratified, deliberately.** The golden sets are stratified on the task label because
their gated metrics demand it. Here the label is adapter success, and stratifying on the
task label would reshape the success base rate the router exists to learn. The pool is a
plain seeded sample, so `medium`-heavy urgency and the long tail of intents arrive in the
proportions the router will actually see.

**The third exclusion is not here.** Hard-cases items (F31/D7) cannot be removed at this
stage — which pairs those are is unknown until the pool has been scored. That cut happens
in `adapterops.router.dataset`, after scoring.

Run:  uv run adapterops router-pool
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adapterops.train.qlora import COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = REPO_ROOT / "data" / "router"
POOL_FILE = POOL_DIR / "pool.parquet"
MANIFEST = REPO_ROOT / "evals" / "ROUTER_POOL.json"

SEED = 20260909
"""Same seed as the mirror and the splits. One number to change, one thing it means."""

PER_TASK = 750
"""4 x 750 = 3,000 pairs, the volume PRD §9 allocates to the router. Every task gets the
same count: the router takes task identity as an input feature (F10), and an unbalanced
pool would let it learn a per-task prior instead of reading the text."""

TASKS = ("intent", "urgency", "pii", "drafting")


@dataclass(frozen=True)
class TaskPool:
    task: str
    text_column: str
    gold_column: str
    source: str
    source_rows: int
    excluded_in_golden: int
    excluded_in_adapter_train: int
    excluded_duplicate_text: int
    eligible: int
    sampled: int

    def as_dict(self) -> dict:
        return {**self.__dict__}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _texts(path: Path, column: str) -> set[str]:
    return set(pd.read_parquet(path)[column].astype(str))


def build_task(task: str, per_task: int = PER_TASK) -> tuple[pd.DataFrame, TaskPool]:
    """Draw one task's pairs, applying both exclusions and counting what each removed."""
    text_col, gold_col = COLUMNS[task]
    source = REPO_ROOT / "data" / task / "split_val.parquet"
    df = pd.read_parquet(source).reset_index(names="source_row")
    df[text_col] = df[text_col].astype(str)

    golden = _texts(REPO_ROOT / "evals" / "golden" / f"{task}.parquet", text_col)
    trained = _texts(REPO_ROOT / "data" / task / "split_train.parquet", text_col)

    in_golden = df[text_col].isin(golden)
    in_trained = df[text_col].isin(trained)
    keep = df[~(in_golden | in_trained)]

    # Third filter, found by the test rather than by design: 92 of drafting's val rows
    # repeat an instruction, and the first draw took one of those pairs. Two identical
    # texts straddling the router's own train/eval split is the same leak one level down.
    deduped = keep.drop_duplicates(subset=text_col)
    n_duplicate = len(keep) - len(deduped)
    keep = deduped

    if len(keep) < per_task:
        msg = (f"{task}: {len(keep)} eligible rows after exclusions, need {per_task}. "
               f"Lower PER_TASK or widen the source — do not relax an exclusion.")
        raise ValueError(msg)

    drawn = keep.sample(n=per_task, random_state=SEED).sort_values("source_row")
    pairs = pd.DataFrame({
        "pair_id": [f"{task}-{i:04d}" for i in range(per_task)],
        "task": task,
        "text": drawn[text_col].to_numpy(),
        "gold": drawn[gold_col].astype(str).to_numpy(),
        "source_file": str(source.relative_to(REPO_ROOT)),
        "source_row": drawn["source_row"].to_numpy(),
    })
    meta = TaskPool(
        task=task,
        text_column=text_col,
        gold_column=gold_col,
        source=str(source.relative_to(REPO_ROOT)),
        source_rows=len(df),
        excluded_in_golden=int(in_golden.sum()),
        excluded_in_adapter_train=int(in_trained.sum()),
        excluded_duplicate_text=n_duplicate,
        eligible=len(keep),
        sampled=per_task,
    )
    return pairs, meta


def verify(pool: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Re-check a built pool against the golden sets and the adapters' training splits.

    Deliberately independent of `build_task`: it reads the frozen parquet back and asks
    the question again from the source files, so a bug in the exclusion logic cannot hide
    behind the same bug in the check. Every count it returns must be zero.
    """
    report: dict[str, dict[str, int]] = {}
    for task in sorted(pool.task.unique()):
        text_col, _ = COLUMNS[task]
        texts = pool.loc[pool.task == task, "text"].astype(str)
        report[task] = {
            "in_golden": int(texts.isin(
                _texts(REPO_ROOT / "evals" / "golden" / f"{task}.parquet", text_col)).sum()),
            "in_adapter_train": int(texts.isin(
                _texts(REPO_ROOT / "data" / task / "split_train.parquet", text_col)).sum()),
            "duplicate_texts": int(len(texts) - texts.nunique()),
        }
    return report


def main(force: bool = False, per_task: int = PER_TASK) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — the pool is frozen.")
        print("  Re-drawing it changes which pairs the router trains on and which")
        print("  failures the hard-cases split is mined from. --force if you mean it.")
        return 1

    frames, metas = [], []
    for task in TASKS:
        pairs, meta = build_task(task, per_task)
        frames.append(pairs)
        metas.append(meta)
        print(f"  {task:9s} val {meta.source_rows:>5,} "
              f"- golden {meta.excluded_in_golden:>3} "
              f"- in-adapter-train {meta.excluded_in_adapter_train:>3} "
              f"- dup {meta.excluded_duplicate_text:>3} "
              f"= eligible {meta.eligible:>5,} -> sampled {meta.sampled:,}")

    pool = pd.concat(frames, ignore_index=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(POOL_FILE)

    checks = verify(pd.read_parquet(POOL_FILE))
    bad = {t: c for t, c in checks.items() if c["in_golden"] or c["in_adapter_train"]}
    if bad:
        POOL_FILE.unlink()
        msg = f"pool leaks and was not frozen: {bad}"
        raise RuntimeError(msg)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "purpose": "Frozen router pool (F7). Re-drawing it moves the router's training "
                   "data and the population the hard-cases split is mined from.",
        "regenerate": "uv run adapterops router-pool --force",
        "seed": SEED,
        "pairs": len(pool),
        "file": str(POOL_FILE.relative_to(REPO_ROOT)),
        "bytes": POOL_FILE.stat().st_size,
        "sha256": _sha256(POOL_FILE),
        "drawn_from": "data/<task>/split_val.parquet — held out from both the adapters "
                      "and the golden sets",
        "exclusions_verified": checks,
        "not_excluded_here": "hard-cases items (F31) — unknown until the pool is scored; "
                             "cut in adapterops.router.dataset",
        "tasks": [m.as_dict() for m in metas],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\n  froze {POOL_FILE.relative_to(REPO_ROOT)} ({len(pool):,} pairs)")
    print(f"  froze {MANIFEST.relative_to(REPO_ROOT)} — all exclusion checks zero")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(force="--force" in sys.argv))
