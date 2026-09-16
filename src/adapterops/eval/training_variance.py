"""Training variance for the tasks whose gates wait on it (F33, D37).

Only intent's regression gate is enforced, because only intent has a measured training spread: two
training runs of one configuration differed by 0.0039. Urgency, PII and drafting have thresholds
built on inference noise alone, which D37 marks provisional — inference noise is the smaller
source, so a gate resting on it would block an equivalent retrain.

This measures the missing spreads the same way the gate sees a release: a regression run of the
served adapters, and a regression run of a second training run of each task at the configuration
that produced it (same seed, rows and epochs), served identically in the same session. The spread
per (task, split) is the absolute difference of the gated metric.

**What the spread contains.** Training nondeterminism under a fixed seed; whatever the session's
library versions change, since the box's torch, transformers and peft are not those the pinned
adapters were trained with (each rerun's training record lists them); and the inference noise the
baseline pair already measures. So it is an upper estimate of a same-configuration retrain's
variance, which is the conservative direction for a gate floor.

**Drafting needs the judge first.** Its gated metric is the distilled judge's mean, and the judge
checkpoint is not on the rented box, so both runs are scored with `adapterops judge-score` before
this will read them.

    uv run adapterops training-variance --runs runs/regression__a10-v8-seed0.json \\
        runs/regression__a10-v8-seed11.json runs/regression__a10-v8-seed22.json \\
        --tasks intent urgency pii drafting
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from adapterops.eval.regression import GATED, SPLITS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "runs" / "training_variance.json"


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spreads(runs: Sequence[dict], tasks: Sequence[str]) -> dict:
    """Per task and split: every run's gated metric, their range, and for three or more runs a
    standard deviation. A missing metric is an error, not a skipped row — a silently absent spread
    would leave a gate provisional with nothing saying why.

    The range stays the gate's floor (`spread`): with any number of runs it is the widest gap the
    measurement has actually seen, which is the conservative choice for a threshold."""
    out: dict = {}
    missing = []
    for task in tasks:
        metric = GATED[task]
        for split in SPLITS:
            values = [run["per_split"].get(task, {}).get(split, {}).get(metric) for run in runs]
            if any(v is None for v in values):
                missing.append(f"{task}/{split}")
                continue
            row = {"metric": metric, "runs": len(values), "values": values,
                   "mean": round(sum(values) / len(values), 6),
                   "spread": round(max(values) - min(values), 6)}
            if len(values) >= 3:
                row["stdev"] = round(statistics.stdev(values), 6)
            if len(values) == 2:
                # The keys the two-run record used, so earlier derivations still read.
                row["original"], row["rerun"] = values
            out.setdefault(task, {})[split] = row
    if missing:
        msg = (f"no gated metric for {', '.join(missing)} — score drafting with "
               "`adapterops judge-score --run <run>` on every run first")
        raise ValueError(msg)
    return out


def main(runs: Sequence[str], tasks: Sequence[str], force: bool = False) -> int:
    if len(runs) < 2:
        msg = "a spread needs at least two runs"
        raise ValueError(msg)
    if OUT.exists() and not force:
        print(f"  {_shown(OUT)} exists — --force to replace a measured variance")
        return 1
    paths = [Path(r) for r in runs]
    per_task = spreads([json.loads(p.read_text()) for p in paths], tasks)
    records = {}
    for task in tasks:
        found = [REPO_ROOT / "runs" / f"{task}-rerun__train.json",
                 *sorted((REPO_ROOT / "runs").glob(f"{task}-seed*__train.json"))]
        rows = [{"file": _shown(p), "sha256": _sha(p)} for p in found if p.exists()]
        if rows:
            records[task] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "purpose": "Training spread per (task, split) for the gate floors (F33, D37).",
        "method": ("Regression runs of the served adapters and of further training runs of each "
                   "task at the configuration that produced it, served identically in one "
                   "session; the spread is the range of the gated metric, and three or more runs "
                   "also give a standard deviation. Runs that differ only in seed measure an "
                   "equivalent retrain; a same-seed rerun measures nondeterminism and library "
                   "drift alone. Either way it includes inference noise, so it is an upper "
                   "estimate."),
        "inputs": [{"file": _shown(p), "sha256": _sha(p)} for p in paths],
        "rerun_training_records": records,
        "per_task": per_task,
    }, indent=2) + "\n", encoding="utf-8")
    for task, splits in per_task.items():
        for split, row in splits.items():
            shown = ", ".join(f"{v}" for v in row["values"])
            extra = f" stdev {row['stdev']}" if "stdev" in row else ""
            print(f"  {task:9s} {split:6s} {row['metric']:18s} {shown}  range {row['spread']}{extra}")
    print(f"\n  wrote {_shown(OUT)} — then `adapterops derive-thresholds --run-a ... --run-b ... "
          "--force`")
    return 0
