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

    uv run adapterops training-variance --original runs/regression__a10-v4-original.json \\
        --rerun runs/regression__a10-v4-rerun.json --tasks urgency pii drafting
"""

from __future__ import annotations

import hashlib
import json
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


def spreads(original: dict, rerun: dict, tasks: Sequence[str]) -> dict:
    """Per task and split: both runs' gated metric and their absolute difference. A missing metric
    is an error, not a skipped row — a silently absent spread would leave a gate provisional with
    nothing saying why."""
    out: dict = {}
    missing = []
    for task in tasks:
        metric = GATED[task]
        for split in SPLITS:
            a = original["per_split"].get(task, {}).get(split, {}).get(metric)
            b = rerun["per_split"].get(task, {}).get(split, {}).get(metric)
            if a is None or b is None:
                missing.append(f"{task}/{split}")
                continue
            out.setdefault(task, {})[split] = {"metric": metric, "original": a, "rerun": b,
                                               "spread": round(abs(a - b), 6)}
    if missing:
        msg = (f"no gated metric for {', '.join(missing)} — score drafting with "
               "`adapterops judge-score --run <run>` on both runs first")
        raise ValueError(msg)
    return out


def main(original: str, rerun: str, tasks: Sequence[str], force: bool = False) -> int:
    if OUT.exists() and not force:
        print(f"  {_shown(OUT)} exists — --force to replace a measured variance")
        return 1
    paths = [Path(original), Path(rerun)]
    runs = [json.loads(p.read_text()) for p in paths]
    per_task = spreads(*runs, tasks)
    records = {}
    for task in tasks:
        record = REPO_ROOT / "runs" / f"{task}-rerun__train.json"
        if record.exists():
            records[task] = {"file": _shown(record), "sha256": _sha(record)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "purpose": "Training spread per (task, split) for the gate floors (F33, D37).",
        "method": ("The served adapters and a second training run of each task at the "
                   "configuration that produced it, regression-scored in one session; spread is "
                   "the absolute difference of the gated metric. Includes library-version drift "
                   "and inference noise, so it is an upper estimate."),
        "inputs": [{"file": _shown(p), "sha256": _sha(p)} for p in paths],
        "rerun_training_records": records,
        "per_task": per_task,
    }, indent=2) + "\n", encoding="utf-8")
    for task, splits in per_task.items():
        for split, row in splits.items():
            print(f"  {task:9s} {split:6s} {row['metric']:18s} {row['original']} → {row['rerun']}"
                  f"  spread {row['spread']}")
    print(f"\n  wrote {_shown(OUT)} — then `adapterops derive-thresholds --run-a ... --run-b ... "
          "--force`")
    return 0
