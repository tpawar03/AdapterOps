"""Deriving the promotion gate's thresholds (F33, D37) from the noise that actually matters.

F33 says: run the regression twice against an unchanged manifest, measure the spread, and set
the threshold to a multiple of it. D37 records why that is not enough on its own. Two runs of one
model measure *inference* noise, which under greedy decoding is close to zero. What decides
whether a retrained-but-equivalent adapter should pass is *training* variance — measured on
intent at ±0.0039 and unmeasured everywhere else.

So each (task, split) gets:

- the inference spread between the two baseline runs;
- the training spread where it has been measured, read from the committed run record, and
  `None` — never a number borrowed from intent — where it has not;
- a floor that is the larger of the two, recording which one bound;
- a threshold of `MULTIPLIER` × floor.

**A zero floor produces no threshold, not a zero threshold.** A zero threshold blocks any drop at
all — the unfalsifiable gate §11 removed. Where nothing has set a floor, the split stays
report-only and the derivation says so.

**A threshold resting on inference noise alone is marked provisional and not enforced.** Inference
noise is the smaller of the two sources, so a threshold built only from it is the one most likely
to block an equivalent retrain. It is reported, and it waits for a training measurement before it
gates anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adapterops.eval.regression import GATED, SPLITS

REPO_ROOT = Path(__file__).resolve().parents[3]
INTENT_RUN = REPO_ROOT / "runs" / "intent__adapter.json"
OUT = REPO_ROOT / "evals" / "GATE_THRESHOLDS.json"

MULTIPLIER = 3.0
"""A stated choice: three times the larger measured spread. With two runs a spread is a single
range rather than a variance estimate, and three ranges is a conservative margin over it. It is
recorded in every derivation so it can be argued with."""


def training_spreads() -> dict[tuple[str, str], dict]:
    """Measured training variance, read from the run records rather than copied into code."""
    out: dict[tuple[str, str], dict] = {}
    if INTENT_RUN.exists():
        record = json.loads(INTENT_RUN.read_text()).get("reproducibility", {})
        if record.get("spread_micro") is not None:
            out[("intent", "random")] = {
                "spread": float(record["spread_micro"]),
                "source": f"{INTENT_RUN.relative_to(REPO_ROOT)} — "
                          f"{record.get('independent_runs', 2)} independent training runs",
            }
    return out


def derive(run_a: dict, run_b: dict, training: dict | None = None,
           multiplier: float = MULTIPLIER) -> dict:
    training = training_spreads() if training is None else training
    out: dict = {}
    for task, metric in GATED.items():
        for split in SPLITS:
            a = run_a["per_split"].get(task, {}).get(split, {}).get(metric)
            b = run_b["per_split"].get(task, {}).get(split, {}).get(metric)
            if a is None and b is None and (task, split) not in training:
                continue
            inference = None if a is None or b is None else round(abs(a - b), 6)
            measured = training.get((task, split))
            train_spread = measured["spread"] if measured else None

            known = [x for x in (inference, train_spread) if x is not None]
            floor = max(known) if known else None
            if not floor:
                row = {"inference_spread": inference, "training_spread": train_spread,
                       "floor": floor, "bound_by": None, "threshold": None,
                       "provisional": None,
                       "note": "no non-zero floor measured — stays report-only rather than "
                               "gating on a zero threshold"}
            else:
                bound_by = ("training" if train_spread is not None
                            and train_spread >= (inference or 0) else "inference")
                row = {"inference_spread": inference, "training_spread": train_spread,
                       "floor": round(floor, 6), "bound_by": bound_by,
                       "threshold": round(multiplier * floor, 6),
                       "provisional": train_spread is None,
                       "note": ("training variance unmeasured — reported, not enforced"
                                if train_spread is None else "enforceable")}
            if measured:
                row["training_source"] = measured["source"]
            out.setdefault(task, {})[split] = row
    return {"multiplier": multiplier, "per_task": out}


def gate_from(derivation: dict) -> dict:
    """Only non-provisional random-split thresholds gate. The hard split stays report-only (F33)."""
    thresholds = {
        task: splits["random"]["threshold"]
        for task, splits in derivation["per_task"].items()
        if "random" in splits and splits["random"]["threshold"] is not None
        and splits["random"]["provisional"] is False
    }
    return {
        "state": "enforcing" if thresholds else "report_only",
        "thresholds": thresholds,
        "multiplier": derivation["multiplier"],
        "why": "D37 — floors take the larger of inference and training spread; thresholds "
               "resting on inference alone are provisional and not enforced",
    }


def main(run_a: str, run_b: str, force: bool = False) -> int:
    """Derive thresholds from two baseline regression runs and write them where manifests read."""
    if OUT.exists() and not force:
        print(f"  {OUT.relative_to(REPO_ROOT)} exists — thresholds are derived. --force to redo.")
        return 1
    runs = [Path(p) for p in (run_a, run_b)]
    derivation = derive(*(json.loads(p.read_text()) for p in runs))
    gate = gate_from(derivation)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "purpose": "Promotion-gate thresholds (F33), derived per D37. Newly built manifests "
                   "carry this gate; the inputs are pinned so the derivation can be re-checked.",
        "inputs": [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in runs],
        "derivation": derivation,
        "gate": gate,
    }, indent=2) + "\n", encoding="utf-8")
    for task, splits in derivation["per_task"].items():
        for split, row in splits.items():
            print(f"  {task:9s} {split:6s} inference {row['inference_spread']} · training "
                  f"{row['training_spread']} · threshold {row['threshold']} · {row['note']}")
    print(f"\n  gate: {gate['state']} · enforced thresholds {gate['thresholds']}")
    return 0
