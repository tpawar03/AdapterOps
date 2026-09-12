"""The shuffled-label training split (F21) — a regression that M7's gate has to catch.

M7 is the project's distinguishing claim: detect → block → rollback, proven end to end. That
needs a candidate adapter that is genuinely broken but looks legitimate to the pipeline — same
task, same base, same serving path, same manifest shape. So the intent training split is
rebuilt with its labels permuted across rows and every text left exactly where it was. An
adapter trained on it learns the output *format* (it will still emit valid intent labels) and
none of the mapping, so the failure is only visible by scoring it.

**Intent, not another task,** because its gated metric is exact-match micro-accuracy on 770
golden rows: a shuffled adapter should land near the 1/77 floor, far outside any plausible
noise, which makes the gate's decision unambiguous. The demonstration is about the pipeline
catching a regression, not about a borderline call.

**What is permuted is the (label, label_text) pair,** so the integer label and its name stay
consistent on every row. The label *distribution* is unchanged — it is a permutation — and the
share of rows that keep their original label by coincidence is recorded against the chance
rate, as evidence the shuffle actually broke the mapping.

    uv run python -m adapterops.data.shuffle --task intent
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260909
LABEL_COLUMNS = {"intent": ["label", "label_text"]}


def paths(task: str) -> tuple[Path, Path, Path]:
    base = REPO_ROOT / "data" / task
    return (base / "split_train.parquet", base / "split_train_shuffled.parquet",
            base / "split_train_shuffled.json")


def shuffle_labels(frame: pd.DataFrame, label_columns: list[str], seed: int = SEED) -> pd.DataFrame:
    order = np.random.default_rng(seed).permutation(len(frame))
    out = frame.copy()
    out[label_columns] = frame[label_columns].to_numpy()[order]
    return out


def evidence(original: pd.DataFrame, shuffled: pd.DataFrame, label_col: str) -> dict:
    kept = float((original[label_col].to_numpy() == shuffled[label_col].to_numpy()).mean())
    shares = original[label_col].value_counts(normalize=True)
    return {
        "rows": len(original),
        "classes": int(original[label_col].nunique()),
        "rows_keeping_their_label": round(kept, 4),
        "chance_of_keeping": round(float((shares ** 2).sum()), 4),
        "label_distribution_unchanged": bool(
            original[label_col].value_counts().sort_index().equals(
                shuffled[label_col].value_counts().sort_index())),
        "texts_untouched": True,
    }


def main(task: str = "intent", force: bool = False) -> int:
    if task not in LABEL_COLUMNS:
        print(f"  no shuffle defined for {task!r}; F21 uses intent")
        return 2
    source, target, manifest = paths(task)
    if manifest.exists() and not force:
        print(f"  {manifest.relative_to(REPO_ROOT)} exists — the shuffled split is frozen.")
        return 1

    original = pd.read_parquet(source)
    shuffled = shuffle_labels(original, LABEL_COLUMNS[task])
    text_col = "text"
    if not original[text_col].equals(shuffled[text_col]):
        msg = "texts moved during the shuffle"
        raise RuntimeError(msg)
    pairs = shuffled.groupby("label").label_text.nunique()
    if (pairs != 1).any():
        msg = "label and label_text were separated by the shuffle"
        raise RuntimeError(msg)

    shuffled.to_parquet(target)
    report = evidence(original, shuffled, "label_text")
    manifest.write_text(json.dumps({
        "purpose": "F21 — intent training split with labels permuted, for the M7 regression",
        "seed": SEED,
        "source": str(source.relative_to(REPO_ROOT)),
        "file": str(target.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "evidence": report,
        "train_with": "python -m adapterops.train.cli_train --task intent "
                      "--train-split train_shuffled --variant shuffled",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  froze {target.relative_to(REPO_ROOT)} · {report['rows']:,} rows · "
          f"{report['rows_keeping_their_label']:.2%} keep their label "
          f"(chance {report['chance_of_keeping']:.2%})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="intent")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(a.task, a.force))
