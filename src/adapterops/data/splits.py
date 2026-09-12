"""Freeze the evaluation splits (F35).

BUILD-PLAN Phase 1 flags this as expensive-to-fix-later: once a golden set is frozen it
must never move, or every later comparison is meaningless. So this module writes
`evals/SPLITS.json` with a sha256 per file and **refuses to overwrite an existing frozen
split** unless `--force` is passed.

Upstream split boundaries are respected rather than reshuffled — where a source ships its
own test/validation split, the golden set is drawn from it.

Run:  uv run adapterops splits
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evals" / "golden"
MANIFEST = REPO_ROOT / "evals" / "SPLITS.json"

SEED = 20260909
"""Same seed as the mirror. Changing it changes every split — a versioned decision."""


@dataclass
class SplitSpec:
    task: str
    label_column: str
    golden_size: int
    stratified: bool
    note: str
    golden_from: str          # which mirrored file the golden set is drawn from
    trainval_from: str        # which mirrored file train/val come from
    val_fraction: float = 0.15
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    """Applied to every split after slicing. Used to derive training targets that are not
    a column of the mirror — PII's `LABEL: value` target, for instance."""
    group_column: str | None = None
    """Split on unique values of this column rather than on rows. Required where the
    source repeats a text: Bitext has 989 instructions appearing up to 8 times, so a
    row-wise split leaves copies of golden texts in train — a leak, caught by
    tests/test_splits_and_harness.py on its first run."""


def urgency_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine subject and body into the single `text` column the task classifies.

    Subject is missing on 1,032 of the English rows, so it is concatenated only when
    present rather than producing a literal "nan" prefix.
    """
    df = df.copy()
    subject = df.subject.fillna("").astype(str).str.strip()
    body = df.body.fillna("").astype(str).str.strip()
    df["text"] = (subject + "\n" + body).str.strip()
    return df


def pii_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `target` column: one `LABEL: value` line per span, in source order.

    Derived from `privacy_mask`, which carries label, value and offsets directly — the
    `masked_text` column is NOT used, because recovering exact boundaries from it is
    impossible in general and caps strict span F1 at 0.897 (STATUS.md D22).
    """
    df = df.copy()
    df["target"] = [
        "\n".join(f"{s['label']}: {s['value']}" for s in sorted(mask, key=lambda s: s["start"]))
        for mask in df.privacy_mask
    ]
    return df


SPECS = [
    SplitSpec(
        task="intent",
        label_column="label",
        golden_size=770,
        stratified=True,
        golden_from="data/intent/test.parquet",
        trainval_from="data/intent/train.parquet",
        note=(
            "770 = exactly 10 per class across 77 classes (PRD §11). The upstream test "
            "split holds 39-40 per class, so 10/class is comfortably drawable. Gated "
            "metric is micro-accuracy; macro-F1 is indicative only at this density."
        ),
    ),
    SplitSpec(
        task="urgency",
        label_column="priority",
        golden_size=300,
        stratified=True,
        golden_from="data/urgency/train.parquet",
        trainval_from="data/urgency/train.parquet",
        transform=urgency_text,
        note=(
            "Single upstream split, so golden is held out first. 300 = 100 per class over "
            "three priorities (low/medium/high) — stratified because the source is "
            "imbalanced (medium 4,951 / high 4,571 / low 2,400) and an unstratified golden "
            "set would under-sample `low`. Gated metric is macro-F1, not accuracy: with "
            "three classes and a 2:1 imbalance, accuracy rewards ignoring the rare class. "
            "Source is cc-by-nc-4.0 — see the README."
        ),
    ),
    SplitSpec(
        task="pii",
        label_column="target",
        golden_size=300,
        stratified=False,
        golden_from="data/pii/validation.parquet",
        trainval_from="data/pii/train.parquet",
        transform=pii_targets,
        note=(
            "Span detection, not classification (PRD v2.5 changelog 24-25). Golden is 300 "
            "documents drawn from the upstream validation split, ~1,800 spans at a median "
            "of six per document. The target is `LABEL: value` lines built from "
            "`privacy_mask`; masked_text is not used (D22). Gated metric is strict span "
            "F1 via adapterops.eval.spans, whose ceiling on this format is 0.9974."
        ),
    ),
    SplitSpec(
        task="drafting",
        label_column="intent",
        golden_size=300,
        stratified=False,
        golden_from="data/drafting/train.parquet",
        trainval_from="data/drafting/train.parquet",
        group_column="instruction",
        note=(
            "Single upstream split, so golden is held out from it first and train/val "
            "come from the remainder. Generative task — scored by the judge, not a label "
            "metric, so the golden set is not stratified. Split is group-aware on "
            "`instruction`: 989 instructions repeat (up to 8x), and a row-wise split "
            "leaves copies of golden texts in train."
        ),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratified_head(df: pd.DataFrame, label_col: str, per_class: int, seed: int) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(label_col, group_keys=False).head(per_class)


def build(spec: SplitSpec) -> dict:
    golden_src = pd.read_parquet(REPO_ROOT / spec.golden_from)
    trainval_src = pd.read_parquet(REPO_ROOT / spec.trainval_from)
    same_source = spec.golden_from == spec.trainval_from

    if spec.group_column:
        groups = golden_src[spec.group_column].drop_duplicates()
        chosen = set(groups.sample(n=spec.golden_size, random_state=SEED))
        golden = (golden_src[golden_src[spec.group_column].isin(chosen)]
                  .drop_duplicates(subset=spec.group_column))
        trainval_src = trainval_src[~trainval_src[spec.group_column].isin(chosen)]
        same_source = False   # exclusion already applied, by group
    elif spec.stratified:
        n_classes = golden_src[spec.label_column].nunique()
        per_class, remainder = divmod(spec.golden_size, n_classes)
        if remainder:
            msg = (
                f"{spec.task}: golden_size {spec.golden_size} is not divisible by "
                f"{n_classes} classes — a stratified split cannot be exact"
            )
            raise ValueError(msg)
        golden = _stratified_head(golden_src, spec.label_column, per_class, SEED)
    else:
        golden = golden_src.sample(n=spec.golden_size, random_state=SEED)

    if same_source:
        trainval_src = trainval_src.drop(index=golden.index)

    trainval = trainval_src.sample(frac=1.0, random_state=SEED)
    n_val = int(len(trainval) * spec.val_fraction)
    val, train = trainval.iloc[:n_val], trainval.iloc[n_val:]

    if spec.transform is not None:
        golden, val, train = (spec.transform(f) for f in (golden, val, train))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = REPO_ROOT / "data" / spec.task
    files = {}
    for name, frame in (("golden", golden), ("val", val), ("train", train)):
        path = (EVAL_DIR / f"{spec.task}.parquet") if name == "golden" else (out_dir / f"split_{name}.parquet")
        frame.reset_index(drop=True).to_parquet(path)
        files[name] = {
            "file": str(path.relative_to(REPO_ROOT)),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    entry = {
        "task": spec.task,
        "seed": SEED,
        "label_column": spec.label_column,
        "stratified": spec.stratified,
        "group_column": spec.group_column,
        "golden_from": spec.golden_from,
        "trainval_from": spec.trainval_from,
        "note": spec.note,
        "splits": files,
    }
    if spec.stratified:
        counts = golden[spec.label_column].value_counts()
        entry["golden_per_class"] = {"min": int(counts.min()), "max": int(counts.max()),
                                     "classes": int(counts.size)}
    return entry


def main(force: bool = False) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — splits are frozen.")
        print("  Re-running would move the bar and invalidate every prior comparison.")
        print("  Pass --force only if you intend that, and record it as a decision.")
        return 1

    entries = [build(spec) for spec in SPECS]
    for e in entries:
        print(f"  {e['task']:9s} golden {e['splits']['golden']['rows']:>5,} · "
              f"val {e['splits']['val']['rows']:>6,} · train {e['splits']['train']['rows']:>6,}"
              + (f"   ({e['golden_per_class']['min']}/class × "
                 f"{e['golden_per_class']['classes']})" if "golden_per_class" in e else ""))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "purpose": "Frozen evaluation splits (F35). Golden sets must never move.",
        "regenerate": "uv run adapterops splits --force  (this invalidates prior comparisons)",
        "pending": [spec.task for spec in SPECS
                    if not (REPO_ROOT / spec.golden_from).exists()],
        "tasks": entries,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  froze {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(force="--force" in sys.argv))
