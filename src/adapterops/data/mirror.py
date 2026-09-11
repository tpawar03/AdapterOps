"""Mirror the public datasets named in PRD §9 into ``data/`` as parquet snapshots.

PRD §9 requires a local mirror committed to the repo, so an upstream takedown or
re-license cannot invalidate the project mid-build (a listed §14 risk). This module
writes one directory per task plus ``data/MANIFEST.json`` recording, for each source,
the Hub revision it came from, the row counts, the columns kept, any filter applied,
and a sha256 of every file written.

Run:  uv run python -m adapterops.data.mirror
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info

SEED = 20260909
"""Fixed sampling seed. Changing it changes the mirror — treat as a versioned decision."""

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Source:
    """One mirrored dataset."""

    task: str
    hf_id: str
    note: str
    keep_columns: list[str] | None = None
    filter_column: str | None = None
    filter_values: list[str] = field(default_factory=list)
    cap_rows: dict[str, int] | None = None
    """Per-split row cap. Applied after filtering, with SEED, so the mirror stays a
    reasonable size in git. Only used where the upstream split dwarfs what the PRD needs."""
    data_files: str | None = None
    """Explicit file within the repo, for datasets that ship several CSVs with no default
    config. Without it `load_dataset` either guesses or refuses."""
    drop_na: list[str] = field(default_factory=list)
    """Columns that must be present; rows missing any of them are dropped."""
    licence: str = ""
    """Recorded in MANIFEST.json. Non-permissive licences must also reach the README."""


SOURCES = [
    Source(
        task="intent",
        hf_id="mteb/banking77",
        note=(
            "Parquet-native mirror of Banking77. PolyAI/banking77 is script-based and "
            "modern `datasets` refuses to load it — see PRD changelog 21."
        ),
    ),
    Source(
        task="drafting",
        hf_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        note="CDLA-Sharing-1.0. Attribution + share-alike; attribution belongs in the README.",
    ),
    Source(
        task="pii",
        hf_id="ai4privacy/pii-masking-openpii-1m",
        note=(
            "English rows only — the source spans 23 languages and is only ~12.5% English, "
            "so an unfiltered subsample would violate non-goal 11 (PRD changelog 22). "
            "mbert_tokens / mbert_token_classes dropped: unused and by far the largest columns. "
            "Capped after filtering: 143,515 English train rows is ~47x what PRD §9 needs "
            "(~3K subsample) and would put 147 MB of parquet in git for no benefit."
        ),
        keep_columns=["source_text", "masked_text", "privacy_mask", "language", "region"],
        filter_column="language",
        filter_values=["en"],
        cap_rows={"train": 20000, "validation": 5000},
        licence="cc-by-4.0",
    ),
    Source(
        task="urgency",
        hf_id="Tobi-Bueck/customer-support-tickets",
        note=(
            "Replaces Kaggle albertobircoci/support-ticket-priority-dataset-50k, which has "
            "NO TICKET TEXT — it is a tabular dataset whose `description_length` column is an "
            "integer and whose description is absent (PRD changelog 27). English rows only: "
            "the source is 60% English / 40% German, the same trap as ai4privacy. Rows "
            "missing body or priority are dropped (1,033 of 11,923). "
            "LICENCE IS cc-by-nc-4.0 — non-commercial, attribution required. This is the "
            "only non-permissive source in the project and the derived adapter inherits it; "
            "disclosed in the README and the adapter card."
        ),
        data_files="dataset-tickets-multi-lang-4-20k.csv",
        keep_columns=["subject", "body", "priority", "language", "type", "queue"],
        filter_column="language",
        filter_values=["en"],
        drop_na=["body", "priority"],
        licence="cc-by-nc-4.0",
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror(source: Source) -> dict:
    out_dir = DATA_DIR / source.task
    out_dir.mkdir(parents=True, exist_ok=True)

    revision = dataset_info(source.hf_id).sha
    dataset = (load_dataset(source.hf_id, data_files=source.data_files)
               if source.data_files else load_dataset(source.hf_id))

    splits, files = {}, []
    for split_name, split in dataset.items():
        before = len(split)
        if source.filter_column:
            split = split.filter(lambda r: r[source.filter_column] in source.filter_values)
        for column in source.drop_na:
            split = split.filter(lambda r, c=column: r[c] is not None and str(r[c]).strip() != "")
        rows_after_filter = len(split)
        cap = (source.cap_rows or {}).get(split_name)
        if cap and len(split) > cap:
            split = split.shuffle(seed=SEED).select(range(cap))
        if source.keep_columns:
            split = split.remove_columns(
                [c for c in split.column_names if c not in source.keep_columns]
            )
        path = out_dir / f"{split_name}.parquet"
        split.to_parquet(path)
        splits[split_name] = {
            "rows_upstream": before,
            "rows_after_filter": rows_after_filter,
            "rows_mirrored": len(split),
            "capped_at": cap,
        }
        files.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return {
        "task": source.task,
        "hf_id": source.hf_id,
        "revision": revision,
        "note": source.note,
        "licence": source.licence or "unrecorded",
        "licence_permissive": source.licence not in {"cc-by-nc-4.0"},
        "columns_kept": source.keep_columns or "all",
        "filter": (
            f"{source.filter_column} in {source.filter_values}" if source.filter_column else None
        ),
        "sampling_seed": SEED if source.cap_rows else None,
        "dropped_rows_missing": source.drop_na or None,
        "splits": splits,
        "files": files,
    }


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    entries = []
    for source in SOURCES:
        print(f"  mirroring {source.task:9s} {source.hf_id}")
        entry = mirror(source)
        entries.append(entry)
        for split_name, counts in entry["splits"].items():
            kept, upstream = counts["rows_mirrored"], counts["rows_upstream"]
            suffix = "" if kept == upstream else f"  (from {upstream:,} upstream)"
            print(f"      {split_name:12s} {kept:>9,} rows{suffix}")

    manifest = {
        "purpose": "Local mirror of the PRD §9 sources; guards against upstream takedown.",
        "regenerate": "uv run python -m adapterops.data.mirror",
        "sources": entries,
    }
    manifest_path = DATA_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(f["bytes"] for e in entries for f in e["files"])
    print(f"\n  wrote {manifest_path.relative_to(REPO_ROOT)}  ·  mirror total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
