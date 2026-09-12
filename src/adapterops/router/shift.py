"""Within-task distribution shift split (F36, M4) — the replacement for a confounded test.

v2 of the PRD specified: train the router on Kaggle-sourced tickets, test on
Banking77-sourced queries. That test could not measure what it claimed. Task identity is
an input feature to the router, and each task is bound to exactly one source — so
splitting by source also splits by task, and the result would have measured the router on
a task value it had never seen. It would have failed for a reason unrelated to
distribution shift, and the failure would have been reported as a shift result (D11).

The replacement holds task constant and moves style *within* each task's own pool. For
each task: TF-IDF the pool text, k-means it into five clusters, and put one whole cluster
plus the top and bottom length deciles on the shift side. Every task appears on both
sides; only phrasing and length distribution move.

**Which cluster, decided by rule rather than by looking.** TF-IDF k-means on short support
text does not produce five equal clusters, so "hold out one cluster" needs a tie-break
that cannot be tuned after seeing the scores. The rule: the cluster whose size is closest
to an even share of the pool, lowest cluster index breaking a tie. Written here, before
any router exists to be helped by a different choice.

**Length deciles are computed over the pool, not the source split**, because the pool is
what is being partitioned — deciles of a population that is not being split would put an
arbitrary number of rows on each side.

The two criteria overlap, and the shift side is their union: a long document inside the
held-out cluster is one row on the shift side, not two.

**The manifest records evidence that the split is shifted, not just that it was built.**
Median length is the wrong check — the shift side holds both tails, so their medians nearly
cancel and a broken split would look fine. What is recorded instead is the length spread on
each side and the share of the shift side's vocabulary that never appears in-distribution.
If that unseen-token rate were near zero the cluster holdout would have moved nothing, and
any M4 degradation would need a different explanation.

Run:  uv run adapterops router-shift
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_FILE = REPO_ROOT / "data" / "router" / "pool.parquet"
SHIFT_FILE = REPO_ROOT / "data" / "router" / "shift.parquet"
MANIFEST = REPO_ROOT / "evals" / "ROUTER_SHIFT.json"

SEED = 20260909
K = 5
"""PRD §11 says k≈5. Five clusters over 750 documents leaves ~150 per cluster at even
split, which is a reportable shift-side size; k=10 would not be."""

DECILE = 0.10

WORD = re.compile(r"[a-z][a-z\']+")
"""Vocabulary comparison only — deliberately cruder than the TF-IDF analyzer, so the
unseen-token evidence is not computed by the same code that produced the clusters."""


def assign_task(texts: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Partition one task's pool texts into `in_distribution` and `shift`."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=5_000, stop_words="english",
                          sublinear_tf=True, min_df=2)
    matrix = vec.fit_transform(texts)
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(matrix)
    cluster = pd.Series(km.labels_, index=texts.index)

    sizes = cluster.value_counts().sort_index()
    even = len(texts) / K
    held = int(min(sizes.index, key=lambda c: (abs(sizes[c] - even), c)))

    length = texts.str.len()
    lo, hi = length.quantile(DECILE), length.quantile(1 - DECILE)
    short, long_ = length <= lo, length >= hi

    in_cluster = cluster == held
    shift = in_cluster | short | long_

    frame = pd.DataFrame({
        "cluster": cluster,
        "chars": length,
        "in_held_cluster": in_cluster,
        "in_short_decile": short,
        "in_long_decile": long_,
        "side": np.where(shift, "shift", "in_distribution"),
    })
    def vocab(series: pd.Series) -> set[str]:
        return {w for text in series for w in WORD.findall(text.lower())}

    seen, unseen_side = vocab(texts[~shift]), vocab(texts[shift])
    novel = unseen_side - seen

    meta = {
        "cluster_sizes": {str(c): int(n) for c, n in sizes.items()},
        "held_cluster": held,
        "held_cluster_rows": int(in_cluster.sum()),
        "short_decile_max_chars": float(lo),
        "long_decile_min_chars": float(hi),
        "short_decile_rows": int(short.sum()),
        "long_decile_rows": int(long_.sum()),
        "overlap_rows": int((in_cluster & (short | long_)).sum()),
        "shift_rows": int(shift.sum()),
        "in_distribution_rows": int((~shift).sum()),
        "shift_fraction": round(float(shift.mean()), 4),
        "evidence_the_split_is_shifted": {
            "unseen_token_rate": round(len(novel) / max(len(unseen_side), 1), 4),
            "unseen_tokens": len(novel),
            "shift_vocabulary": len(unseen_side),
            "chars_in_distribution_p10_p50_p90": [
                float(length[~shift].quantile(q)) for q in (0.1, 0.5, 0.9)],
            "chars_shift_p10_p50_p90": [
                float(length[shift].quantile(q)) for q in (0.1, 0.5, 0.9)],
        },
    }
    share = meta["held_cluster_rows"] / len(texts)
    if not 0.05 <= share <= 0.40:
        meta["warning"] = (
            f"held cluster is {share:.1%} of the pool — k-means split this task very "
            f"unevenly, so the cluster criterion contributes little or dominates")
    return frame, meta


def main(force: bool = False) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — the shift split is frozen.")
        print("  Re-drawing it moves the bar M4 is reported against. --force if you mean it.")
        return 1
    if not POOL_FILE.exists():
        print("  no router pool — run `uv run adapterops router-pool` first.")
        return 2

    pool = pd.read_parquet(POOL_FILE)
    frames, metas = [], {}
    for task in sorted(pool.task.unique()):
        rows = pool[pool.task == task]
        frame, meta = assign_task(rows.text.reset_index(drop=True))
        frame.insert(0, "pair_id", rows.pair_id.to_numpy())
        frame.insert(1, "task", task)
        frames.append(frame)
        metas[task] = meta
        print(f"  {task:9s} clusters {list(meta['cluster_sizes'].values())} "
              f"-> held #{meta['held_cluster']} ({meta['held_cluster_rows']}) "
              f"+ deciles -> shift {meta['shift_rows']:>3} "
              f"({meta['shift_fraction']:.0%}) · "
              f"unseen tokens {meta['evidence_the_split_is_shifted']['unseen_token_rate']:.1%} · "
              f"chars p10/p90 "
              f"{meta['evidence_the_split_is_shifted']['chars_in_distribution_p10_p50_p90'][0]:.0f}/"
              f"{meta['evidence_the_split_is_shifted']['chars_in_distribution_p10_p50_p90'][2]:.0f}"
              f" -> "
              f"{meta['evidence_the_split_is_shifted']['chars_shift_p10_p50_p90'][0]:.0f}/"
              f"{meta['evidence_the_split_is_shifted']['chars_shift_p10_p50_p90'][2]:.0f}")
        if "warning" in meta:
            print(f"            ! {meta['warning']}")

    out = pd.concat(frames, ignore_index=True)
    SHIFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SHIFT_FILE)

    MANIFEST.write_text(json.dumps({
        "purpose": "Frozen within-task shift split (F36). Task held constant; phrasing "
                   "and length distribution moved. Degradation is reported per task "
                   "(M4), never blended.",
        "regenerate": "uv run adapterops router-shift --force",
        "seed": SEED,
        "k": K,
        "decile": DECILE,
        "cluster_rule": "the cluster whose size is closest to an even share of the pool; "
                        "lowest index breaks a tie. Fixed before any router exists.",
        "shift_side": "union of the held cluster and the top and bottom length deciles",
        "file": str(SHIFT_FILE.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(SHIFT_FILE.read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(POOL_FILE.read_bytes()).hexdigest(),
        "tasks": metas,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  froze {SHIFT_FILE.relative_to(REPO_ROOT)} and "
          f"{MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(force="--force" in sys.argv))
