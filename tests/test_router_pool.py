"""Guards on the router pool's two exclusions (F7).

Both failure modes are silent and both inflate the result, so the checks that matter are
the ones asserting a *number the pool is not allowed to have*. The last test is the
unusual one: it asserts that an exclusion is still doing work, because a filter that
removes nothing looks identical to a filter that has been quietly broken.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "evals" / "ROUTER_POOL.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="router pool not frozen yet")


def manifest():
    return json.loads(MANIFEST.read_text())


def pool():
    return pd.read_parquet(ROOT / manifest()["file"])


def test_frozen_pool_matches_its_manifest():
    """The committed parquet is the pool the manifest describes, byte for byte."""
    import hashlib

    m = manifest()
    path = ROOT / m["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"]
    assert len(pd.read_parquet(path)) == m["pairs"]


def test_pool_never_overlaps_golden_or_adapter_training_data():
    """Recomputed from the source files, not read back from the manifest.

    A golden row here trains the router on an eval item; a training row here teaches the
    router a success rate the adapter only achieves from memory.
    """
    from adapterops.router.pool import verify

    for task, counts in verify(pool()).items():
        assert counts["in_golden"] == 0, f"{task}: golden rows in the router pool"
        assert counts["in_adapter_train"] == 0, f"{task}: adapter training rows in the pool"
        assert counts["duplicate_texts"] == 0, f"{task}: duplicate texts in the pool"


def test_every_task_is_equally_represented_in_both_slices():
    """Task is an input feature (F10); an unbalanced pool teaches a per-task prior."""
    from adapterops.router.pool import PER_TASK, PER_TASK_MINING, TASKS

    counts = pool().groupby(["task", "purpose"]).size().unstack()
    assert set(counts.index) == set(TASKS)
    assert set(counts["router"]) == {PER_TASK}
    assert set(counts["mining"]) == {PER_TASK_MINING}


def test_router_and_mining_slices_share_no_rows():
    """The F31 exclusion, enforced by allocation rather than subtraction (D29). If these
    two slices ever overlap, the hard-cases split is mined from router training data and
    D7's leak is back with nothing to announce it."""
    p = pool()
    router = set(p.loc[p.purpose == "router", "text"])
    mining = set(p.loc[p.purpose == "mining", "text"])
    assert not router & mining
    # source_row is an index into each task's own val split, so uniqueness is only
    # meaningful per task — across tasks the numbers collide by construction.
    per_slice = p.groupby(["task", "purpose"]).source_row.nunique()
    per_slice_rows = p.groupby(["task", "purpose"]).size()
    assert (per_slice == per_slice_rows).all()


def test_pairs_are_uniquely_identified_and_complete():
    p = pool()
    assert p.pair_id.is_unique
    assert not p.text.str.strip().eq("").any()
    assert not p.gold.isna().any()


def test_drafting_exclusion_is_load_bearing():
    """The in-adapter-train filter removes 467 drafting rows, and must keep doing so.

    The frozen drafting split is group-aware only where the golden set was carved out, so
    11.7% of its val rows share an `instruction` with the adapter's training data (D24).
    If this count ever reads zero the filter has been broken, not satisfied.
    """
    entry = next(t for t in manifest()["tasks"] if t["task"] == "drafting")
    assert entry["excluded_in_adapter_train"] == 467

    val = pd.read_parquet(ROOT / "data/drafting/split_val.parquet")
    train = set(pd.read_parquet(ROOT / "data/drafting/split_train.parquet").instruction)
    assert int(val.instruction.isin(train).sum()) == entry["excluded_in_adapter_train"]


def test_pool_refuses_to_be_redrawn_without_force():
    from adapterops.router.pool import main as pool_main

    assert pool_main(force=False) == 1


def test_oversized_draw_raises_rather_than_relaxing_an_exclusion():
    from adapterops.router.pool import build_task

    with pytest.raises(ValueError, match="do not relax an exclusion"):
        build_task("intent", per_task=10_000)
