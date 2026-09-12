"""Guards on the within-task shift split (F36).

The requirement F36 states — *every task present on both sides* — is the one that makes
this test a shift test rather than D11's confounded source split, so it is asserted
directly. The last test is the one that would catch a shift set that was built correctly
and shifts nothing.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "evals" / "ROUTER_SHIFT.json"
POOL = ROOT / "data" / "router" / "pool.parquet"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="shift split not frozen yet")


def manifest():
    return json.loads(MANIFEST.read_text())


def shift():
    return pd.read_parquet(ROOT / manifest()["file"])


def test_frozen_shift_matches_its_manifest_and_the_pool_it_was_cut_from():
    """A re-drawn pool with a stale shift split misaligns silently — so pin both hashes."""
    m = manifest()
    assert hashlib.sha256((ROOT / m["file"]).read_bytes()).hexdigest() == m["sha256"]
    assert hashlib.sha256(POOL.read_bytes()).hexdigest() == m["pool_sha256"], \
        "the pool has been re-frozen since the shift split was cut; re-run router-shift"


def test_shift_covers_the_router_slice_exactly_once_and_nothing_else():
    """Mining rows feed F31, never a router eval, so a shift side for them would be a
    column nothing reads — and would invite someone to read it."""
    pool = pd.read_parquet(POOL)
    s = shift()
    assert s.pair_id.is_unique
    assert set(s.pair_id) == set(pool.loc[pool.purpose == "router", "pair_id"])
    assert not set(s.pair_id) & set(pool.loc[pool.purpose == "mining", "pair_id"])


def test_every_task_appears_on_both_sides():
    """F36's actual requirement. D11 failed precisely here: source and task were collinear,
    so the held-out side contained task values the router had never seen, and the result
    would have been reported as distribution shift."""
    counts = shift().groupby(["task", "side"]).size().unstack(fill_value=0)
    assert set(counts.columns) == {"in_distribution", "shift"}
    assert (counts > 0).all().all(), f"a task is missing from one side:\n{counts}"


def test_side_is_exactly_the_union_of_the_three_criteria():
    s = shift()
    union = s.in_held_cluster | s.in_short_decile | s.in_long_decile
    assert (s.side.eq("shift") == union).all()


def test_the_shift_side_is_actually_shifted():
    """Median length cannot show this — the shift side holds both tails, so the medians
    nearly cancel. Vocabulary can: a quarter to a half of the shift side's words never
    appear in-distribution, which is the cluster holdout doing its job."""
    for task, meta in manifest()["tasks"].items():
        evidence = meta["evidence_the_split_is_shifted"]
        assert evidence["unseen_token_rate"] > 0.10, \
            f"{task}: shift side shares almost all its vocabulary with training"
        lo_id, _, hi_id = evidence["chars_in_distribution_p10_p50_p90"]
        lo_sh, _, hi_sh = evidence["chars_shift_p10_p50_p90"]
        assert lo_sh < lo_id and hi_sh > hi_id, f"{task}: length deciles did not widen the tails"


def test_shift_refuses_to_be_redrawn_without_force():
    from adapterops.router.shift import main as shift_main

    assert shift_main(force=False) == 1
