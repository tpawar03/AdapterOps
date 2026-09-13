"""Guards on D38's relabelling: change drafting's label, and nothing else."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.router import relabel as rl


def pool():
    rows = []
    for i in range(6):
        rows.append({"pair_id": f"drafting-r{i}", "task": "drafting", "purpose": "router",
                     "success": i < 3, "label_rule": "proxy", "label_is_proxy": True})
        rows.append({"pair_id": f"drafting-m{i}", "task": "drafting", "purpose": "mining",
                     "success": i < 3, "label_rule": "proxy", "label_is_proxy": True})
        rows.append({"pair_id": f"intent-r{i}", "task": "intent", "purpose": "router",
                     "success": i % 2 == 0, "label_rule": "exact", "label_is_proxy": False})
    return pd.DataFrame(rows)


def grades():
    ids = [f"drafting-{s}{i}" for s in ("r", "m") for i in range(6)]
    return pd.Series([5, 2, 4, 3, 5, 4] * 2, index=ids, dtype=float)


def test_only_drafting_labels_change_and_the_old_label_is_kept():
    before = pool()
    after = rl.relabel(before, grades())
    intent = after.task == "intent"
    assert after.loc[intent, "success"].tolist() == before.loc[intent, "success"].tolist()
    drafting = after[after.task == "drafting"]
    assert drafting.success.tolist() == [g >= 4 for g in [5, 2, 4, 3, 5, 4] for _ in (0, 1)][:0] \
        or drafting.set_index("pair_id").success.to_dict() == {
            p: grades()[p] >= 4 for p in drafting.pair_id}
    assert drafting.proxy_success.astype(bool).tolist() == \
        before[before.task == "drafting"].success.tolist()
    assert not drafting.label_is_proxy.any()


def test_a_drafting_pair_without_a_grade_is_refused_rather_than_mixed():
    with pytest.raises(ValueError, match="no teacher grade"):
        rl.relabel(pool(), grades().drop("drafting-r0"))


def test_router_split_membership_is_unchanged(tmp_path):
    """Re-stratifying on new labels would move rows between splits and confound the comparison."""
    relabelled = rl.relabel(pool(), grades())
    base = pool()
    for split, ids in (("train", ["drafting-r0", "drafting-r1", "intent-r0"]),
                       ("eval", ["drafting-r2", "intent-r1"]),
                       ("shift_eval", ["drafting-r3", "drafting-r4", "intent-r2"])):
        base[base.pair_id.isin(ids)].to_parquet(tmp_path / f"router_{split}.parquet")
    out = rl.relabel_router_splits(relabelled, router_dir=tmp_path)
    for split in rl.SPLITS:
        frozen = pd.read_parquet(tmp_path / f"router_{split}.parquet")
        assert out[split].pair_id.tolist() == frozen.pair_id.tolist()


def test_the_new_drafting_bucket_is_mining_judge_failures_only():
    bucket = rl.drafting_hard_bucket(rl.relabel(pool(), grades()), cap=150)
    assert set(bucket.purpose) == {"mining"}
    assert not bucket.success.any()
    assert set(bucket.failure_type) <= {"judge_grade_2", "judge_grade_3"}


def test_rebuilding_the_hard_split_leaves_other_tasks_untouched():
    hard = pd.DataFrame([
        {"pair_id": "intent-m1", "task": "intent", "quarantine": False},
        {"pair_id": "drafting-mOLD", "task": "drafting", "quarantine": False},
    ])
    bucket = rl.drafting_hard_bucket(rl.relabel(pool(), grades()), cap=150)
    rebuilt = rl.rebuild_hard_split(hard, bucket)
    assert "drafting-mOLD" not in set(rebuilt.pair_id)
    assert rebuilt[rebuilt.task == "intent"].pair_id.tolist() == ["intent-m1"]
    assert set(rebuilt[rebuilt.task == "drafting"].applicability) == {"not_applicable"}
