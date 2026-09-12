"""Guards on F21's shuffled-label split, and on training it without clobbering the real adapter.

The shuffle has to break the text-to-label mapping while changing nothing else, or the M7
regression it feeds is a different model rather than a broken one.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.data import shuffle


@pytest.fixture(scope="module")
def pair():
    original = pd.read_parquet(ROOT / "data/intent/split_train.parquet")
    return original, shuffle.shuffle_labels(original, ["label", "label_text"])


def test_texts_stay_put_and_only_labels_move(pair):
    original, shuffled = pair
    assert original.text.equals(shuffled.text)
    assert sorted(original.label_text) == sorted(shuffled.label_text)


def test_label_and_its_name_stay_together(pair):
    _, shuffled = pair
    assert (shuffled.groupby("label").label_text.nunique() == 1).all()


def test_the_mapping_is_actually_broken(pair):
    """With 77 balanced classes a row keeps its label by chance about 1.3% of the time."""
    original, shuffled = pair
    report = shuffle.evidence(original, shuffled, "label_text")
    assert report["label_distribution_unchanged"]
    assert report["rows_keeping_their_label"] < 3 * report["chance_of_keeping"]


def test_the_shuffle_is_deterministic():
    original = pd.read_parquet(ROOT / "data/intent/split_train.parquet")
    a = shuffle.shuffle_labels(original, ["label", "label_text"])
    b = shuffle.shuffle_labels(original, ["label", "label_text"])
    pd.testing.assert_frame_equal(a, b)


def test_training_a_non_default_split_without_a_variant_is_refused():
    """Otherwise the shuffled adapter overwrites checkpoints/intent, its M11 checkpoint and
    runs/intent__train.json — the real adapter's provenance."""
    from adapterops.train import cli_train

    assert cli_train.main(["--task", "intent", "--train-split", "train_shuffled"]) == 2


def test_the_training_config_defaults_to_the_real_split():
    from adapterops.train.qlora import config_for

    assert config_for("intent").train_split == "train"
