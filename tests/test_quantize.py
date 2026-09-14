"""Guards on the int8 comparison's accounting (F27) — the parts a wrong answer would hide in."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from adapterops.eval import quantize as q


class Tied(torch.nn.Module):
    """An embedding tied to an output head, as Qwen2.5-1.5B ties them, plus one ordinary Linear."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.proj = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 10, bias=False)
        self.head.weight = self.embed.weight


def test_round_trip_lands_on_int8_levels_one_scale_per_row():
    torch.manual_seed(0)
    w = torch.randn(3, 5)
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    levels = q.round_trip_int8(w) / scale
    assert torch.allclose(levels, levels.round(), atol=1e-4)
    assert levels.abs().max() <= 127 + 1e-4
    assert (q.round_trip_int8(w) - w).abs().max() <= scale.max() / 2 + 1e-6


def test_fp32_size_counts_a_tied_matrix_once():
    assert q.parameter_bytes(Tied()) == (40 + 16 + 4) * 4


def test_weight_only_simulation_breaks_the_tie_and_keeps_the_embedding_exact():
    torch.manual_seed(0)
    model = Tied()
    embedding = model.embed.weight.detach().clone()
    size = q.simulate_weight_only_int8(model)

    assert torch.equal(model.embed.weight, embedding), "the embedding lookup was quantized"
    assert model.head.weight.data_ptr() != model.embed.weight.data_ptr()
    # embedding (40) and proj bias (4) stay fp32; proj 16 int8 + 4 row scales; head 40 int8 + 10 scales
    assert size == 40 * 4 + 4 * 4 + (16 + 4 * 4) + (40 + 10 * 4)


def test_a_quick_run_keeps_every_class():
    import pandas as pd

    frame = pd.DataFrame({"label": ["a"] * 5 + ["b"] * 5, "text": list("abcdefghij")})
    sample = q.balanced_sample(frame, "label", 2)
    assert sample.label.value_counts().to_dict() == {"a": 2, "b": 2}
