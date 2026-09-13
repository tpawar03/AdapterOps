"""Guards on the derived cost figures (D41).

The break-even test is the one that matters: it pins the reading that local serving is cheaper
only above a sustained request rate, rather than by a fixed multiple.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import economics as ec

PRICE = {"input": 0.15, "output": 0.60}


def test_frontier_usd_prices_input_and_output_separately():
    assert ec.frontier_usd(1_000_000, 1_000_000, PRICE) == pytest.approx(0.75)


def test_local_cost_per_1k_from_hourly_rate_and_throughput():
    # $0.75/h at 23.6 rps is 84,960 requests an hour
    assert ec.local_usd_per_1k(0.75, 23.6) == pytest.approx(0.75 / 84.96)


def test_break_even_is_where_an_hour_of_gpu_buys_the_same_frontier_calls():
    rps = ec.break_even_rps(0.75, frontier_usd_per_1k=0.0573)
    assert ec.local_usd_per_1k(0.75, rps) == pytest.approx(0.0573)


def test_a_pair_called_twice_is_priced_once():
    rows = [
        {"pair_id": "intent-r0001", "prompt_tokens": 400, "completion_tokens": 5, "error": None},
        {"pair_id": "intent-r0001", "prompt_tokens": 400, "completion_tokens": 5, "error": None},
        {"pair_id": "pii-r0001", "prompt_tokens": 200, "completion_tokens": 50, "error": None},
    ]
    costs = ec.frontier_costs(rows, PRICE)
    assert costs["pooled"]["requests"] == 2
    assert costs["per_task"]["intent"]["prompt_tokens"] == 400


def test_errored_calls_are_not_priced():
    rows = [{"pair_id": "pii-r0001", "prompt_tokens": 200, "completion_tokens": 0,
             "error": "RateLimitError"}]
    assert ec.frontier_costs(rows, PRICE)["pooled"] is None


def test_footprint_compares_one_base_plus_adapters_with_a_full_copy_per_task():
    f = ec.footprint(3_000, {"a": 100, "b": 100})
    assert f["one_base_plus_adapters_bytes"] == 3_200
    assert f["full_copy_per_task_bytes"] == 6_000
    assert "not GPU memory" in f["note"]


def test_derive_records_what_was_not_measured():
    rows = [{"pair_id": "intent-r0001", "prompt_tokens": 400, "completion_tokens": 5,
             "error": None}]
    out = ec.derive(rows, {"throughput_rps": 23.6}, PRICE, 3_000, {"intent": 100})
    assert set(out["not_measured"]) == {"gpu_memory_at_serving", "frontier_latency",
                                        "max_local_throughput"}
    assert out["local"]["per_task"] is None
