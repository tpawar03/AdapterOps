import json

import pandas as pd

from adapterops.serve import latency_profile as lp


class FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self.lines


def test_a_stream_is_split_into_first_token_and_decode(monkeypatch):
    import requests

    lines = [b"data: " + json.dumps({"choices": [{"text": ""}]}).encode(), b"",
             b"data: " + json.dumps({"choices": [{"text": "NAME"}]}).encode(),
             b"data: " + json.dumps({"choices": [{"text": ": Jo"}]}).encode(),
             b"data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 3}}).encode(),
             b"data: [DONE]"]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream(lines))
    row = lp.stream_one("http://x", "pii", "Jo called")
    assert row["tokens"] == 3 and row["ttft_ms"] <= row["total_ms"] and row["tpot_ms"] is not None


def test_the_summary_counts_pairs_inside_the_target():
    rows = pd.DataFrame({"task": ["pii"] * 4, "ttft_ms": [100, 200, 300, 900],
                         "total_ms": [400, 1000, 2000, 3000], "tokens": [1, 50, 80, 120],
                         "tpot_ms": [None, 16, 21, 17]})
    s = lp.summarise(rows)["pii"]
    assert s["ttft_within_target"] == 0.75 and s["total_within_target"] == 0.25
