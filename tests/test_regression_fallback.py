"""F17 — a regression run records its fallback rate instead of aborting on a failed request."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval import regression
from adapterops.router import generate


def test_failed_requests_score_as_wrong_answers_and_are_counted(monkeypatch):
    calls = {"n": 0}

    def complete(base_url, adapter, prompt, max_tokens, timeout=180.0):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise ConnectionError("server dropped the request")
        return {"prediction": "card_arrival"}

    monkeypatch.setattr(generate, "complete", complete)
    predictor = regression.http_predictor("http://localhost:8000", concurrency=1)
    out = predictor("intent", ["a", "b", "c", "d", "e", "f", "g", "h"])

    assert out.count("") == 2 and len(out) == 8
    summary = regression.fallback_summary(predictor.stats)
    assert summary == {**summary, "requests": 8, "errors": 2, "fallback_rate": 0.25}


def test_no_requests_means_no_rate_rather_than_zero():
    assert regression.fallback_summary({"requests": 0, "errors": 0})["fallback_rate"] is None
