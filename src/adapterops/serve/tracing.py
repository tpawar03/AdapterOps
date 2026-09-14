"""Request tracing for the online path (F25, N4).

Three tracers, one interface — `trace(ticket_id, text, result)`:

- `NullTracer` — nothing recorded.
- `JsonlTracer` — one JSON line per ticket in a local file, for a laptop or a GPU session.
- `LangfuseTracer` — Langfuse's public ingestion API: one trace per ticket, one span per
  (ticket, task) pair carrying the route, score, threshold, latency and cost.

**The Langfuse tracer talks to the HTTP API, not the SDK.** `POST /api/public/ingestion` with the
project's public and secret keys as basic auth is the documented, versioned surface both SDK
generations sit on, so this works without adding a dependency whose client API changed between
major versions. It has been exercised against a mock transport, not a live Langfuse project.

**Ticket text is not sent unless asked for.** Traces carry task outputs, routes and costs; the raw
ticket is replaced by its length. The PII adapter's whole job is finding personal data in that
text, and a tracing backend is the wrong place to collect it by default.

**A tracer never breaks a request.** Export failures are counted and swallowed: losing a trace is
a monitoring gap, failing a customer request because a trace did not export is an outage.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path


class NullTracer:
    name = "none"

    def trace(self, ticket_id: str, text: str, result: dict) -> None:
        return None


class JsonlTracer:
    name = "jsonl"

    def __init__(self, path: Path, store_text: bool = False) -> None:
        self.path = Path(path)
        self.store_text = store_text
        self._lock = threading.Lock()

    def trace(self, ticket_id: str, text: str, result: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ticket_id": ticket_id, "at": datetime.now(UTC).isoformat(),
               "input": text if self.store_text else {"chars": len(text)}, "result": result}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


class LangfuseTracer:
    name = "langfuse"

    def __init__(self, host: str, public_key: str, secret_key: str, client=None,
                 store_text: bool = False, timeout: float = 5.0) -> None:
        import httpx

        self.url = host.rstrip("/") + "/api/public/ingestion"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
        self.client = client or httpx.Client(timeout=timeout)
        self.store_text = store_text
        self.sent = 0
        self.failures = 0
        self.last_error: str | None = None

    @classmethod
    def from_env(cls, **kwargs) -> LangfuseTracer:
        public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if not (public and secret):
            msg = "set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and LANGFUSE_HOST if self-hosted)"
            raise RuntimeError(msg)
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return cls(host, public, secret, **kwargs)

    def events(self, ticket_id: str, text: str, result: dict) -> list[dict]:
        end = datetime.now(UTC)
        start = end - timedelta(milliseconds=result.get("latency_ms", 0.0))
        trace_id = str(uuid.uuid4())

        def event(kind: str, body: dict) -> dict:
            return {"id": str(uuid.uuid4()), "timestamp": end.isoformat(), "type": kind,
                    "body": body}

        batch = [event("trace-create", {
            "id": trace_id,
            "name": "ticket",
            "timestamp": start.isoformat(),
            "input": text if self.store_text else {"chars": len(text)},
            "output": {task: p["output"] for task, p in result["pairs"].items()},
            "metadata": {"ticket_id": ticket_id, "manifest_version": result.get("manifest_version"),
                         "policy": result.get("policy"), "cost_usd": result.get("cost_usd"),
                         "latency_ms": result.get("latency_ms")},
            "tags": ["adapterops"],
        })]
        for task, pair in result["pairs"].items():
            pair_end = start + timedelta(milliseconds=pair["latency_ms"])
            batch.append(event("span-create", {
                "id": str(uuid.uuid4()),
                "traceId": trace_id,
                "name": task,
                "startTime": start.isoformat(),
                "endTime": pair_end.isoformat(),
                "output": pair["output"],
                "level": "ERROR" if pair["served_by"] == "none" else "DEFAULT",
                "metadata": {k: pair[k] for k in ("route", "served_by", "policy", "score",
                                                  "threshold", "latency_ms", "cost_usd",
                                                  "judge_score", "local_error",
                                                  "frontier_error")},
            }))
        return batch

    def trace(self, ticket_id: str, text: str, result: dict) -> None:
        try:
            r = self.client.post(self.url, headers=self.headers,
                                 content=json.dumps({"batch": self.events(ticket_id, text, result)}))
            r.raise_for_status()
            self.sent += 1
        except Exception as exc:                          # noqa: BLE001 - counted, never raised
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"


def make_tracer(kind: str, path: Path | None = None):
    if kind == "none":
        return NullTracer()
    if kind == "jsonl":
        return JsonlTracer(path or Path("runs/traces/requests.jsonl"))
    if kind == "langfuse":
        return LangfuseTracer.from_env()
    msg = f"unknown tracer {kind!r}; expected none, jsonl or langfuse"
    raise ValueError(msg)
