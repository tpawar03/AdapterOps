"""FastAPI front of the request path (PRD §8, §12's orchestration layer).

Thin on purpose: every decision lives in `serve/pipeline.py`, which the demo and the tests drive
directly. This module only turns it into HTTP.

    POST /v1/tickets   {"text": "...", "tasks": ["intent", "pii"]}   one routed answer per task
    GET  /v1/metrics   live frontier-call rate, fallback rate, P95 latency, cost per 1K
    GET  /v1/service   manifest version, policy and threshold, backends
    GET  /healthz

    uv run adapterops serve-api --backend vllm --base-url http://localhost:8000
    uv run adapterops serve-api --backend transformers --no-frontier      # laptop, no spend
"""

# No `from __future__ import annotations`: FastAPI resolves the request model from the live
# annotation, and a deferred string annotation on a module-level model is where that breaks.

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from adapterops import __version__
from adapterops.serve.pipeline import Service

Task = Literal["intent", "urgency", "pii", "drafting"]
MAX_THREADS = 256
"""Concurrent tickets the API will run. A sync endpoint runs in anyio's worker pool, which defaults
to 40 threads — so without raising it, any load test past 40 concurrent requests measures that pool
rather than vLLM. Each ticket blocks one thread for the whole of its generation."""


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    tasks: list[Task] | None = None
    ticket_id: str | None = Field(default=None, max_length=64)


def create_app(service: Service, max_threads: int = MAX_THREADS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = max_threads
        yield

    app = FastAPI(title="AdapterOps request path", version=__version__, lifespan=lifespan)

    @app.post("/v1/tickets")
    def tickets(body: TicketIn) -> dict:
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text is blank")
        tasks = tuple(dict.fromkeys(body.tasks)) if body.tasks else None
        return service.handle(body.text, tasks or ("intent", "urgency", "pii", "drafting"),
                              ticket_id=body.ticket_id)

    @app.get("/v1/metrics")
    def metrics() -> dict:
        return service.metrics.snapshot()

    @app.get("/v1/service")
    def describe() -> dict:
        return service.describe()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "manifest_version": service.manifest.get("version")}

    return app


def main(backend: str, base_url: str, policy: str, frontier: bool, judge: bool, trace: str,
         host: str, port: int, benchmark_caps: bool = False,
         max_threads: int = MAX_THREADS, frontier_per_minute: int | None = None,
         frontier_per_day: int | None = None, pii_guard: bool = True, domain_gate: bool = True) -> int:
    import uvicorn

    from adapterops.serve.pipeline import build_service
    from adapterops.serve.tracing import make_tracer

    service = build_service(backend=backend, base_url=base_url, policy=policy, frontier=frontier,
                            judge=judge, tracer=make_tracer(trace), benchmark_caps=benchmark_caps,
                            frontier_per_minute=frontier_per_minute,
                            frontier_per_day=frontier_per_day, pii_guard=pii_guard,
                            domain_gate=domain_gate)
    print("  " + "  ·  ".join(f"{k}: {v}" for k, v in service.describe().items()))
    uvicorn.run(create_app(service, max_threads=max_threads), host=host, port=port)
    return 0
