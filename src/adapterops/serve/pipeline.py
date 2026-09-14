"""The online request path (PRD §8) — one ticket in, one routed answer per task out.

Everything before this module evaluated routing offline, on recorded pairs. This is the path §8
drew: a ticket is split into (ticket, task) pairs, each pair is routed, answered locally or by
GPT-4o-mini, the drafting reply is scored by the distilled judge, and the six live quantities
§11 monitors — frontier-call rate, fallback rate, latency and cost among them — are counted as
requests go by rather than derived afterwards.

**Routing decisions reuse the committed operating point, not a new threshold.** Each policy's
threshold is the one its operating curve recorded at a 20% escalation budget
(`runs/router__operating_curve__judged.json`), so the live path makes the decision the published
curve measured. Confidence is the default because it is the policy that paid (57% of the oracle's
gain); the learned router is selectable and loses on the same evidence.

- **Confidence** is a cascade: the adapter always runs, and the pair escalates when its mean
  token log-probability is low enough — `-mean_logprob >= threshold`, the curve's own ordering.
- **The learned router** decides before the adapter runs, on `P(fail) >= threshold`.

**A fallback is not an escalation** (§11). An escalation is the policy choosing the frontier. A
fallback is local inference failing — an exception, a timeout, or output the task cannot use: an
intent or urgency label outside the label set, or PII lines that do not parse. Both reach
GPT-4o-mini; they are counted apart, because a rising fallback rate is a serving defect and a
rising escalation rate is a routing one.

**Thresholds were measured on vLLM log-probabilities.** The transformers backend computes the same
quantity on CPU or MPS, in a different precision, so its decisions near the threshold can differ.
It is the demo path; the vLLM backend is the one the operating point describes.

**Serving is pinned by `manifests/system.json`** (M6): adapter revisions, the router checkpoint and
the judge checkpoint come from the manifest, and a router or judge file whose sha256 no longer
matches its pin refuses to load.

    uv run adapterops serve-api --backend vllm --base-url http://localhost:8000
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import pandas as pd

from adapterops.eval.spans import parse_model_output
from adapterops.train.qlora import COLUMNS, PROMPTS

REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM = REPO_ROOT / "manifests" / "system.json"
CURVE = REPO_ROOT / "runs" / "router__operating_curve__judged.json"
ECONOMICS = REPO_ROOT / "runs" / "economics.json"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"

TASKS = ("intent", "urgency", "pii", "drafting")
OPERATING_BUDGET = 0.2
POLICIES = ("confidence", "router", "never")
SERVICE_SPEND_CAP_USD = 1.00
"""Frontier spend a single service process may incur before refusing to escalate. A backstop under
the account cap, like the batch runs' — a request path that escalates in a loop should stop itself."""

Judge = Callable[[Sequence[str], Sequence[str]], Sequence[float]]


@dataclass
class Generation:
    text: str
    mean_logprob: float | None = None
    n_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    def generate(self, task: str, text: str) -> Generation: ...


@dataclass
class Decision:
    escalate: bool
    score: float | None
    threshold: float
    policy: str


# ------------------------------------------------------------------ manifest


def load_manifest(path: Path = SYSTEM) -> dict:
    return json.loads(path.read_text())


def adapter_components(manifest: dict) -> dict:
    """Base model plus the four adapter pins, in the shape `manifests/adapters.json` uses."""
    components = manifest["components"]
    return {"base_model": components["base_model"], **components["adapters"]}


def pinned_checkpoint(manifest: dict, component: str, root: Path = REPO_ROOT) -> Path | None:
    """The directory of a pinned router or judge, after checking its weights still match the pin."""
    pins = manifest["components"].get(component)
    if not pins:
        return None
    weights = next(v for k, v in pins.items() if k.endswith("model.safetensors"))
    path = root / weights["file"]
    if not path.exists():
        msg = f"{component} pinned at {weights['file']} is not on disk"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != weights["sha256"]:
        msg = (f"{weights['file']} is not the {component} manifest v{manifest['version']} pins "
               f"({digest[:12]} vs {weights['sha256'][:12]}) — refusing to serve an unpinned model")
        raise ValueError(msg)
    return path.parent


def operating_threshold(policy: str, budget: float = OPERATING_BUDGET,
                        population: str = "router_in_distribution", path: Path = CURVE) -> float:
    """The threshold a policy's committed operating curve used at `budget`."""
    curve = json.loads(path.read_text())["populations"][population]["all_tasks"]["curve"]
    row = next((r for r in curve if r["policy"] == policy and r["budget"] == budget), None)
    if row is None or row["threshold"] is None:
        msg = f"no {policy} threshold at budget {budget} in {path.name}"
        raise KeyError(msg)
    return float(row["threshold"])


def load_vocab(golden_dir: Path = GOLDEN_DIR) -> dict[str, frozenset[str]]:
    """Label sets for the two classification tasks, read from the frozen golden sets."""
    vocab = {}
    for task in ("intent", "urgency"):
        column = COLUMNS[task][1]
        labels = pd.read_parquet(golden_dir / f"{task}.parquet")[column].astype(str).str.strip()
        vocab[task] = frozenset(labels)
    return vocab


# ------------------------------------------------------------------ outputs


def first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def malformed(task: str, text: str, vocab: dict[str, frozenset[str]]) -> str | None:
    """Why an output is unusable for its task, or None. A usable output that is *wrong* is not
    malformed — that is a quality question for the gate, not a fallback."""
    if task in ("intent", "urgency"):
        label = first_line(text)
        return None if label in vocab[task] else f"not a {task} label: {label[:40]!r}"
    if task == "pii":
        lines = [line for line in text.splitlines() if line.strip()]
        unparsed = len(lines) - len(parse_model_output(text))
        return f"{unparsed} unparseable PII line(s)" if unparsed else None
    return None if text.strip() else "empty reply"


def parse_output(task: str, text: str) -> dict:
    if task in ("intent", "urgency"):
        return {"label": first_line(text)}
    if task == "pii":
        return {"spans": [{"label": label, "value": value}
                          for label, value in parse_model_output(text)]}
    return {"reply": text.strip()}


# ------------------------------------------------------------------ backends


class VLLMBackend:
    """The served adapters, addressed by task name as `scripts/phase2_serve.sh` registers them."""

    name = "vllm"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, task: str, text: str) -> Generation:
        from adapterops.router.generate import MAX_TOKENS, complete

        out = complete(self.base_url, task, PROMPTS[task].format(text=text), MAX_TOKENS[task],
                       timeout=self.timeout)
        return Generation(text=out["prediction"], mean_logprob=out["mean_logprob"],
                          n_tokens=out["n_tokens"], latency_s=out["latency_s"],
                          finish_reason=out["finish_reason"])


class TransformersBackend:
    """The same pinned adapters with transformers + peft, on CPU or MPS — the demo path, not the
    benchmark. Adapters share one base model, so generation is serialised."""

    name = "transformers"
    MAX_NEW_TOKENS: ClassVar[dict[str, int]] = {"intent": 12, "urgency": 6, "pii": 384,
                                                "drafting": 200}
    """Benchmark caps, except drafting: 448 on the A10, 200 here to keep a CPU reply short."""

    def __init__(self, components: dict, benchmark_caps: bool = False) -> None:
        """`benchmark_caps` uses the generation run's caps (drafting 448). A drafting reply cut at
        200 tokens has a different mean log-probability from the one the operating curve ranked,
        so a run measuring routing against the curve needs them; the demo does not."""
        self.components = components
        if benchmark_caps:
            from adapterops.router.generate import MAX_TOKENS

            self.MAX_NEW_TOKENS = dict(MAX_TOKENS)
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base = self.components["base_model"]
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "mps" else torch.float32
            tok = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
            model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"],
                                                         dtype=dtype)
            for i, task in enumerate(TASKS):
                entry = self.components[task]
                # A local checkpoint (a retrained candidate, never published) loads from disk.
                path = entry.get("path") or snapshot_download(
                    entry["repo"], revision=entry["revision"],
                    allow_patterns=["adapter_model.safetensors", "adapter_config.json"])
                if i == 0:
                    model = PeftModel.from_pretrained(model, path, adapter_name=task)
                else:
                    model.load_adapter(path, adapter_name=task)
            self._model = (tok, model.to(device).eval(), device)
        return self._model

    def generate(self, task: str, text: str) -> Generation:
        import torch

        with self._lock:
            tok, model, device = self._load()
            model.set_adapter(task)
            cap = self.MAX_NEW_TOKENS[task]
            ids = tok(PROMPTS[task].format(text=text), return_tensors="pt",
                      add_special_tokens=False).to(device)
            started = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=cap, do_sample=False,
                                     pad_token_id=tok.eos_token_id, output_logits=True,
                                     return_dict_in_generate=True)
            elapsed = time.perf_counter() - started
            new = out.sequences[0, ids["input_ids"].shape[1]:]
            # Raw logits, not `out.scores`: Qwen's generation_config applies repetition_penalty
            # 1.1 even to greedy decoding, and the processed scores it leaves behind put drafting's
            # mean log-probability 0.115 above vLLM's on the same replies (teacher-forced, 40
            # recorded replies; raw logits sit 0.003 from vLLM). Short labels barely repeat, so
            # only drafting moved — and with it most of the curve's escalations.
            scores = model.compute_transition_scores(out.sequences, out.logits,
                                                     normalize_logits=True)[0]
            logprobs = [float(x) for x in scores[:len(new)] if math.isfinite(float(x))]
        cut = len(new) >= cap and int(new[-1]) != tok.eos_token_id
        return Generation(text=tok.decode(new, skip_special_tokens=True),
                          mean_logprob=statistics.fmean(logprobs) if logprobs else None,
                          n_tokens=len(new), latency_s=elapsed,
                          finish_reason="length" if cut else "stop")


class OpenAIFrontier:
    """GPT-4o-mini with the escalation arm's prompts (F8) — the arm the operating curve measured."""

    name = "gpt-4o-mini"

    def __init__(self, intent_vocab: Sequence[str], client=None, timeout: float = 30.0,
                 spend_cap: float = SERVICE_SPEND_CAP_USD) -> None:
        from adapterops.router.frontier import Ledger

        self.intent_vocab = ", ".join(sorted(intent_vocab))
        self.ledger = Ledger(cap=spend_cap, request_cap=5_000)
        self._client = client
        self.timeout = timeout

    def generate(self, task: str, text: str) -> Generation:
        from adapterops.router.frontier import MAX_TOKENS, MODEL, build_prompt

        if self.ledger.stopped:
            msg = f"frontier disabled for this process: {self.ledger.reason}"
            raise RuntimeError(msg)
        if self._client is None:
            from openai import OpenAI

            # One attempt: a request path that retries a rate limit inside a request only
            # lengthens the request. The pair fails, and that is counted.
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        self.ledger.count_request()
        started = time.perf_counter()
        r = self._client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": build_prompt(task, text,
                                                                            self.intent_vocab)}],
            max_completion_tokens=MAX_TOKENS[task], temperature=0.0)
        elapsed = time.perf_counter() - started
        self.ledger.add(r.usage.prompt_tokens, r.usage.completion_tokens)
        return Generation(text=(r.choices[0].message.content or "").strip(), latency_s=elapsed,
                          finish_reason=r.choices[0].finish_reason,
                          prompt_tokens=r.usage.prompt_tokens,
                          completion_tokens=r.usage.completion_tokens)


# ------------------------------------------------------------------ policies


class ConfidencePolicy:
    """Escalate when the adapter's own mean log-probability is low (F9). Runs after the adapter."""

    name = "confidence"
    before_local = False

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def decide_after(self, task: str, text: str, generation: Generation) -> Decision:
        if generation.mean_logprob is None:
            # Nothing to rank on. Keep the local answer and say so, rather than guessing.
            return Decision(False, None, self.threshold, self.name)
        score = -generation.mean_logprob
        return Decision(score >= self.threshold, score, self.threshold, self.name)


class RouterPolicy:
    """The learned DeBERTa router (F10): escalate when P(adapter fails) is high. Runs before it."""

    name = "router_p_fail"
    before_local = True

    def __init__(self, checkpoint: Path, threshold: float, max_length: int = 256) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.threshold = threshold
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, dtype=torch.float32).eval()
        self._lock = threading.Lock()

    def decide_before(self, task: str, text: str) -> Decision:
        import torch

        from adapterops.router.train import to_text

        enc = self.tok(to_text(pd.DataFrame({"task": [task], "text": [text]})), truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        with self._lock, torch.no_grad():
            p_fail = float(torch.softmax(self.model(**enc).logits, dim=-1)[0, 1])
        return Decision(p_fail >= self.threshold, p_fail, self.threshold, self.name)


class NeverEscalate:
    """Always-cheap (F8). Local failures still fall back — a fallback is not a routing choice."""

    name = "never"
    before_local = False
    threshold = math.inf

    def decide_after(self, task: str, text: str, generation: Generation) -> Decision:
        return Decision(False, None, self.threshold, self.name)


# ------------------------------------------------------------------ cost and metrics


@dataclass
class Prices:
    local_usd_per_request: float
    frontier_input_per_1m: float
    frontier_output_per_1m: float
    source: str

    @classmethod
    def from_economics(cls, path: Path = ECONOMICS) -> Prices:
        """Derived, not billed (D41): the A10 hour spread over M1's measured throughput."""
        a = json.loads(path.read_text())["assumptions"]
        return cls(local_usd_per_request=a["gpu_usd_per_hour"] / 3600 / a["local_throughput_rps"],
                   frontier_input_per_1m=a["frontier_price_per_1m"]["input"],
                   frontier_output_per_1m=a["frontier_price_per_1m"]["output"],
                   source=str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT)
                   else str(path))

    def frontier(self, g: Generation) -> float:
        return (g.prompt_tokens * self.frontier_input_per_1m
                + g.completion_tokens * self.frontier_output_per_1m) / 1e6


class Metrics:
    """Live counters since process start — §11's frontier-call rate, fallback rate, P95 latency and
    cost per 1K, measured on the traffic this process served."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tickets = 0
        self.cost_usd = 0.0
        self.per_task: dict[str, dict] = {}

    def record(self, pairs: Sequence[dict]) -> None:
        with self._lock:
            self.tickets += 1
            for p in pairs:
                t = self.per_task.setdefault(p["task"], {"pairs": 0, "escalated": 0,
                                                         "fallbacks": 0, "unanswered": 0,
                                                         "latency_ms": []})
                t["pairs"] += 1
                t["escalated"] += p["route"] == "escalated"
                t["fallbacks"] += p["route"] == "fallback"
                t["unanswered"] += p["served_by"] == "none"
                t["latency_ms"].append(p["latency_ms"])
                self.cost_usd += p["cost_usd"]

    def snapshot(self) -> dict:
        with self._lock:
            pairs = sum(t["pairs"] for t in self.per_task.values())

            def rate(key: str, rows: Sequence[dict]) -> float | None:
                n = sum(r["pairs"] for r in rows)
                return round(sum(r[key] for r in rows) / n, 4) if n else None

            def p95(values: list[float]) -> float | None:
                if not values:
                    return None
                ordered = sorted(values)
                return round(ordered[int(0.95 * (len(ordered) - 1))], 1)

            return {
                "since": "process start",
                "tickets": self.tickets,
                "pairs": pairs,
                "frontier_call_rate": rate("escalated", list(self.per_task.values())),
                "fallback_rate": rate("fallbacks", list(self.per_task.values())),
                "cost_usd": round(self.cost_usd, 6),
                "cost_per_1k_pairs_usd": round(self.cost_usd / pairs * 1000, 4) if pairs else None,
                "cost_per_1k_tickets_usd": (round(self.cost_usd / self.tickets * 1000, 4)
                                            if self.tickets else None),
                "per_task": {
                    task: {"pairs": t["pairs"],
                           "frontier_call_rate": rate("escalated", [t]),
                           "fallback_rate": rate("fallbacks", [t]),
                           "unanswered": t["unanswered"],
                           "p95_ms": p95(t["latency_ms"])}
                    for task, t in sorted(self.per_task.items())
                },
            }


# ------------------------------------------------------------------ the service


class Service:
    def __init__(self, local: Backend, policy, frontier: Backend | None = None,
                 judge: Judge | None = None, tracer=None, prices: Prices | None = None,
                 vocab: dict[str, frozenset[str]] | None = None,
                 manifest: dict | None = None) -> None:
        from adapterops.serve.tracing import NullTracer

        self.local = local
        self.policy = policy
        self.frontier = frontier
        self.judge = judge
        self.tracer = tracer or NullTracer()
        self.prices = prices or Prices.from_economics()
        self.vocab = vocab or load_vocab()
        self.manifest = manifest or {}
        self.metrics = Metrics()

    def describe(self) -> dict:
        return {
            "manifest_version": self.manifest.get("version"),
            "local_backend": getattr(self.local, "name", type(self.local).__name__),
            "policy": self.policy.name,
            "threshold": self.policy.threshold,
            "operating_point": f"{OPERATING_BUDGET:.0%} escalation budget, {CURVE.name}",
            "frontier": getattr(self.frontier, "name", None),
            "judge": self.judge is not None,
            "prices": self.prices.source,
        }

    def handle(self, text: str, tasks: Sequence[str] = TASKS, ticket_id: str | None = None) -> dict:
        unknown = sorted(set(tasks) - set(TASKS))
        if unknown:
            msg = f"unknown task(s) {unknown}; expected a subset of {list(TASKS)}"
            raise ValueError(msg)
        ticket_id = ticket_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            pairs = list(pool.map(lambda task: self._pair(task, text), tasks))
        result = {
            "ticket_id": ticket_id,
            "manifest_version": self.manifest.get("version"),
            "policy": self.policy.name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "cost_usd": round(sum(p["cost_usd"] for p in pairs), 8),
            "pairs": {p["task"]: p for p in pairs},
        }
        self.metrics.record(pairs)
        self.tracer.trace(ticket_id, text, result)
        return result

    def _run_local(self, task: str, text: str) -> tuple[Generation | None, str | None]:
        try:
            generation = self.local.generate(task, text)
        except Exception as exc:                          # noqa: BLE001 - a fallback, counted
            return None, f"{type(exc).__name__}: {exc}"
        problem = malformed(task, generation.text, self.vocab)
        return generation, (f"malformed output: {problem}" if problem else None)

    def _pair(self, task: str, text: str) -> dict:
        started = time.perf_counter()
        cost, decision, local, local_error, notes = 0.0, None, None, None, []

        if self.policy.before_local:
            decision = self.policy.decide_before(task, text)
        if not (decision and decision.escalate):
            local, local_error = self._run_local(task, text)
            cost += self.prices.local_usd_per_request
            if local_error is None and not self.policy.before_local:
                decision = self.policy.decide_after(task, text, local)

        if decision and decision.escalate:
            route = "escalated"
        elif local_error:
            route = "fallback"
        else:
            route = "local"

        final, served_by, frontier_error = local, "local", None
        if route != "local":
            if self.frontier is None:
                frontier_error = "no frontier configured"
                if local is None and local_error is None:
                    # The router escalated before the adapter ran; answer locally instead.
                    local, local_error = self._run_local(task, text)
                    cost += self.prices.local_usd_per_request
                    final = local
                notes.append("would reach GPT-4o-mini, but no frontier is configured — answered "
                             "locally" if local_error is None else
                             "local inference failed and no frontier is configured")
            else:
                try:
                    final = self.frontier.generate(task, text)
                    cost += self.prices.frontier(final)
                    served_by = "frontier"
                except Exception as exc:                  # noqa: BLE001 - reported per pair
                    frontier_error = f"{type(exc).__name__}: {exc}"
                    final = local
            if served_by == "local" and (final is None or local_error):
                served_by = "none"

        judge_score = None
        if task == "drafting" and served_by == "local" and self.judge is not None:
            # The judge scores the local adapter only (§10): it was calibrated on adapter-like
            # replies and fails on another generator's (changelog 33).
            judge_score = round(float(self.judge([text], [final.text])[0]), 3)

        confidence = (math.exp(local.mean_logprob)
                      if local is not None and local.mean_logprob is not None else None)
        return {
            "task": task,
            "route": route,
            "served_by": served_by,
            "policy": self.policy.name,
            "score": None if decision is None or decision.score is None else round(decision.score, 6),
            "threshold": None if math.isinf(self.policy.threshold) else self.policy.threshold,
            "predicted_success": (round(1 - decision.score, 4)
                                  if decision and decision.policy == "router_p_fail"
                                  and decision.score is not None else None),
            "mean_token_probability": None if confidence is None else round(confidence, 4),
            "output": None if final is None else parse_output(task, final.text),
            "valid": final is not None and malformed(task, final.text, self.vocab) is None,
            "finish_reason": None if final is None else final.finish_reason,
            "judge_score": judge_score,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "cost_usd": round(cost, 8),
            "local_error": local_error,
            "frontier_error": frontier_error,
            "notes": notes,
        }


def make_policy(name: str, manifest: dict, curve: Path = CURVE):
    if name == "confidence":
        return ConfidencePolicy(operating_threshold("confidence", path=curve))
    if name == "router":
        checkpoint = pinned_checkpoint(manifest, "router")
        if checkpoint is None:
            msg = "the manifest pins no router"
            raise ValueError(msg)
        return RouterPolicy(checkpoint, operating_threshold("router_p_fail", path=curve))
    if name == "never":
        return NeverEscalate()
    msg = f"unknown policy {name!r}; expected one of {POLICIES}"
    raise ValueError(msg)


def build_service(backend: str = "vllm", base_url: str = "http://localhost:8000",
                  policy: str = "confidence", frontier: bool = True, judge: bool = True,
                  tracer=None, local: Backend | None = None,
                  benchmark_caps: bool = False) -> Service:
    """A service wired from the manifest. The frontier is used only if an OpenAI key is present."""
    manifest = load_manifest()
    vocab = load_vocab()
    if local is None:
        local = (VLLMBackend(base_url) if backend == "vllm"
                 else TransformersBackend(adapter_components(manifest),
                                          benchmark_caps=benchmark_caps))
    front = None
    if frontier:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        if os.environ.get("OPENAI_API_KEY"):
            front = OpenAIFrontier(vocab["intent"])
    scorer = None
    if judge:
        checkpoint = pinned_checkpoint(manifest, "judge")
        if checkpoint is not None:
            from adapterops.judge.score import load_judge

            scorer = load_judge(checkpoint)
    return Service(local=local, policy=make_policy(policy, manifest), frontier=front,
                   judge=scorer, tracer=tracer, vocab=vocab, manifest=manifest)
