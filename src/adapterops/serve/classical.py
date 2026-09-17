"""A non-LoRA model served behind the adapter interface (TODO.md §2, urgency against TF-IDF).

Urgency's adapter loses to TF-IDF + logistic regression on every split, and its three seeds scatter so widely
that its gate is barely a check (`runs/urgency__tfidf.json`, `runs/training_variance.json`). So a task can be
pinned to a scikit-learn model instead of a LoRA adapter, and every path that serves or scores a task honours it.

**Pinned the way the router and judge are: a file in the repository and its sha256.** A manifest entry
`{"kind": "sklearn", "path": ..., "sha256": ...}` is loaded only if the file still hashes to the pin, so what
is served cannot drift from what was scored. vLLM never sees the task; the request path and the regression
runner answer it locally.

**It never escalates on confidence.** The confidence threshold was calibrated on adapter log-probabilities, and
a classifier has none, so `mean_logprob` is None and `ConfidencePolicy` keeps the local answer — the same
outcome the routing study argued for independently, since GPT-4o-mini is worse than the local model on urgency.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KIND = "sklearn"


def is_classical(entry: dict | None) -> bool:
    return bool(entry) and entry.get("kind") == KIND


def load_verified(entry: dict, root: Path = REPO_ROOT):
    """The pinned model, after checking the file still hashes to its pin."""
    import joblib

    path = Path(entry["path"])
    path = path if path.is_absolute() else root / path
    if not path.exists():
        msg = f"{entry['path']} is pinned but not on disk"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        msg = (f"{entry['path']} is not the pinned model ({digest[:12]} vs {entry['sha256'][:12]}) — "
               "refusing to serve an unpinned model")
        raise ValueError(msg)
    return joblib.load(path)


def classical_models(components: dict, root: Path = REPO_ROOT) -> dict:
    return {task: load_verified(entry, root) for task, entry in components.items()
            if task != "base_model" and is_classical(entry)}


class ClassicalBackend:
    name = "classical"

    def __init__(self, models: dict) -> None:
        self.models = models

    def generate(self, task: str, text: str):
        from adapterops.serve.pipeline import Generation

        started = time.perf_counter()
        label = str(self.models[task].predict([text])[0])
        return Generation(text=label, mean_logprob=None, n_tokens=0,
                          latency_s=time.perf_counter() - started, finish_reason="stop")


class TaskRoutedBackend:
    """The default backend for every task except those pinned to a classical model."""

    def __init__(self, default, classical: ClassicalBackend) -> None:
        self.default, self.classical = default, classical
        self.name = getattr(default, "name", "local")

    def generate(self, task: str, text: str):
        if task in self.classical.models:
            return self.classical.generate(task, text)
        return self.default.generate(task, text)


def with_classical_backend(local, components: dict, root: Path = REPO_ROOT):
    models = classical_models(components, root)
    return TaskRoutedBackend(local, ClassicalBackend(models)) if models else local


def with_classical_predictor(predictor: Callable[[str, Sequence[str]], list[str]], components: dict,
                             root: Path = REPO_ROOT):
    """A regression predictor that answers classical tasks locally and the rest through `predictor`."""
    models = classical_models(components, root)
    if not models:
        return predictor

    def predict(task: str, texts: Sequence[str]) -> list[str]:
        if task in models:
            return [str(label) for label in models[task].predict(list(texts))]
        return predictor(task, texts)

    predict.stats = getattr(predictor, "stats", {})
    return predict
