"""Guards on serving a task from a pinned scikit-learn model behind the adapter interface."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import classical as cl


def pinned_model(tmp_path, label="high"):
    model = DummyClassifier(strategy="constant", constant=label).fit([[0], [1]], [label, "low"])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return {"kind": "sklearn", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Adapter:
    name = "vllm"

    def __init__(self):
        self.calls = []

    def generate(self, task, text):
        from adapterops.serve.pipeline import Generation

        self.calls.append(task)
        return Generation(text="adapter answer", mean_logprob=-0.2)


def test_a_modified_model_file_is_refused(tmp_path):
    entry = pinned_model(tmp_path)
    Path(entry["path"]).write_bytes(Path(entry["path"]).read_bytes() + b"x")
    with pytest.raises(ValueError, match="not the pinned model"):
        cl.load_verified(entry)


def test_only_the_classical_task_bypasses_the_adapter_backend(tmp_path):
    adapter = Adapter()
    components = {"base_model": {"repo": "b"}, "intent": {"repo": "i", "revision": "r"},
                  "urgency": pinned_model(tmp_path)}
    backend = cl.with_classical_backend(adapter, components)
    urgent = backend.generate("urgency", "server down, customers blocked")
    intent = backend.generate("intent", "where is my card")
    assert urgent.text == "high" and urgent.mean_logprob is None
    assert intent.text == "adapter answer" and adapter.calls == ["intent"]


def test_no_classical_pin_leaves_the_backend_untouched(tmp_path):
    adapter = Adapter()
    assert cl.with_classical_backend(adapter, {"intent": {"repo": "i", "revision": "r"}}) is adapter


def test_the_regression_predictor_answers_classical_tasks_locally_and_keeps_stats(tmp_path):
    seen = []

    def http(task, texts):
        seen.append(task)
        return ["x"] * len(texts)

    http.stats = {"intent": {"fallbacks": 0}}
    predict = cl.with_classical_predictor(http, {"urgency": pinned_model(tmp_path, "low")})
    assert predict("urgency", ["a", "b"]) == ["low", "low"] and seen == []
    assert predict("intent", ["a"]) == ["x"] and seen == ["intent"]
    assert predict.stats == {"intent": {"fallbacks": 0}}


def test_confidence_routing_keeps_a_classical_answer_local(tmp_path):
    from adapterops.serve.pipeline import ConfidencePolicy

    backend = cl.ClassicalBackend({"urgency": cl.load_verified(pinned_model(tmp_path))})
    decision = ConfidencePolicy(0.39).decide_after("urgency", "t", backend.generate("urgency", "t"))
    assert decision.escalate is False


def test_vllm_launch_skips_a_classical_pin(tmp_path, monkeypatch):
    from adapterops.serve import launch

    pins = tmp_path / "pins.json"
    local_adapter = tmp_path / "intent"
    local_adapter.mkdir()
    pins.write_text(json.dumps({"components": {"base_model": {"repo": "b", "revision": "r"},
                                               "intent": {"path": str(local_adapter)},
                                               "urgency": pinned_model(tmp_path)}}))
    assert set(launch.materialise(pins)) == {"intent"}
