"""The out-of-domain gate in the request path (TODO.md §3; evidence in `runs/ood__detector.json` and
`runs/ood__frontier.json`).

On tickets unlike their training data the adapters were confidently wrong — confidence routing escalated none of
intent's wrong answers — while GPT-4o-mini beat them on intent, urgency and drafting. PII was the exception: it
transferred, and GPT-4o-mini leaked 29% of spans against its 0.6%. So before a pair runs locally, the gate asks
whether the ticket resembles the task's training texts, and a flagged intent, urgency or drafting pair goes
straight to the frontier. PII is never gated.

**Rebuilt at start-up, verified against a pin.** The check (TF-IDF character 3-5 grams, mean cosine of the 5
nearest training texts, or word count) is deterministic in the training and validation splits, which are already
in git and pinned by sha256. Storing the fitted index would add 40 MB to the repository; instead
`manifests/domain_gate.json` records each split's sha256 and each task's thresholds, and the gate refuses to start
if a split has changed or a recomputed threshold differs from the pinned one — so what serves is what was measured.

**Its own route.** A gated pair is `route: out_of_domain`, counted apart from confidence escalations, so the
frontier-call rate the operating point was tuned on stays comparable with every earlier measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_FILE = REPO_ROOT / "manifests" / "domain_gate.json"
GATED_TASKS = ("intent", "urgency", "drafting")
"""PII stays local whatever the gate says: it transferred, and GPT-4o-mini is far worse at it."""
TOLERANCE = 1e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_file(task: str, split: str) -> Path:
    from adapterops.eval.ood_detect import TRAIN_SPLIT

    name = TRAIN_SPLIT[task] if split == "train" else split
    return REPO_ROOT / "data" / task / f"split_{name}.parquet"


def build_pin(tasks: tuple[str, ...] = GATED_TASKS) -> dict:
    """Fit each task's check and record what it was fitted on and the thresholds it produced."""
    from adapterops.eval import ood_detect as od

    pin = {"purpose": "Out-of-domain gate: rebuilt at start-up from these splits and refused if a threshold moves.",
           "method": {"signal": f"mean cosine of {od.NEIGHBOURS} nearest training texts, TF-IDF char 3-5 grams, "
                                "or word count", "flag_quantile_on_validation": od.FLAG_QUANTILE},
           "evidence": ["runs/ood__detector.json", "runs/ood__frontier.json"], "tasks": {}}
    for task in tasks:
        train, val = _split_file(task, "train"), _split_file(task, "val")
        similarity = od.Similarity(od.texts(task, od.TRAIN_SPLIT[task]))
        val_texts = od.texts(task, "val")
        limits = od.thresholds(similarity.score(val_texts), od.length(val_texts))
        pin["tasks"][task] = {
            "train_split": {"file": str(train.relative_to(REPO_ROOT)), "sha256": _sha(train)},
            "val_split": {"file": str(val.relative_to(REPO_ROOT)), "sha256": _sha(val)},
            "thresholds": limits}
    return pin


class DomainGate:
    def __init__(self, pin: dict) -> None:
        from adapterops.eval import ood_detect as od

        self._od = od
        self.checks: dict[str, tuple] = {}
        for task, spec in pin["tasks"].items():
            for part in ("train_split", "val_split"):
                path = REPO_ROOT / spec[part]["file"]
                if _sha(path) != spec[part]["sha256"]:
                    msg = f"{spec[part]['file']} has changed since the domain gate was pinned — rebuild the pin"
                    raise ValueError(msg)
            similarity = od.Similarity(od.texts(task, od.TRAIN_SPLIT[task]))
            val_texts = od.texts(task, "val")
            recomputed = od.thresholds(similarity.score(val_texts), od.length(val_texts))
            for key, pinned in spec["thresholds"].items():
                if not math.isclose(recomputed[key], pinned, rel_tol=0, abs_tol=TOLERANCE):
                    msg = (f"{task} {key} recomputes to {recomputed[key]!r}, the pin says {pinned!r} — refusing to "
                           "serve a gate that is not the one measured")
                    raise ValueError(msg)
            self.checks[task] = (similarity, spec["thresholds"])

    def flags(self, task: str, text: str) -> bool:
        if task not in self.checks:
            return False
        similarity, limits = self.checks[task]
        return bool(similarity.score([text])[0] < limits["similarity_below"]
                    or len(text.split()) > limits["length_above"])


def load(manifest: dict) -> DomainGate | None:
    """The gate a manifest pins, or None if it pins none."""
    pin = (manifest.get("components") or {}).get("domain_gate")
    if not pin:
        return None
    path = REPO_ROOT / pin["file"]
    if _sha(path) != pin["sha256"]:
        msg = f"{pin['file']} is not the domain gate manifest v{manifest.get('version')} pins"
        raise ValueError(msg)
    return DomainGate(json.loads(path.read_text()))


def main() -> int:
    pin = build_pin()
    PIN_FILE.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    for task, spec in pin["tasks"].items():
        print(f"  {task:9s} similarity below {spec['thresholds']['similarity_below']:.6f} · length above "
              f"{spec['thresholds']['length_above']}")
    print(f"  wrote {PIN_FILE.relative_to(REPO_ROOT)}")
    return 0
