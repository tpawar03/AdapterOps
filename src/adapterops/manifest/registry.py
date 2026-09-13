"""Pinned adapter registry (F16, first artifact) — revisions, not `main`.

PRD changelog 29 records why this file exists, and it is not a hypothetical: a stale
clone re-ran training against an old config and force-pushed over a published adapter's
`main`. The recorded score then described weights that were no longer being served, and
nothing in the repo could have detected it. The PII repo still shows the whole sequence —
`5405e954` held the scored weights, `dbb236a9` overwrote them, `3b38a284` restored them.

So a pin here is two hashes, not one. The **revision** identifies a commit; the
**weight digest** identifies the tensors. Only the second survives a force-push: a repo
can be rewritten so that a revision no longer exists, or a different revision serves the
same weights, and `verify()` reports those two cases differently because they mean
different things.

    uv run adapterops pin-adapters     # resolve main -> write the pin
    uv run adapterops verify-pins      # has anything moved since?

`verify-pins` is the check to run before a GPU session that spends money on inference, and
before believing any score attributed to a pinned adapter.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "manifests" / "adapters.json"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TASKS = ("intent", "urgency", "pii", "drafting")
HF_USER = "Tanny03"
WEIGHT_FILE = "adapter_model.safetensors"
CANDIDATES_DIR = REPO_ROOT / "manifests" / "candidates"


def _resolve(repo: str, revision: str = "main") -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    weights = {
        s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id)
        for s in info.siblings
        if s.rfilename == WEIGHT_FILE
    }
    return {
        "repo": repo,
        "revision": info.sha,
        "weight_file": WEIGHT_FILE,
        "weight_sha256": weights.get(WEIGHT_FILE),
        "last_modified": str(info.lastModified),
    }


def pin(force: bool = False) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — adapters are pinned.")
        print("  Re-pinning adopts whatever `main` points at now, which is exactly the")
        print("  move changelog 29 is about. Run verify-pins first. --force if you mean it.")
        return 1

    entries = {"base_model": _resolve(BASE_MODEL)}
    for task in TASKS:
        entries[task] = _resolve(f"{HF_USER}/adapterops-{task}")
        print(f"  {task:9s} {entries[task]['revision'][:8]} "
              f"weights {str(entries[task]['weight_sha256'])[:12]}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "purpose": "Pinned component revisions (F16). Serving and evaluation resolve "
                   "adapters through this file, never through `main`.",
        "why": "changelog 29 — a stale clone force-pushed over a published adapter's "
               "main, and the recorded score then described weights nobody was serving.",
        "verify": "uv run adapterops verify-pins",
        "components": entries,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  pinned {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


def verify(pins: Path | None = None) -> tuple[int, dict]:
    """Compare the pin against the Hub. Returns (exit code, per-component findings)."""
    if not MANIFEST.exists():
        print("  nothing pinned — run `uv run adapterops pin-adapters` first.")
        return 2, {}

    pinned = json.loads((pins or MANIFEST).read_text())["components"]
    findings, worst = {}, "ok"
    for name, entry in pinned.items():
        try:
            at_pin = _resolve(entry["repo"], entry["revision"])
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            findings[name] = {"state": "revision_gone", "detail": f"{type(exc).__name__}"}
            worst = "revision_gone"
            continue

        head = _resolve(entry["repo"])
        same_weights = at_pin["weight_sha256"] == entry["weight_sha256"]
        head_is_pin = head["revision"] == entry["revision"]

        if not same_weights:
            state = "weights_rewritten"      # the pinned revision serves different tensors
            worst = state
        elif head_is_pin:
            state = "ok"
        elif head["weight_sha256"] == entry["weight_sha256"]:
            state = "main_moved_same_weights"
        else:
            state = "main_moved_new_weights"
            worst = worst if worst != "ok" else state

        findings[name] = {
            "state": state,
            "pinned_revision": entry["revision"][:8],
            "head_revision": head["revision"][:8],
            "weight_sha256": str(entry["weight_sha256"])[:12],
        }

    for name, f in findings.items():
        mark = "ok " if f["state"] == "ok" else "!! "
        print(f"  {mark}{name:11s} {f['state']:24s} "
              f"pinned {f.get('pinned_revision', '-')} head {f.get('head_revision', '-')}")

    if worst in ("revision_gone", "weights_rewritten"):
        print("\n  A pinned revision no longer serves the weights it was pinned to.")
        print("  Every score attributed to it is unverified until this is resolved.")
        return 1, findings
    if any(f["state"] != "ok" for f in findings.values()):
        print("\n  `main` has moved past a pin. Not an error — the pin is what is served —")
        print("  but a newer adapter exists and nothing is using it.")
    return 0, findings


def main(action: str = "verify", force: bool = False) -> int:
    if action == "pin":
        return pin(force=force)
    return verify()[0]


def candidate_components(base: dict, task: str, resolved: dict) -> dict:
    """The current pin set with exactly one adapter replaced, recording what it replaced."""
    if task == "base_model" or task not in base:
        msg = f"{task!r} is not a swappable adapter in this pin set"
        raise ValueError(msg)
    out = copy.deepcopy(base)
    out[task] = {**resolved, "candidate_of": (base[task] or {}).get("repo")}
    return out


def pin_candidate(task: str, repo: str, name: str, force: bool = False) -> int:
    """Pin a candidate adapter set for M7 or M11: the current pins with one adapter swapped.

    A candidate is served under the real task name, so the regression run addresses it
    exactly as it would the adapter it might replace. The file records which adapter it
    stands in for, so a served candidate is never mistaken for the pinned system.
    """
    path = CANDIDATES_DIR / f"{name}.json"
    if path.exists() and not force:
        print(f"  {path.relative_to(REPO_ROOT)} exists — --force to re-resolve it")
        return 1
    resolved = _resolve(repo)
    if not resolved["weight_sha256"]:
        print(f"  {repo} has no {WEIGHT_FILE} at main — not a pinnable adapter")
        return 2
    base = json.loads(MANIFEST.read_text())["components"]
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "purpose": "Candidate pin set — the pinned system with one adapter swapped. Served "
                   "with PINS=<this file> for M7 (regressed adapter) or M11 (under-trained).",
        "candidate": name,
        "replaces": task,
        "base_pins": str(MANIFEST.relative_to(REPO_ROOT)),
        "components": candidate_components(base, task, resolved),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  {name}: {task} -> {repo} @ {resolved['revision'][:8]} "
          f"weights {str(resolved['weight_sha256'])[:12]}")
    return 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def combine_candidates(names: list[str], name: str, force: bool = False) -> int:
    """Merge already-pinned candidates into one set, so several swaps share one server.

    Each GPU-session restart costs minutes of load time, and the four M11 checkpoints can be
    served together because the regression run scores every task independently. The merge reads
    the revisions already pinned in each candidate file rather than re-resolving the Hub, so the
    combined set serves exactly the weights the individual pins recorded — and it refuses a file
    built against different base pins, or two files that replace the same task.
    """
    path = CANDIDATES_DIR / f"{name}.json"
    if path.exists() and not force:
        print(f"  {_rel(path)} exists — --force to rebuild it")
        return 1
    base = json.loads(MANIFEST.read_text())["components"]
    combined = copy.deepcopy(base)
    replaced: list[str] = []
    for candidate in names:
        doc = json.loads((CANDIDATES_DIR / f"{candidate}.json").read_text())
        tasks = doc["replaces"] if isinstance(doc["replaces"], list) else [doc["replaces"]]
        for task in tasks:
            if task in replaced:
                msg = f"{task} is replaced by more than one candidate"
                raise ValueError(msg)
        for key, component in doc["components"].items():
            if key not in tasks and component != base.get(key):
                msg = f"{candidate} was built against different pins for {key}"
                raise ValueError(msg)
        for task in tasks:
            combined[task] = doc["components"][task]
            replaced.append(task)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "purpose": "Combined candidate pin set — several swaps served together on one server.",
        "candidate": name,
        "replaces": replaced,
        "combined_from": names,
        "base_pins": _rel(MANIFEST),
        "components": combined,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  {name}: replaces {', '.join(replaced)} — from {', '.join(names)}")
    return 0
