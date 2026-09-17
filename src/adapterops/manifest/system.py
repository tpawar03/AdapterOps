"""System manifest (F16) — one file pinning every component of one deployable version.

The adapter registry pins weights. This pins *a system*: which adapters, which router,
which judge, and — the part that is easy to leave out and the reason half the value is
here — **which eval splits**. A score is a statement about a model measured against a bar.
Pinning the model and letting the bar move means a changed number has two explanations and
no way to separate them, so the splits are pinned beside the models and a moved bar shows
up as a diff rather than as an unexplained result.

**Promotion is blocked, not warned about** (F19), and **rollback restores a previous
version whole** (F20). Together those are M7, the project's distinguishing claim.

**The gate starts report-only, deliberately.** PRD §11 corrects v2.1 on exactly this: the
hard-cases split sits by construction near the decision boundary and will swing several
points between identical runs, so "any drop blocks" reintroduces the unfalsifiable gate
that raising the golden sets from 100 to 300 existed to remove. A threshold is legitimate
only once it has been derived from two baseline runs against an unchanged manifest (F33).
Until then `gate.state` is `report_only`, and `promote()` refuses to enforce a threshold
that does not exist rather than inventing one.

**The manifest is what gets served (M6).** `serve/launch.py` and the request path read adapter
revisions, the router and the judge from `system.json`, so a promotion or a rollback changes the
next launch.

    uv run adapterops manifest show
    uv run adapterops manifest promote --note "router v1"
    uv run adapterops manifest promote --note "..." --evidence router=runs/router__operating_curve__judged.json
    uv run adapterops manifest rollback
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "manifests"
CURRENT = MANIFEST_DIR / "system.json"
HISTORY = MANIFEST_DIR / "history"
ADAPTERS = MANIFEST_DIR / "adapters.json"

SPLIT_FILES = {
    "golden_intent": "evals/golden/intent.parquet",
    "golden_urgency": "evals/golden/urgency.parquet",
    "golden_pii": "evals/golden/pii.parquet",
    "golden_drafting": "evals/golden/drafting.parquet",
    "router_pool": "data/router/pool.parquet",
    "router_shift": "data/router/shift.parquet",
    "hard_cases": "evals/hard/hard_cases.parquet",      # Phase 4; absent until mined
}

ROUTER_FILES = ("checkpoints/router__judged/model.safetensors",
                "checkpoints/router__judged/config.json")
"""The judge-labelled router (D38), the one every published routing number describes. Manifests
v1–v3 pinned `checkpoints/router`, the proxy-labelled router no result used (D46)."""

JUDGE_FILES = ("checkpoints/judge/model.safetensors",
               "checkpoints/judge/config.json")

EVIDENCE_COMPONENTS = ("router", "judge")
"""Components an adapter regression run cannot measure. The router is judged by its operating
curve and the judge by its calibration, so moving either needs that report attached instead (D46)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_file(rel: str) -> dict | None:
    path = REPO_ROOT / rel
    if not path.exists():
        return None
    return {"file": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def current_gate() -> dict:
    """The gate a newly built manifest carries (F33, D37).

    Derived thresholds when `evals/GATE_THRESHOLDS.json` exists; report-only otherwise. Never an
    invented threshold — before the baseline runs exist there is nothing to derive one from, and
    a guessed threshold is the unfalsifiable gate PRD §11 removed.
    """
    path = REPO_ROOT / "evals" / "GATE_THRESHOLDS.json"
    if not path.exists():
        return {
            "state": "report_only",
            "why": "F33 — thresholds are derived from two baseline regression runs against an "
                   "unchanged manifest, with D37's training-variance floor, and those runs have "
                   "not happened. A threshold invented before then is the unfalsifiable gate "
                   "PRD §11 removed.",
            "thresholds": {},
        }
    derived = json.loads(path.read_text())
    return {**derived["gate"], "derivation": str(path.relative_to(REPO_ROOT))}


def build(note: str = "", pins: Path | None = None) -> dict:
    """Assemble a candidate manifest from what is on disk right now.

    Components that do not exist yet are pinned as `null` rather than omitted, so the
    manifest's shape says what a complete system needs and a diff shows a component
    *arriving* rather than a key appearing from nowhere.
    """
    source = pins or ADAPTERS
    # A candidate pin set (M7, M11) builds a manifest exactly as the real pins would, so the
    # gate judges the weights that were actually served and regressed.
    adapters = json.loads(source.read_text())["components"] if source.exists() else {}
    router = {name: _pin_file(name) for name in ROUTER_FILES}
    judge = {name: _pin_file(name) for name in JUDGE_FILES}
    return {
        "version": next_version(),
        "note": note,
        "pins": str(pins) if pins else None,
        "components": {
            "base_model": adapters.get("base_model"),
            "adapters": {t: adapters.get(t) for t in
                         ("intent", "urgency", "pii", "drafting")},
            "router": router if any(router.values()) else None,
            "judge": judge if any(judge.values()) else None,
            # Rebuilt at start-up from pinned splits and refused if a threshold moves (serve/domain_gate.py).
            "domain_gate": _pin_file("manifests/domain_gate.json"),
        },
        "eval_splits": {name: _pin_file(rel) for name, rel in SPLIT_FILES.items()},
        "gate": current_gate(),
        "provenance": {
            name: _pin_file(f"runs/{name}")
            for name in ("intent__adapter.json", "pii__adapter.json",
                         "urgency__adapter.json", "m1_serving.json",
                         "router__train__judged.json", "router__operating_curve__judged.json",
                         "judge__report.json")
        },
    }


def next_version() -> int:
    """One more than any version ever written — current or archived.

    The current manifest alone is not enough after a rollback. Rolling v3 back to v2 leaves v3 in
    history, and a promotion numbered from the current v2 becomes a second "v3" whose archiving
    later overwrites the first — which here is the forced release M7 records. Found promoting
    D46's manifest, before anything was overwritten.
    """
    versions = [int(json.loads(CURRENT.read_text())["version"])] if CURRENT.exists() else []
    if HISTORY.exists():
        versions += [int(p.stem.split("-")[1]) for p in HISTORY.glob("system-*.json")]
    return max(versions, default=0) + 1


def load_current() -> dict | None:
    return json.loads(CURRENT.read_text()) if CURRENT.exists() else None


def diff(old: dict | None, new: dict) -> dict:
    """What moved between two manifests, flattened to dotted keys.

    Reported as a diff rather than a boolean because "something changed" is not actionable:
    a promotion that moves an adapter and a promotion that moves an eval split are the same
    boolean and very different events.
    """
    def flatten(obj, prefix=""):
        out = {}
        for k, v in (obj or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and not {"sha256", "revision"} & set(v):
                out.update(flatten(v, f"{key}."))
            else:
                out[key] = (v or {}).get("sha256") or (v or {}).get("revision") if isinstance(v, dict) else v
        return out

    a = flatten({"components": (old or {}).get("components"),
                 "eval_splits": (old or {}).get("eval_splits")})
    b = flatten({"components": new.get("components"), "eval_splits": new.get("eval_splits")})
    changed = {k: {"from": a.get(k), "to": b.get(k)}
               for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    return {
        "changed": changed,
        "eval_splits_moved": sorted(k for k in changed if k.startswith("eval_splits.")),
    }


def blocking_reasons(new: dict, regression: dict | None,
                     refreeze: str | None = None,
                     evidence: dict[str, Path] | None = None) -> list[str]:
    """Why this manifest must not be promoted (F19). Empty means promote.

    `regression` is the result of a regression run — the on-demand script from Phase 4.
    Absent, promotion of a manifest whose *models* moved is refused: promoting an unmeasured
    model change is precisely what the gate exists to stop. Promoting a manifest that only
    re-pins splits or notes is allowed, because there is nothing to regress.

    **The first manifest is exempt**, and the check that says so is not a special case so
    much as the definition: a regression is a comparison against a previous version, and v1
    has none. The first attempt at this blocked v1 for having "moved" every component away
    from nothing. **A component arriving later is exempt for the same reason (D39)** — v1 was
    pinned before the router existed, and a regression for it has nothing to compare against.

    **Splits move only by a deliberate re-freeze (D39).** `refreeze` names the decision that
    records it, and is refused when an already-pinned model moves in the same promotion: a
    regression measured across a moved bar compares nothing.

    **A router or judge moves on its own evidence, not on an adapter regression run (D46).** A
    regression run scores adapters against the golden sets and says nothing about a router's
    escalation decisions or a judge's calibration, so accepting one would let any adapter run
    license a router change. The evidence is the component's own report — the operating curve,
    the calibration report — attached by path and pinned by sha256 in the promoted manifest.
    """
    reasons = []
    old = load_current()
    moved = diff(old, new)
    if old is None:
        return reasons          # baseline: there is no previous version to regress from

    evidence = evidence or {}
    moved_models = [k for k, change in moved["changed"].items()
                    if k.startswith("components.") and change["from"] is not None]
    evidenced = [k for k in moved_models if k.split(".")[1] in EVIDENCE_COMPONENTS]
    model_keys = [k for k in moved_models if k not in evidenced]
    if model_keys and regression is None:
        reasons.append(
            f"components moved ({', '.join(model_keys)}) with no regression run attached — "
            f"run the regression script and pass its result")
    for component in sorted({k.split(".")[1] for k in evidenced}):
        report = evidence.get(component)
        if report is None:
            reasons.append(
                f"{component} moved with no evaluation report attached — pass it with "
                f"--evidence {component}=<path>; an adapter regression run cannot measure a "
                f"{component}")
        elif not Path(report).exists():
            reasons.append(f"{component} evaluation report {report} does not exist")

    if refreeze and not moved["eval_splits_moved"]:
        reasons.append(f"--refreeze-splits {refreeze} given, but no eval split moved")
    elif refreeze and moved_models:
        reasons.append(
            f"re-freezing splits ({', '.join(moved['eval_splits_moved'])}) while models move "
            f"({', '.join(moved_models)}) — re-freeze in a promotion of its own first")
    elif moved["eval_splits_moved"] and not refreeze:
        reasons.append(
            f"eval splits moved ({', '.join(moved['eval_splits_moved'])}) — a score "
            f"measured against a different bar is not comparable to the previous one. "
            f"Re-freeze deliberately and record it as a decision, or revert the split.")

    if regression and new["gate"]["state"] == "enforcing":
        for task, result in (regression.get("per_task") or {}).items():
            drop = result.get("drop")
            limit = new["gate"]["thresholds"].get(task)
            if drop is not None and limit is not None and drop > limit:
                reasons.append(f"{task}: regression {drop:.4f} exceeds threshold {limit:.4f}")
        # The variance gate asks "did it drop more than noise"; this asks "does it still beat prompting".
        # Urgency's threshold (0.1224 on three-seed variance) is far wider than its margin over the prompted
        # base (0.0134), so the first question alone passes an under-trained urgency checkpoint (drop 0.087);
        # only this check blocks it.
        floors = prompted_floors()
        for task, result in (regression.get("per_task") or {}).items():
            candidate = (result.get("random") or {}).get("candidate")
            floor = floors.get(task)
            if candidate is not None and floor is not None and candidate < floor:
                reasons.append(f"{task}: candidate {candidate:.4f} is below the prompted base model's "
                               f"{floor:.4f} — the adapter would no longer beat prompting")
    return reasons


PROMPTED_BASELINES = {
    "intent": ("runs/intent__prompted-per-class.json", "micro_accuracy"),
    "urgency": ("runs/urgency__prompted-fewshot.json", "macro_f1"),
    "pii": ("runs/pii__prompted-fewshot.json", "span_f1_strict"),
}
"""The prompted base model's recorded golden score per task, on each task's gated metric. Drafting's
prompted comparison is a GPT-4o grade, not the distilled judge the gate uses, so it has no floor here."""


def prompted_floors(root: Path = REPO_ROOT) -> dict[str, float]:
    floors = {}
    for task, (path, metric) in PROMPTED_BASELINES.items():
        file = root / path
        if file.exists():
            value = json.loads(file.read_text()).get("metrics", {}).get(metric)
            if value is not None:
                floors[task] = float(value)
    return floors


def promote(note: str = "", regression: dict | None = None, force: bool = False,
            pins: Path | None = None, refreeze: str | None = None,
            evidence: dict[str, Path] | None = None) -> int:
    new = build(note, pins)
    reasons = blocking_reasons(new, regression, refreeze, evidence)
    if reasons and not force:
        print("  PROMOTION BLOCKED:")
        for r in reasons:
            print(f"    · {r}")
        return 1

    old = load_current()
    if old is not None:
        HISTORY.mkdir(parents=True, exist_ok=True)
        (HISTORY / f"system-{old['version']:04d}.json").write_text(
            json.dumps(old, indent=2) + "\n", encoding="utf-8")

    new["promoted_over"] = (old or {}).get("version")
    new["diff_from_previous"] = diff(old, new)
    if reasons and force:
        new["promoted_despite"] = reasons
    if refreeze and new["diff_from_previous"]["eval_splits_moved"]:
        new["refrozen_splits"] = {"decision": refreeze,
                                  "splits": new["diff_from_previous"]["eval_splits_moved"]}
    if evidence:
        new["component_evidence"] = {component: _pin_file(_rel(Path(path)))
                                     for component, path in sorted(evidence.items())
                                     if Path(path).exists()}
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    print(f"  promoted manifest v{new['version']}"
          + (f" over v{old['version']}" if old else " (first)"))
    for key, change in new["diff_from_previous"]["changed"].items():
        print(f"    {key}: {str(change['from'])[:12]} -> {str(change['to'])[:12]}")
    return 0


def rollback(to: int | None = None) -> int:
    """Restore a previous manifest whole (F20). The current one is archived first."""
    current = load_current()
    if current is None:
        print("  nothing to roll back — no current manifest.")
        return 2
    archived = sorted(HISTORY.glob("system-*.json")) if HISTORY.exists() else []
    if not archived:
        print("  nothing to roll back to — no history.")
        return 2

    target = (HISTORY / f"system-{to:04d}.json") if to else archived[-1]
    if not target.exists():
        print(f"  no archived manifest v{to}")
        return 2

    (HISTORY / f"system-{current['version']:04d}.json").write_text(
        json.dumps(current, indent=2) + "\n", encoding="utf-8")
    restored = json.loads(target.read_text())
    restored["rolled_back_from"] = current["version"]
    CURRENT.write_text(json.dumps(restored, indent=2) + "\n", encoding="utf-8")
    print(f"  rolled back to manifest v{restored['version']} (from v{current['version']})")
    return 0


def show() -> int:
    current = load_current()
    if current is None:
        print("  no system manifest — `uv run adapterops manifest promote` creates v1.")
        return 2
    print(f"  manifest v{current['version']}  ·  gate: {current['gate']['state']}")
    if current.get("note"):
        print(f"  note: {current['note']}")
    for task, pin in (current["components"]["adapters"] or {}).items():
        rev = (pin or {}).get("revision", "—")
        print(f"    adapter {task:9s} {str(rev)[:8]}")
    for component in ("router", "judge"):
        pins = current["components"].get(component) or {}
        weights = next((v for k, v in pins.items() if k.endswith("model.safetensors")), None)
        if weights:
            print(f"    {component:7s} {weights['file']}  {weights['sha256'][:12]}")
    for name, pin in current["eval_splits"].items():
        state = (pin or {}).get("sha256", "not present")[:12]
        print(f"    split   {name:16s} {state}")
    missing = [k for k, v in current["components"].items() if v is None]
    if missing:
        print(f"  not yet pinned: {', '.join(missing)}")
    return 0
