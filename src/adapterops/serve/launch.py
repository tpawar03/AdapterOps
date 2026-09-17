"""Materialise a pinned adapter set and emit the vLLM arguments that serve it (F5, M1, M7).

Serving through `--lora-modules name=Tanny03/adapterops-intent` resolves to whatever `main`
points at on the box, at the moment the box starts — which is the failure mode changelog 29
already produced once. So pins are downloaded *at their revision* into local directories
first, and vLLM is pointed at paths. What gets served is a property of a pin file, not of
the Hub's current state.

**A candidate is served by swapping the pin file, not the adapter name.** M7 and M11 both need
a different set of intent weights served *as* `intent`, so the regression run addresses them
exactly as it would the real adapter. `--pins` takes a candidate pin set built by
`adapterops pin-candidate`; without it, the adapter pins in `manifests/system.json` are served —
the manifest decides what runs (M6).

**Downloads are namespaced by pin set.** Both pin sets define `intent`. Downloaded into the same
`_adapters/intent` directory, the second set would overwrite the first, and vLLM would serve
whichever weights landed last — under a name that says nothing about which they were.

    eval "$(uv run python -m adapterops.serve.launch --print-args)"
    eval "$(uv run python -m adapterops.serve.launch --print-args --pins manifests/candidates/intent-m11.json)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapterops.manifest.registry import REPO_ROOT

ADAPTER_DIR = REPO_ROOT / "_adapters"
"""Gitignored scratch on the rented box. Named without a leading dot on purpose (D20)."""

SYSTEM = REPO_ROOT / "manifests" / "system.json"


def load_components(pins: Path | None = None) -> dict:
    """What gets served. **Without `--pins`, the system manifest decides (M6)**: editing the adapter
    revisions in `manifests/system.json` — which only a promotion or a rollback does — changes
    what the next launch serves. `manifests/adapters.json` is the registry the manifest is built
    from, not the thing served."""
    if pins is not None:
        return json.loads(Path(pins).read_text())["components"]
    components = json.loads(SYSTEM.read_text())["components"]
    return {"base_model": components["base_model"], **components["adapters"]}


def adapter_dir(pins: Path | None, task: str) -> Path:
    """One directory per (pin set, task), so two pin sets can never share downloaded weights."""
    return ADAPTER_DIR / ("baseline" if pins is None else Path(pins).stem) / task


def materialise(pins: Path | None = None,
                tasks: tuple[str, ...] | None = None) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    out: dict[str, Path] = {}
    for task, entry in load_components(pins).items():
        if task == "base_model" or entry is None or (tasks and task not in tasks):
            continue
        if entry.get("kind") == "sklearn":
            continue  # not a LoRA adapter: the request path and regress answer it locally
        if entry.get("path"):
            # A locally trained adapter — a training-variance rerun — is served from disk. It was
            # never published, and naming it by path in a pin set is how it gets regression-
            # tested exactly as the served adapter is, without a public upload first.
            local = Path(entry["path"])
            out[task] = local if local.is_absolute() else REPO_ROOT / local
            continue
        path = snapshot_download(
            repo_id=entry["repo"],
            revision=entry["revision"],
            local_dir=adapter_dir(pins, task),
            allow_patterns=["adapter_model.safetensors", "adapter_config.json",
                            "*.json", "*.txt"],
            ignore_patterns=["checkpoint-*/*"],
        )
        out[task] = Path(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-args", action="store_true",
                    help="emit shell variables for scripts/phase2_serve.sh")
    ap.add_argument("--pins", default=None,
                    help="serve a candidate pin set instead of manifests/adapters.json")
    args = ap.parse_args(argv)

    pins = Path(args.pins) if args.pins else None
    components = load_components(pins)
    paths = materialise(pins)
    modules = " ".join(f"{task}={path}" for task, path in sorted(paths.items()))

    if args.print_args:
        print(f'BASE="{components["base_model"]["repo"]}"')
        print(f'BASE_REVISION="{components["base_model"]["revision"]}"')
        print(f'LORA_MODULES="{modules}"')
        shown = SYSTEM.relative_to(REPO_ROOT) if SYSTEM.is_relative_to(REPO_ROOT) else SYSTEM
        print(f'PIN_SET="{pins or shown}"')
    else:
        for task, path in sorted(paths.items()):
            print(f"  {task:9s} {(components[task].get('revision') or 'local')[:8]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
