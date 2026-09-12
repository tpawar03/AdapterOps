"""Materialise the pinned adapters and emit the vLLM arguments that serve them (F5, M1).

Serving through `--lora-modules name=Tanny03/adapterops-intent` resolves to whatever
`main` points at on the box, at the moment the box starts — which is the failure mode
changelog 29 already produced once. So the pins are downloaded *at their revision* into
local directories first, and vLLM is pointed at paths. What gets served is then a
property of `manifests/adapters.json`, not of the Hub's current state.

    eval "$(uv run python -m adapterops.serve.launch --print-args)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapterops.manifest.registry import MANIFEST, REPO_ROOT

ADAPTER_DIR = REPO_ROOT / "_adapters"
"""Gitignored scratch on the rented box. Named without a leading dot on purpose (D20)."""


def materialise(tasks: tuple[str, ...] | None = None) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    components = json.loads(MANIFEST.read_text())["components"]
    out: dict[str, Path] = {}
    for task, entry in components.items():
        if task == "base_model" or (tasks and task not in tasks):
            continue
        path = snapshot_download(
            repo_id=entry["repo"],
            revision=entry["revision"],
            local_dir=ADAPTER_DIR / task,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json",
                            "*.json", "*.txt"],
            ignore_patterns=["checkpoint-*/*"],
        )
        out[task] = Path(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-args", action="store_true",
                    help="emit shell variables for scripts/phase2_serve.sh")
    args = ap.parse_args()

    components = json.loads(MANIFEST.read_text())["components"]
    paths = materialise()
    modules = " ".join(f"{task}={path}" for task, path in sorted(paths.items()))

    if args.print_args:
        print(f'BASE="{components["base_model"]["repo"]}"')
        print(f'BASE_REVISION="{components["base_model"]["revision"]}"')
        print(f'LORA_MODULES="{modules}"')
    else:
        for task, path in sorted(paths.items()):
            print(f"  {task:9s} {components[task]['revision'][:8]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
