"""Assemble a self-contained Hugging Face Space bundle in `demo/_space/` (M8).

The Space gets the app, the package source it imports, the pin file and the rendered dashboard —
and nothing else: no data, no checkpoints, no runs beyond `DASHBOARD.md`. The bundle is gitignored
and rebuilt from the repo, so it can never hold a number the repo does not.

**The Space runs the versions this machine tested, read from what is installed.** The card's
`sdk_version` is the local gradio, and `requirements.txt` pins the local transformers, peft, torch
and friends exactly. `demo/requirements.txt` stays a loose floor for local installs; a Space built
from floors would resolve whatever is newest on the day, which is a different app from the one that
was checked. Gradio is left out of the Space's requirements because `sdk_version` installs it, and
torch comes from the CPU wheel index — the default Linux wheel carries CUDA libraries a CPU Space
downloads and never uses.

    uv run adapterops economics && uv run adapterops dashboard
    uv run python demo/build_space.py
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "_space"

PINNED = ("torch", "transformers", "peft", "huggingface-hub", "pandas")
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

CARD = """---
title: AdapterOps
emoji: 🧩
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {gradio}
python_version: "{python}"
app_file: app.py
pinned: false
license: other
short_description: Four LoRA adapters on one base, with the negative results kept
---

Demo for [AdapterOps](https://github.com/tpawar03/AdapterOps). Runs the pinned adapters with
transformers + peft on this Space's CPU — not the benchmark path. Benchmark numbers are on the
Results tab and in the repo. The urgency adapter is trained on CC BY-NC 4.0 data and is for
non-commercial use only; the PII adapter was trained on synthetic data only.
"""


def requirements() -> str:
    """Exact local versions — a local build label such as `+cpu` is dropped so any platform matches."""
    lines = [f"--extra-index-url {CPU_TORCH_INDEX}"]
    lines += [f"{name}=={md.version(name).split('+')[0]}" for name in PINNED]
    return "\n".join(lines) + "\n"


def card() -> str:
    python = ".".join(platform.python_version_tuple()[:2])
    return CARD.format(gradio=md.version("gradio"), python=python)


def main() -> int:
    for need in (ROOT / "runs" / "DASHBOARD.md", ROOT / "manifests" / "adapters.json"):
        if not need.exists():
            print(f"missing {need.relative_to(ROOT)} — see the docstring")
            return 2
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "runs").mkdir(parents=True)
    (OUT / "manifests").mkdir()
    shutil.copy(HERE / "app.py", OUT / "app.py")
    shutil.copy(ROOT / "runs" / "DASHBOARD.md", OUT / "runs" / "DASHBOARD.md")
    shutil.copy(ROOT / "manifests" / "adapters.json", OUT / "manifests" / "adapters.json")
    shutil.copytree(ROOT / "src" / "adapterops", OUT / "src" / "adapterops",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "requirements.txt").write_text(requirements())
    (OUT / "README.md").write_text(card())
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
