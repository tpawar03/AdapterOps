"""Assemble a self-contained Hugging Face Space bundle in `demo/_space/` (M8).

The Space gets the app, the package source it imports, the pin file and the rendered dashboard —
and nothing else: no data, no checkpoints, no runs beyond `DASHBOARD.md`. The bundle is gitignored
and rebuilt from the repo, so it can never hold a number the repo does not.

    uv run adapterops economics && uv run adapterops dashboard
    uv run python demo/build_space.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "_space"

CARD = """---
title: AdapterOps
emoji: 🧩
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: other
---

Demo for [AdapterOps](https://github.com/tpawar03/AdapterOps). Runs the pinned adapters with
transformers + peft on this Space's CPU — not the benchmark path. Benchmark numbers are on the
Results tab and in the repo. The urgency adapter is trained on CC BY-NC 4.0 data and is for
non-commercial use only; the PII adapter was trained on synthetic data only.
"""


def main() -> int:
    for need in (ROOT / "runs" / "DASHBOARD.md", ROOT / "manifests" / "adapters.json"):
        if not need.exists():
            print(f"missing {need.relative_to(ROOT)} — see the docstring")
            return 2
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "runs").mkdir(parents=True)
    (OUT / "manifests").mkdir()
    shutil.copy(HERE / "app.py", OUT / "app.py")
    shutil.copy(HERE / "requirements.txt", OUT / "requirements.txt")
    shutil.copy(ROOT / "runs" / "DASHBOARD.md", OUT / "runs" / "DASHBOARD.md")
    shutil.copy(ROOT / "manifests" / "adapters.json", OUT / "manifests" / "adapters.json")
    shutil.copytree(ROOT / "src" / "adapterops", OUT / "src" / "adapterops",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "README.md").write_text(CARD)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
