"""Thin entrypoint so `python -m adapterops.train.cli_train --task X` works on a GPU box
without installing the package. Mirrors `adapterops train`, which needs the console script.
"""

from __future__ import annotations

import argparse
import json

from adapterops.train.qlora import config_for, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="override the per-task training-row cap")
    args = ap.parse_args()

    overrides = {"hub_repo": args.hub_repo}
    if args.subsample is not None:
        overrides["train_subsample"] = args.subsample
    cfg = config_for(args.task, **overrides)
    print(f"  {args.task}: seq {cfg.max_seq_length}, batch {cfg.batch_size}x{cfg.grad_accum}, "
          f"{cfg.epochs:.0f} epochs, subsample {cfg.train_subsample}")

    summary = train(cfg)
    out = f"runs/{args.task}__train.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
