"""Thin entrypoint so `python -m adapterops.train.cli_train --task X` works on a GPU box
without installing the package. Mirrors `adapterops train`, which needs the console script.
"""

from __future__ import annotations

import argparse
import json

from adapterops.train.qlora import config_for, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="override the per-task training-row cap")
    ap.add_argument("--train-split", default="train",
                    help="frozen split to train on, e.g. train_shuffled for F21")
    ap.add_argument("--variant", default=None,
                    help="names the outputs; required for any split other than train")
    ap.add_argument("--epochs", type=float, default=None,
                    help="override the per-task epochs — e.g. to repeat a run whose config was "
                         "set by override rather than TASK_CONFIGS (PII's 8,000 rows, 3 epochs)")
    ap.add_argument("--report-to", default=None, choices=["wandb"],
                    help="also log the run to Weights & Biases (needs WANDB_API_KEY)")
    args = ap.parse_args(argv)

    # A non-default split trained under the task's own names would overwrite the real
    # adapter's checkpoint, its M11 checkpoint and runs/<task>__train.json.
    if args.train_split != "train" and not args.variant:
        print(f"  --train-split {args.train_split} needs --variant, or it would overwrite "
              f"the real {args.task} adapter's outputs")
        return 2
    name = f"{args.task}-{args.variant}" if args.variant else args.task

    overrides = {"hub_repo": args.hub_repo, "train_split": args.train_split,
                 "output_dir": f"checkpoints/{name}"}
    if args.subsample is not None:
        overrides["train_subsample"] = args.subsample
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.report_to:
        overrides["report_to"] = [args.report_to]
    cfg = config_for(args.task, **overrides)
    print(f"  {args.task}: seq {cfg.max_seq_length}, batch {cfg.batch_size}x{cfg.grad_accum}, "
          f"{cfg.epochs:.0f} epochs, subsample {cfg.train_subsample}")

    summary = train(cfg)
    out = f"runs/{name}__train.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
