"""Command-line entrypoint. Subcommands land as each phase does (BUILD-PLAN.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapterops import __version__


def _cmd_splits(args: argparse.Namespace) -> int:
    from adapterops.data.splits import main as splits_main

    return splits_main(force=args.force)


def _cmd_mirror(_: argparse.Namespace) -> int:
    from adapterops.data.mirror import main as mirror_main

    return mirror_main()


def _cmd_pii_task(_: argparse.Namespace) -> int:
    from adapterops.data.pii_task import main as pii_main

    return pii_main()


def _cmd_train(args: argparse.Namespace) -> int:
    from adapterops.train.qlora import TrainConfig, train

    cfg = TrainConfig(task=args.task, output_dir=f"checkpoints/{args.task}", hub_repo=args.hub_repo)
    summary = train(cfg)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import pandas as pd

    from adapterops.eval.harness import evaluate, majority_predictor, record

    split = Path(args.split).resolve()
    if args.system != "majority":
        print(f"  system {args.system!r} is not wired up yet — see BUILD-PLAN Phase 0/1")
        return 2

    train = pd.read_parquet(Path(f"data/{args.task}/split_train.parquet"))
    _, label_col = __import__(
        "adapterops.eval.harness", fromlist=["TASK_COLUMNS"]
    ).TASK_COLUMNS[args.task]
    result = evaluate(split, args.task, args.system, majority_predictor(train[label_col]))
    path = record(result)
    print(f"  {result.task} · {result.system} · n={result.n}")
    for k, v in result.metrics.items():
        marker = "  <- gated" if k == result.gated_metric else ""
        print(f"    {k:16s} {v:.4f}{marker}")
    print(f"  wrote {path.relative_to(Path.cwd())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adapterops", description=__doc__)
    parser.add_argument("--version", action="version", version=f"adapterops {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_mirror = sub.add_parser("mirror", help="mirror the PRD §9 datasets into data/")
    p_mirror.set_defaults(func=_cmd_mirror)

    p_splits = sub.add_parser("splits", help="freeze the evaluation splits (F35)")
    p_splits.add_argument("--force", action="store_true",
                          help="overwrite frozen splits — invalidates prior comparisons")
    p_splits.set_defaults(func=_cmd_splits)

    p_pii = sub.add_parser("pii-task", help="build the PII binary task and probe its confound")
    p_pii.set_defaults(func=_cmd_pii_task)

    p_train = sub.add_parser("train", help="QLoRA-train a task adapter (needs CUDA)")
    p_train.add_argument("--task", required=True, choices=["intent", "urgency"])
    p_train.add_argument("--hub-repo", default=None, help="push to this HF Hub repo when done")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="score a system on a split (F6)")
    p_eval.add_argument("--split", required=True, help="path to a split parquet")
    p_eval.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
    p_eval.add_argument("--system", default="majority",
                        help="majority (floor) | adapter | prompted | frontier")
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
