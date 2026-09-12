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


def _cmd_router_pool(args: argparse.Namespace) -> int:
    from adapterops.router.pool import main as pool_main

    return pool_main(force=args.force)


def _cmd_router_shift(args: argparse.Namespace) -> int:
    from adapterops.router.shift import main as shift_main

    return shift_main(force=args.force)


def _cmd_manifest(args: argparse.Namespace) -> int:
    from adapterops.manifest import system

    if args.action == "show":
        return system.show()
    if args.action == "promote":
        return system.promote(note=args.note, force=args.force)
    return system.rollback(to=args.to)


def _cmd_router_train(_: argparse.Namespace) -> int:
    from adapterops.router.train import main as rtrain_main

    return rtrain_main()


def _cmd_router_report(_: argparse.Namespace) -> int:
    from adapterops.router.report import main as report_main

    return report_main()


def _cmd_frontier(args: argparse.Namespace) -> int:
    from adapterops.router.frontier import main as frontier_main

    return frontier_main(limit=args.limit, workers=args.workers, purpose=args.purpose)


def _cmd_prompted(args: argparse.Namespace) -> int:
    from adapterops.eval.prompted import main as prompted_main

    return prompted_main(task=args.task, recipe=args.recipe, base_url=args.base_url,
                         max_model_len=args.max_model_len, budget_only=args.budget_only,
                         force=args.force)


def _cmd_regress(args: argparse.Namespace) -> int:
    from adapterops.eval.regression import main as regress_main

    return regress_main(base_url=args.base_url, name=args.name, baseline=args.baseline,
                        save_predictions=args.save_predictions)


def _cmd_hard_cases(args: argparse.Namespace) -> int:
    from adapterops.router.adjudicate import main as hard_main

    return hard_main(force=args.force)


def _cmd_judge_report(_: argparse.Namespace) -> int:
    from adapterops.judge.report import main as judge_report_main

    return judge_report_main()


def _cmd_judge_train(_: argparse.Namespace) -> int:
    from adapterops.judge.train import main as judge_train_main

    return judge_train_main()


def _cmd_judge_label(args: argparse.Namespace) -> int:
    from adapterops.judge.label import main as judge_main

    return judge_main(project=args.project, include_frontier=args.include_frontier,
                      limit=args.limit, workers=args.workers, spend_cap=args.spend_cap)


def _cmd_router_dataset(args: argparse.Namespace) -> int:
    from adapterops.router.dataset import main as dataset_main

    return dataset_main(force=args.force)


def _cmd_pin_adapters(args: argparse.Namespace) -> int:
    from adapterops.manifest.registry import pin

    return pin(force=args.force)


def _cmd_verify_pins(args: argparse.Namespace) -> int:
    from adapterops.manifest.registry import verify

    return verify(Path(args.pins) if args.pins else None)[0]


def _cmd_pin_candidate(args: argparse.Namespace) -> int:
    from adapterops.manifest.registry import pin_candidate

    return pin_candidate(task=args.task, repo=args.repo, name=args.name, force=args.force)


def _cmd_train(args: argparse.Namespace) -> int:
    from adapterops.train.qlora import config_for, train

    cfg = config_for(args.task, hub_repo=args.hub_repo)
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

    p_pool = sub.add_parser("router-pool", help="freeze the router (ticket, task) pool (F7)")
    p_pool.add_argument("--force", action="store_true",
                        help="re-draw the pool — moves the router's training data")
    p_pool.set_defaults(func=_cmd_router_pool)

    p_shift = sub.add_parser("router-shift", help="freeze the within-task shift split (F36)")
    p_shift.add_argument("--force", action="store_true",
                         help="re-draw the shift split — moves the bar M4 is reported against")
    p_shift.set_defaults(func=_cmd_router_shift)

    p_rds = sub.add_parser("router-dataset",
                           help="assemble router train/eval with the F31 exclusion")
    p_rds.add_argument("--force", action="store_true", help="rebuild a frozen dataset")
    p_rds.set_defaults(func=_cmd_router_dataset)

    p_man = sub.add_parser("manifest", help="system manifest: pin, promote, roll back (F16)")
    p_man.add_argument("action", choices=["show", "promote", "rollback"])
    p_man.add_argument("--note", default="", help="what this version changes")
    p_man.add_argument("--force", action="store_true",
                       help="promote despite blocking reasons — recorded in the manifest")
    p_man.add_argument("--to", type=int, default=None, help="roll back to this version")
    p_man.set_defaults(func=_cmd_manifest)

    p_rt = sub.add_parser("router-train", help="train the DeBERTa router (F10)")
    p_rt.set_defaults(func=_cmd_router_train)

    p_rr = sub.add_parser("router-report", help="operating curve vs all baselines (F11/M3)")
    p_rr.set_defaults(func=_cmd_router_report)

    p_fr = sub.add_parser("frontier", help="measure frontier success on the pool (F8/F11)")
    p_fr.add_argument("--limit", type=int, default=None,
                      help="pairs per task — project the full cost before paying it")
    p_fr.add_argument("--workers", type=int, default=6)
    p_fr.add_argument("--purpose", choices=["router", "mining"], default=None)
    p_fr.set_defaults(func=_cmd_frontier)

    p_pb = sub.add_parser("prompted", help="prompted baseline on the golden set (M2)")
    p_pb.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
    p_pb.add_argument("--recipe", default="fewshot", choices=["fewshot", "per-class"])
    p_pb.add_argument("--base-url", default="http://localhost:8000")
    p_pb.add_argument("--max-model-len", type=int, default=1536)
    p_pb.add_argument("--budget-only", action="store_true",
                      help="check the prompt fits the server's context; needs no server")
    p_pb.add_argument("--force", action="store_true", help="replace a recorded baseline")
    p_pb.set_defaults(func=_cmd_prompted)

    p_rg = sub.add_parser("regress", help="on-demand regression run, both splits (F18)")
    p_rg.add_argument("--base-url", default="http://localhost:8000")
    p_rg.add_argument("--name", required=True, help="names runs/regression__<name>.json")
    p_rg.add_argument("--baseline", default=None,
                      help="a previous regression run to compare against")
    p_rg.add_argument("--save-predictions", action="store_true",
                      help="keep every prediction, so drafting can be judged after the GPU is gone")
    p_rg.set_defaults(func=_cmd_regress)

    p_hc = sub.add_parser("hard-cases",
                          help="adjudicate mined failures and freeze the hard split (F30/F31)")
    p_hc.add_argument("--force", action="store_true",
                      help="re-screen a frozen hard split — moves the bar it gates on")
    p_hc.set_defaults(func=_cmd_hard_cases)

    p_jr = sub.add_parser("judge-report",
                          help="D28 proxy vs judge, same-family check, escalation re-scored")
    p_jr.set_defaults(func=_cmd_judge_report)

    p_jt = sub.add_parser("judge-train", help="train the distilled drafting judge (F14)")
    p_jt.set_defaults(func=_cmd_judge_train)

    p_jl = sub.add_parser("judge-label", help="GPT-4o judgments on drafting outputs (F13)")
    p_jl.add_argument("--project", action="store_true",
                      help="count tokens and price the run locally — makes no API call")
    p_jl.add_argument("--include-frontier", action="store_true",
                      help="also grade the 750 frontier drafts (evaluation only)")
    p_jl.add_argument("--limit", type=int, default=None, help="items per source")
    p_jl.add_argument("--workers", type=int, default=4)
    p_jl.add_argument("--spend-cap", type=float, default=8.0,
                      help="hard stop in USD, derived from token counts")
    p_jl.set_defaults(func=_cmd_judge_label)

    p_pin = sub.add_parser("pin-adapters", help="pin adapter revisions in manifests/ (F16)")
    p_pin.add_argument("--force", action="store_true",
                       help="re-pin to whatever `main` points at now")
    p_pin.set_defaults(func=_cmd_pin_adapters)

    p_verify = sub.add_parser("verify-pins", help="has a pinned adapter moved on the Hub?")
    p_verify.add_argument("--pins", default=None, help="verify a candidate pin set instead")
    p_verify.set_defaults(func=_cmd_verify_pins)

    p_cand = sub.add_parser("pin-candidate",
                            help="pin a candidate set with one adapter swapped (M7, M11)")
    p_cand.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
    p_cand.add_argument("--repo", required=True, help="Hub repo of the candidate adapter")
    p_cand.add_argument("--name", required=True, help="names manifests/candidates/<name>.json")
    p_cand.add_argument("--force", action="store_true")
    p_cand.set_defaults(func=_cmd_pin_candidate)

    p_train = sub.add_parser("train", help="QLoRA-train a task adapter (needs CUDA)")
    p_train.add_argument("--task", required=True, choices=["intent", "urgency", "pii", "drafting"])
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
