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
        regression = None
        if args.regression:
            run = json.loads(Path(args.regression).read_text())
            if "comparison" not in run:
                print(f"  {args.regression} has no comparison — run regress with --baseline")
                return 2
            regression = run["comparison"]
        evidence = {}
        for item in args.evidence or []:
            component, sep, path = item.partition("=")
            if not sep or component not in ("router", "judge"):
                print(f"  --evidence takes router=<path> or judge=<path>, got {item!r}")
                return 2
            evidence[component] = Path(path).resolve()
        return system.promote(note=args.note, force=args.force, regression=regression,
                              pins=Path(args.pins) if args.pins else None,
                              refreeze=args.refreeze_splits, evidence=evidence or None)
    return system.rollback(to=args.to)


def _cmd_router_train(_: argparse.Namespace) -> int:
    from adapterops.router.train import main as rtrain_main

    return rtrain_main()


def _cmd_router_report(args: argparse.Namespace) -> int:
    from adapterops.router.report import main as report_main

    return report_main(judged=args.judged)


def _cmd_router_plot(_: argparse.Namespace) -> int:
    from adapterops.router.plot import main as plot_main

    return plot_main()


def _cmd_router_misroutes(_: argparse.Namespace) -> int:
    from adapterops.router.misroute import main as misroute_main

    return misroute_main()


def _cmd_router_latency(_: argparse.Namespace) -> int:
    from adapterops.router.latency import main as latency_main

    return latency_main()


def _cmd_pii_false_positives(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from adapterops.eval.pii_negatives import main as negatives_main

    return negatives_main(limit=args.limit, batch=args.batch, force_set=args.force_set,
                          set_name=args.set, adapter_dir=args.adapter_dir, name=args.name)


def _cmd_pii_realistic(args: argparse.Namespace) -> int:
    from adapterops.eval.pii_realistic import main as realistic_main

    return realistic_main(batch=args.batch, adapter_dir=args.adapter_dir, name=args.name)


def _cmd_pii_realistic_compare(args: argparse.Namespace) -> int:
    from adapterops.eval.pii_realistic import compare

    return compare(args.candidate)


def _cmd_pii_realistic_split(args: argparse.Namespace) -> int:
    from adapterops.data.pii_realistic_split import main as split_main

    return split_main(force=args.force)


def _cmd_urgency_tfidf(args: argparse.Namespace) -> int:
    from adapterops.eval.urgency_tfidf import main as tfidf_main

    return tfidf_main(save=args.save)


def _cmd_hard_shared(_: argparse.Namespace) -> int:
    from adapterops.eval.shared_hard import main as shared_main

    return shared_main()


def _cmd_pii_relabel(_: argparse.Namespace) -> int:
    from adapterops.eval.pii_relabel import main as relabel_main

    return relabel_main()


def _cmd_pii_guard(_: argparse.Namespace) -> int:
    from adapterops.eval.pii_guard import main as guard_main

    return guard_main()


def _cmd_router_per_task(_: argparse.Namespace) -> int:
    from adapterops.router.per_task import main as per_task_main

    return per_task_main()


def _cmd_pii_errors(_: argparse.Namespace) -> int:
    from adapterops.eval.pii_errors import main as errors_main

    return errors_main()


def _cmd_router_rescore(args: argparse.Namespace) -> int:
    from adapterops.router.rescore import main as rescore_main

    return rescore_main(task=args.task, batch=args.batch, limit=args.limit)


def _cmd_pii_negatives_split(args: argparse.Namespace) -> int:
    from adapterops.data.pii_negatives_split import main as split_main

    return split_main(force=args.force)


def _cmd_training_variance(args: argparse.Namespace) -> int:
    from adapterops.eval.training_variance import main as variance_main

    runs = args.runs or [args.original, args.rerun]
    if not all(runs):
        raise SystemExit("pass --runs A B [C ...], or --original and --rerun")
    return variance_main(runs=runs, tasks=args.tasks, force=args.force)


def _cmd_request_path_run(args: argparse.Namespace) -> int:
    from adapterops.serve.request_run import main as run_main
    from adapterops.serve.request_run import resummarise

    if args.resummarise:
        print(f"  rewrote {resummarise(args.name)}")
        return 0

    return run_main(url=args.url, name=args.name, concurrency=args.concurrency, limit=args.limit,
                    population=args.population, duration=args.duration, window=args.window,
                    direct_vllm=args.direct_vllm)


def _cmd_quantize_compare(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from adapterops.eval.quantize import main as quantize_main

    return quantize_main(limit=args.limit)


def _cmd_serve_api(args: argparse.Namespace) -> int:
    from adapterops.serve.api import main as api_main

    return api_main(backend=args.backend, base_url=args.base_url, policy=args.policy,
                    frontier=not args.no_frontier, judge=not args.no_judge, trace=args.trace,
                    host=args.host, port=args.port, benchmark_caps=args.benchmark_caps,
                    max_threads=args.max_threads, frontier_per_minute=args.frontier_per_minute,
                    frontier_per_day=args.frontier_per_day, pii_guard=not args.no_pii_guard)


def _cmd_regression_redaction(args: argparse.Namespace) -> int:
    from adapterops.eval.regression import add_redaction

    for run in args.run:
        pii = add_redaction(Path(run))
        print(f"  {run}: " + " · ".join(
            f"{split} fully masked {v.get('docs_fully_masked')}, wholly unmasked spans "
            f"{v.get('gold_spans_wholly_unmasked')}" for split, v in pii.items()))
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    from adapterops.router.frontier import main as frontier_main

    return frontier_main(limit=args.limit, workers=args.workers, purpose=args.purpose)


def _cmd_prompted(args: argparse.Namespace) -> int:
    from adapterops.eval.prompted import main as prompted_main

    return prompted_main(task=args.task, recipe=args.recipe, base_url=args.base_url,
                         max_model_len=args.max_model_len, budget_only=args.budget_only,
                         force=args.force, save_predictions=args.save_predictions)


def _cmd_derive_thresholds(args: argparse.Namespace) -> int:
    from adapterops.eval.thresholds import main as thresholds_main

    return thresholds_main(run_a=args.run_a, run_b=args.run_b, force=args.force)


def _cmd_regress(args: argparse.Namespace) -> int:
    from adapterops.eval.regression import main as regress_main

    return regress_main(base_url=args.base_url, name=args.name, baseline=args.baseline,
                        save_predictions=args.save_predictions, pins=args.pins)


def _cmd_hard_cases(args: argparse.Namespace) -> int:
    from adapterops.router.adjudicate import main as hard_main

    return hard_main(force=args.force)


def _cmd_judge_score(args: argparse.Namespace) -> int:
    from adapterops.judge.score import main as judge_score_main

    return judge_score_main(run_path=args.run, baseline=args.baseline, force=args.force)


def _cmd_judge_m2(args: argparse.Namespace) -> int:
    from adapterops.judge.m2 import main as m2_main

    return m2_main(project=args.project, sides=args.sides, workers=args.workers,
                   spend_cap=args.spend_cap)


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


def _cmd_combine_candidates(args: argparse.Namespace) -> int:
    from adapterops.manifest.registry import combine_candidates

    return combine_candidates(names=args.sources, name=args.name, force=args.force)


def _cmd_pin_candidate(args: argparse.Namespace) -> int:
    from adapterops.manifest.registry import pin_candidate

    return pin_candidate(task=args.task, repo=args.repo, name=args.name, force=args.force)


def _cmd_economics(_: argparse.Namespace) -> int:
    from adapterops.eval.economics import main as economics_main

    return economics_main()


def _cmd_dashboard(_: argparse.Namespace) -> int:
    from adapterops.eval.dashboard import main as dashboard_main

    return dashboard_main()


def _cmd_train(args: argparse.Namespace) -> int:
    from adapterops.train.qlora import config_for, train

    overrides = {"hub_repo": args.hub_repo}
    if args.seed is not None:
        overrides |= {"seed": args.seed, "output_dir": f"checkpoints/{args.task}-seed{args.seed}"}
    cfg = config_for(args.task, **overrides)
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
    p_man.add_argument("--pins", default=None,
                       help="build from a candidate pin set instead of manifests/adapters.json")
    p_man.add_argument("--regression", default=None,
                       help="a regression run (with --baseline) to attach to the promotion")
    p_man.add_argument("--refreeze-splits", default=None, metavar="DECISION",
                       help="re-pin moved eval splits deliberately, citing the decision that "
                            "records it; refused if a pinned model moves in the same promotion")
    p_man.add_argument("--evidence", action="append", default=None, metavar="COMPONENT=PATH",
                       help="the report that licenses moving the router or judge (D46), e.g. "
                            "router=runs/router__operating_curve__judged.json; repeatable")
    p_man.set_defaults(func=_cmd_manifest)

    p_rt = sub.add_parser("router-train", help="train the DeBERTa router (F10)")
    p_rt.set_defaults(func=_cmd_router_train)

    p_rr = sub.add_parser("router-report", help="operating curve vs all baselines (F11/M3)")
    p_rr.add_argument("--judged", action="store_true",
                      help="D38: drafting graded by GPT-4o on both arms, from the __judged files")
    p_rr.set_defaults(func=_cmd_router_report)

    p_rp = sub.add_parser("router-plot",
                          help="draw the operating curve from the committed report (F11/M3)")
    p_rp.set_defaults(func=_cmd_router_plot)

    p_rm = sub.add_parser("router-misroutes",
                          help="persist router-misroute records at the operating point (F29/F37)")
    p_rm.set_defaults(func=_cmd_router_misroutes)

    p_rl = sub.add_parser("router-latency",
                          help="time the pinned router's decisions on CPU (PRD §7, < 50 ms)")
    p_rl.set_defaults(func=_cmd_router_latency)

    p_api = sub.add_parser("serve-api", help="the online request path over HTTP (PRD §8)")
    p_api.add_argument("--backend", choices=["vllm", "transformers"], default="vllm",
                       help="vllm: the served adapters; transformers: the same pins on CPU/MPS")
    p_api.add_argument("--base-url", default="http://localhost:8000", help="the vLLM server")
    p_api.add_argument("--policy", choices=["confidence", "router", "never"],
                       default="confidence")
    p_api.add_argument("--no-frontier", action="store_true",
                       help="never call GPT-4o-mini, even with OPENAI_API_KEY set")
    p_api.add_argument("--no-judge", action="store_true")
    p_api.add_argument("--benchmark-caps", action="store_true",
                       help="transformers backend: generate drafting to 448 tokens, as the curve "
                            "did, instead of the demo's 200")
    p_api.add_argument("--max-threads", type=int, default=256,
                       help="concurrent tickets the API runs; anyio's default of 40 would cap a "
                            "load test before vLLM does")
    p_api.add_argument("--frontier-per-minute", type=int, default=400,
                       help="GPT-4o-mini requests allowed per minute; over it, pairs stay local")
    p_api.add_argument("--frontier-per-day", type=int, default=8000,
                       help="GPT-4o-mini requests allowed per day, under the account's 10,000 cap")
    p_api.add_argument("--no-pii-guard", action="store_true",
                       help="do not add pattern-matched identifiers and dates the PII answer left untagged")

    p_red = sub.add_parser("regression-redaction",
                           help="backfill PII redaction leakage into committed regression runs")
    p_red.add_argument("--run", nargs="+", required=True, help="regression runs with saved predictions")
    p_red.set_defaults(func=_cmd_regression_redaction)
    p_api.add_argument("--trace", choices=["none", "jsonl", "langfuse"], default="jsonl")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8080)
    p_api.set_defaults(func=_cmd_serve_api)

    p_rpr = sub.add_parser("request-path-run",
                           help="send the curve's 392 recorded pairs through a running serve-api")
    p_rpr.add_argument("--url", default="http://127.0.0.1:8080", help="the serve-api to drive")
    p_rpr.add_argument("--name", required=True, help="names runs/request_path__<name>.json")
    p_rpr.add_argument("--concurrency", type=int, nargs="+", default=[1],
                       help="client concurrency levels; each is one full pass over the pairs")
    p_rpr.add_argument("--limit", type=int, default=None, help="pairs per task, for a smoke run")
    p_rpr.add_argument("--population", choices=["router_in_distribution", "router_shift"],
                       default="router_in_distribution",
                       help="which curve population to send: 392 in-distribution or 1,047 shifted")
    p_rpr.add_argument("--duration", type=float, default=None,
                       help="seconds per concurrency level, cycling the pairs; default is one pass")
    p_rpr.add_argument("--window", type=float, default=60.0,
                       help="with --duration: seconds per summary window")
    p_rpr.add_argument("--direct-vllm", default=None, metavar="URL",
                       help="send straight to vLLM — no API, routing or frontier — for a ceiling")
    p_rpr.add_argument("--resummarise", action="store_true",
                       help="recompute runs/request_path__<name>.json from its per-request file; "
                            "sends nothing")
    p_rpr.set_defaults(func=_cmd_request_path_run)

    p_q = sub.add_parser("quantize-compare",
                         help="fp32 vs dynamic int8 on the intent adapter, CPU (F27, N2)")
    p_q.add_argument("--limit", type=int, default=None,
                     help="golden items to use, kept class-balanced (default: all 770)")
    p_q.set_defaults(func=_cmd_quantize_compare)

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
    p_pb.add_argument("--save-predictions", action="store_true",
                      help="keep every reply, so drafting can be judged after the GPU is gone")
    p_pb.set_defaults(func=_cmd_prompted)

    p_dt = sub.add_parser("derive-thresholds",
                          help="gate thresholds from two baseline regression runs (F33, D37)")
    p_dt.add_argument("--run-a", required=True, help="first baseline regression run")
    p_dt.add_argument("--run-b", required=True, help="second baseline regression run")
    p_dt.add_argument("--force", action="store_true")
    p_dt.set_defaults(func=_cmd_derive_thresholds)

    p_tv = sub.add_parser("training-variance",
                          help="training spread across two or more runs of one configuration (F33, D37)")
    p_tv.add_argument("--runs", nargs="+", help="two or more regression runs of the same "
                      "configuration, scored in one session")
    p_tv.add_argument("--original", help="regression run of the served adapters (with --rerun)")
    p_tv.add_argument("--rerun", help="regression run of the retrained adapters")
    p_tv.add_argument("--tasks", nargs="+", required=True,
                      choices=["intent", "urgency", "pii", "drafting"])
    p_tv.add_argument("--force", action="store_true", help="replace a measured variance")
    p_tv.set_defaults(func=_cmd_training_variance)

    p_pn = sub.add_parser("pii-false-positives",
                          help="the PII adapter on PII-free text: how often it reports some (ch. 53)")
    p_pn.add_argument("--limit", type=int, default=None, help="texts per source, for a smoke run")
    p_pn.add_argument("--batch", type=int, default=16)
    p_pn.add_argument("--force-set", action="store_true",
                      help="rebuild the frozen PII-free set — moves the bar the rate is measured on")
    p_pn.add_argument("--set", choices=["golden_screened", "ai4privacy_val"],
                      default="golden_screened",
                      help="the 928 screened golden texts, or span-free PII validation sentences")
    p_pn.add_argument("--adapter-dir", default=None,
                      help="score a local PII checkpoint instead of the manifest's adapter")
    p_pn.add_argument("--name", default=None,
                      help="names runs/pii__false_positives__<name>.json (default: the served run)")
    p_pn.set_defaults(func=_cmd_pii_false_positives)

    p_ps = sub.add_parser("pii-negatives-split",
                          help="PII training split with PII-free sentences and empty answers")
    p_ps.add_argument("--force", action="store_true", help="rebuild the frozen split")
    p_ps.set_defaults(func=_cmd_pii_negatives_split)

    p_rs = sub.add_parser("router-rescore",
                          help="re-score a task's routing pairs with the adapter a manifest replaced it with")
    p_rs.add_argument("--task", default="pii", choices=["intent", "urgency", "pii", "drafting"])
    p_rs.add_argument("--batch", type=int, default=8)
    p_rs.add_argument("--limit", type=int, default=None, help="pairs per population, for a smoke run")
    p_rs.set_defaults(func=_cmd_router_rescore)

    p_pe = sub.add_parser("pii-errors",
                          help="classify every PII span error: label, boundary, spurious, invented, missed")
    p_pe.set_defaults(func=_cmd_pii_errors)

    p_pt = sub.add_parser("router-per-task",
                          help="per-task confidence thresholds chosen on training pairs, scored held out")
    p_pt.set_defaults(func=_cmd_router_per_task)

    p_pg = sub.add_parser("pii-guard",
                          help="regex spans added where the PII adapter tagged nothing: leaks removed vs false spans")
    p_pg.set_defaults(func=_cmd_pii_guard)

    p_rl = sub.add_parser("pii-relabel",
                          help="relabel GENDER/SEX and ID-number spans by their nearest cue word; measure it")
    p_rl.set_defaults(func=_cmd_pii_relabel)

    p_hs = sub.add_parser("hard-shared",
                          help="mine a hard split from never-gated systems' failures; re-run M11's check on it")
    p_hs.set_defaults(func=_cmd_hard_shared)

    p_ut = sub.add_parser("urgency-tfidf",
                          help="TF-IDF urgency against the served adapter, under a frozen rule")
    p_ut.add_argument("--save", action="store_true",
                      help="write models/urgency-tfidf/model.joblib and score the reloaded file")
    p_ut.set_defaults(func=_cmd_urgency_tfidf)

    p_pr = sub.add_parser("pii-realistic",
                          help="served PII adapter on Nemotron-PII formats and real TAB court text (local)")
    p_pr.add_argument("--batch", type=int, default=8)
    p_pr.add_argument("--adapter-dir", default=None, help="score a local checkpoint instead of the served revision")
    p_pr.add_argument("--name", default=None, help="names the candidate's output files")
    p_pr.set_defaults(func=_cmd_pii_realistic)

    p_prc = sub.add_parser("pii-realistic-compare",
                           help="paired bootstrap of a candidate's realistic-format leaks against the served adapter")
    p_prc.add_argument("--candidate", required=True, help="the --name given to pii-realistic")
    p_prc.set_defaults(func=_cmd_pii_realistic_compare)

    p_prs = sub.add_parser("pii-realistic-split",
                           help="PII training split with Nemotron-PII documents; frozen with its decision rule")
    p_prs.add_argument("--force", action="store_true", help="rebuild the frozen split")
    p_prs.set_defaults(func=_cmd_pii_realistic_split)

    p_rg = sub.add_parser("regress", help="on-demand regression run, both splits (F18)")
    p_rg.add_argument("--base-url", default="http://localhost:8000")
    p_rg.add_argument("--name", required=True, help="names runs/regression__<name>.json")
    p_rg.add_argument("--baseline", default=None,
                      help="a previous regression run to compare against")
    p_rg.add_argument("--save-predictions", action="store_true",
                      help="keep every prediction, so drafting can be judged after the GPU is gone")
    p_rg.add_argument("--pins", default=None,
                      help="the pin set vLLM serves; tasks it pins to a classical model are answered locally")
    p_rg.set_defaults(func=_cmd_regress)

    p_hc = sub.add_parser("hard-cases",
                          help="adjudicate mined failures and freeze the hard split (F30/F31)")
    p_hc.add_argument("--force", action="store_true",
                      help="re-screen a frozen hard split — moves the bar it gates on")
    p_hc.set_defaults(func=_cmd_hard_cases)

    p_js = sub.add_parser("judge-score",
                          help="score a regression run's saved drafting replies with the judge")
    p_js.add_argument("--run", required=True, help="a regress or prompted run made with --save-predictions")
    p_js.add_argument("--baseline", default=None, help="recompute the run's comparison against this")
    p_js.add_argument("--force", action="store_true")
    p_js.set_defaults(func=_cmd_judge_score)

    p_m2 = sub.add_parser("judge-m2",
                          help="GPT-4o grades on golden drafting, adapter vs prompted (M2)")
    p_m2.add_argument("--project", action="store_true",
                      help="count tokens and price the run locally — makes no API call")
    p_m2.add_argument("--sides", nargs="+", default=["adapter", "prompted"],
                      choices=["adapter", "prompted", "frontier"])
    p_m2.add_argument("--workers", type=int, default=4)
    p_m2.add_argument("--spend-cap", type=float, default=1.0,
                      help="hard stop in USD, derived from token counts")
    p_m2.set_defaults(func=_cmd_judge_m2)

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

    p_comb = sub.add_parser("combine-candidates",
                            help="merge pinned candidates to serve several swaps at once")
    p_comb.add_argument("--from", dest="sources", nargs="+", required=True,
                        help="candidate names under manifests/candidates/")
    p_comb.add_argument("--name", required=True)
    p_comb.add_argument("--force", action="store_true")
    p_comb.set_defaults(func=_cmd_combine_candidates)

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

    p_econ = sub.add_parser("economics",
                            help="derived cost per 1K requests and weight footprint (D41)")
    p_econ.set_defaults(func=_cmd_economics)

    p_dash = sub.add_parser("dashboard",
                            help="render runs/DASHBOARD.md from committed runs (F17, F32)")
    p_dash.set_defaults(func=_cmd_dashboard)
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
