"""Join the local and frontier runs and produce the operating curve (F11, M3, M4).

Three inputs meet here: the local run's per-pair success and confidence, the frontier run's
per-pair success, and the learned router's per-pair failure probability. Everything before
this point was careful to keep them comparable, and this is where being careless cashes out.

**Three invariants it enforces rather than assumes.**

*One drafting cut, taken from the local run, applied to both arms.* Cutting each arm at its
own median would hand the frontier a 50% drafting success rate whatever it produced, and
the curve would compare two different bars while looking entirely normal (D28).

*Every policy is scored on the same rows.* The learned router only has predictions for the
held-out eval set, so a curve computed over the whole pool would rank the router on missing
values while ranking confidence on everything — two policies, two populations, one chart.
The population is therefore named in the output, and when a router exists the population
*is* its eval set.

*The curve is reported with and without drafting.* D28 committed to that before any result
existed. If the two readings disagree, the disagreement is the finding: it says how much of
the headline rests on the one task whose label is a stand-in.

    uv run adapterops router-report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adapterops.judge.report import JUDGE_SUCCESS_MIN
from adapterops.router.baselines import (
    DEFAULT_BUDGETS,
    compare,
    headroom_captured,
    operating_curve,
)
from adapterops.router.scoring import PROXY_LABELLED_TASKS, drafting_cut_from, label

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
OUT_JSON = REPO_ROOT / "runs" / "router__operating_curve.json"
OUT_CSV = REPO_ROOT / "runs" / "router__operating_curve.csv"
JUDGMENTS = REPO_ROOT / "data" / "judge" / "judgments.parquet"


def join(local: pd.DataFrame, frontier: pd.DataFrame,
         router_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pair, carrying both arms' success under a single set of rules."""
    cut = drafting_cut_from(local)
    local = label(local, drafting_cut=cut)

    front = frontier.rename(columns={c: c.removeprefix("frontier_")
                                     for c in frontier.columns
                                     if c.startswith("frontier_")})
    front = label(front, drafting_cut=cut)[["pair_id", "success"]]
    front = front.rename(columns={"success": "frontier_success"})

    joined = local.merge(front, on="pair_id", how="inner", validate="one_to_one")
    if router_scores is not None:
        # `side` lives in the router's eval splits, not in the scored pool. Merging only
        # router_p_fail dropped it, so populations() never split and the report blended
        # in-distribution and shift rows — M3 and M4 measured as one number.
        keep = ["pair_id", "router_p_fail"]
        if "side" in router_scores.columns and "side" not in joined.columns:
            keep.append("side")
        joined = joined.merge(router_scores[keep],
                              on="pair_id", how="left", validate="one_to_one")
    joined.attrs["drafting_cut"] = cut
    return joined


def populations(joined: pd.DataFrame,
                router_scores: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Which rows each curve is computed over, named so the reader can check.

    With a trained router the populations are its eval splits, because that is where every
    policy can be scored on identical rows. Without one, the whole router slice is used and
    said so — the confidence and oracle curves are still meaningful there.
    """
    if router_scores is None:
        return {"router_slice__no_router_yet": joined}
    scored = joined[joined.router_p_fail.notna()]
    by_side = {}
    if "side" in scored.columns:
        for side, frame in scored.groupby("side"):
            by_side[f"router_{side}"] = frame
    return by_side or {"router_eval": scored}


def curves(frame: pd.DataFrame, policies: tuple[str, ...],
           budgets=DEFAULT_BUDGETS,
           proxy_tasks: tuple[str, ...] = PROXY_LABELLED_TASKS) -> dict:
    """The comparison over one population, computed twice: all tasks, and real labels only."""
    out: dict = {}
    for view, rows in (("all_tasks", frame),
                       ("real_labels_only", frame[~frame.task.isin(proxy_tasks)])):
        if rows.empty or rows.success.nunique() < 2:
            out[view] = {"pairs": len(rows),
                         "note": "too few rows, or no failures — no curve to draw"}
            continue
        table = compare(rows, policies=policies, budgets=budgets)
        oracle = operating_curve(rows, "oracle", budgets)
        out[view] = {
            "pairs": len(rows),
            "tasks": sorted(rows.task.unique()),
            "local_quality": round(float(rows.success.mean()), 4),
            "frontier_quality": round(float(rows.frontier_success.mean()), 4),
            "curve": table.to_dict("records"),
            "headroom_captured_at_0.20": {
                p: headroom_captured(table[table.policy == p], oracle, 0.20)
                for p in policies if p != "oracle"
            },
        }
    return out


def build_report(joined: pd.DataFrame, router_scores: pd.DataFrame | None,
                 policies: tuple[str, ...],
                 proxy_tasks: tuple[str, ...] = PROXY_LABELLED_TASKS) -> dict:
    return {
        "note": "Escalation quality is measured, not assumed — frontier_success comes from "
                "an actual GPT-4o-mini run over the same pairs (F8).",
        "drafting_cut": joined.attrs.get("drafting_cut"),
        "drafting_cut_note": "taken from the local run and applied to both arms, so the two "
                             "are scored against one bar (D28)",
        "policies": list(policies),
        "pairs_joined": len(joined),
        "populations": {
            name: curves(frame, policies, proxy_tasks=proxy_tasks)
            for name, frame in populations(joined, router_scores).items()
        },
    }


def join_judged(local: pd.DataFrame, frontier: pd.DataFrame, frontier_grades: pd.Series,
                router_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """D38: drafting success from GPT-4o grades on **both** arms; every other task as `join`.

    `join` re-derives every label with the proxy rule, so handing it D38's relabelled pool would
    quietly put drafting back on token-F1. Worse, it would grade the frontier's drafts by the proxy
    and the adapter's by the judge — two bars on one chart, the thing D28's invariant exists to
    prevent. So drafting is overwritten after the join from the one source both arms share: the
    GPT-4o grade.
    """
    drafting = local.task == "drafting"
    if "label_is_proxy" not in local or local.loc[drafting, "label_is_proxy"].astype(bool).any():
        msg = "local drafting labels are still the token-F1 proxy — pass the D38 __judged pool"
        raise ValueError(msg)
    judged_local = local.loc[drafting].set_index("pair_id").success.astype(bool)

    base = join(local, frontier, router_scores)
    rows = base.task == "drafting"
    missing = set(base.loc[rows, "pair_id"]) - set(frontier_grades.index)
    if missing:
        msg = f"{len(missing)} frontier drafting replies have no GPT-4o grade"
        raise ValueError(msg)
    base.loc[rows, "success"] = base.loc[rows, "pair_id"].map(judged_local)
    base.loc[rows, "frontier_success"] = (
        base.loc[rows, "pair_id"].map(frontier_grades) >= JUDGE_SUCCESS_MIN)
    base["success"] = base.success.astype(bool)
    base["frontier_success"] = base.frontier_success.astype(bool)
    base.loc[rows, "label_is_proxy"] = False
    base.attrs["drafting_labels"] = (f"GPT-4o grade >= {JUDGE_SUCCESS_MIN} on both arms (D38)")
    return base


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(judged: bool = False) -> int:
    """`judged=True` reads D38's __judged pool and router scores and grades drafting by the judge."""
    suffix = "__judged" if judged else ""
    local_path = DATA_DIR / f"scored{suffix}.parquet"
    frontier_path = DATA_DIR / "frontier.parquet"
    hints = {
        local_path: ("run `python -m adapterops.router.relabel` (D38)" if judged
                     else "the pool has not been scored through the adapters (GPU session)"),
        frontier_path: "run `adapterops frontier`",
    }
    for path, hint in hints.items():
        if not path.exists():
            print(f"  no {_shown(path)} — {hint}")
            return 2

    local = pd.read_parquet(local_path)
    local = local[local.purpose == "router"]
    frontier = pd.read_parquet(frontier_path)

    # Both router eval splits, so M3 (in-distribution) and M4 (shift) are one report.
    frames = [DATA_DIR / f"router_{n}_scored{suffix}.parquet" for n in ("eval", "shift_eval")]
    present = [pd.read_parquet(f) for f in frames if f.exists()]
    router_scores = pd.concat(present, ignore_index=True) if present else None

    policies = ("random", "confidence", "oracle")
    if router_scores is not None:
        policies = ("random", "confidence", "router_p_fail", "oracle")

    if judged:
        grades = pd.read_parquet(JUDGMENTS)
        grades = (grades[(grades.source == "frontier") & grades.parsed_ok.astype(bool)]
                  .drop_duplicates("pair_id").set_index("pair_id").score.astype(float))
        joined = join_judged(local, frontier, grades, router_scores)
        proxy_tasks: tuple[str, ...] = ()
    else:
        joined = join(local, frontier, router_scores)
        proxy_tasks = PROXY_LABELLED_TASKS
    report = build_report(joined, router_scores, policies, proxy_tasks=proxy_tasks)
    report["drafting_labels"] = joined.attrs.get("drafting_labels", "token-F1 proxy (D28)")

    out_json = OUT_JSON.with_name(f"router__operating_curve{suffix}.json")
    out_csv = OUT_CSV.with_name(f"router__operating_curve{suffix}.csv")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [{"population": pop, **r}
            for pop, views in report["populations"].items()
            for r in views.get("all_tasks", {}).get("curve", [])]
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"  drafting labels: {report['drafting_labels']}")
    for pop, views in report["populations"].items():
        print(f"\n  === {pop} ===")
        for view, block in views.items():
            if "curve" not in block:
                print(f"  {view}: {block['note']}")
                continue
            print(f"  -- {view} ({block['pairs']:,} pairs) · "
                  f"local {block['local_quality']:.4f} · "
                  f"frontier {block['frontier_quality']:.4f}")
            table = pd.DataFrame(block["curve"])
            print(table.pivot(index="budget", columns="policy", values="quality").to_string())
            print(f"  headroom captured at 20%: {block['headroom_captured_at_0.20']}")
    print(f"\n  wrote {_shown(out_json)}")
    return 0
