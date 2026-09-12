"""What the GPT-4o grades say — the three questions Phase 3 exists to answer.

**1 · D28, the commitment made before a judge existed.** Drafting's router label is a
stand-in: token-F1 against the Bitext reference, cut at the pool median. D28 said that when
the real metric arrived, the proxy's agreement with it would be measured and published
whatever it came to. That comparison uses the adapter's own router-slice replies only, since
those are the pairs that carry the proxy label.

**2 · The §10 same-family caveat.** GPT-4o grades the adapter's replies and GPT-4o-mini's
replies to the same requests. A gap favouring the frontier may be quality or family
preference, and the two cannot be separated without human labels — so the gap is reported
with that caveat attached, not resolved in either direction.

**3 · The fairer escalation comparison.** Phase 2 found escalation lowered drafting quality
under token-F1, and attributed it to house-style conformance rather than capability. A
reference-free judge is the metric that tests that attribution. The same pairs are shown
under both metrics side by side, so a reversal is visible as a reversal.

**A partial run is labelled, not hidden.** If grading has not finished, the report still
builds — partial numbers are informative — but `coverage.complete` is false and the output
says every figure is provisional.

    uv run python -m adapterops.judge.report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adapterops.judge.calibrate import proxy_vs_judge, same_family_check
from adapterops.router.scoring import drafting_cut_from

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGMENTS = REPO_ROOT / "data" / "judge" / "judgments.parquet"
SCORED = REPO_ROOT / "data" / "router" / "scored.parquet"
OUT = REPO_ROOT / "runs" / "judge__report.json"

JUDGE_SUCCESS_MIN = 4
"""A grade of 4 or 5 counts as a successful draft. A stated choice, recorded in every output:
4 is the rubric's "helpful and correct, with minor gaps", the lowest grade a reply could be
sent on without rework."""
EXPECTED_ROUTER_PAIRS = 750


def graded(judgments: pd.DataFrame, source: str, purpose: str = "router") -> pd.Series:
    rows = judgments[(judgments.source == source) & (judgments.purpose == purpose)
                     & judgments.parsed_ok.astype(bool)]
    return rows.set_index("pair_id").score.astype(float)


def _rate(mask: pd.Series) -> float:
    return round(float(mask.mean()), 4) if len(mask) else float("nan")


def escalation_under_judge(local: pd.Series, frontier: pd.Series,
                           local_proxy: pd.Series, frontier_proxy: pd.Series,
                           cut: float) -> dict:
    """Local versus frontier drafts on the same requests, under the judge and under the proxy."""
    ids = local.index.intersection(frontier.index)
    ids = ids.intersection(local_proxy.index).intersection(frontier_proxy.index)
    l_ok = local.loc[ids] >= JUDGE_SUCCESS_MIN
    f_ok = frontier.loc[ids] >= JUDGE_SUCCESS_MIN
    lp, fp = local_proxy.loc[ids], frontier_proxy.loc[ids]
    lp_ok = (lp > 0) & (lp >= cut)
    fp_ok = (fp > 0) & (fp >= cut)
    return {
        "pairs": len(ids),
        "judge": {
            "local_success": _rate(l_ok), "frontier_success": _rate(f_ok),
            "rescued": _rate(~l_ok & f_ok), "broken": _rate(l_ok & ~f_ok),
            "mean_local": round(float(local.loc[ids].mean()), 4) if len(ids) else None,
            "mean_frontier": round(float(frontier.loc[ids].mean()), 4) if len(ids) else None,
        },
        "token_f1_proxy_on_the_same_pairs": {
            "local_success": _rate(lp_ok), "frontier_success": _rate(fp_ok),
            "rescued": _rate(~lp_ok & fp_ok), "broken": _rate(lp_ok & ~fp_ok),
        },
        "judge_success_min": JUDGE_SUCCESS_MIN,
        "proxy_cut": cut,
    }


def build_report(judgments: pd.DataFrame, scored: pd.DataFrame) -> dict:
    router = scored[scored.purpose == "router"]
    cut = drafting_cut_from(router)
    local_proxy = router[router.task == "drafting"].set_index("pair_id").proxy_token_f1

    local = graded(judgments, "local")
    frontier = graded(judgments, "frontier")
    frontier_proxy = (judgments[judgments.source == "frontier"]
                      .drop_duplicates("pair_id").set_index("pair_id").proxy_token_f1)

    d28_ids = local.index.intersection(local_proxy.index)
    if len(d28_ids) >= 10:
        d28 = proxy_vs_judge(local_proxy.loc[d28_ids].to_numpy(),
                             local.loc[d28_ids].to_numpy(),
                             proxy_cut=cut, judge_success_min=JUDGE_SUCCESS_MIN)
    else:
        d28 = {"n": len(d28_ids), "note": "too few graded router drafting pairs"}

    complete = len(local) >= EXPECTED_ROUTER_PAIRS and len(frontier) >= EXPECTED_ROUTER_PAIRS
    return {
        "coverage": {
            "local_router_graded": len(local),
            "frontier_router_graded": len(frontier),
            "expected_each": EXPECTED_ROUTER_PAIRS,
            "complete": complete,
            "note": None if complete else "grading unfinished — every figure is provisional",
        },
        "score_distribution": {
            src: {int(k): int(v) for k, v in graded(judgments, src).value_counts()
                  .sort_index().items()}
            for src in ("local", "frontier")
        },
        "d28_proxy_vs_judge": d28,
        "same_family_check": same_family_check(local, frontier),
        "escalation_under_judge": escalation_under_judge(
            local, frontier, local_proxy, frontier_proxy, cut),
    }


def main() -> int:
    if not JUDGMENTS.exists():
        print("  no data/judge/judgments.parquet — run `uv run adapterops judge-label` first.")
        return 2
    report = build_report(pd.read_parquet(JUDGMENTS), pd.read_parquet(SCORED))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["coverage"]["complete"]:
        print("  PARTIAL — grading has not finished; every number below is provisional")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
