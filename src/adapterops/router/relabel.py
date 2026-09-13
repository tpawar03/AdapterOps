"""D38 — replace drafting's proxy labels with the judge's, and move nothing else.

Phase 3 closed D28 badly for the token-F1 proxy (Spearman 0.28, κ 0.14), and that proxy had
been used twice: to label the router's drafting pairs, and to mine drafting's hard cases.
43% of the router's drafting labels flip under the judge, and 73% of the graded drafting hard
cases are judge successes. This rebuilds both from GPT-4o's teacher grades.

**The router's split membership is kept exactly as frozen; only drafting's labels change.**
Re-running the dataset build would re-stratify on the new labels and move rows between train,
eval and shift, so a before-and-after comparison would sit on different rows — two changes
where one was intended. With membership fixed, any change in the router is the label's doing.

**Success is a teacher grade of 4 or more** — the rubric's "helpful and correct, with minor
gaps", the lowest grade a reply could be sent on without rework. The old label is kept beside
the new one as `proxy_success`, so every flip stays auditable.

**The drafting hard bucket is rebuilt from judge failures in the mining slice**, all 700 of
which are now graded, capped as before. The intent, urgency and PII buckets are copied through
untouched.

Everything is written beside the frozen artifacts with a `__judged` suffix. Replacing the frozen
versions is a separate decision, recorded when it is made.

    uv run python -m adapterops.router.relabel
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adapterops.judge.report import JUDGE_SUCCESS_MIN
from adapterops.router.dataset import HARD_CAP_PER_TASK, SEED

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER_DIR = REPO_ROOT / "data" / "router"
SCORED = ROUTER_DIR / "scored.parquet"
JUDGMENTS = REPO_ROOT / "data" / "judge" / "judgments.parquet"
HARD = REPO_ROOT / "evals" / "hard" / "hard_cases.parquet"
SPLITS = ("train", "eval", "shift_eval")
LABEL_RULE = (f"GPT-4o teacher grade >= {JUDGE_SUCCESS_MIN} "
              f"(reference-free judge, D34; replaces the token-F1 proxy, D38)")


def teacher_grades(judgments: pd.DataFrame) -> pd.Series:
    """The adapter's graded drafting replies, as pair_id -> 1-5 grade."""
    local = judgments[(judgments.source == "local") & judgments.parsed_ok.astype(bool)]
    return local.drop_duplicates("pair_id").set_index("pair_id").score.astype(float)


def relabel(scored: pd.DataFrame, grades: pd.Series) -> pd.DataFrame:
    out = scored.copy()
    drafting = out.task == "drafting"
    ungraded = drafting & ~out.pair_id.isin(grades.index)
    if ungraded.any():
        msg = f"{int(ungraded.sum())} drafting pairs have no teacher grade — refusing to mix labels"
        raise ValueError(msg)
    out["proxy_success"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[drafting, "proxy_success"] = out.loc[drafting, "success"].astype(bool)
    out["judge_score"] = out.pair_id.map(grades).where(drafting)
    out.loc[drafting, "success"] = out.loc[drafting, "judge_score"] >= JUDGE_SUCCESS_MIN
    out["success"] = out["success"].astype(bool)
    out.loc[drafting, "label_rule"] = LABEL_RULE
    out.loc[drafting, "label_is_proxy"] = False
    return out


def relabel_router_splits(relabelled: pd.DataFrame,
                          router_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Same rows in every split as frozen; drafting's `success` taken from the relabelled pool."""
    lookup = relabelled.set_index("pair_id").success
    out = {}
    for split in SPLITS:
        frame = pd.read_parquet((router_dir or ROUTER_DIR) / f"router_{split}.parquet").copy()
        drafting = frame.task == "drafting"
        frame.loc[drafting, "success"] = frame.loc[drafting, "pair_id"].map(lookup)
        frame["success"] = frame["success"].astype(bool)
        if "label_is_proxy" in frame:
            frame.loc[drafting, "label_is_proxy"] = False
        out[split] = frame
    return out


def drafting_hard_bucket(relabelled: pd.DataFrame, cap: int = HARD_CAP_PER_TASK,
                         seed: int = SEED) -> pd.DataFrame:
    """Judge failures among mining-slice drafting replies, capped and balanced by grade."""
    mining = relabelled[(relabelled.task == "drafting") & (relabelled.purpose == "mining")]
    failures = mining[~mining.success.astype(bool)].copy()
    failures["failure_type"] = "judge_grade_" + failures.judge_score.astype(int).astype(str)
    if len(failures) <= cap:
        return failures
    grades = sorted(failures.failure_type.unique())
    per = max(1, cap // len(grades))
    take = pd.concat([g.sample(n=min(per, len(g)), random_state=seed)
                      for _, g in failures.groupby("failure_type")])
    if len(take) < cap:
        rest = failures.drop(index=take.index).sample(frac=1.0, random_state=seed)
        take = pd.concat([take, rest.head(cap - len(take))])
    return take


def rebuild_hard_split(hard: pd.DataFrame, bucket: pd.DataFrame) -> pd.DataFrame:
    kept = hard[hard.task != "drafting"]
    fresh = bucket.copy()
    fresh["applicability"] = "not_applicable"
    fresh["quarantine"] = False
    for column in ("has_frontier_answer", "frontier_rejects_gold",
                   "independent_agreement_against_gold"):
        fresh[column] = False
    combined = pd.concat([kept, fresh], ignore_index=True)
    for column in ("quarantine", "has_frontier_answer", "frontier_rejects_gold",
                   "independent_agreement_against_gold"):
        if column in combined:
            combined[column] = combined[column].fillna(False).astype(bool)
    return combined


def main() -> int:
    scored = pd.read_parquet(SCORED)
    relabelled = relabel(scored, teacher_grades(pd.read_parquet(JUDGMENTS)))
    splits = relabel_router_splits(relabelled)
    hard = pd.read_parquet(HARD)
    bucket = drafting_hard_bucket(relabelled)
    rebuilt = rebuild_hard_split(hard, bucket)

    relabelled.to_parquet(ROUTER_DIR / "scored__judged.parquet")
    for split, frame in splits.items():
        frame.to_parquet(ROUTER_DIR / f"router_{split}__judged.parquet")
    rebuilt.to_parquet(HARD.with_name("hard_cases__judged.parquet"))

    drafting = relabelled[relabelled.task == "drafting"]
    old_bucket = set(hard[hard.task == "drafting"].pair_id)
    summary = {
        "decision": "D38 — drafting labels from GPT-4o teacher grades; router membership fixed",
        "label_rule": LABEL_RULE,
        "drafting_success_rate": {
            purpose: {"proxy": round(float(g.proxy_success.astype(bool).mean()), 4),
                      "judge": round(float(g.success.mean()), 4)}
            for purpose, g in drafting.groupby("purpose")
        },
        "router_split_flips": {
            split: int((frame[frame.task == "drafting"].success.to_numpy()
                        != pd.read_parquet(ROUTER_DIR / f"router_{split}.parquet")
                        .query("task == 'drafting'").success.astype(bool).to_numpy()).sum())
            for split, frame in splits.items()
        },
        "hard_split": {
            "drafting_bucket_before": len(old_bucket),
            "drafting_bucket_after": len(bucket),
            "cap": HARD_CAP_PER_TASK,
            "cap_bound": len(bucket) >= HARD_CAP_PER_TASK,
            "judge_failures_in_mining_slice": int((~drafting[drafting.purpose == "mining"]
                                                   .success).sum()),
            "grades_in_new_bucket": {k: int(v) for k, v in
                                     bucket.failure_type.value_counts().sort_index().items()},
            "old_proxy_hard_cases_still_hard": len(old_bucket & set(bucket.pair_id)),
            "other_tasks_unchanged": int(len(rebuilt[rebuilt.task != "drafting"])
                                         == len(hard[hard.task != "drafting"])),
            "total_before": len(hard),
            "total_after": len(rebuilt),
        },
        "written": ["data/router/scored__judged.parquet"]
                   + [f"data/router/router_{s}__judged.parquet" for s in SPLITS]
                   + ["evals/hard/hard_cases__judged.parquet"],
        "not_yet_done": "replacing the frozen artifacts, and retraining the router on these labels",
    }
    out = REPO_ROOT / "evals" / "RELABEL_D38.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
