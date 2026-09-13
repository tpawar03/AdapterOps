"""Router v2 — three pre-registered variants against the confidence baseline (D42).

The v1 router lost to the adapter's own confidence for two diagnosed reasons:

1. **Wrong target.** It predicted local *failure*, but escalation only pays where the frontier then
   succeeds. At a 20% budget it escalated 78% of urgency, where GPT-4o-mini (0.42) is worse than the
   adapter (0.51). The oracle already ranks by gain (D32); the router's label never did.
2. **Wrong information.** It read the ticket text alone. The confidence policy reads the adapter's
   output log-probabilities, which a cascade has for free — the adapter always runs first.

Each variant addresses one or both, and all three were fixed in `evals/ROUTER_V2_PREREG.json` before
any was trained or scored. `prepare` freezes that file's hash; `evaluate` refuses to run if it moved.

- `gain_text` — the v1 DeBERTa router with its label changed to *rescue*. Three seeds, each judged.
- `cascade_lr` — logistic regression on per-task generation signals, label rescue.
- `confidence_per_task` — confidence standardised within each task on training statistics. No label.

**Nothing new is trusted until the old numbers reproduce.** `evaluate` recomputes the confidence,
oracle and v1-router curves from these frames and fails if any quality differs from the published
judged report — the check that the rescue labels were joined under the report's own rules.

    uv run python -m adapterops.router.cascade prepare
    uv run python -m adapterops.router.cascade train-text     # three seeds, CPU
    uv run python -m adapterops.router.cascade evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.router.baselines import compare, headroom_captured, operating_curve
from adapterops.router.report import frontier_grades, join_judged

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
PREREG = REPO_ROOT / "evals" / "ROUTER_V2_PREREG.json"
PREREG_HASH = REPO_ROOT / "evals" / "ROUTER_V2_PREREG.sha256"
PUBLISHED = REPO_ROOT / "runs" / "router__operating_curve__judged.json"
OUT_FILE = REPO_ROOT / "runs" / "router__v2.json"

SPLITS = {"eval": "router_in_distribution", "shift_eval": "router_shift"}
TASKS = ("drafting", "intent", "pii", "urgency")
SIGNALS = ("mean_logprob", "min_logprob", "log_tokens", "hit_cap")
BUDGET = 0.20
REPRODUCED = ("confidence", "oracle", "router_v1")
"""`random` is left out: its draw depends on row order, and the report's frames were ordered by the
join, not by the split files."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prereg() -> dict:
    if not PREREG_HASH.exists():
        msg = "no frozen pre-registration hash — run `cascade prepare` first"
        raise FileNotFoundError(msg)
    digest = sha256(PREREG)
    if PREREG_HASH.read_text().strip() != digest:
        msg = ("evals/ROUTER_V2_PREREG.json changed after it was frozen. A pre-registration that "
               "can be edited after scoring is not one — record the change as a new decision")
        raise ValueError(msg)
    return {**json.loads(PREREG.read_text()), "_sha256": digest}


# ---------------------------------------------------------------- labels


def joined_router_rows() -> pd.DataFrame:
    """Both arms' success for every router pair, under the judged report's rules (D38)."""
    local = pd.read_parquet(DATA_DIR / "scored__judged.parquet")
    local = local[local.purpose == "router"]
    frontier = pd.read_parquet(DATA_DIR / "frontier.parquet")
    return join_judged(local, frontier, frontier_grades())


def with_rescue(split_frame: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    """Attach `frontier_success` and `rescue`, refusing if the two sources disagree on `success`."""
    arms = joined.set_index("pair_id")[["success", "frontier_success"]]
    missing = set(split_frame.pair_id) - set(arms.index)
    if missing:
        msg = f"{len(missing)} split pairs are absent from the join"
        raise ValueError(msg)
    frame = split_frame.join(arms.rename(columns={"success": "joined_success"}), on="pair_id")
    if (frame.success.astype(bool) != frame.joined_success.astype(bool)).any():
        msg = "the split's success label disagrees with the join — built under different rules"
        raise ValueError(msg)
    frame = frame.drop(columns="joined_success")
    frame["frontier_success"] = frame.frontier_success.astype(bool)
    frame["rescue"] = frame.frontier_success & ~frame.success.astype(bool)
    return frame


def prepare() -> int:
    if not PREREG_HASH.exists():
        PREREG_HASH.write_text(sha256(PREREG) + "\n")
        print(f"  froze {PREREG.relative_to(REPO_ROOT)} at {sha256(PREREG)[:12]}")
    load_prereg()
    joined = joined_router_rows()
    for split in ("train", *SPLITS):
        frame = with_rescue(pd.read_parquet(DATA_DIR / f"router_{split}__judged.parquet"), joined)
        frame.to_parquet(DATA_DIR / f"router_{split}__rescue.parquet")
        if split == "train":   # base rates on train only — eval stays unread until `evaluate`
            print("  train rescue rate by task:",
                  frame.groupby("task").rescue.mean().round(3).to_dict())
    print("  wrote router_{train,eval,shift_eval}__rescue.parquet")
    return 0


# ---------------------------------------------------------------- variants


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-task intercept and per-task slope on every generation signal.

    A pooled slope would repeat v1-confidence's assumption that a log-probability means the same
    thing for a 5-token intent label and a 120-token reply.
    """
    signals = pd.DataFrame({
        "mean_logprob": frame.mean_logprob.astype(float),
        "min_logprob": frame.min_logprob.astype(float),
        "log_tokens": np.log1p(frame.n_tokens.astype(float)),
        "hit_cap": (frame.finish_reason == "length").astype(float),
    }, index=frame.index)
    if signals.isna().any().any():
        msg = "missing generation signals — the cascade cannot score a pair it has no output for"
        raise ValueError(msg)
    cols: dict[str, pd.Series] = {}
    for task in TASKS:
        on = (frame.task == task).astype(float)
        cols[task] = on
        for name in SIGNALS:
            cols[f"{task}__{name}"] = on * signals[name]
    return pd.DataFrame(cols, index=frame.index)


def fit_cascade(train: pd.DataFrame, spec: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    strata = train.task + "|" + train.rescue.astype(str)
    folds = StratifiedKFold(n_splits=spec["cv_folds"], shuffle=True, random_state=spec["seed"])
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000)),
        {"logisticregression__C": spec["C_grid"]},
        scoring=spec["cv_scoring"],
        cv=list(folds.split(train, strata)),
    )
    search.fit(design(train), train.rescue.astype(int))
    return search


def confidence_per_task(train: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    stats = train.groupby("task").mean_logprob.agg(["mean", "std"])
    z = ((frame.mean_logprob - frame.task.map(stats["mean"]))
         / frame.task.map(stats["std"]))
    return -z


# ---------------------------------------------------------------- scoring


def quality_at_budget(local: np.ndarray, frontier: np.ndarray, scores: np.ndarray,
                      id_rank: np.ndarray, budget: float) -> float:
    """`baselines.evaluate_at_budget`'s quality, in numpy for the bootstrap: escalate the top
    `budget` share by score, ties broken by pair_id ascending."""
    n = round(budget * len(local))
    realised = local.copy()
    if n:
        top = np.lexsort((id_rank, -scores))[:n]
        realised[top] = frontier[top]
    return float(realised.mean())


def paired_bootstrap(frame: pd.DataFrame, scores_a: np.ndarray, scores_b: np.ndarray,
                     budget: float, resamples: int, seed: int) -> dict:
    """quality(a) − quality(b), resampling pairs within task, budget re-applied on every draw."""
    local = frame.success.to_numpy(float)
    front = frame.frontier_success.to_numpy(float)
    id_rank = frame.pair_id.rank(method="dense").to_numpy()
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    tasks = frame.task.to_numpy()
    groups = [np.flatnonzero(tasks == t) for t in sorted(set(tasks))]
    rng = np.random.default_rng(seed)

    diffs = np.empty(resamples)
    for i in range(resamples):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        diffs[i] = (quality_at_budget(local[idx], front[idx], a[idx], id_rank[idx], budget)
                    - quality_at_budget(local[idx], front[idx], b[idx], id_rank[idx], budget))
    point = (quality_at_budget(local, front, a, id_rank, budget)
             - quality_at_budget(local, front, b, id_rank, budget))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "resamples": resamples}


def decide(in_distribution: dict, shift: dict) -> str:
    lo, hi = in_distribution["ci95"]
    if lo > 0 and shift["difference"] > 0:
        return "beats confidence"
    if hi < 0:
        return "loses to confidence"
    return "inconclusive"


def combine(outcomes: list[str]) -> str:
    """gain_text's rule: a verdict only if every seed reaches it."""
    return outcomes[0] if len(set(outcomes)) == 1 else "inconclusive"


def rescue_auc(frame: pd.DataFrame, scores: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score

    def one(mask: pd.Series) -> float | None:
        y = frame.rescue[mask].astype(int)
        return None if y.nunique() < 2 else round(float(roc_auc_score(y, scores[mask])), 4)

    return {"overall": one(frame.task.notna()),
            **{t: one(frame.task == t) for t in sorted(frame.task.unique())}}


def attach_scores(frame: pd.DataFrame, split: str, train: pd.DataFrame, search,
                  seeds: list[int]) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame["confidence_per_task"] = confidence_per_task(train, frame)
    frame["cascade_lr"] = search.predict_proba(design(frame))[:, 1]
    sources = {f"gain_text_s{s}": f"router_{split}_scored__rescue_s{s}.parquet" for s in seeds}
    sources["router_v1"] = f"router_{split}_scored__judged.parquet"
    for column, name in sources.items():
        path = DATA_DIR / name
        if not path.exists():
            msg = f"no {path.relative_to(REPO_ROOT)} — run `cascade train-text` first"
            raise FileNotFoundError(msg)
        scores = pd.read_parquet(path)[["pair_id", "router_p_fail"]]
        frame = frame.merge(scores.rename(columns={"router_p_fail": column}), on="pair_id",
                            how="left", validate="one_to_one")
        if frame[column].isna().any():
            msg = f"{column} is missing scores for some pairs"
            raise ValueError(msg)
    return frame


def check_reproduces(table: pd.DataFrame, population: str) -> None:
    published = json.loads(PUBLISHED.read_text())["populations"][population]["all_tasks"]["curve"]
    names = {"router_p_fail": "router_v1"}
    for row in published:
        policy = names.get(row["policy"], row["policy"])
        if policy not in REPRODUCED:
            continue
        mine = table[(table.policy == policy) & (table.budget == row["budget"])].quality.iloc[0]
        if abs(mine - row["quality"]) > 1e-4:
            msg = (f"{population}: {policy} at budget {row['budget']} gives {mine}, published "
                   f"{row['quality']} — the frames do not match the report, so nothing new here "
                   f"can be compared with it")
            raise ValueError(msg)


def evaluate() -> int:
    prereg = load_prereg()
    variants = prereg["variants"]
    seeds = variants["gain_text"]["seeds"]
    boot = prereg["uncertainty"]

    train = pd.read_parquet(DATA_DIR / "router_train__rescue.parquet")
    search = fit_cascade(train, variants["cascade_lr"])
    candidates = ["confidence_per_task", "cascade_lr", *(f"gain_text_s{s}" for s in seeds)]
    policies = ("random", "confidence", "router_v1", *candidates, "oracle")

    out: dict = {
        "decision": "D42",
        "prereg": str(PREREG.relative_to(REPO_ROOT)),
        "prereg_sha256": prereg["_sha256"],
        "cascade_cv": {
            "best_C": search.best_params_["logisticregression__C"],
            "mean_cv_neg_log_loss": {
                str(c): round(float(s), 4)
                for c, s in zip(variants["cascade_lr"]["C_grid"],
                                search.cv_results_["mean_test_score"], strict=True)},
        },
        "populations": {},
    }
    comparisons: dict[str, dict[str, dict]] = {c: {} for c in candidates}
    for split, population in SPLITS.items():
        frame = pd.read_parquet(DATA_DIR / f"router_{split}__rescue.parquet")
        frame = attach_scores(frame, split, train, search, seeds)
        table = compare(frame, policies=policies)
        check_reproduces(table, population)
        oracle = operating_curve(frame, "oracle")

        block = {
            "pairs": len(frame),
            "rescue_rate": round(float(frame.rescue.mean()), 4),
            "local_quality": round(float(frame.success.mean()), 4),
            "frontier_quality": round(float(frame.frontier_success.mean()), 4),
            "reproduced_published": list(REPRODUCED),
            "headroom_captured": {
                str(b): {p: headroom_captured(table[table.policy == p], oracle, b)
                         for p in policies if p != "oracle"} for b in (0.10, BUDGET)},
            "rescue_auc": {p: rescue_auc(frame, frame[p])
                           for p in ("router_v1", *candidates)},
            "at_budget_0.20": table[table.budget == BUDGET].to_dict("records"),
            "curve": table.to_dict("records"),
            "vs_confidence_at_0.20": {},
        }
        for c in candidates:
            result = paired_bootstrap(frame, frame[c].to_numpy(), (-frame.mean_logprob).to_numpy(),
                                      BUDGET, boot["resamples"], boot["seed"])
            block["vs_confidence_at_0.20"][c] = result
            comparisons[c][population] = result
        out["populations"][population] = block

    verdicts = {c: decide(r["router_in_distribution"], r["router_shift"])
                for c, r in comparisons.items()}
    out["verdicts"] = {
        "confidence_per_task": verdicts["confidence_per_task"],
        "cascade_lr": verdicts["cascade_lr"],
        "gain_text": {"per_seed": {s: verdicts[f"gain_text_s{s}"] for s in seeds},
                      "combined": combine([verdicts[f"gain_text_s{s}"] for s in seeds])},
    }
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")

    for population, block in out["populations"].items():
        print(f"\n  === {population} ({block['pairs']} pairs) ===")
        print(f"  headroom captured at 0.20: {block['headroom_captured'][str(BUDGET)]}")
        for c, r in block["vs_confidence_at_0.20"].items():
            print(f"  {c:22s} vs confidence {r['difference']:+.4f}  95% CI {r['ci95']}")
    print(f"\n  verdicts: {json.dumps(out['verdicts'])}")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0


def train_text() -> int:
    from adapterops.router.train import RouterConfig, train

    for seed in load_prereg()["variants"]["gain_text"]["seeds"]:
        tag = f"rescue_s{seed}"
        # Never the default output_dir: checkpoints/router is the checkpoint manifest v2 pins.
        train(RouterConfig(seed=seed, target="rescue", data_suffix="__rescue", tag=tag,
                           output_dir=str(REPO_ROOT / "checkpoints" / f"router__{tag}")))
        print(f"  trained gain_text seed {seed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adapterops.router.cascade")
    ap.add_argument("step", choices=["prepare", "train-text", "evaluate"])
    step = ap.parse_args(argv).step
    return {"prepare": prepare, "train-text": train_text, "evaluate": evaluate}[step]()


if __name__ == "__main__":
    raise SystemExit(main())
