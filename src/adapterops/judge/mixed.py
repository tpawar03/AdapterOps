"""A drafting judge trained on every generator's replies (TODO.md §3, a judge that tracks GPT-4o on any generator).

The served judge learned GPT-4o's grades on the adapter's replies alone. It tracks GPT-4o there (Spearman 0.73), but
not on another generator's (0.33), and on out-of-domain tickets it ranks replies in the opposite order (-0.30,
-0.19, `runs/ood.json`) — so drafting's gate and the request path's judge scores cannot see a bad reply off the
training distribution. The data to fix that already exists and was paid for: GPT-4o grades on GPT-4o-mini's and the
prompted base model's replies, not only the adapter's.

**Frozen before training** (`data/judge/mixed_split.json`):

- **Training:** every graded in-domain reply from all three generators, minus the holdouts.
- **H1 — adapter calibration:** the served judge's own 150-item holdout, unchanged, so its 0.73 stays comparable.
- **H2 — across generators:** 30% of the M2 golden requests, with all three generators' replies to each.
- **H3 — out of domain:** all 120 graded replies on ABCD and CFPB tickets (60 adapter, 60 GPT-4o-mini). Never
  trained on: whether training on more generators alone transfers to unseen tickets is the question.

Every split is grouped by request text, so a request's replies never sit on both sides of a split.

**The rule, written before any score is read:** serve the mixed judge only if its H1 Spearman is no more than 0.05
below the served judge's on the same items, and it beats the served judge on both H2 and H3 with the paired
bootstrap's 95% interval for the difference above zero.

    uv run adapterops judge-mixed --build
    uv run adapterops judge-mixed --train      # CPU, background
    uv run adapterops judge-mixed --compare
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from adapterops.judge.train import JUDGE_DIR, REPO_ROOT, SEED, JudgeConfig

DATA_FILE = JUDGE_DIR / "mixed.parquet"
RECORD = JUDGE_DIR / "mixed_split.json"
OUT_DIR = REPO_ROOT / "checkpoints" / "judge-mixed"
RUN_FILE = REPO_ROOT / "runs" / "judge__mixed.json"


def paths(version: int = 1) -> dict:
    """Version 1 trains on in-domain replies from every generator; version 2 adds out-of-domain training replies
    (`judge/ood_train.py`) and keeps version 1's holdouts and rule unchanged, so the two compare directly."""
    if version == 1:
        return {"data": DATA_FILE, "record": RECORD, "out": OUT_DIR, "run": RUN_FILE, "tag": "mixed"}
    return {"data": JUDGE_DIR / f"mixed_v{version}.parquet", "record": JUDGE_DIR / f"mixed_v{version}_split.json",
            "out": REPO_ROOT / "checkpoints" / f"judge-mixed-v{version}",
            "run": REPO_ROOT / "runs" / f"judge__mixed_v{version}.json", "tag": f"mixed-v{version}"}
H2_FRACTION = 0.30
H1_TOLERANCE = 0.05
RESAMPLES = 10_000
RULE = ("Serve the mixed judge only if its H1 Spearman is no more than 0.05 below the served judge's on the same "
        "items, and it beats the served judge on both H2 and H3 with the paired bootstrap's 95% interval for the "
        "Spearman difference above zero.")


def _group(instruction: str) -> str:
    return hashlib.sha256(" ".join(str(instruction).split()).lower().encode()).hexdigest()[:16]


def in_domain() -> pd.DataFrame:
    judgments = pd.read_parquet(JUDGE_DIR / "judgments.parquet")
    judgments = judgments[judgments.parsed_ok.astype(bool) & judgments.score.notna()]
    router = pd.DataFrame({
        "item_id": judgments.item_id, "generator": judgments.source.map({"local": "adapter", "frontier": "gpt4o_mini"}),
        "origin": "router_study", "instruction": judgments.instruction, "reply": judgments.reply,
        "score": judgments.score.astype(float), "calibration": judgments.split == "calibration"})
    m2 = pd.concat([pd.read_parquet(JUDGE_DIR / "m2_golden.parquet"),
                    pd.read_parquet(JUDGE_DIR / "m2_golden__frontier.parquet")], ignore_index=True)
    m2 = m2[m2.score.notna()]
    golden = pd.DataFrame({
        "item_id": "m2:" + m2.item_id.astype(str),
        "generator": m2.source.map({"adapter": "adapter", "prompted": "prompted_base", "frontier": "gpt4o_mini"}),
        "origin": "m2_golden", "instruction": m2.instruction, "reply": m2.reply, "score": m2.score.astype(float),
        "calibration": False})
    frame = pd.concat([router, golden], ignore_index=True).assign(domain="in")
    if frame.generator.isna().any():
        msg = "a graded reply has an unknown generator"
        raise ValueError(msg)
    return frame


def out_of_domain() -> pd.DataFrame:
    from adapterops.eval import ood_frontier as of
    from adapterops.eval.ood import OUTPUTS_FILE, SAMPLE_FILE
    from adapterops.eval.ood_label import LABELS_FILE
    from adapterops.judge.label import parse_judgment

    sample = pd.read_parquet(SAMPLE_FILE)
    text = sample.set_index("id").text
    graded = of.graded_ids(sample)
    outputs = pd.read_parquet(OUTPUTS_FILE)
    adapter_reply = {r.id: json.loads(r.output).get("reply", "") for r in outputs[outputs.task == "drafting"].itertuples()}
    labels = pd.read_parquet(LABELS_FILE)
    adapter_grade = {r.id: r.label for r in labels[labels.task == "drafting_grade"].itertuples()}
    cache = of.load_cache()
    mini_reply = {i["id"]: cache[i["key"]]["raw"] for i in of.answer_items(sample)
                  if i["task"] == "drafting" and i["key"] in cache}
    mini_grade = {i["id"]: parse_judgment(cache[i["key"]]["raw"]) for i in of.grade_items(sample, mini_reply)
                  if i["key"] in cache}
    rows = []
    for ticket in graded:
        for generator, replies, grades in (("adapter", adapter_reply, adapter_grade), ("gpt4o_mini", mini_reply, mini_grade)):
            grade = grades.get(ticket)
            if grade is None or pd.isna(grade):
                continue
            rows.append({"item_id": f"ood:{generator}:{ticket}", "generator": generator, "origin": ticket.split(":")[0],
                         "instruction": text[ticket], "reply": replies.get(ticket, ""), "score": float(grade),
                         "calibration": False, "domain": "ood"})
    return pd.DataFrame(rows)


def assign_splits(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    frame = frame.assign(group=frame.instruction.map(_group))
    frame["split"] = "train"
    frame.loc[frame.domain == "ood", "split"] = "h3_out_of_domain"
    frame.loc[frame.calibration & (frame.domain == "in"), "split"] = "h1_adapter_calibration"
    m2_groups = pd.Series(sorted(frame[frame.origin == "m2_golden"].group.unique()))
    held = set(m2_groups.sample(frac=H2_FRACTION, random_state=seed))
    frame.loc[(frame.origin == "m2_golden") & frame.group.isin(held), "split"] = "h2_across_generators"
    # A request whose reply is held out anywhere leaves training entirely, so no request is on both sides.
    held_groups = set(frame[frame.split != "train"].group)
    leaking = (frame.split == "train") & frame.group.isin(held_groups)
    frame.loc[leaking, "split"] = "dropped_shares_a_held_out_request"
    train_groups = pd.Series(sorted(frame[frame.split == "train"].group.unique()))
    val = set(train_groups.sample(frac=0.10, random_state=seed))
    frame.loc[(frame.split == "train") & frame.group.isin(val), "split"] = "validation"
    return frame


def check_disjoint(frame: pd.DataFrame) -> None:
    groups = {s: set(g.group) for s, g in frame.groupby("split")}
    fit = groups.get("train", set()) | groups.get("validation", set())
    for name in ("h1_adapter_calibration", "h2_across_generators", "h3_out_of_domain"):
        if fit & groups.get(name, set()):
            msg = f"{name} shares requests with training"
            raise RuntimeError(msg)
    if groups.get("train", set()) & groups.get("validation", set()):
        msg = "validation shares requests with training"
        raise RuntimeError(msg)


def build() -> int:
    frame = assign_splits(pd.concat([in_domain(), out_of_domain()], ignore_index=True))
    check_disjoint(frame)
    frame.to_parquet(DATA_FILE, index=False)
    counts = frame.groupby(["split", "generator"]).size().unstack(fill_value=0)
    RECORD.write_text(json.dumps({
        "purpose": "Mixed-generator drafting judge: data, splits and decision rule, frozen before training.",
        "file": str(DATA_FILE.relative_to(REPO_ROOT)), "sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "seed": SEED, "counts": {s: row.to_dict() for s, row in counts.iterrows()},
        "preregistered": {"rule": RULE, "h1_tolerance": H1_TOLERANCE, "bootstrap_resamples": RESAMPLES,
                          "known_limits": [("GPT-4o is the teacher and grades GPT-4o-mini, so family preference is "
                                            "in the labels, as it is for the served judge"),
                                           "H3 has 120 replies, so its interval will be wide"]},
    }, indent=2) + "\n", encoding="utf-8")
    print(counts.to_string())
    print(f"  wrote {DATA_FILE.relative_to(REPO_ROOT)} and {RECORD.relative_to(REPO_ROOT)}")
    return 0


def build_v2(seed: int = SEED) -> int:
    """Version 1's frozen frame plus the graded out-of-domain training replies, holdouts untouched."""
    from adapterops.judge.ood_train import GRADED

    v1 = frames_frame(1)
    extra = pd.read_parquet(GRADED).assign(group=lambda f: f.instruction.map(_group))
    held_groups = set(v1[v1.split.str.startswith("h")].group)
    if shared := set(extra.group) & held_groups:
        msg = f"{len(shared)} out-of-domain training requests are also held-out requests"
        raise ValueError(msg)
    groups = pd.Series(sorted(extra.group.unique()))
    val = set(groups.sample(frac=0.10, random_state=seed))
    extra["split"] = ["validation" if g in val else "train" for g in extra.group]
    frame = pd.concat([v1, extra[v1.columns]], ignore_index=True)
    check_disjoint(frame)
    out = paths(2)
    frame.to_parquet(out["data"], index=False)
    counts = frame.groupby(["split", "generator"]).size().unstack(fill_value=0)
    out["record"].write_text(json.dumps({
        "purpose": "Mixed judge, version 2: version 1's data plus graded out-of-domain training replies.",
        "file": str(out["data"].relative_to(REPO_ROOT)), "sha256": hashlib.sha256(out["data"].read_bytes()).hexdigest(),
        "built_on": {"file": str(DATA_FILE.relative_to(REPO_ROOT)), "sha256": json.loads(RECORD.read_text())["sha256"]},
        "holdouts": "identical to version 1, item for item", "seed": seed,
        "counts": {s: row.to_dict() for s, row in counts.iterrows()},
        "preregistered": {"rule": RULE, "h1_tolerance": H1_TOLERANCE, "bootstrap_resamples": RESAMPLES,
                          "known_limits": [("the out-of-domain training tickets share their two sources with H3, so a "
                                            "pass shows those sources' failure was learned, not any new domain")]},
    }, indent=2) + "\n", encoding="utf-8")
    print(counts.to_string())
    return 0


def frames_frame(version: int = 1) -> pd.DataFrame:
    p = paths(version)
    record = json.loads(p["record"].read_text())
    if hashlib.sha256(p["data"].read_bytes()).hexdigest() != record["sha256"]:
        msg = f"{p['data'].name} is not the frozen mixed split"
        raise ValueError(msg)
    return pd.read_parquet(p["data"])


def frames(version: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frames_frame(version)
    return (frame[frame.split == "train"], frame[frame.split == "validation"],
            frame[frame.split == "h1_adapter_calibration"])


def train(version: int = 1) -> int:
    from adapterops.judge.train import train as train_judge

    p = paths(version)
    summary = train_judge(JudgeConfig(tag=p["tag"], output_dir=str(p["out"])), frames=frames(version))
    print(json.dumps(summary["calibration"], indent=2))
    return 0


def paired_spearman_bootstrap(gold: np.ndarray, old: np.ndarray, new: np.ndarray, resamples: int = RESAMPLES,
                              seed: int = SEED) -> dict:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = len(gold)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        a, b = spearmanr(gold[idx], old[idx])[0], spearmanr(gold[idx], new[idx])[0]
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"n": n, "served": round(float(spearmanr(gold, old)[0]), 4), "mixed": round(float(spearmanr(gold, new)[0]), 4),
            "difference": round(float(spearmanr(gold, new)[0] - spearmanr(gold, old)[0]), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def compare(version: int = 1) -> int:
    from adapterops.judge.score import load_judge

    p = paths(version)
    frame = frames_frame(version)
    held = frame[frame.split.str.startswith("h")].reset_index(drop=True)
    served = load_judge(REPO_ROOT / "checkpoints" / "judge")(held.instruction.tolist(), held.reply.tolist())
    mixed = load_judge(p["out"])(held.instruction.tolist(), held.reply.tolist())
    held = held.assign(served=np.asarray(served, dtype=float), mixed=np.asarray(mixed, dtype=float))
    result = {"rule": RULE, "holdouts": {}, "means_by_generator": {}}
    for split, g in held.groupby("split"):
        result["holdouts"][split] = paired_spearman_bootstrap(g.score.to_numpy(), g.served.to_numpy(), g.mixed.to_numpy())
        result["means_by_generator"][split] = {
            gen: {"gpt4o": round(float(x.score.mean()), 3), "served": round(float(x.served.mean()), 3),
                  "mixed": round(float(x.mixed.mean()), 3), "n": len(x)} for gen, x in g.groupby("generator")}
    h = result["holdouts"]
    result["decision"] = {
        "h1_within_tolerance": h["h1_adapter_calibration"]["mixed"] >= h["h1_adapter_calibration"]["served"] - H1_TOLERANCE,
        "h2_better": h["h2_across_generators"]["ci95"][0] > 0,
        "h3_better": h["h3_out_of_domain"]["ci95"][0] > 0}
    result["decision"]["serve_mixed_judge"] = all(result["decision"].values())
    if version > 1:
        # Out-of-domain by source: H3's CFPB half is the same source as the new training tickets, as is its ABCD half.
        g = held[held.split == "h3_out_of_domain"]
        result["h3_by_source"] = {src: paired_spearman_bootstrap(x.score.to_numpy(), x.served.to_numpy(),
                                                                 x.mixed.to_numpy())
                                  for src, x in g.groupby("origin")}
    p["run"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split, r in h.items():
        print(f"  {split:24s} served {r['served']:+.4f} · mixed {r['mixed']:+.4f} · diff {r['difference']:+.4f} CI {r['ci95']}")
    print(f"  decision {result['decision']} · wrote {p['run'].relative_to(REPO_ROOT)}")
    return 0
